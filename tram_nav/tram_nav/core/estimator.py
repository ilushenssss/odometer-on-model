"""Adaptive model-based navigator (the core of the solution).

State vector of the augmented extended Kalman filter::

    x = [ s,  v,  a,  d,  eta ]
          |   |   |   |   +-- inverse relative mass  m_nom / m   (unknown load)
          |   |   |   +------ unmodelled specific force (disturbance) [m/s^2]
          |   |   +---------- actuator state: lagged commanded specific force
          |   +-------------- speed [m/s]
          +------------------ travelled distance along the track [m]

Continuous-time process model::

    s'   = v
    v'   = sat_mu( eta * a )  +  eta * a_tb(u, v)
           - [ c0*sgn(v) + c1*v + eta*c2*v|v|/m_nom + g*sin(theta(s)) + k_c*|kappa(s)| ] / (1+gamma)
           + d  +  r_hat(v, u)                             <- RBF approximator
    a'   = ( a_cmd(u, v) - a ) / tau                       <- traction / brake lag
    d'   = -d / tau_d + w_d                                <- Gauss-Markov disturbance
    eta' = w_eta                                           <- random walk (mass)

``a_cmd`` contains the hyperbolic traction characteristic, ED/friction brake
blending and handle dead-zone; ``sat_mu`` is a smooth adhesion limit with an
*adaptively estimated* adhesion coefficient ``mu_hat``.  At standstill the
holding brake keeps the vehicle at rest (hybrid model).

Measurement model of wheel channel ``i`` (creep-aware)::

    z_i = c_i * omega_i * r_nom / k  =  v * (1 + lambda_i(x))  + n_i

``lambda_i`` - expected micro-slip (creep) of that axle, ``c_i`` - on-line
relative wheel calibration, ``k`` - common odometric scale (mean wheel wear)
estimated in an outer loop from map-aided stop matching.  The noise ``n_i`` is
Huber-robust and Sage-Husa adaptive.  Zero-velocity updates at standstill.

When all odometry is lost the filter keeps integrating the adapted model -
dead reckoning from the controller handle only.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from . import physics as ph
from .fdi import OK, STATUS_CODES, SUSPECT, OdometryFDI
from .params import G, NavigatorParams, TramParams
from .rbf import RBFApproximator
from .track import TrackMap

MODE_NOMINAL, MODE_DEGRADED, MODE_BLIND, MODE_STANDSTILL = "NOMINAL", "DEGRADED", "BLIND", "STANDSTILL"
MODE_CODES = {MODE_NOMINAL: 0, MODE_DEGRADED: 1, MODE_BLIND: 2, MODE_STANDSTILL: 3}


@dataclass
class NavState:
    t: float = 0.0
    s: float = 0.0
    v: float = 0.0
    a: float = 0.0                 # estimated body acceleration
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    sigma_s: float = 0.0
    sigma_v: float = 0.0
    eta: float = 1.0
    mass_est: float = 0.0
    dist: float = 0.0
    mu_hat: float = 0.3
    rbf: float = 0.0
    scale: float = 1.0
    mode: str = MODE_NOMINAL
    blind_time: float = 0.0
    n_used: int = 0
    n_fixes: int = 0
    slip: bool = False
    sensor_status: List[str] = field(default_factory=list)
    sensor_calib: List[float] = field(default_factory=list)
    cov: Optional[np.ndarray] = None


class TramNavigator:
    NX = 5

    def __init__(self, tram: Optional[TramParams] = None, nav: Optional[NavigatorParams] = None,
                 track: Optional[TrackMap] = None, n_sensors: int = 3,
                 sensor_powered: Optional[Sequence[bool]] = None, s0: float = 0.0):
        self.p = tram or TramParams()
        self.np = nav or NavigatorParams()
        self.track = track
        self.n = n_sensors
        self.powered = list(sensor_powered) if sensor_powered is not None else [True] * n_sensors
        self.fdi = OdometryFDI(n_sensors, self.np, self.powered, self.p.v_max)
        self.rbf = RBFApproximator(self.p.v_max, self.np.rbf_v_centers, self.np.rbf_u_centers,
                                   self.np.rbf_forget) if self.np.rbf_enable else None
        n = self.np
        self.x = np.array([s0, 0.0, 0.0, 0.0, 1.0])
        self.P = np.diag([n.p0_pos, n.p0_vel, n.p0_trac, n.p0_dist, n.p0_eta]).astype(float)
        self.Qd = np.array([n.q_pos, n.q_vel, n.q_trac, n.q_dist, n.q_eta])
        self.R = [n.r_wheel] * n_sensors
        self.calib = [1.0] * n_sensors
        self.k = 1.0                          # common odometric scale
        self.sigma_k = math.sqrt(n.p0_scale)
        self.mu_hat = n.mu_prior
        self.u = 0.0
        self.u_raw = 0.0
        self._u_hist = collections.deque()
        self.readings: List[Optional[tuple]] = [None] * n_sensors
        self.t: Optional[float] = None
        self.gamma1 = 1.0 + self.p.rot_mass_factor
        self.blind_time = 0.0
        self.stop_time = 0.0
        self._stop_inflated = False
        self._snapped = False
        self._standstill = False
        self._slip_since_fix = 0.0           # accumulated slip time since the last fix
        self._last_slip_t = -1e9
        self._snapped_blind = False
        self._held = False
        self._held_time = 0.0
        self.n_fixes = 0
        self._fix_var = 0.0
        self._fix = None                      # (s_station, s_est_after_fix, blind)
        self._k_step = 0
        self._rbf_val = 0.0
        self._I = np.eye(self.NX)
        self.state = NavState(mu_hat=self.mu_hat, sensor_status=["DROPOUT"] * n_sensors,
                              sensor_calib=list(self.calib), s=s0)

    # ------------------------------------------------------------------ inputs
    def set_handle(self, u: float) -> None:
        if u == u:  # not NaN
            self.u_raw = max(-1.0, min(1.0, float(u)))

    def _delayed_handle(self, t: float) -> float:
        """Handle position ``handle_delay`` seconds ago (transport delay)."""
        hist = self._u_hist
        hist.append((t, self.u_raw))
        td = t - self.np.handle_delay
        while len(hist) > 1 and hist[1][0] <= td + 1e-9:
            hist.popleft()
        return hist[0][1]

    def set_wheel_speed(self, i: int, stamp: float, omega: float) -> None:
        """Angular wheel speed [rad/s] of channel ``i`` measured at ``stamp``."""
        if 0 <= i < self.n:
            self.readings[i] = (stamp, float(omega) * self.p.wheel_radius)

    def reset(self, s0: float = 0.0, v0: float = 0.0) -> None:
        n = self.np
        self.x[:] = [s0, v0, 0.0, 0.0, 1.0]
        self.P = np.diag([n.p0_pos, n.p0_vel, n.p0_trac, n.p0_dist, n.p0_eta]).astype(float)
        self._fix = None

    # ------------------------------------------------------------------ model
    def _a_cmd(self, u: float, v: float):
        Fw, Ft = ph.commanded_force(u, v, self.p)
        m = self.p.mass_nominal * self.gamma1
        return Fw / m, Ft / m

    def _adh_limit(self, a_wheel: float, v: float) -> float:
        p = self.p
        na, npw = p.n_axles, p.n_powered
        if a_wheel >= 0.0:
            frac = npw / na
        else:
            ed = ph.ed_share(v, p)
            frac = 1.0 / (ed * na / npw + (1.0 - ed))
        return self.mu_hat * G * frac / self.gamma1

    def _dynamics(self, x, u, grade, kappa):
        """Right-hand side of the process model and its Jacobian."""
        p, n = self.p, self.np
        s, v, a, d, eta = x
        a_cmd, a_tb = self._a_cmd(u, v)
        aw = eta * a
        lim = self._adh_limit(aw, v)
        sat, dsat = ph.smooth_sat(aw, lim)
        c2m = p.res_c2 / p.mass_nominal
        res = (p.res_c0 * ph.motion_sign(v) + p.res_c1 * v + eta * c2m * v * abs(v)
               + G * math.sin(grade) + p.curve_coeff * abs(kappa) * ph.motion_sign(v, 0.2)) / self.gamma1
        vdot = sat + eta * a_tb - res + d + self._rbf_val
        tau = p.traction_tau if a_cmd >= 0.0 else p.brake_tau
        adot = (a_cmd - a) / tau
        ddot = -d / n.dist_decay_tau
        J = np.zeros((5, 5))
        J[0, 1] = 1.0
        J[2, 2] = -1.0 / tau
        J[3, 3] = -1.0 / n.dist_decay_tau
        # standstill: the holding brake keeps the tram at rest unless the
        # tractive effort overcomes the resistance; it never rolls backwards
        if v < 0.05 and (u <= p.handle_deadzone or vdot < 0.0):
            self._held = True
            return 0.0, adot, ddot, J
        self._held = False
        dms = (1.0 / 0.05) if abs(v) < 0.05 else 0.0
        J[1, 1] = -(p.res_c0 * dms + p.res_c1 + 2.0 * eta * c2m * abs(v)) / self.gamma1
        J[1, 2] = dsat * eta
        J[1, 3] = 1.0
        J[1, 4] = dsat * a + a_tb - c2m * v * abs(v) / self.gamma1
        h = 0.05
        a_cmd2, _ = self._a_cmd(u, v + h)
        J[2, 1] = (a_cmd2 - a_cmd) / h / tau
        return vdot, adot, ddot, J

    def _creep(self, x) -> List[float]:
        """Expected micro-slip (creep) of every sensed axle.

        Axles transmitting tractive / braking force roll slightly faster /
        slower than the body.  Inverse of the rising branch of the creep curve
        with the current adhesion estimate:
        ``lambda = lambda_p * (1 - sqrt(1 - F_axle / (mu_hat * N_axle)))``.
        """
        p = self.p
        lp = self.np.creep_slip_peak
        mu = max(self.mu_hat, 0.03)
        spec = x[4] * x[2] * self.gamma1 / G       # total wheel force / (m g)
        if abs(spec) < 1e-4 or x[1] < 0.3:
            return [0.0] * self.n
        na, npw = p.n_axles, p.n_powered
        out = []
        for i in range(self.n):
            if spec > 0.0:
                r = spec * na / npw if self.powered[i] else 0.0
            else:
                ed = ph.ed_share(x[1], p)
                r = -spec * ((ed * na / npw if self.powered[i] else 0.0) + (1.0 - ed))
            lam = lp * (1.0 - math.sqrt(1.0 - min(r / mu, 0.999)))
            out.append(lam if spec > 0.0 else -lam)
        return out

    def _kf_update(self, H: np.ndarray, e: float, R: float, freeze_pos: bool = False) -> None:
        P = self.P
        if not freeze_pos and H.sum() == 1.0 and H.max() == 1.0:
            # unit measurement vector with the optimal gain: cheap standard form
            j = int(H.argmax())
            S = P[j, j] + R
            K = P[:, j] / S
            self.x += K * e
            self.P = P - np.outer(K, P[j, :])
            return
        PH = P @ H
        S = float(H @ PH) + R
        K = PH / S
        if freeze_pos:
            K[0] = 0.0
        self.x += K * e
        IKH = self._I - np.outer(K, H)
        self.P = IKH @ P @ IKH.T + np.outer(K, K) * R   # Joseph form (valid for any gain)

    # ------------------------------------------------------------------ main step
    def step(self, t: float) -> NavState:
        if self.t is None:
            self.t = t
            dt = 1.0 / self.np.rate
        else:
            dt = t - self.t
            if dt <= 0.0:
                return self.state
            self.t = t
            dt = min(dt, 0.2)
        n, p = self.np, self.p
        x = self.x
        u = self.u = self._delayed_handle(t)
        grade = kappa = 0.0
        if self.track is not None and n.use_track_grade:
            grade = self.track.grade_at(x[0])
            kappa = self.track.curvature_at(x[0])

        # ---------------- prediction ----------------
        if self.rbf is not None:
            self._rbf_val = self.rbf.predict(x[1], u)
        vdot, adot, ddot, J = self._dynamics(x, u, grade, kappa)
        F = self._I + J * dt
        F[0, :] += 0.5 * dt * dt * J[1, :]
        x[0] += x[1] * dt + 0.5 * vdot * dt * dt
        x[1] += vdot * dt
        x[2] += adot * dt
        x[3] += ddot * dt
        Q = self.Qd * dt
        slipping = self.fdi.any_slip(t)
        if slipping:
            self._slip_since_fix += dt
            self._last_slip_t = t
        if slipping:  # adhesion-limited operation: the model is less certain
            Q = Q.copy()
            Q[1] *= 10.0
            Q[3] *= 5.0
        P = F @ self.P @ F.T
        P[np.diag_indices(5)] += Q
        self.P = P
        x[1] = max(0.0, x[1])

        # ---------------- fault detection ----------------
        aw = x[2] * x[4]
        mode = 1 if aw > 0.05 else (-1 if aw < -0.05 else 0)
        creep = self._creep(x)
        readings = []
        for i in range(self.n):
            rd = self.readings[i]
            if rd is not None:
                z = rd[1] * self.calib[i] / self.k / (1.0 + creep[i]) + vdot * n.meas_delay
                rd = (rd[0], z, rd[1])
            readings.append(rd)
        dec = self.fdi.assess(t, dt, readings, x[1], P[1, 1], vdot, mode, self.R)
        if self.fdi.reference_faulted:
            self.mu_hat = n.mu_prior      # the "slips" were an artefact of the faulty reference
        self._adapt_adhesion(mode, dt)

        # ---------------- standstill detection ----------------
        # (with hysteresis: enter when all channels ~0, leave on motion or traction)
        fresh = sorted(abs(d.value) for d in dec if d.status in (OK, SUSPECT) and d.new)
        if not fresh or u > p.handle_deadzone:
            self._standstill = False
        elif self._standstill:
            self._standstill = fresh[len(fresh) // 2] < n.zupt_exit_speed
        else:
            self._standstill = fresh[-1] < n.zupt_speed and x[1] < 1.0
        standstill = self._standstill

        # ---------------- measurement update ----------------
        n_used = 0
        H = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        if standstill:
            # zero-velocity update; position is held (no creeping at stops)
            self._kf_update(H, -x[1], 1e-4, freeze_pos=True)
            n_used = len(fresh)
            self.stop_time += dt
            if self.stop_time > 5.0 and not self._stop_inflated:
                self.P[4, 4] += n.stop_eta_inflation  # passengers boarding -> mass may change
                self._stop_inflated = True
            if (n.station_snap and not self._snapped and self.track is not None
                    and self.track.stations and self.stop_time > n.snap_min_stop):
                self._snapped = True
                self._station_fix()
        else:
            self.stop_time = 0.0
            self._stop_inflated = False
            self._snapped = False
            if self.blind_time > n.blind_decorrelate_time and any(d.accept for d in dec):
                # odometry is back after a long blind phase: a noisy speed snapshot
                # must not rewrite the dead-reckoned distance through the (poorly
                # known) s-v correlation -> keep sigma_s, drop the correlation
                self.P[0, 1:] = 0.0
                self.P[1:, 0] = 0.0
            for i, d in enumerate(dec):
                if not d.accept:
                    continue
                e = readings[i][1] - x[1]
                R = max(self.R[i], self.fdi.noise_var(i)) * d.r_scale
                S = self.P[1, 1] + R
                nu = abs(e) / math.sqrt(S)
                if nu > n.huber_k:  # Huber weighting
                    R *= nu / n.huber_k
                self._kf_update(H, e, R)
                n_used += 1
                if d.r_scale == 1.0:  # Sage-Husa adaptation of the channel noise
                    r_new = (1 - n.r_adapt_forget) * self.R[i] + n.r_adapt_forget * max(e * e - S + self.R[i], 0.0)
                    self.R[i] = min(max(r_new, n.r_wheel_min), n.r_wheel_max)
        if n_used:
            self.blind_time = 0.0
        else:
            self.blind_time += dt
        # blind stop: brakes applied and the model has come to rest -> the tram
        # stands still (it cannot be pushed backwards by the brakes)
        self._held_time = self._held_time + dt if (self._held and u < -p.handle_deadzone) else 0.0
        if self.blind_time > 1.0 and self._held_time > 2.0:
            self._kf_update(H, -x[1], n.blind_zupt_r, freeze_pos=True)
            if (n.station_snap and not self._snapped_blind and self.track is not None and self.track.stations
                    and self._held_time > n.snap_min_stop_blind):
                self._snapped_blind = True
                self._station_fix(blind=True)
        elif not self._held:
            self._snapped_blind = False

        # ---------------- constraints ----------------
        x[1] = max(0.0, x[1])
        x[3] = min(max(x[3], -n.dist_bound), n.dist_bound)
        x[4] = min(max(x[4], n.eta_bounds[0]), n.eta_bounds[1])
        self.P = 0.5 * (self.P + self.P.T)

        # ---------------- online learning ----------------
        self._k_step += 1
        if n_used and not slipping and not standstill:
            self._calibrate(dec, creep, vdot)
            coasting = abs(x[2] * x[4]) < n.rbf_coast_accel or not n.rbf_coast_only
            if (self.rbf is not None and x[1] > n.rbf_min_speed and self._k_step % n.rbf_update_every == 0
                    and self.P[3, 3] < 0.02 and coasting):
                self.rbf.update(x[1], u, x[3] + self._rbf_val)

        return self._output(t, vdot, dec, n_used, slipping, standstill)

    # ------------------------------------------------------------------ adaptation
    def _adapt_adhesion(self, mode: int, dt: float) -> None:
        """Adhesion limit: a slip onset proves that the demanded axle force
        exceeded the available adhesion -> ``mu_hat`` is an upper bound."""
        p, n, x = self.p, self.np, self.x
        for i in self.fdi.slip_onsets:
            if mode == 0 or (mode > 0 and not self.powered[i]):
                continue
            demand = abs(x[4] * x[2]) * self.gamma1
            if mode > 0:
                demand *= p.n_axles / p.n_powered
            else:
                ed = ph.ed_share(x[1], p)
                demand *= ed * p.n_axles / p.n_powered + (1.0 - ed)
            # bounded step: one event can lower the estimate by at most 30 %
            self.mu_hat = max(n.mu_min, 0.7 * self.mu_hat, min(self.mu_hat, 0.9 * demand / G))
        self.mu_hat += (n.mu_prior - self.mu_hat) * dt / n.mu_recover_tau

    def _calibrate(self, dec, creep, vdot) -> None:
        """Relative wheel-radius calibration (only the relative scale is
        observable from odometry; the common scale ``k`` comes from fixes)."""
        n, x = self.np, self.x
        used = [i for i, d in enumerate(dec) if d.accept and self.readings[i] is not None]
        if len(used) < 2 or x[1] < 3.0 or abs(vdot) > 0.3:
            return
        for i in used:
            raw = self.readings[i][1] / self.k / (1.0 + creep[i])
            if raw > 1.0:
                self.calib[i] += n.calib_gain * (x[1] / raw - self.calib[i])
        mean_c = sum(self.calib) / self.n
        self.calib = [min(max(c / mean_c, 1 - n.calib_bound), 1 + n.calib_bound) for c in self.calib]

    def _station_fix(self, blind: bool = False) -> None:
        """Map-aided stop matching.

        A long stop close to a platform from the track map is treated as a
        position measurement ``s = s_station``.  Consecutive fixes also give the
        common odometric scale ``k`` = odometer distance / map distance.
        """
        n, x = self.np, self.x
        st = self.track.stations
        s_near = min(st, key=lambda v: abs(v - x[0]))
        e = s_near - x[0]
        var_scale = (self.sigma_k * self._dist_since_fix()) ** 2
        # after slip / slide the stopping point at the platform is less certain
        recent_slip = self.t is not None and self.t - self._last_slip_t < n.snap_slip_window
        R = (n.snap_sigma * (n.snap_slip_factor if recent_slip else 1.0)) ** 2
        S = self.P[0, 0] + var_scale + R
        gate = n.snap_gate_blind if blind else n.snap_gate_max
        if abs(e) > gate:
            return
        if e * e > n.snap_nsigma ** 2 * S:
            # robust (Huber-like) fix: never discard a plausible stop, but
            # inflate its noise so that the innovation sits at n_sigma
            R = e * e / n.snap_nsigma ** 2 - self.P[0, 0] - var_scale
            S = self.P[0, 0] + var_scale + R
        # scale estimation from two consecutive fixes
        if self._fix is not None and not blind and not self._fix[2] and self._slip_since_fix < n.scale_max_slip:
            L = s_near - self._fix[0]
            D = x[0] - self._fix[1]
            if L > 150.0 and D > 0.0:
                rho = D / L
                if abs(rho - 1.0) < n.scale_gate:
                    r_rho = 2.0 * (n.snap_scale_sigma / L) ** 2 + self._fix_var / L ** 2
                    w = self.sigma_k ** 2 / (self.sigma_k ** 2 + r_rho)
                    step = max(-n.scale_max_step, min(n.scale_max_step, w * (rho - 1.0)))
                    self.k *= 1.0 + step
                    self.k = min(max(self.k, n.scale_bounds[0]), n.scale_bounds[1])
                    self.sigma_k = max(math.sqrt(1.0 - w) * self.sigma_k, n.scale_sigma_min)
        self.P[0, 0] += var_scale
        self._kf_update(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), e, R)
        self._fix_var = float(self.P[0, 0])
        self._fix = (s_near, float(self.x[0]), blind)
        self._slip_since_fix = 0.0
        self.n_fixes += 1

    def _dist_since_fix(self) -> float:
        if self._fix is None:
            return abs(self.x[0])
        return abs(self.x[0] - self._fix[1])

    # ------------------------------------------------------------------ output
    def _output(self, t, vdot, dec, n_used, slipping, standstill) -> NavState:
        x, P, p = self.x, self.P, self.p
        st = self.state
        st.t, st.s, st.v, st.a = t, float(x[0]), float(x[1]), float(vdot)
        var_s = P[0, 0] + (self.sigma_k * self._dist_since_fix()) ** 2
        st.sigma_s = math.sqrt(max(var_s, 0.0))
        st.sigma_v = math.sqrt(max(P[1, 1] + (self.sigma_k * x[1]) ** 2, 0.0))
        st.eta, st.dist, st.mu_hat, st.rbf = float(x[4]), float(x[3]), self.mu_hat, self._rbf_val
        st.mass_est = p.mass_nominal / float(x[4])
        st.scale = self.k
        st.n_used = n_used
        st.n_fixes = self.n_fixes
        st.slip = slipping
        st.blind_time = self.blind_time
        st.sensor_status = [d.status for d in dec]
        st.sensor_calib = list(self.calib)
        st.cov = P
        if standstill:
            st.mode = MODE_STANDSTILL
        elif self.blind_time > 0.5:
            st.mode = MODE_BLIND
        elif all(d.status == OK for d in dec):
            st.mode = MODE_NOMINAL
        else:
            st.mode = MODE_DEGRADED
        if self.track is not None:
            st.x, st.y, st.yaw = self.track.pose(st.s)
        else:
            st.x, st.y, st.yaw = st.s, 0.0, 0.0
        return st

    @staticmethod
    def status_code(status: str) -> int:
        return STATUS_CODES.get(status, 9)

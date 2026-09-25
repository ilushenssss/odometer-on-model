"""High-fidelity "ground truth" tram simulator.

This is *not* used by the navigator - it replaces the real vehicle for
development, testing and demonstration.  It deliberately contains more
physics than the navigator's model:

* per-axle wheel dynamics with the nonlinear creep-force curve; hybrid
  creep / macro-slip model -> realistic wheel slip in traction and wheel
  slide in braking on low adhesion;
* on-board anti-slip / wheel-slide-protection (WSP) controller which modulates
  the axle torque (sawtooth wheel-speed oscillations);
* spatially correlated random adhesion, weather transitions;
* actuator lags + transport delay, ED/friction brake blending, track brake;
* unknown and changing mass (passenger exchange), traction degradation,
  resistance uncertainty, individually worn wheels;
* incremental encoders with pulse quantisation, noise and injected faults.
"""
from __future__ import annotations

import collections
import math
from typing import List, Optional

import numpy as np

from . import physics as ph
from .driver import DriverModel
from .params import G, WEATHER, AdhesionParams, ScenarioConfig, SensorParams, TramParams
from .track import TrackMap


class TramSimulator:
    def __init__(self, track: TrackMap, scenario: Optional[ScenarioConfig] = None,
                 params: Optional[TramParams] = None, sensors: Optional[SensorParams] = None,
                 dt: float = 0.005, driver: Optional[DriverModel] = None):
        self.track = track
        self.sc = scenario or ScenarioConfig()
        self.p = params or TramParams()
        self.sp = sensors or SensorParams()
        self.dt = dt
        self.rng = np.random.default_rng(self.sc.seed)
        self.driver = driver if driver is not None else DriverModel(track, self.p, self.rng)

        na = self.p.n_axles
        self.t = 0.0
        self.s = track.stations[0] if track.stations else 0.0
        self.v = 0.0
        self.a = 0.0
        self.mass = self.sc.mass
        self.eff = self.sc.traction_efficiency
        self.u = 0.0
        self.F_w = 0.0                 # lagged wheel force demand
        self.F_tb = 0.0                # lagged track brake force
        self.vw = [0.0] * na           # wheel peripheral speeds
        self.slip_mode = [False] * na
        self.k_as = [1.0] * na         # anti-slip torque modulation
        self.theta = [0.0] * na        # wheel rotation angles
        self.mu_fluct = 0.0
        self.mu_fluct_axle = [0.0] * na
        self.delay_buf = collections.deque([0.0] * max(1, int(0.1 / dt)))
        self.m_eq = [1400.0 if i in self.p.powered_axles else 450.0 for i in range(na)]
        if self.sc.wheel_wear:
            dr = self.rng.normal(-0.004, self.sp.radius_error_std, na)
        else:
            dr = np.zeros(na)
        self.r_true = [self.p.wheel_radius + float(d) for d in dr]
        self._ap_cur = self._weather_at(0.0)
        # sensors
        self._pulse = 2.0 * math.pi / self.sp.pulses_per_rev
        nwin = max(1, int(round(self.sp.window * self.sp.rate)))
        self._counts = [collections.deque([0] * (nwin + 1), maxlen=nwin + 1) for _ in self.sp.axles]
        self._last_meas: List[Optional[float]] = [0.0] * len(self.sp.axles)
        self._stop_timer = 0.0
        self._mass_changed_at_stop = False
        self._emerg = sorted(self.sc.emergency_brakes)
        self._mass_events = sorted(self.sc.mass_changes)

    # ------------------------------------------------------------------ helpers
    def _weather_at(self, t: float) -> AdhesionParams:
        prof = self.sc.weather_profile
        cur = WEATHER[prof[0][1]]
        for k, (tk, name) in enumerate(prof):
            if t >= tk:
                prev = cur
                cur = WEATHER[name]
                blend = min(1.0, (t - tk) / 20.0) if k > 0 else 1.0
                if blend < 1.0:
                    cur = AdhesionParams(
                        mu_max=prev.mu_max + blend * (cur.mu_max - prev.mu_max),
                        slip_peak=cur.slip_peak,
                        slide_ratio=prev.slide_ratio + blend * (cur.slide_ratio - prev.slide_ratio),
                        slip_decay=cur.slip_decay, speed_coeff=cur.speed_coeff,
                        fluctuation=prev.fluctuation + blend * (cur.fluctuation - prev.fluctuation),
                        fluct_length=cur.fluct_length)
        return cur

    @staticmethod
    def _schedule(t: float, base: float, changes) -> float:
        val = base
        for tk, vk in changes:
            if t >= tk:
                val = vk
        return val

    @property
    def weather_name(self) -> str:
        name = self.sc.weather_profile[0][1]
        for tk, n in self.sc.weather_profile:
            if self.t >= tk:
                name = n
        return name

    # ------------------------------------------------------------------ dynamics
    def step(self, u_ext: Optional[float] = None) -> None:
        p, dt = self.p, self.dt
        t = self.t
        if int(t / 0.1) != int((t - dt) / 0.1) or t == 0.0:
            self._ap_cur = self._weather_at(t)
        ap = self._ap_cur
        while self._mass_events and t >= self._mass_events[0][0]:
            self.mass = self._mass_events.pop(0)[1]
        self.eff = self._schedule(t, self.sc.traction_efficiency, self.sc.efficiency_changes)

        # --- driver / handle ------------------------------------------------
        if u_ext is None:
            u = self.driver.update(t, self.s, self.v, dt)
            while self._emerg and t >= self._emerg[0]:
                self.driver.force_emergency(t, 4.0)
                self._emerg.pop(0)
        else:
            u = u_ext
        self.u = u
        self.delay_buf.append(u)
        u_d = self.delay_buf.popleft()

        # --- actuators ------------------------------------------------------
        F_cmd, F_tb_cmd = ph.commanded_force(u_d, self.v, p)
        if F_cmd > 0.0:
            F_cmd *= self.eff
        tau = p.traction_tau if F_cmd > self.F_w and F_cmd > 0 else p.brake_tau
        self.F_w += (F_cmd - self.F_w) * min(1.0, dt / tau)
        self.F_tb += (F_tb_cmd - self.F_tb) * min(1.0, dt / 0.3)

        # --- axle force demands --------------------------------------------
        na, npw = p.n_axles, p.n_powered
        demands = [0.0] * na
        if self.F_w >= 0.0:
            for i in p.powered_axles:
                demands[i] = self.F_w / npw
        else:
            B = -self.F_w
            ed = ph.ed_share(self.v, p)
            for i in range(na):
                share = B * (1.0 - ed) / na
                if i in p.powered_axles:
                    share += B * ed / npw
                demands[i] = -share

        # --- adhesion field ------------------------------------------------
        ds = self.v * dt
        if ds > 0.0:
            L = ap.fluct_length
            a_ = ds / L
            self.mu_fluct += -self.mu_fluct * a_ + math.sqrt(2 * a_) * float(self.rng.standard_normal())
            for i in range(na):
                self.mu_fluct_axle[i] += (-self.mu_fluct_axle[i] * a_ * 4
                                          + math.sqrt(8 * a_) * float(self.rng.standard_normal()))
        N = self.mass * G / na
        v = self.v
        F_adh_total = 0.0
        for i in range(na):
            local = math.exp(ap.fluctuation * (self.mu_fluct + 0.3 * self.mu_fluct_axle[i]))
            mu_pk = ph.mu_peak(v, ap, local)
            cap = mu_pk * N
            Fd = demands[i] * self.k_as[i]
            if not self.slip_mode[i]:
                if abs(Fd) <= cap:
                    F_adh = Fd
                    lam = ph.creep_slip_for_force(Fd / cap if cap > 0 else 0.0, ap)
                    self.vw[i] = max(0.0, v * (1.0 + lam))
                else:
                    self.slip_mode[i] = True
            if self.slip_mode[i]:
                vref = max(v, 0.5)
                lam = (self.vw[i] - v) / vref
                if abs(lam) <= ap.slip_peak:
                    F_adh = math.copysign(cap, Fd if Fd != 0 else lam)
                else:
                    F_adh = N * ph.creep_curve(lam, mu_pk, ap)
                self.vw[i] += (Fd - F_adh) / self.m_eq[i] * dt
                self.vw[i] = min(max(self.vw[i], 0.0), 3.0 * v + 5.0)
                lam = (self.vw[i] - v) / vref
                if abs(Fd) <= cap and (abs(lam) <= ap.slip_peak or lam * Fd <= 0):
                    self.slip_mode[i] = False
            # on-board anti-slip / WSP controller (uses the true slip)
            lam_meas = (self.vw[i] - v) / max(v, 1.0)
            if (abs(lam_meas) > 0.03 and v > 0.3) or abs(self.vw[i] - v) > 0.4:
                self.k_as[i] = max(0.02, self.k_as[i] - (2.0 + 20.0 * abs(lam_meas)) * dt)
            else:
                self.k_as[i] = min(1.0, self.k_as[i] + 0.6 * dt)
            F_adh_total += F_adh

        # --- body ------------------------------------------------------------
        grade = self.track.grade_at(self.s)
        kappa = self.track.curvature_at(self.s)
        F_res = ph.resistance_force(v, self.mass, p, self.sc.resistance_scale)
        F = F_adh_total + self.F_tb - F_res - self.mass * ph.grade_curve_accel(v, grade, kappa, p)
        m_eff = self.mass * (1.0 + p.rot_mass_factor)
        a = F / m_eff
        v_new = v + a * dt
        if v_new <= 0.0:
            v_new = 0.0
            a = -v / dt if v > 0 else 0.0
            if v_new == 0.0 and self.F_w <= 0.0:
                for i in range(na):
                    self.vw[i] = 0.0
                    self.slip_mode[i] = False
        self.a = a
        self.s += 0.5 * (v + v_new) * dt
        self.v = v_new
        for i in range(na):
            self.theta[i] += self.vw[i] / self.r_true[i] * dt

        # --- passenger exchange at stops ------------------------------------
        if self.v == 0.0:
            self._stop_timer += dt
            if self.sc.passenger_exchange and self._stop_timer > 5.0 and not self._mass_changed_at_stop:
                self.mass = float(np.clip(self.mass + self.rng.normal(0.0, 4000.0),
                                          p.mass_empty, p.mass_empty + 20_000.0))
                self._mass_changed_at_stop = True
        else:
            self._stop_timer = 0.0
            self._mass_changed_at_stop = False
        self.t = t + dt

    # ------------------------------------------------------------------ sensors
    def sample_sensors(self) -> List[Optional[float]]:
        """Angular speeds [rad/s] of the encoders (``None`` = no message).

        Must be called at ``SensorParams.rate``.
        """
        out: List[Optional[float]] = []
        t = self.t
        for k, axle in enumerate(self.sp.axles):
            cnt = int(math.floor(self.theta[axle] / self._pulse))
            buf = self._counts[k]
            buf.append(cnt)
            omega = (buf[-1] - buf[0]) * self._pulse / self.sp.window
            omega += float(self.rng.normal(0.0, self.sp.noise_std)) / self.p.wheel_radius
            val: Optional[float] = omega
            for f in self.sc.faults:
                if f.sensor != k or not (f.t_start <= t < f.t_end):
                    continue
                if f.kind == "dropout":
                    val = None
                elif f.kind == "stuck":
                    val = self._last_meas[k]
                elif f.kind == "zero":
                    val = 0.0
                elif f.kind == "scale" and val is not None:
                    val *= f.value
                elif f.kind == "noise" and val is not None:
                    val += float(self.rng.normal(0.0, f.value)) / self.p.wheel_radius
                elif f.kind == "spikes" and val is not None:
                    if self.rng.random() < (f.value or 0.05):
                        val += float(self.rng.choice([-1, 1]) * self.rng.uniform(2, 8)) / self.p.wheel_radius
            if val is not None and not any(f.kind == "stuck" and f.sensor == k and f.t_start <= t < f.t_end
                                           for f in self.sc.faults):
                self._last_meas[k] = val
            out.append(val)
        return out

    def truth(self) -> dict:
        x, y, yaw = self.track.pose(self.s)
        return dict(t=self.t, s=self.s, v=self.v, a=self.a, x=x, y=y, yaw=yaw, mass=self.mass,
                    u=self.u, wheel=list(self.vw), slip=list(self.slip_mode), weather=self.weather_name,
                    eff=self.eff)

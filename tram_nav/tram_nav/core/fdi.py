"""Fault detection and isolation (FDI) for the wheel-speed sensors.

Each odometry channel passes a cascade of tests every navigation cycle:

1. **Freshness** - no message for ``timeout`` seconds -> ``DROPOUT``.
2. **Validity** - NaN / inf / out of the physical range -> ``INVALID``.
3. **Stuck-at** - bit-identical value for ``stuck_window`` while the model
   says the tram moves -> ``STUCK``.
4. **Wheel slip / slide** - wheel acceleration (0.2 s baseline) differs from
   the model-predicted body acceleration by more than ``slip_accel``; or the
   wheel is faster (traction) / slower (braking) than the consensus on a
   motored axle.  Slip is a *transient* condition: the channel is excluded
   for ``slip_hold`` seconds after the last indication -> ``SLIP``.
5. **Consensus** - deviation from the median of the healthy channels.
6. **Innovation gate** - normalised innovation squared against the EKF
   prediction, with an asymmetric (slip-aware) threshold.

Persistent inconsistencies (4.-6. not explained by slip) for more than
``fault_confirm`` seconds latch the channel as ``FAULT``; it is released after
``recover_time`` seconds of consistent behaviour.

If every channel has been rejected for ``reacquire_time`` seconds (e.g. the
prediction drifted during a long all-axle slide), channels that agree with each
other, show no slip and lie within a *wide* gate are force-accepted with an
inflated noise so that the filter re-converges instead of locking itself out.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .params import NavigatorParams

OK, SLIP, SUSPECT, FAULT, DROPOUT, STUCK, INVALID = "OK", "SLIP", "SUSPECT", "FAULT", "DROPOUT", "STUCK", "INVALID"
STATUS_CODES = {OK: 0, SLIP: 1, SUSPECT: 2, FAULT: 3, DROPOUT: 4, STUCK: 5, INVALID: 6}


@dataclass
class Decision:
    accept: bool = False
    r_scale: float = 1.0
    status: str = DROPOUT
    nis: float = 0.0
    new: bool = False       # a new measurement was available this cycle
    value: float = 0.0


class _Channel:
    __slots__ = ("hist", "raw", "last_stamp", "slip_until", "bad_time", "good_time", "latched", "status",
                 "slip_start", "slip_confirmed", "slip_accel_seen", "noise_var", "prev")

    def __init__(self):
        self.hist = collections.deque(maxlen=200)
        self.raw = collections.deque(maxlen=200)
        self.last_stamp = -1e9
        self.slip_until = -1e9
        self.bad_time = 0.0
        self.good_time = 0.0
        self.latched = False
        self.status = DROPOUT
        self.slip_start = -1e9
        self.slip_confirmed = False
        self.slip_accel_seen = False
        self.noise_var = 0.0     # on-line estimate of the channel noise variance
        self.prev = None


class OdometryFDI:
    def __init__(self, n: int, np_: NavigatorParams, powered: Sequence[bool], v_max: float = 22.0):
        self.n = n
        self.p = np_
        self.powered = list(powered)
        self.v_max = v_max
        self.ch = [_Channel() for _ in range(n)]
        self.blind_time = 0.0
        self.slip_onsets: List[int] = []   # channels whose slip was confirmed this cycle
        self.reference_faulted = False

    # ------------------------------------------------------------------
    def _deriv(self, c: _Channel, t: float, base: float = 0.2) -> Optional[float]:
        if len(c.hist) < 2:
            return None
        t1, z1 = c.hist[-1]
        for (t0, z0) in reversed(c.hist):
            if t1 - t0 >= base:
                return (z1 - z0) / (t1 - t0)
        return None

    def _stuck(self, c: _Channel, t: float) -> bool:
        if len(c.raw) < 5:
            return False
        z_last = c.raw[-1][1]
        t_first = None
        for (tk, zk) in reversed(c.raw):
            if zk != z_last:
                return False
            t_first = tk
            if t - tk >= self.p.stuck_window:
                return True
        return t_first is not None and t - t_first >= self.p.stuck_window

    # ------------------------------------------------------------------
    def assess(self, t: float, dt: float, readings: Sequence[Optional[Tuple[float, float]]],
               v_pred: float, var_pred: float, a_model: float, mode: int,
               r_meas: Sequence[float]) -> List[Decision]:
        """Classify all channels.

        ``readings[i]`` = ``(stamp, speed_mps[, raw_value])`` of the latest message
        or None (``raw_value`` - the unprocessed reading, used by the stuck test).
        ``mode``: +1 traction, -1 braking, 0 coasting.
        """
        p = self.p
        dec = [Decision() for _ in range(self.n)]
        self.slip_onsets = []
        cand = []
        for i, c in enumerate(self.ch):
            d = dec[i]
            rd = readings[i]
            if rd is None or t - rd[0] > p.timeout:
                c.status = DROPOUT
                d.status = DROPOUT
                c.hist.clear()
                c.raw.clear()
                continue
            stamp, z = rd[0], rd[1]
            d.new = stamp > c.last_stamp
            d.value = z
            if d.new:
                c.last_stamp = stamp
                if not math.isfinite(z):
                    c.status = d.status = INVALID
                    continue
                # model-free noise estimate from first differences (the model
                # acceleration removes the deterministic part)
                if c.prev is not None and 0.0 < stamp - c.prev[0] < 0.2:
                    dz = z - c.prev[1] - a_model * (stamp - c.prev[0])
                    c.noise_var += p.noise_est_alpha * (min(0.5 * dz * dz, 4.0) - c.noise_var)
                c.prev = (stamp, z)
                c.hist.append((stamp, z))
                c.raw.append((stamp, rd[2] if len(rd) > 2 else z))
            if not math.isfinite(z) or z < -0.5 or z > 1.5 * self.v_max:
                c.status = d.status = INVALID
                continue
            if self._stuck(c, t) and v_pred > 1.0:
                c.status = d.status = STUCK
                c.latched = True
                c.good_time = 0.0
                continue
            # wheel slip from wheel acceleration
            aw = self._deriv(c, t)
            thr_a = max(p.slip_accel, 5.0 * math.sqrt(2.0 * c.noise_var) / 0.2)
            slip_now = aw is not None and abs(aw - a_model) > thr_a and (v_pred > 0.3 or z > 0.5)
            if slip_now:
                if t > c.slip_until:
                    c.slip_start, c.slip_confirmed = t, False
                c.slip_until = t + p.slip_hold
                c.slip_accel_seen = True
            elif t > c.slip_until:
                c.slip_accel_seen = False
            cand.append(i)

        # consensus reference (median of the channels that are neither slipping nor faulty)
        slip_active = any(t <= self.ch[i].slip_until for i in cand)
        vals = sorted(dec[i].value for i in cand if t > self.ch[i].slip_until and not self.ch[i].latched)
        ref = None
        if len(vals) >= 3:
            m = len(vals)
            ref = vals[m // 2] if m % 2 else 0.5 * (vals[m // 2 - 1] + vals[m // 2])
        elif len(vals) == 2 and mode != 0:
            # traction: slipping wheels are faster -> the slower is more credible,
            # a non-motored axle most of all
            trailer = [dec[i].value for i in cand if not self.powered[i] and t > self.ch[i].slip_until
                       and not self.ch[i].latched]
            ref = (trailer[0] if trailer else vals[0]) if mode > 0 else vals[1]

        # directional reference: the least slipping wheel is the slowest one in
        # traction and the fastest one in braking
        plaus = max(1.0, 4.0 * math.sqrt(var_pred + p.r_wheel))
        dch = [i for i in cand if not self.ch[i].latched and self.ch[i].bad_time < 0.5
               and abs(dec[i].value - v_pred) < plaus and t > self.ch[i].slip_until]
        dvals = [dec[i].value for i in dch]
        dref, dref_var = None, 0.0
        if len(dvals) >= 2 and mode != 0 and (v_pred > 0.15 or max(dvals) > 0.15):
            trailer = [i for i in dch if not self.powered[i]]
            pool = trailer if (mode > 0 and trailer) else dch   # a non-motored axle cannot spin in traction
            j = (min if mode > 0 else max)(pool, key=lambda k: dec[k].value)
            dref, dref_var = dec[j].value, self.ch[j].noise_var
        any_accept = False
        for i in cand:
            c, d = self.ch[i], dec[i]
            z = d.value
            e = z - v_pred
            S = var_pred + max(r_meas[i], c.noise_var)
            d.nis = e * e / S
            thr_c = (max(p.consensus_abs, p.consensus_rel * abs(ref), 3.0 * math.sqrt(c.noise_var))
                     if ref is not None else None)
            dev = (z - ref) if ref is not None else 0.0
            # directional slip evidence (only powered axles slip in traction)
            if dref is not None and not c.latched and (self.powered[i] or mode < 0):
                base = p.slip_dev_abs_low if dref < 2.0 else p.slip_dev_abs
                thr_d = max(base, p.slip_dev_rel * dref, 3.0 * math.sqrt(c.noise_var + dref_var))
                if (mode > 0 and z - dref > thr_d) or (mode < 0 and dref - z > thr_d):
                    if t > c.slip_until:
                        c.slip_start, c.slip_confirmed, c.slip_accel_seen = t, False, False
                    c.slip_until = t + p.slip_hold
                    # a physical slip starts with a wheel-acceleration anomaly,
                    # lasts, goes in the direction of the applied force and is
                    # moderate (anti-slip control); wheel-radius offsets, spikes
                    # and dead sensors are not
                    if (not c.slip_confirmed and c.slip_accel_seen and t - c.slip_start >= p.slip_confirm_time
                            and abs(z - dref) < 0.3 * max(dref, 1.0)):
                        c.slip_confirmed = True
                        self.slip_onsets.append(i)
            in_slip = t <= c.slip_until
            # asymmetric gate: deviations in the slip direction are suspicious
            gate = p.gate_chi2
            if (mode > 0 and e > 0) or (mode < 0 and e < 0):
                gate *= 0.6
            inconsistent = thr_c is not None and abs(dev) > thr_c
            gate_fail = d.nis > gate
            if slip_active and not self.powered[i] and mode > 0 and dev <= 0.0:
                # a non-motored axle cannot slip in traction: while motored
                # wheels spin it is the reference, never the culprit
                inconsistent = gate_fail = False
            if in_slip:
                c.status = d.status = SLIP
                c.good_time = 0.0
                continue
            if inconsistent or gate_fail:
                c.bad_time += dt
                c.good_time = 0.0
            else:
                c.bad_time = max(0.0, c.bad_time - 0.5 * dt)
                c.good_time += dt
            if c.bad_time > p.fault_confirm and (inconsistent or ref is None):
                c.latched = True
            if c.latched and c.good_time > p.recover_time:
                c.latched = False
                c.bad_time = 0.0
            if c.latched:
                c.status = d.status = FAULT
                continue
            if inconsistent or (gate_fail and ref is None):
                c.status = d.status = SUSPECT
                continue
            # agreeing with the majority of independent channels outweighs a
            # failed model gate (the prediction may have been pulled off by a
            # fault that is not isolated yet) -> accept with inflated noise
            c.status = d.status = OK
            d.accept = d.new
            if gate_fail:
                d.r_scale = d.nis / gate
            any_accept = any_accept or d.accept

        # persistent "slip" of every motored channel against a non-motored
        # reference is not physical (the anti-slip system limits real slip to a
        # few seconds) -> the reference itself is faulty (e.g. scale drift)
        self.reference_faulted = False
        if mode > 0:
            pw = [i for i in cand if self.powered[i] and not self.ch[i].latched]
            if len(pw) >= 2 and all(t <= self.ch[i].slip_until and t - self.ch[i].slip_start > p.slip_max_time
                                    for i in pw):
                zs = [dec[i].value for i in pw]
                if max(zs) - min(zs) < max(p.consensus_abs, p.consensus_rel * max(zs)):
                    for j in cand:
                        if not self.powered[j] and not self.ch[j].latched and dec[j].value < min(zs):
                            self.ch[j].latched, self.ch[j].good_time = True, 0.0
                            self.ch[j].status = dec[j].status = FAULT
                            dec[j].accept = False
                            self.reference_faulted = True
                    if self.reference_faulted:
                        for i in pw:     # the motored channels were right all along
                            self.ch[i].slip_until = t

        # re-acquisition after a long blind phase
        if any_accept:
            self.blind_time = 0.0
        else:
            self.blind_time += dt
            if self.blind_time > p.reacquire_time:
                ok = [i for i in cand if t > self.ch[i].slip_until and dec[i].new
                      and self.ch[i].status != STUCK]
                if ok:
                    zs = [dec[i].value for i in ok]
                    spread = max(zs) - min(zs)
                    wide = 25.0 * (var_pred + max(r_meas[i] for i in ok))
                    zc = sorted(zs)[len(zs) // 2]
                    if (len(ok) >= 2 and spread < 2 * p.consensus_abs) or len(ok) == 1:
                        if (zc - v_pred) ** 2 < wide:
                            for i in ok:
                                if abs(dec[i].value - zc) < 2 * p.consensus_abs:
                                    dec[i].accept, dec[i].r_scale, dec[i].status = True, 10.0, OK
                                    self.ch[i].latched = False
                                    self.ch[i].bad_time = 0.0
                                    self.ch[i].status = OK
                            self.blind_time = 0.0
        return dec

    def noise_var(self, i: int) -> float:
        return self.ch[i].noise_var

    def any_slip(self, t: float) -> bool:
        return any(t <= c.slip_until for c in self.ch)

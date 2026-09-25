"""Closed-loop evaluation: simulator -> (handle, odometry) -> navigators.

Three estimators are compared on the same data:

* ``proposed``  - :class:`TramNavigator` (adaptive model + FDI + RBF);
* ``odometry``  - classical odometry: mean of all reporting wheel sensors with
  the nominal wheel radius, integrated (no fault handling, dropouts skipped);
* ``model``     - open-loop dead reckoning from the handle with the nominal
  model only (what is left if every sensor dies at the start).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .estimator import TramNavigator
from .params import NavigatorParams, ScenarioConfig, SensorParams, TramParams
from .simulator import TramSimulator
from .track import TrackMap, generate_demo_track


@dataclass
class RunResult:
    scenario: ScenarioConfig
    data: Dict[str, np.ndarray]
    step_times: np.ndarray
    navigator: TramNavigator

    def metrics(self) -> Dict[str, float]:
        d = self.data
        out = {}
        for k in ("proposed", "odometry", "model"):
            ev = d[f"{k}_v"] - d["v"]
            es = d[f"{k}_s"] - d["s"]
            out[f"{k}_rmse_v"] = float(np.sqrt(np.mean(ev ** 2)))
            out[f"{k}_max_ev"] = float(np.max(np.abs(ev)))
            out[f"{k}_final_es"] = float(abs(es[-1]))
            out[f"{k}_max_es"] = float(np.max(np.abs(es)))
            out[f"{k}_rel_es_pct"] = float(100.0 * np.max(np.abs(es)) / max(d["s"][-1] - d["s"][0], 1.0))
        out["distance"] = float(d["s"][-1] - d["s"][0])
        out["step_mean_us"] = float(np.mean(self.step_times) * 1e6)
        out["step_p99_us"] = float(np.percentile(self.step_times, 99) * 1e6)
        out["step_max_us"] = float(np.max(self.step_times) * 1e6)
        # consistency: fraction of time the true error is within 3 sigma
        out["within_3sigma_pct"] = float(100.0 * np.mean(np.abs(d["proposed_s"] - d["s"]) <= 3 * d["sigma_s"] + 0.5))
        return out

    def trajectory(self, track: TrackMap) -> Dict[str, Dict[str, float]]:
        """2-D trajectory metrics of the three estimators (see core/trajectory.py)."""
        from .trajectory import trajectory_from_s, trajectory_metrics
        d = self.data
        xy_true = np.column_stack([d["x"], d["y"]])
        res = {"proposed": trajectory_metrics(track, d["t"], d["s"], d["proposed_s"], xy_true,
                                              np.column_stack([d["px"], d["py"]]), d["yaw"], d["pyaw"],
                                              d["v"], d["sigma_s"])}
        for k in ("odometry", "model"):
            tr = trajectory_from_s(track, d[f"{k}_s"])
            res[k] = trajectory_metrics(track, d["t"], d["s"], d[f"{k}_s"], xy_true, tr[:, :2], d["yaw"],
                                        tr[:, 2], d["v"], check_on_track=False)
        return res


def run_scenario(sc: ScenarioConfig, track: Optional[TrackMap] = None,
                 tram: Optional[TramParams] = None, nav: Optional[NavigatorParams] = None,
                 sensors: Optional[SensorParams] = None, true_tram: Optional[TramParams] = None,
                 log_every: int = 5) -> RunResult:
    track = track or generate_demo_track()
    tram = tram or TramParams()
    sensors = sensors or SensorParams()
    sim = TramSimulator(track, sc, true_tram or tram, sensors)
    powered = [ax in tram.powered_axles for ax in sensors.axles]
    s0 = sim.s
    nav_p = TramNavigator(tram, nav, track, len(sensors.axles), powered, s0=s0)
    nav_m = TramNavigator(tram, nav, track, len(sensors.axles), powered, s0=s0)
    odo_s, odo_v = s0, 0.0

    sub = max(1, int(round(1.0 / (sensors.rate * sim.dt))))
    n_steps = int(sc.duration / sim.dt)
    log = {k: [] for k in ("t", "s", "v", "a", "u", "mass", "eff", "mu_true", "proposed_s", "proposed_v",
                           "odometry_s", "odometry_v", "model_s", "model_v", "sigma_s", "sigma_v",
                           "eta", "mass_est", "dist", "mu_hat", "rbf", "mode", "n_used", "slip_true",
                           "slip_est", "z0", "z1", "z2", "st0", "st1", "st2", "x", "y", "px", "py", "yaw", "pyaw",
                           "blind")}
    step_times = []
    k_nav = 0
    for k in range(n_steps):
        sim.step()
        if (k + 1) % sub:
            continue
        t = sim.t
        meas = sim.sample_sensors()
        nav_p.set_handle(sim.u)
        nav_m.set_handle(sim.u)
        vals = []
        for i, om in enumerate(meas):
            if om is not None:
                nav_p.set_wheel_speed(i, t, om)
                vals.append(om * tram.wheel_radius)
        t0 = time.perf_counter()
        st = nav_p.step(t)
        step_times.append(time.perf_counter() - t0)
        stm = nav_m.step(t)
        dt = 1.0 / sensors.rate
        if vals:
            odo_v = float(np.mean(vals))
        odo_s += odo_v * dt
        k_nav += 1
        if k_nav % log_every:
            continue
        tr = sim.truth()
        L = log
        L["t"].append(t); L["s"].append(sim.s); L["v"].append(sim.v); L["a"].append(sim.a)
        L["u"].append(sim.u); L["mass"].append(sim.mass); L["eff"].append(sim.eff)
        L["mu_true"].append(sim._ap_cur.mu_max)
        L["proposed_s"].append(st.s); L["proposed_v"].append(st.v)
        L["odometry_s"].append(odo_s); L["odometry_v"].append(odo_v)
        L["model_s"].append(stm.s); L["model_v"].append(stm.v)
        L["sigma_s"].append(st.sigma_s); L["sigma_v"].append(st.sigma_v)
        L["eta"].append(st.eta); L["mass_est"].append(st.mass_est); L["dist"].append(st.dist)
        L["mu_hat"].append(st.mu_hat); L["rbf"].append(st.rbf)
        L["mode"].append({"NOMINAL": 0, "DEGRADED": 1, "BLIND": 2, "STANDSTILL": 3}[st.mode])
        L["n_used"].append(st.n_used); L["slip_true"].append(any(tr["slip"])); L["slip_est"].append(st.slip)
        L["blind"].append(st.blind_time)
        for i in range(3):
            v_i = meas[i] * tram.wheel_radius if i < len(meas) and meas[i] is not None else np.nan
            L[f"z{i}"].append(v_i)
            L[f"st{i}"].append(TramNavigator.status_code(st.sensor_status[i]) if i < len(st.sensor_status) else 9)
        L["x"].append(tr["x"]); L["y"].append(tr["y"]); L["px"].append(st.x); L["py"].append(st.y)
        L["yaw"].append(tr["yaw"]); L["pyaw"].append(st.yaw)
    data = {k: np.asarray(v) for k, v in log.items()}
    return RunResult(sc, data, np.asarray(step_times), nav_p)

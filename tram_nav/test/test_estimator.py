import math

import numpy as np
import pytest

from tram_nav.core.estimator import MODE_BLIND, MODE_STANDSTILL, TramNavigator
from tram_nav.core.evaluation import run_scenario
from tram_nav.core.params import NavigatorParams, ScenarioConfig, SensorFault, TramParams
from tram_nav.core.track import generate_demo_track

TRACK = generate_demo_track()


def feed(nav, t, u, omegas):
    nav.set_handle(u)
    for i, om in enumerate(omegas):
        if om is not None:
            nav.set_wheel_speed(i, t, om)
    return nav.step(t)


def test_standstill_no_drift():
    nav = TramNavigator(track=TRACK, sensor_powered=[True, True, False])
    rng = np.random.default_rng(0)
    st = None
    for k in range(3000):                               # 60 s, noisy zero readings, brake applied
        t = k * 0.02
        st = feed(nav, t, -0.5, list(rng.normal(0, 0.05, 3)))
    assert st.mode == MODE_STANDSTILL
    assert abs(st.s) < 0.05 and st.v < 0.01


def test_constant_speed_tracking():
    nav = TramNavigator(track=None, nav=NavigatorParams(station_snap=False), sensor_powered=[True, True, False])
    r = TramParams().wheel_radius
    rng = np.random.default_rng(1)
    v = 10.0
    st = None
    s_true = 0.0
    for k in range(1, 1500):
        t = k * 0.02
        v_true = min(v, 1.0 * t)
        s_true += v_true * 0.02
        st = feed(nav, t, 0.3 if v_true < v else 0.05, list((v_true + rng.normal(0, 0.02, 3)) / r))
    assert st.v == pytest.approx(v, abs=0.05)
    assert st.s == pytest.approx(s_true, abs=1.0)


def test_nan_handle_ignored():
    nav = TramNavigator(track=TRACK)
    nav.set_handle(0.5)
    nav.set_handle(float("nan"))
    assert nav.u_raw == 0.5


def test_blind_mode_flag_and_growing_uncertainty():
    nav = TramNavigator(track=TRACK, sensor_powered=[True, True, False])
    r = TramParams().wheel_radius
    for k in range(1, 500):
        feed(nav, k * 0.02, 0.4, [min(8.0, 0.8 * k * 0.02) / r] * 3)
    sig0 = nav.state.sigma_s
    st = None
    for k in range(500, 1000):                          # all sensors silent
        st = feed(nav, k * 0.02, 0.1, [None] * 3)
    assert st.mode == MODE_BLIND
    assert st.sigma_s > sig0
    assert st.v > 1.0                                    # model keeps the tram moving


@pytest.mark.parametrize("name,kwargs,max_pos_err,max_rmse_v", [
    ("nominal", dict(), 15.0, 0.12),   # k (wheel wear) not yet learned in 300 s
    ("autumn", dict(weather_profile=[(0.0, "dry"), (120.0, "leaves")]), 30.0, 0.2),
    ("faults", dict(faults=[SensorFault(0, "stuck", 60, 100), SensorFault(1, "zero", 120, 160),
                            SensorFault(2, "spikes", 170, 230, 0.1)]), 20.0, 0.25),
])
def test_closed_loop_accuracy(name, kwargs, max_pos_err, max_rmse_v):
    res = run_scenario(ScenarioConfig(name=name, duration=300, **kwargs), track=TRACK)
    m = res.metrics()
    assert m["proposed_rmse_v"] < max_rmse_v
    assert m["proposed_max_es"] < max_pos_err
    # the proposed navigator must beat plain odometry in position and speed
    assert m["proposed_max_es"] < m["odometry_max_es"]
    assert m["proposed_rmse_v"] < m["odometry_rmse_v"]


def test_mass_identification():
    res = run_scenario(ScenarioConfig(duration=400, mass=60_000.0, wheel_wear=False), track=TRACK,
                       nav=NavigatorParams(rbf_enable=False))
    mass_est = res.data["mass_est"][-100:].mean()
    assert mass_est == pytest.approx(60_000.0, rel=0.08)


def test_real_time_budget():
    res = run_scenario(ScenarioConfig(duration=120), track=TRACK)
    m = res.metrics()
    # 50 Hz -> 20 ms period; the filter must use only a small fraction of it
    assert m["step_mean_us"] < 2000.0
    assert np.percentile(res.step_times, 99) < 5e-3

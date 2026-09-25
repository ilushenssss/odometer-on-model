"""Tests of the 2-D trajectory built from the navigator output.

The navigator estimates the along-track distance ``s``; the trajectory is
``s -> (x, y, yaw)`` through the track map.  These tests check

* the geometry of the map projection (points on the rails, correct heading);
* the metric helpers;
* the accuracy of the reconstructed trajectory in closed loop against the
  simulator ground truth (nominal, low adhesion, sensor faults with a total
  odometry blackout, changing plant, a different line geometry);
* continuity (no jumps while moving) and consistency with the reported
  uncertainty.

Run separately:  python3 -m pytest -v test/test_trajectory.py
"""
import math

import numpy as np
import pytest

from tram_nav.core.evaluation import run_scenario
from tram_nav.core.params import ScenarioConfig, SensorFault
from tram_nav.core.track import TrackMap, generate_demo_track
from tram_nav.core.trajectory import distance_to_track, trajectory_from_s, trajectory_metrics, wrap_angle

TRACK = generate_demo_track()


# ---------------------------------------------------------------- geometry
def test_distance_to_track_straight_line():
    tr = TrackMap([0, 100, 200], [0, 0, 0])
    assert distance_to_track(tr, 50.0, 3.0) == pytest.approx(3.0)
    assert distance_to_track(tr, 50.0, -2.0, s_hint=50.0) == pytest.approx(2.0)
    # beyond the ends the map continues along the tangent
    assert distance_to_track(tr, 260.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert distance_to_track(tr, -40.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_projection_lies_on_rails_everywhere():
    s = np.linspace(-30.0, TRACK.length + 30.0, 3000)
    xyz = trajectory_from_s(TRACK, s)
    d = [distance_to_track(TRACK, x, y, sv) for (x, y, _), sv in zip(xyz, s)]
    assert max(d) < 1e-6


def test_projection_heading_matches_track_tangent():
    s = np.linspace(1.0, TRACK.length - 1.0, 2000)
    pose = trajectory_from_s(TRACK, s)
    ahead = trajectory_from_s(TRACK, s + 0.5)
    behind = trajectory_from_s(TRACK, s - 0.5)
    tangent = np.arctan2(ahead[:, 1] - behind[:, 1], ahead[:, 0] - behind[:, 0])
    err = np.degrees(np.abs(wrap_angle(pose[:, 2] - tangent)))
    assert np.percentile(err, 99) < 3.0


def test_projection_is_arc_length_parameterised():
    # moving 1 m along s moves the point by ~1 m in the plane (0.5 % on tight curves)
    s = np.linspace(0.0, TRACK.length - 1.0, 1500)
    a, b = trajectory_from_s(TRACK, s), trajectory_from_s(TRACK, s + 1.0)
    step = np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
    assert np.all(step > 0.99) and np.all(step < 1.0 + 1e-6)


def test_metrics_identity_and_chord_property():
    t = np.arange(0, 100, 0.1)
    s = np.linspace(0, 1000, len(t))
    xy = trajectory_from_s(TRACK, s)
    m = trajectory_metrics(TRACK, t, s, s, xy[:, :2], xy[:, :2], xy[:, 2], xy[:, 2], np.full(len(t), 10.0))
    assert m["max_2d"] == 0.0 and m["heading_max_deg"] == 0.0 and m["max_off_track"] < 1e-6
    s_bad = s + 25.0
    xyb = trajectory_from_s(TRACK, s_bad)
    m = trajectory_metrics(TRACK, t, s, s_bad, xy[:, :2], xyb[:, :2])
    assert m["chord_le_arc"] == 1.0            # |xy error| <= |s error|
    assert 0.0 < m["max_2d"] <= 25.0 + 1e-6


# ---------------------------------------------------------------- closed loop
def _run(sc, track=TRACK):
    res = run_scenario(sc, track=track, log_every=2)
    return res, res.trajectory(track)


CASES = [
    # name, scenario, max 2-D error [m], rms 2-D error [m]
    ("nominal", ScenarioConfig(name="nominal", duration=400), 12.0, 6.0),
    ("autumn", ScenarioConfig(name="autumn", duration=400,
                              weather_profile=[(0.0, "dry"), (100.0, "leaves")]), 40.0, 20.0),
    ("faults_blackout", ScenarioConfig(name="faults_blackout", duration=450, faults=[
        SensorFault(0, "stuck", 60, 100), SensorFault(1, "zero", 120, 160), SensorFault(2, "spikes", 170, 200, 0.1),
        SensorFault(0, "dropout", 220, 370), SensorFault(1, "dropout", 220, 370),
        SensorFault(2, "dropout", 220, 370)]), 60.0, 25.0),
    ("param_change", ScenarioConfig(name="param_change", duration=400, mass=58_000, passenger_exchange=True,
                                    efficiency_changes=[(150.0, 0.67)], resistance_scale=1.15), 20.0, 10.0),
]


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def closed_loop(request):
    name, sc, max_err, rms_err = request.param
    res, tm = _run(sc)
    return name, res, tm, max_err, rms_err


def test_trajectory_accuracy(closed_loop):
    name, res, tm, max_err, rms_err = closed_loop
    p = tm["proposed"]
    assert p["max_2d"] < max_err, p
    assert p["rms_2d"] < rms_err, p


def test_trajectory_better_than_odometry(closed_loop):
    _, _, tm, _, _ = closed_loop
    assert tm["proposed"]["rms_2d"] < tm["odometry"]["rms_2d"]
    assert tm["proposed"]["max_2d"] < tm["odometry"]["max_2d"]


def test_trajectory_stays_on_rails(closed_loop):
    _, _, tm, _, _ = closed_loop
    assert tm["proposed"]["max_off_track"] < 0.01


def test_trajectory_heading(closed_loop):
    _, _, tm, _, _ = closed_loop
    assert tm["proposed"]["heading_p50_deg"] < 0.5
    assert tm["proposed"]["heading_p95_deg"] < 10.0


def test_trajectory_continuous_while_moving(closed_loop):
    """While the tram moves, consecutive published points are never farther apart
    than the speed allows - with one legitimate exception: when odometry
    returns after a BLIND phase the filter corrects the dead-reckoned position.
    Such a correction must happen right after BLIND and must reduce the error.
    (Station fixes happen only at standstill.)"""
    _, res, _, _, _ = closed_loop
    d = res.data
    step = np.hypot(np.diff(d["px"]), np.diff(d["py"]))
    allowed = (d["v"][1:] + 5.0) * np.diff(d["t"]) + 1.0
    moving = d["v"][1:] > 1.0
    for i in np.nonzero(moving & (step > allowed))[0]:
        was_blind = np.any(d["mode"][max(0, i - 25):i + 1] == 2)
        err_before = abs(d["proposed_s"][i] - d["s"][i])
        err_after = abs(d["proposed_s"][i + 1] - d["s"][i + 1])
        assert was_blind, f"jump of {step[i]:.1f} m at t={d['t'][i + 1]:.1f} s while moving"
        assert err_after < err_before, f"correction at t={d['t'][i + 1]:.1f} s increased the error"


def test_trajectory_error_within_reported_uncertainty(closed_loop):
    _, _, tm, _, _ = closed_loop
    assert tm["proposed"]["within_3sigma"] > 0.9


def test_trajectory_follows_rails_during_blackout():
    sc = CASES[2][1]
    res, _ = _run(sc)
    d = res.data
    blind = (d["t"] > 225) & (d["t"] < 370)
    assert np.all(d["mode"][blind][50:] == 2)          # BLIND: no odometry at all
    # the dead-reckoned position keeps advancing with the real tram
    ds_true = d["s"][blind][-1] - d["s"][blind][0]
    ds_est = d["proposed_s"][blind][-1] - d["proposed_s"][blind][0]
    assert ds_true > 300.0
    assert abs(ds_est - ds_true) / ds_true < 0.15
    for x, y, s in zip(d["px"][blind][::25], d["py"][blind][::25], d["proposed_s"][blind][::25]):
        assert distance_to_track(TRACK, x, y, s) < 0.01


def _loop_track():
    """A different line: a closed-ish loop with an S-bend and grades."""
    pts, h, x, y = [(0.0, 0.0)], 0.0, 0.0, 0.0
    for length, radius, direction in [(400, 0, 0), (157, 100, 1), (300, 0, 0), (79, 50, -1), (79, 50, 1),
                                      (500, 0, 0), (236, 150, 1), (600, 0, 0), (120, 40, 1), (400, 0, 0)]:
        for _ in range(int(length)):
            if radius:
                h += direction / radius
            x, y = x + math.cos(h), y + math.sin(h)
            pts.append((x, y))
    xy = np.array(pts)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))])
    z = 12.0 * np.sin(s / 450.0)
    stations = [0.0, 700.0, 1400.0, 2100.0, float(s[-1]) - 5.0]
    return TrackMap(xy[:, 0], xy[:, 1], z, s, stations)


def test_trajectory_on_a_different_line():
    track = _loop_track()
    res, tm = _run(ScenarioConfig(name="loop", duration=420, weather_profile=[(0.0, "wet")]), track)
    p = tm["proposed"]
    assert res.data["s"][-1] > 1500.0
    assert p["max_2d"] < 25.0 and p["rms_2d"] < 12.0, p
    assert p["max_off_track"] < 0.01
    assert p["rms_2d"] < tm["odometry"]["rms_2d"]

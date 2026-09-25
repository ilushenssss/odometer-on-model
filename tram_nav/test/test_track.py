import math

import pytest

from tram_nav.core.track import TrackMap, generate_demo_track


def test_straight_track_pose_and_extrapolation():
    tr = TrackMap([0, 100, 200], [0, 0, 0], [0, 1, 2])
    assert tr.length == pytest.approx(200)
    x, y, yaw = tr.pose(50)
    assert (x, y, yaw) == pytest.approx((50, 0, 0))
    assert tr.pose(250)[0] == pytest.approx(250)       # tangent extrapolation
    assert tr.grade_at(100) == pytest.approx(math.atan(0.01), rel=1e-3)


def test_circle_curvature():
    R = 50.0
    th = [i * 0.01 for i in range(157)]
    tr = TrackMap([R * math.sin(a) for a in th], [R - R * math.cos(a) for a in th], smooth_window=1.0)
    assert tr.curvature_at(tr.length / 2) == pytest.approx(1 / R, rel=0.02)


def test_csv_roundtrip(tmp_path):
    tr = generate_demo_track()
    p = tmp_path / "t.csv"
    tr.to_csv(str(p))
    tr2 = TrackMap.from_csv(str(p))
    assert tr2.length == pytest.approx(tr.length, abs=5)
    assert tr2.stations == pytest.approx(tr.stations)
    for s in [10.0, 777.0, 3210.0]:
        assert tr2.pose(s)[:2] == pytest.approx(tr.pose(s)[:2], abs=0.2)
        assert tr2.grade_at(s) == pytest.approx(tr.grade_at(s), abs=2e-3)


def test_next_station():
    tr = generate_demo_track()
    assert tr.next_station(0.0) == tr.stations[1]
    assert tr.next_station(tr.length + 10) is None


def test_invalid_track():
    with pytest.raises(ValueError):
        TrackMap([0, 0], [0, 0])

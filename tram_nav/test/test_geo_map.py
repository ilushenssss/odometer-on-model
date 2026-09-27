import pytest

from tram_nav.core.geo import LocalFrame
from tram_nav.core.track import TrackMap, generate_demo_track


def test_local_frame_roundtrip_and_scale():
    f = LocalFrame(55.7558, 37.6173)
    lat, lon = f.to_latlon(1234.5, -678.9)
    x, y = f.to_xy(lat, lon)
    assert (x, y) == pytest.approx((1234.5, -678.9), abs=1e-6)
    # 1 degree of latitude ~ 111.4 km at 55.8 N
    assert f.to_xy(56.7558, 37.6173)[1] == pytest.approx(111_450.0, rel=2e-3)


def test_project_point_onto_line():
    tr = generate_demo_track()
    for s in [0.0, 333.3, 2500.0, 5800.0]:
        x, y, yaw = tr.pose(s)
        assert tr.project(x, y) == pytest.approx(s, abs=0.05)


def test_track_csv_with_latlon(tmp_path):
    tr = generate_demo_track()
    f = LocalFrame(55.0, 37.0)
    p = tmp_path / "t.csv"
    with open(p, "w") as fh:
        fh.write("lat,lon,alt,station\n")
        for i, s in enumerate(range(0, int(tr.length), 5)):
            x, y, _ = tr.pose(float(s))
            la, lo = f.to_latlon(x, y)
            fh.write(f"{la:.9f},{lo:.9f},0,{int(i % 100 == 0)}\n")
    t2 = TrackMap.from_csv(str(p))
    assert t2.geo is not None
    assert t2.length == pytest.approx(tr.length, abs=6.0)
    assert len(t2.stations) == len(range(0, int(tr.length), 500))
    la, lo = t2.geo.to_latlon(*t2.pose(1000.0)[:2])
    assert f.to_xy(la, lo) == pytest.approx(tr.pose(1000.0)[:2], abs=0.2)

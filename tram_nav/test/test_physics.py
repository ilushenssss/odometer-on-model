import math

import pytest

from tram_nav.core import physics as ph
from tram_nav.core.params import WEATHER, TramParams

P = TramParams()


def test_handle_levels():
    assert ph.handle_to_levels(0.0, P) == (0.0, 0.0, False)
    tr, br, em = ph.handle_to_levels(1.0, P)
    assert tr == pytest.approx(1.0) and br == 0.0 and not em
    tr, br, em = ph.handle_to_levels(-0.5, P)
    assert tr == 0.0 and 0.4 < br < 0.5 and not em
    assert ph.handle_to_levels(-1.0, P)[2] is True
    assert ph.handle_to_levels(float("nan"), P) == (0.0, 0.0, False)
    assert ph.handle_to_levels(5.0, P)[0] == pytest.approx(1.0)


def test_traction_envelope_constant_force_then_power():
    v_corner = P.traction_power / P.traction_force_max
    assert ph.traction_envelope(0.0, P) == P.traction_force_max
    assert ph.traction_envelope(v_corner * 0.99, P) == P.traction_force_max
    assert ph.traction_envelope(2 * v_corner, P) == pytest.approx(P.traction_force_max / 2)
    # continuity at the corner speed
    assert ph.traction_envelope(v_corner * 1.0001, P) == pytest.approx(P.traction_force_max, rel=1e-3)


def test_braking_opposes_motion_and_vanishes_at_rest():
    Fw, Ft = ph.commanded_force(-0.8, 10.0, P)
    assert Fw < 0 and Ft == 0.0
    Fw0, _ = ph.commanded_force(-0.8, 0.0, P)
    assert Fw0 == 0.0
    Fw, Ft = ph.commanded_force(-1.0, 10.0, P)  # emergency: track brake
    assert Ft == pytest.approx(-P.track_brake_force)


def test_ed_share_monotone():
    vals = [ph.ed_share(v, P) for v in [0.0, 0.3, 0.7, 1.2, 1.5, 5.0]]
    assert vals[0] == 0.0 and vals[-1] == 1.0
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_creep_curve_peak_and_inverse():
    ap = WEATHER["dry"]
    mu_pk = ph.mu_peak(0.0, ap)
    lam = [i * 1e-3 for i in range(0, 300)]
    mus = [ph.creep_curve(x, mu_pk, ap) for x in lam]
    i_max = max(range(len(mus)), key=mus.__getitem__)
    assert lam[i_max] == pytest.approx(ap.slip_peak, abs=2e-3)
    assert max(mus) == pytest.approx(mu_pk, rel=1e-3)
    assert mus[-1] < mu_pk  # falling branch -> unstable slip
    assert ph.creep_curve(-0.05, mu_pk, ap) == pytest.approx(-ph.creep_curve(0.05, mu_pk, ap))
    for r in [0.0, 0.2, 0.5, 0.9, 1.0]:
        lam_r = ph.creep_slip_for_force(r, ap)
        assert ph.creep_curve(lam_r, mu_pk, ap) / mu_pk == pytest.approx(r, abs=1e-9)


def test_adhesion_decreases_with_speed_and_weather():
    assert ph.mu_peak(20.0, WEATHER["dry"]) < ph.mu_peak(0.0, WEATHER["dry"])
    assert WEATHER["ice"].mu_max < WEATHER["leaves"].mu_max < WEATHER["wet"].mu_max < WEATHER["dry"].mu_max


def test_smooth_sat():
    for x in [0.0, 0.1, -0.1]:
        y, dy = ph.smooth_sat(x, 1.0)
        assert y == pytest.approx(x, abs=1e-3) and dy == pytest.approx(1.0, abs=1e-2)
    y, dy = ph.smooth_sat(10.0, 1.0)
    assert 0.99 < y <= 1.0 and dy < 1e-3
    y, _ = ph.smooth_sat(-10.0, 1.0)
    assert -1.0 <= y < -0.99
    # derivative consistency
    h = 1e-6
    for x in [0.5, 0.9, 1.3]:
        y1, dy = ph.smooth_sat(x, 1.0)
        y2, _ = ph.smooth_sat(x + h, 1.0)
        assert (y2 - y1) / h == pytest.approx(dy, rel=1e-3)


def test_resistance_signs():
    assert ph.resistance_force(10.0, 5e4, P) > 0
    assert ph.resistance_force(0.0, 5e4, P) == 0.0
    assert ph.grade_curve_accel(5.0, math.radians(2), 0.0, P) > 0
    assert ph.grade_curve_accel(5.0, -math.radians(2), 0.0, P) < 0

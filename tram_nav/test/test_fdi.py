from tram_nav.core.fdi import DROPOUT, FAULT, INVALID, OK, SLIP, STUCK, OdometryFDI
from tram_nav.core.params import NavigatorParams

NP = NavigatorParams()


def run(fdi, T, fn, v=10.0, a=0.0, mode=0, dt=0.02):
    dec = None
    t = 0.0
    while t < T:
        t += dt
        readings = [fn(i, t) for i in range(fdi.n)]
        dec = fdi.assess(t, dt, readings, v, 0.01, a, mode, [NP.r_wheel] * fdi.n)
    return dec


def noisy(i, t, v=10.0):
    return (t, v + 0.01 * ((int(t * 1000) * (i + 3)) % 7 - 3))


def test_all_healthy():
    fdi = OdometryFDI(3, NP, [True, True, False])
    dec = run(fdi, 2.0, noisy)
    assert [d.status for d in dec] == [OK, OK, OK]
    assert all(d.accept for d in dec)


def test_dropout_and_invalid():
    fdi = OdometryFDI(3, NP, [True, True, False])
    dec = run(fdi, 2.0, lambda i, t: None if i == 0 else ((t, float("nan")) if i == 1 else noisy(i, t)))
    assert dec[0].status == DROPOUT and dec[1].status == INVALID and dec[2].status == OK


def test_stuck_sensor_detected():
    fdi = OdometryFDI(3, NP, [True, True, False])
    dec = run(fdi, 3.0, lambda i, t: (t, 9.87) if i == 1 else noisy(i, t))
    assert dec[1].status == STUCK and not dec[1].accept
    assert dec[0].accept and dec[2].accept


def test_zero_sensor_isolated():
    fdi = OdometryFDI(3, NP, [True, True, False])
    dec = run(fdi, 3.0, lambda i, t: (t, 0.0 + 0.001 * (int(t * 50) % 2)) if i == 2 else noisy(i, t))
    assert dec[2].status in (FAULT, "SUSPECT") and not dec[2].accept
    assert dec[0].accept and dec[1].accept


def test_wheel_slip_in_traction():
    fdi = OdometryFDI(3, NP, [True, True, False])

    def slip(i, t):
        if i == 0 and t > 1.0:
            return (t, 10.0 + 4.0 * (t - 1.0))     # wheel spins up at 4 m/s^2
        return noisy(i, t)
    dec = run(fdi, 1.5, slip, mode=1, a=0.0)
    assert dec[0].status == SLIP and not dec[0].accept
    assert dec[2].accept


def test_reacquisition_after_long_blind_phase():
    fdi = OdometryFDI(2, NP, [True, False])
    # prediction is off by 0.3 m/s (> 3 sigma) but sensors agree -> accepted after reacquire_time
    dec = run(fdi, NP.reacquire_time + 1.0, lambda i, t: (t, noisy(i, t, 10.3)[1]), v=10.0)
    assert any(d.accept for d in dec)

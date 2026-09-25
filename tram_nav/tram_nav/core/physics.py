"""Nonlinear longitudinal physics of a tram.

The functions here are shared by the "true" simulator and by the navigator
(which evaluates them with *nominal* parameters).  All of them are scalar and
allocation-free so they can be evaluated at kHz rates from pure Python.

Nonlinearities that are modelled:

* hyperbolic traction characteristic ``F = min(F0, P / v)`` (constant force,
  then constant power);
* blending of electrodynamic (ED) and friction brakes with ED fade-out at low
  speed; magnetic track brake in emergency (adhesion independent);
* Davis-type motion resistance ``m (c0 + c1 v) + c2 v^2`` with smoothed
  static friction around ``v = 0``;
* grade and curve resistance from the track map;
* wheel-rail creep-force (adhesion) curve ``mu(lambda, v)`` with an unstable
  falling branch -> wheel slip / slide;
* adhesion limitation of the tractive / braking effort (smooth saturation).
"""
from __future__ import annotations

import math

from .params import G, AdhesionParams, TramParams


def handle_to_levels(u: float, p: TramParams):
    """Split the driver's controller handle position into traction / brake levels.

    ``u`` in [-1, 1]; positive - traction, negative - braking.
    Returns ``(traction_level, brake_level, emergency)`` with levels in [0, 1].
    """
    if u != u:  # NaN guard
        u = 0.0
    u = max(-1.0, min(1.0, u))
    dz = p.handle_deadzone
    if u > dz:
        return (u - dz) / (1.0 - dz), 0.0, False
    if u < -dz:
        emergency = u <= p.emergency_threshold
        lvl = min(1.0, (-u - dz) / (1.0 - dz))
        return 0.0, (1.0 if emergency else lvl), emergency
    return 0.0, 0.0, False


def traction_envelope(v: float, p: TramParams) -> float:
    """Maximum tractive effort at speed ``v`` (constant force / constant power)."""
    v = abs(v)
    if v * p.traction_force_max <= p.traction_power:
        return p.traction_force_max
    return p.traction_power / v


def ed_share(v: float, p: TramParams) -> float:
    """Fraction of braking provided by the electrodynamic brake (smoothstep fade)."""
    x = abs(v) / p.ed_fade_speed
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


def motion_sign(v: float, eps: float = 0.05) -> float:
    """Smooth sign function used for Coulomb-like forces near standstill."""
    x = v / eps
    if x >= 1.0:
        return 1.0
    if x <= -1.0:
        return -1.0
    return x


def commanded_force(u: float, v: float, p: TramParams):
    """Force commanded by the handle position, before adhesion limitation.

    Returns ``(F_wheel, F_track)``: the signed force demanded through the
    wheels (traction positive, braking negative) and the adhesion-independent
    track-brake force (<= 0).
    """
    tr, br, emerg = handle_to_levels(u, p)
    sgn = motion_sign(v)
    if tr > 0.0:
        return tr * traction_envelope(v, p), 0.0
    if br > 0.0:
        ft = -p.track_brake_force * sgn if emerg else 0.0
        return -br * p.brake_force_max * sgn, ft
    return 0.0, 0.0


def resistance_force(v: float, mass: float, p: TramParams, scale: float = 1.0) -> float:
    """Davis resistance (always opposing motion)."""
    av = abs(v)
    return scale * (mass * (p.res_c0 * motion_sign(v) + p.res_c1 * v) + p.res_c2 * av * v)


def grade_curve_accel(v: float, grade: float, curvature: float, p: TramParams) -> float:
    """Specific (per unit mass) resistance from grade and curves."""
    return G * math.sin(grade) + p.curve_coeff * abs(curvature) * motion_sign(v, 0.2)


# ---------------------------------------------------------------------------
# Adhesion
# ---------------------------------------------------------------------------

def mu_peak(v: float, ap: AdhesionParams, local_factor: float = 1.0) -> float:
    """Peak adhesion coefficient at speed ``v`` (decreases with speed)."""
    return ap.mu_max * local_factor / (1.0 + ap.speed_coeff * abs(v))


def creep_curve(slip: float, mu_pk: float, ap: AdhesionParams) -> float:
    """Signed adhesion coefficient for a given slip ratio.

    Parabolic rising branch up to ``slip_peak``, exponential decay to the
    sliding value afterwards.
    """
    a = abs(slip)
    lp = ap.slip_peak
    if a <= lp:
        r = a / lp
        mu = mu_pk * r * (2.0 - r)
    else:
        sr = ap.slide_ratio
        mu = mu_pk * (sr + (1.0 - sr) * math.exp(-ap.slip_decay * (a - lp)))
    return mu if slip >= 0.0 else -mu


def creep_slip_for_force(force_ratio: float, ap: AdhesionParams) -> float:
    """Inverse of the rising branch: slip that transmits ``|F|/(mu N) = ratio``."""
    r = min(1.0, abs(force_ratio))
    lam = ap.slip_peak * (1.0 - math.sqrt(1.0 - r))
    return lam if force_ratio >= 0.0 else -lam


def smooth_sat(x: float, lim: float, p: int = 6):
    """Smooth saturation ``x / (1 + |x/lim|^p)^(1/p)`` and its derivative."""
    if lim <= 1e-9:
        return 0.0, 0.0
    r = abs(x) / lim
    if r < 0.3:  # fast path, error < 1e-3 relative
        return x, 1.0
    base = 1.0 + r ** p
    y = x * base ** (-1.0 / p)
    dy = base ** (-1.0 / p - 1.0)
    return y, dy

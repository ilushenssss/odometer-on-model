"""Parameter containers for the tram model, the simulator and the navigator.

All units are SI unless explicitly stated otherwise:
    mass [kg], force [N], speed [m/s], acceleration [m/s^2], power [W],
    distance [m], time [s], angle [rad].

The default values describe a typical modern 100 % low-floor articulated tram
(three sections, four axles, three of them motored), close to the class of
vehicles operated in large Russian cities (e.g. 71-931M, 71-911).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Dict, List, Tuple

G = 9.80665


@dataclass
class TramParams:
    """Nominal ("catalogue") parameters of the vehicle.

    These are the values the navigator *believes* in.  The simulator perturbs
    them to create the real, unknown plant (mass with passengers, worn wheels,
    degraded traction equipment, ...).
    """

    # --- mass -----------------------------------------------------------------
    mass_empty: float = 42_000.0          # tare mass
    mass_nominal: float = 50_000.0        # design mass used by the model
    rot_mass_factor: float = 0.08         # rotating-mass allowance (gamma)

    # --- axles ------------------------------------------------------------------
    n_axles: int = 4
    powered_axles: Tuple[int, ...] = (0, 1, 2)
    wheel_radius: float = 0.330           # nominal (new) wheel radius

    # --- traction ---------------------------------------------------------------
    traction_force_max: float = 70_000.0  # starting tractive effort
    traction_power: float = 560_000.0     # continuous power at the wheel rim
    traction_tau: float = 0.45            # traction actuator time constant
    handle_deadzone: float = 0.04         # |u| below this is coasting

    # --- braking ----------------------------------------------------------------
    brake_force_max: float = 75_000.0     # service brake (ED + friction)
    ed_fade_speed: float = 1.5            # below this speed ED brake fades out
    brake_tau: float = 0.35               # brake actuator time constant
    emergency_threshold: float = -0.95    # handle below -> emergency brake
    track_brake_force: float = 60_000.0   # magnetic track brake (adhesion free)

    # --- motion resistance: F = m*(c0 + c1*v) + c2*v^2 -----------------------
    res_c0: float = 0.018                 # rolling resistance [m/s^2]
    res_c1: float = 4.0e-4                # [1/s]
    res_c2: float = 5.5                   # aerodynamic 0.5*rho*Cd*A [kg/m]
    curve_coeff: float = 2.5              # curve resistance a = curve_coeff/R

    # --- limits -----------------------------------------------------------------
    v_max: float = 22.0                   # design speed (~80 km/h)
    a_phys_max: float = 3.5               # max physically plausible |a| of body

    @property
    def n_powered(self) -> int:
        return len(self.powered_axles)

    def copy(self, **kw) -> "TramParams":
        return replace(self, **kw)


@dataclass
class AdhesionParams:
    """Wheel-rail adhesion (friction) characteristic.

    mu(lambda) rises almost linearly up to the peak ``mu_max`` at
    ``slip_peak`` and then decays towards ``mu_max * slide_ratio`` - the
    classic creep-force curve that makes wheel-slip unstable.
    """

    mu_max: float = 0.30
    slip_peak: float = 0.02
    slide_ratio: float = 0.55
    slip_decay: float = 8.0               # decay rate of the falling branch
    speed_coeff: float = 0.012            # mu reduction with speed (Curtius-Kniffler like)
    fluctuation: float = 0.12             # relative std of spatial mu variations
    fluct_length: float = 25.0            # correlation length of variations [m]


WEATHER: Dict[str, AdhesionParams] = {
    "dry": AdhesionParams(mu_max=0.30, fluctuation=0.08),
    "wet": AdhesionParams(mu_max=0.18, fluctuation=0.15),
    "leaves": AdhesionParams(mu_max=0.11, fluctuation=0.30, slide_ratio=0.5),
    "ice": AdhesionParams(mu_max=0.09, fluctuation=0.25, slide_ratio=0.45),
}


@dataclass
class SensorFault:
    """A fault injected into one odometry sensor during a time interval."""

    sensor: int
    kind: str          # 'dropout' | 'stuck' | 'zero' | 'spikes' | 'scale' | 'noise'
    t_start: float
    t_end: float
    value: float = 0.0  # meaning depends on the kind (scale factor, noise std...)


@dataclass
class SensorParams:
    """Wheel speed (odometry) sensors - incremental encoders on axles."""

    axles: Tuple[int, ...] = (0, 2, 3)     # axle carrying each sensor
    pulses_per_rev: int = 200
    rate: float = 50.0                     # publication rate [Hz]
    window: float = 0.1                    # pulse counting window [s]
    noise_std: float = 0.02                # additive speed noise [m/s]
    radius_error_std: float = 0.003        # wheel wear -> true radius spread [m]


@dataclass
class NavigatorParams:
    """Tuning of the adaptive estimator."""

    rate: float = 50.0
    handle_delay: float = 0.1             # transport delay handle -> traction converter [s]
    # process noise spectral densities
    q_pos: float = 1e-4
    q_vel: float = 2e-2
    q_trac: float = 0.02
    q_dist: float = 4e-3
    q_eta: float = 1e-5
    q_scale: float = 1e-9
    # measurement noise
    r_wheel: float = 0.03 ** 2
    r_wheel_min: float = 0.01 ** 2
    r_wheel_max: float = 1.0
    r_adapt_forget: float = 0.01          # Sage-Husa forgetting of R
    huber_k: float = 2.0                  # Huber threshold [sigma]
    gate_chi2: float = 11.0               # innovation gate (~ 99.9 % for 1 dof)
    # initial uncertainty
    p0_pos: float = 1.0
    p0_vel: float = 0.25
    p0_trac: float = 0.1
    p0_dist: float = 0.05
    p0_eta: float = 0.02
    p0_scale: float = 0.03 ** 2            # prior of the odometric scale (wheel wear)
    scale_bounds: Tuple[float, float] = (0.9, 1.1)
    eta_bounds: Tuple[float, float] = (0.6, 1.4)
    dist_bound: float = 1.0
    dist_decay_tau: float = 60.0          # disturbance decay when blind
    # fault detection
    timeout: float = 0.3
    slip_accel: float = 1.5               # |wheel accel - model accel| limit
    slip_dev_abs: float = 0.35            # directional slip threshold [m/s]
    slip_dev_rel: float = 0.02
    slip_dev_abs_low: float = 0.18        # ... below 2 m/s (wheel spin at start-up)
    noise_est_alpha: float = 0.02         # EWMA gain of the per-channel noise estimate
    slip_max_time: float = 4.0            # longer "slip" of all motored axles -> reference fault
    slip_confirm_time: float = 0.3        # slip must persist to adapt the adhesion estimate
    consensus_abs: float = 0.35
    consensus_rel: float = 0.06
    fault_confirm: float = 1.5
    recover_time: float = 3.0
    slip_hold: float = 1.0
    stuck_window: float = 1.5
    # adhesion adaptation
    mu_prior: float = 0.30
    mu_min: float = 0.04
    mu_recover_tau: float = 120.0
    # calibration
    calib_gain: float = 2e-4
    calib_bound: float = 0.05
    # RBF approximator
    rbf_enable: bool = False              # optional RBF residual approximator (see README, ablation)
    rbf_v_centers: int = 7
    rbf_u_centers: int = 1                # 1 -> residual over speed only (identifiable w.r.t. eta)
    rbf_forget: float = 0.9995
    rbf_min_speed: float = 1.0
    rbf_update_every: int = 5
    rbf_coast_only: bool = False          # learn only without tractive / brake force
    rbf_coast_accel: float = 0.1          # |eta*a| below this = coasting [m/s^2]
    zupt_speed: float = 0.10
    zupt_exit_speed: float = 0.25
    blind_decorrelate_time: float = 5.0   # after a longer blind phase: drop the s-v correlation
    blind_zupt_r: float = 0.05 ** 2        # model-inferred standstill when blind
    creep_slip_peak: float = 0.02        # creep: lambda = slip_peak/(2 mu_hat) * F_axle/(m_axle g)
    meas_delay: float = 0.05              # group delay of the pulse-counting window
    stop_eta_inflation: float = 4e-3      # P_eta increase at stops (passenger exchange)
    reacquire_time: float = 2.0
    use_track_grade: bool = True
    # map-aided stop matching ("station snap")
    station_snap: bool = True
    snap_min_stop: float = 4.0            # stop duration before matching [s]
    snap_sigma: float = 2.5               # stopping-point spread at a platform [m]
    snap_gate: float = 20.0               # |innovation| always accepted up to ... [m]
    snap_gate_max: float = 60.0           # ... and never beyond (3 sigma in between)
    snap_gate_blind: float = 150.0        # ... for model-inferred stops without odometry
    snap_min_stop_blind: float = 10.0
    snap_slip_window: float = 30.0        # slip within this time before a stop ...
    snap_slip_factor: float = 3.0         # ... inflates the stopping-point sigma
    snap_nsigma: float = 3.0              # innovation gate [sigma]
    snap_scale_sigma: float = 8.0         # stopping-point spread assumed for scale estimation [m]
    scale_sigma_min: float = 0.004
    scale_max_step: float = 0.01          # max relative scale change per station pair        # floor of the scale uncertainty
    scale_max_slip: float = 2.0           # max slip time between two fixes for a scale update [s]
    scale_gate: float = 0.08              # max |D/L - 1| accepted for scale update


def params_to_dict(p) -> dict:
    return {f.name: getattr(p, f.name) for f in fields(p)}


def update_from_dict(p, d: dict):
    """Return a copy of dataclass ``p`` with fields overridden from ``d``."""
    kw = {}
    names = {f.name for f in fields(p)}
    for k, v in d.items():
        if k in names:
            cur = getattr(p, k)
            if isinstance(cur, tuple):
                v = tuple(v)
            kw[k] = v
    return replace(p, **kw)


@dataclass
class ScenarioConfig:
    name: str = "nominal"
    duration: float = 600.0
    weather_profile: List[Tuple[float, str]] = field(default_factory=lambda: [(0.0, "dry")])
    faults: List[SensorFault] = field(default_factory=list)
    mass: float = 50_000.0
    mass_changes: List[Tuple[float, float]] = field(default_factory=list)  # (t, new mass)
    traction_efficiency: float = 1.0
    efficiency_changes: List[Tuple[float, float]] = field(default_factory=list)
    resistance_scale: float = 1.0
    passenger_exchange: bool = False          # random mass change at every stop
    emergency_brakes: List[float] = field(default_factory=list)  # times of emergency stops
    wheel_wear: bool = True                   # random true wheel radii
    seed: int = 1

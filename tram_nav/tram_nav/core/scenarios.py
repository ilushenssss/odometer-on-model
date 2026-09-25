"""Named test scenarios (used by the simulator node, tests and figures)."""
from __future__ import annotations

from typing import Callable, Dict

from .params import ScenarioConfig, SensorFault


def _nominal(seed: int = 1) -> ScenarioConfig:
    return ScenarioConfig(name="nominal", duration=900, seed=seed)


def _wet(seed: int = 1) -> ScenarioConfig:
    return ScenarioConfig(name="wet", duration=900, weather_profile=[(0.0, "wet")], seed=seed)


def _autumn(seed: int = 1) -> ScenarioConfig:
    """Dry -> leaves on the rails -> wet: heavy slip / slide."""
    return ScenarioConfig(name="autumn", duration=900,
                          weather_profile=[(0.0, "dry"), (200.0, "leaves"), (600.0, "wet")], seed=seed)


def _ice(seed: int = 1) -> ScenarioConfig:
    return ScenarioConfig(name="ice", duration=1100, weather_profile=[(0.0, "ice")], seed=seed)


def _sensor_faults(seed: int = 1) -> ScenarioConfig:
    """A sequence of odometry faults ending with a total odometry blackout."""
    return ScenarioConfig(name="sensor_faults", duration=900, seed=seed, faults=[
        SensorFault(0, "stuck", 100, 160),
        SensorFault(1, "zero", 200, 260),
        SensorFault(2, "spikes", 300, 400, 0.1),
        SensorFault(2, "scale", 450, 550, 0.9),
        SensorFault(0, "dropout", 600, 900),
        SensorFault(1, "dropout", 600, 900),
        SensorFault(2, "dropout", 600, 900),
    ])


def _blackout(seed: int = 1) -> ScenarioConfig:
    """All odometry lost for 5 minutes in wet weather with a heavy tram."""
    return ScenarioConfig(name="blackout", duration=900, seed=seed, mass=56_000,
                          weather_profile=[(0.0, "wet")],
                          faults=[SensorFault(i, "dropout", 300, 600) for i in range(3)])


def _param_change(seed: int = 1) -> ScenarioConfig:
    """Changing plant: passenger exchange at every stop, traction degradation
    (one of three traction converters lost) and 15 % higher resistance."""
    return ScenarioConfig(name="param_change", duration=900, seed=seed, mass=58_000,
                          passenger_exchange=True, efficiency_changes=[(300.0, 0.67)],
                          resistance_scale=1.15)


def _combined(seed: int = 1) -> ScenarioConfig:
    """Everything at once: bad weather, faults, changing mass, emergency stops."""
    return ScenarioConfig(name="combined", duration=900, seed=seed, mass=55_000, passenger_exchange=True,
                          weather_profile=[(0.0, "wet"), (300.0, "leaves"), (650.0, "wet")],
                          efficiency_changes=[(400.0, 0.8)], emergency_brakes=[160.0, 520.0],
                          faults=[SensorFault(2, "noise", 50, 200, 0.3),
                                  SensorFault(0, "stuck", 250, 300),
                                  SensorFault(1, "spikes", 350, 450, 0.08),
                                  SensorFault(0, "dropout", 700, 820),
                                  SensorFault(1, "dropout", 700, 820),
                                  SensorFault(2, "dropout", 700, 820)])


SCENARIOS: Dict[str, Callable[..., ScenarioConfig]] = {
    "nominal": _nominal,
    "wet": _wet,
    "autumn": _autumn,
    "ice": _ice,
    "sensor_faults": _sensor_faults,
    "blackout": _blackout,
    "param_change": _param_change,
    "combined": _combined,
}


def get_scenario(name: str, seed: int = 1) -> ScenarioConfig:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}', available: {', '.join(SCENARIOS)}")
    return SCENARIOS[name](seed)

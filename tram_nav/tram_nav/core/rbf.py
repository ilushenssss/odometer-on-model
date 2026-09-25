"""Online adaptive nonlinear approximator (normalised RBF network + RLS).

Learns the part of the tram dynamics that the physical model does not
describe - the residual specific force ``r(v, u)`` [m/s^2] as a smooth
function of speed and handle position (unknown resistance, traction
characteristic errors, brake blending errors, efficiency, ...).

    r_hat(v, u) = w^T phi(v, u),   phi_j = exp(-|z - c_j|^2_Sigma) / sum_k exp(...)

Weights are identified by exponentially-weighted recursive least squares
(forgetting factor ``lambda``) which tracks slow parameter drift.  The
covariance is bounded (trace limit + regularisation towards the prior) to avoid
wind-up in directions that are not excited - the network is only trained while
the odometry is trustworthy and is then used to *extrapolate* the model when
all sensors are lost.
"""
from __future__ import annotations

import numpy as np


class RBFApproximator:
    def __init__(self, v_max: float = 22.0, n_v: int = 7, n_u: int = 5, forget: float = 0.9995,
                 p0: float = 0.5, trace_max: float = 50.0, w_bound: float = 1.0):
        self.cv = np.linspace(0.0, v_max, n_v)
        self.cu = np.linspace(-1.0, 1.0, n_u) if n_u > 1 else np.zeros(1)
        self.sv = (self.cv[1] - self.cv[0]) if n_v > 1 else v_max
        self.su = (self.cu[1] - self.cu[0]) if n_u > 1 else 1.0
        self.n = n_v * n_u
        self.w = np.zeros(self.n)
        self.P = np.eye(self.n) * p0
        self.p0 = p0
        self.lam = forget
        self.trace_max = trace_max
        self.w_bound = w_bound
        self.n_updates = 0

    def features(self, v: float, u: float) -> np.ndarray:
        ev = np.exp(-0.5 * ((v - self.cv) / self.sv) ** 2)
        eu = np.exp(-0.5 * ((u - self.cu) / self.su) ** 2) if len(self.cu) > 1 else np.ones(1)
        phi = np.outer(ev, eu).ravel()
        s = phi.sum()
        return phi / s if s > 1e-12 else phi

    def predict(self, v: float, u: float) -> float:
        return float(self.w @ self.features(v, u))

    def update(self, v: float, u: float, y: float, weight: float = 1.0) -> float:
        """One RLS step towards target ``y``; returns the a-priori error."""
        phi = self.features(v, u)
        Pphi = self.P @ phi
        denom = self.lam / max(weight, 1e-6) + phi @ Pphi
        k = Pphi / denom
        err = y - self.w @ phi
        self.w += k * err
        np.clip(self.w, -self.w_bound, self.w_bound, out=self.w)
        self.P = (self.P - np.outer(k, Pphi)) / self.lam
        tr = np.trace(self.P)
        if tr > self.trace_max:  # anti wind-up
            self.P *= self.trace_max / tr
        self.n_updates += 1
        return float(err)

    def reset(self) -> None:
        self.w[:] = 0.0
        self.P = np.eye(self.n) * self.p0
        self.n_updates = 0

    def surface(self, v_grid, u_grid) -> np.ndarray:
        return np.array([[self.predict(v, u) for v in v_grid] for u in u_grid])

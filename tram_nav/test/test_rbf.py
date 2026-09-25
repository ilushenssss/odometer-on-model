import numpy as np

from tram_nav.core.rbf import RBFApproximator


def test_partition_of_unity():
    net = RBFApproximator()
    for v, u in [(0, 0), (10, 0.5), (22, -1), (5, 1)]:
        assert abs(net.features(v, u).sum() - 1.0) < 1e-12


def test_learns_smooth_residual():
    rng = np.random.default_rng(0)
    net = RBFApproximator(forget=0.9999)
    f = lambda v, u: 0.05 * np.sin(v / 4.0) + 0.04 * u * u - 0.02   # noqa: E731
    for _ in range(6000):
        v, u = rng.uniform(0, 22), rng.uniform(-1, 1)
        net.update(v, u, f(v, u) + rng.normal(0, 0.01))
    err = [net.predict(v, u) - f(v, u) for v in np.linspace(1, 21, 15) for u in np.linspace(-0.9, 0.9, 7)]
    assert np.sqrt(np.mean(np.square(err))) < 0.01


def test_bounded_weights_and_covariance():
    net = RBFApproximator(w_bound=0.5, trace_max=10.0)
    for _ in range(2000):
        net.update(5.0, 0.0, 100.0)      # absurd target
    assert np.all(np.abs(net.w) <= 0.5 + 1e-12)
    assert np.trace(net.P) <= 10.0 + 1e-9
    net.reset()
    assert net.predict(5.0, 0.0) == 0.0

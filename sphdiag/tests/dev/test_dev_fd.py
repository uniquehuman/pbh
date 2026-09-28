"""Finite differences: Fornberg weights, convergence orders, parity ghosts, error estimate."""
import numpy as np
import pytest

from sphdiag import RadialOperator, fornberg_weights
from devbg import stretched


def test_fornberg_known_weights():
    h = 0.1
    x = np.array([[-2, -1, 0, 1, 2]]) * h
    c = fornberg_weights([0.0], x, 2)
    np.testing.assert_allclose(c[0, :, 1], np.array([1, -8, 0, 8, -1]) / (12 * h), atol=1e-12)
    np.testing.assert_allclose(c[0, :, 2], np.array([-1, 16, -30, 16, -1]) / (12 * h * h),
                               atol=1e-9)
    # one-sided 5-point first derivative
    x = np.array([[0, 1, 2, 3, 4]]) * h
    c = fornberg_weights([0.0], x, 1)
    np.testing.assert_allclose(c[0, :, 1], np.array([-25, 48, -36, 16, -3]) / (12 * h),
                               atol=1e-10)


def test_fornberg_exact_on_polynomials_nonuniform():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 1, size=(20, 7)), axis=1)
    z = x[:, 3] + 0.01
    c = fornberg_weights(z, x, 1)[:, :, 1]
    for p in range(7):
        d = np.sum(c * x ** p, axis=1)
        np.testing.assert_allclose(d, p * z ** max(p - 1, 0) if p else 0 * z, atol=1e-8)


def _orders(errs):
    e = np.asarray(errs)
    return np.log2(e[:-1] / e[1:])


@pytest.mark.parametrize("center", ["vertex", "cell", "none"])
@pytest.mark.parametrize("parity", [1, -1])
def test_convergence_order_4_and_6(center, parity):
    e4, e6 = [], []
    for N in (40, 80, 160):
        r = stretched(N, 0.0, 2.0, cell=(center == "cell"))
        if center == "none":
            r = r + 0.3
        op = RadialOperator(r, center)
        f, fp = (np.cos(r), -np.sin(r)) if parity == 1 else (np.sin(r), np.cos(r))
        e4.append(np.max(np.abs(op.d(f, parity, 4) - fp)))
        e6.append(np.max(np.abs(op.d(f, parity, 6) - fp)))
    o4, o6 = _orders(e4), _orders(e6)
    print(center, parity, "order4", o4, "order6", o6)
    assert np.all(o4 > 3.6), o4
    assert np.all(o6[:1] > 5.4), o6


@pytest.mark.parametrize("center", ["vertex", "cell", "none"])
def test_error_estimate_tracks_true_error(center):
    for N in (60, 120):
        r = stretched(N, 0.0, 3.0, cell=(center == "cell"))
        if center == "none":
            r = r + 0.5
        op = RadialOperator(r, center)
        f = np.sin(r) * np.exp(-0.1 * r ** 2)             # odd, as the parity requires
        fp = (np.cos(r) - 0.2 * r * np.sin(r)) * np.exp(-0.1 * r ** 2)
        d4, sig = op.d_err(f, f, -1)
        err = np.abs(d4 - fp)
        # the estimate is within a factor ~2 of the true max error and bounds it (x3) pointwise
        assert 0.5 < sig.max() / err.max() < 2.0
        assert np.all(err <= 3 * sig + 1e-14)


def test_parity_ghosts_vertex():
    r = np.linspace(0, 1, 21)
    op = RadialOperator(r, "vertex")
    # odd function: derivative at 0 is exact for polynomials up to degree 4
    f = r + r ** 3
    assert abs(op.d(f, -1)[0] - 1.0) < 1e-13
    # even function: derivative zero at the centre (to round-off)
    assert abs(op.d(np.cos(r), 1)[0]) < 1e-14


def test_cumint_order4():
    errs = []
    for N in (40, 80, 160):
        r = stretched(N, 0.0, 2.0)
        op = RadialOperator(r, "vertex")
        g = r ** 2 * np.cos(r)                 # even
        exact = (r ** 2 - 2) * np.sin(r) + 2 * r * np.cos(r)
        errs.append(np.max(np.abs(op.cumint(g, 1) - exact)))
    o = _orders(errs)
    print("cumint orders", o)
    assert np.all(o > 3.6)

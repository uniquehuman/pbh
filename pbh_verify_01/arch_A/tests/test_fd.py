"""Finite-difference layer (contract section 8): Fornberg weights, orders, ghosts, quadrature."""
import numpy as np
import pytest

from conftest import grid, order
from sphdiag.fd import FDGrid, fornberg_weights, even_extrapolate_to_zero


def test_fornberg_known_weights():
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    c = fornberg_weights(np.array([0.0]), x, 2)
    assert np.allclose(c[1, 0], [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
    assert np.allclose(c[2, 0], [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])


@pytest.mark.parametrize("center", ["vertex", "cell", "none"])
@pytest.mark.parametrize("p", [4, 6])
def test_exact_on_polynomials(center, p):
    """Order-p stencils differentiate polynomials of degree p exactly (incl. ghosts / one-sided)."""
    rng = np.random.default_rng(1)
    r = np.sort(rng.uniform(0.05, 1.0, 20))
    if center == "vertex":
        r = np.concatenate([[0.0], r])
    elif center == "none":
        r = r - 0.5
    g = FDGrid(r, center)
    for deg in range(p + 1):
        for parity in (+1, -1):
            if center != "none" and (deg % 2 == 0) != (parity == 1):
                continue
            f = r ** deg
            d1, _ = g.deriv(f, parity, p, 1)
            d2, _ = g.deriv(f, parity, p, 2)
            e1 = deg * r ** (deg - 1) if deg > 0 else 0 * r
            e2 = deg * (deg - 1) * r ** (deg - 2) if deg > 1 else 0 * r
            assert np.allclose(d1, e1, atol=1e-8, rtol=1e-8), (deg, parity)
            assert np.allclose(d2, e2, atol=1e-6, rtol=1e-6), (deg, parity)


@pytest.mark.parametrize("center", ["vertex", "cell", "none"])
def test_convergence_orders_uniform_and_stretched(center):
    Ns = [40, 80, 160, 320]
    for stretched in (False, True):
        e1, e2, eI = [], [], []
        for N in Ns:
            x = grid(center, N, 2.0, rmin=0.5)
            r = x + 0.15 * np.sin(x) if stretched else x      # smooth odd map keeps parity ghosts smooth
            g = FDGrid(r, center)
            f = np.sin(1.3 * r)
            d1, _ = g.deriv(f, -1, 4, 1)
            d2, _ = g.deriv(f, -1, 4, 2)
            e1.append(np.max(np.abs(d1 - 1.3 * np.cos(1.3 * r))))
            e2.append(np.max(np.abs(d2 + 1.69 * np.sin(1.3 * r))))
            eI.append(np.max(np.abs(g.cumint(np.cos(r)) - (np.sin(r) - np.sin(r[0])))))
        for e in (e1, e2, eI):
            assert np.all(order(e, Ns) > 3.6), (center, stretched, order(e, Ns))


def test_parity_ghosts_matter():
    """Without the correct parity the centre derivative is wrong: guards against parity bugs."""
    r = np.linspace(0, 1, 41)
    g = FDGrid(r, "vertex")
    f = np.cos(r)                   # even
    good, _ = g.deriv(f, +1)
    bad, _ = g.deriv(f, -1)
    assert abs(good[0]) < 1e-12
    assert abs(bad[0]) > 1.0


def test_even_extrapolation():
    r = np.array([0.1, 0.2, 0.3])
    assert abs(even_extrapolate_to_zero(r, np.cos(r)) - 1.0) < 1e-6

"""Constraint residuals: convergence, F10 identity (sign +), violation detection, masses."""
import numpy as np
import pytest

from sphdiag import Projections, SphericalSlice, diagnose
import devbg as bg


def generic(N, center):
    if center == "cell":
        r = 4.0 * (np.arange(N) + 0.5) / N
    else:
        r = bg.stretched(N, 0, 4.0)
    kw, ex = bg.generic_slice(r)
    d = diagnose(SphericalSlice(center=center, **kw),
                 Projections(ex["rho"], ex["j_r"], 0 * r, 0 * r))
    return d, ex, kw


def orders(e):
    e = np.asarray(e)
    return np.log2(e[:-1] / e[1:])


@pytest.mark.parametrize("center", ["vertex", "cell"])
def test_generic_slice_convergence(center):
    res = {k: [] for k in ("ham_bulk", "ham_all", "mom", "hamM", "hamM_all", "M", "M_int")}
    for N in (50, 100, 200, 400):
        d, ex, kw = generic(N, center)
        assert d.status["data"] == "OK"
        assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION", d.status["n_violated"]
        res["ham_bulk"].append(d.norms["ham"]["bulk"]["Linf"])
        res["ham_all"].append(d.norms["ham"]["all"]["Linf"])
        res["mom"].append(d.norms["mom"]["all"]["Linf"])
        res["hamM"].append(d.norms["hamM"]["bulk"]["Linf"])
        res["hamM_all"].append(d.norms["hamM"]["all"]["Linf"])
        res["M"].append(np.max(np.abs(d.M - ex["M"])))
        res["M_int"].append(np.max(np.abs(d.M_int - ex["M"])))
    o = {k: orders(v) for k, v in res.items()}
    print(center, {k: np.round(v, 2) for k, v in o.items()})
    assert np.all(o["ham_bulk"][1:] > 3.5)
    assert np.all(o["mom"][1:] > 3.5)
    assert np.all(o["M"][1:] > 3.5)
    assert np.all(o["M_int"][1:] > 3.5)
    assert np.all(o["hamM"][1:] > 3.5)
    # M' = D(M) is a nested derivative (M contains Gamma = D(R)/A): where the stencil shifts
    # (outer edge, one-sided) the error of Gamma is not smooth and M' drops to ~3rd order
    assert np.all(o["hamM_all"][1:] > 2.8)
    # near the centre the H residual converges only as h^2 (1 - Gamma^2 ~ r^2 cancellation)
    assert np.all(o["ham_all"][1:] > 1.7)


def test_F10_sign_via_matter_perturbation():
    """Perturb rho by a bump: dH = -16 pi drho, dH_M = -4 pi R^2 R' drho = +(R^2 R'/4) dH."""
    d0, ex, kw = generic(200, "vertex")
    r = kw["r"]
    drho = 1e-3 * np.exp(-((r - 1.5) / 0.3) ** 2)
    d1 = diagnose(SphericalSlice(center="vertex", **kw),
                  Projections(ex["rho"] + drho, ex["j_r"], 0 * r, 0 * r))
    dH = d1.ham - d0.ham
    dHM = d1.hamM - d0.hamM
    m = r > 0
    np.testing.assert_allclose(dH[m], -16 * np.pi * drho[m], atol=1e-12)
    np.testing.assert_allclose(dHM, (d0.R ** 2 * d0.dR / 4) * np.where(m, dH, 0), atol=1e-12)
    assert d1.status["constraints"] == "VIOLATED"
    assert d1.status["n_violated"]["ham"] > 0 and d1.status["n_violated"]["hamM"] > 0
    assert d1.status["n_violated"]["mom"] == 0
    # the integral mass sees the extra matter, M_MS does not
    assert abs((d1.M_int[-1] - d0.M_int[-1])
               - np.trapezoid(4 * np.pi * d0.R ** 2 * drho * d0.dR, r)) < 1e-6


def test_F10_identity_on_violated_data():
    """H_M = +(R^2 R'/4) H + R^3 K_B M holds up to truncation for arbitrary matter."""
    d0, ex, kw = generic(400, "vertex")
    r = kw["r"]
    d = diagnose(SphericalSlice(center="vertex", **kw),
                 Projections(0.3 * ex["rho"] + 0.01, 2 * ex["j_r"] + 0.01 * r, 0 * r, 0 * r))
    m = (r > 0.2) & (r < 3.5)
    rhs = d.R ** 2 * d.dR / 4 * d.ham + d.R ** 3 * kw["KB"] * d.mom
    assert np.max(np.abs(d.hamM[m] - rhs[m])) < 1e-6 * np.max(np.abs(d.hamM[m]))
    wrong = -d.R ** 2 * d.dR / 4 * d.ham + d.R ** 3 * kw["KB"] * d.mom
    assert np.max(np.abs(d.hamM[m] - wrong[m])) > 1e-2 * np.max(np.abs(d.hamM[m]))


def test_momentum_violation_detected():
    d0, ex, kw = generic(200, "cell")
    r = kw["r"]
    d = diagnose(SphericalSlice(center="cell", **kw),
                 Projections(ex["rho"], ex["j_r"] + 1e-4 * r * np.exp(-r ** 2), 0 * r, 0 * r))
    assert d.status["constraints"] == "VIOLATED"
    assert d.status["n_violated"]["mom"] > 10


def test_sigma_bounds_residual_on_coarse_grids():
    """On consistent data |X| <= kappa_c sigma(X) + floor everywhere (no false VIOLATED)."""
    for N in (30, 60):
        for center in ("vertex", "cell"):
            d, ex, kw = generic(N, center)
            assert d.status["n_violated"] == {"ham": 0, "mom": 0, "hamM": 0}


def test_relative_residual_definition():
    d, ex, kw = generic(100, "vertex")
    i = 30
    G, R, A = d.Gamma[i], d.R[i], kw["A"][i]
    KA, KB, rho = kw["KA"][i], kw["KB"][i], ex["rho"][i]
    terms = [2 * (1 - G ** 2) / R ** 2, -4 * d.dGamma[i] / (A * R), 4 * KA * KB, 2 * KB ** 2,
             -16 * np.pi * rho]
    assert abs(d.ham[i] - sum(terms)) < 1e-12
    assert abs(d.ham_rel[i] - abs(sum(terms)) / sum(abs(t) for t in terms)) < 1e-12


def test_M_flux_matches_milligan_form():
    d, ex, kw = generic(200, "vertex")
    r = kw["r"]
    integrand = 4 * np.pi * d.R ** 2 * d.U * ex["j_r"]
    assert abs(d.M_flux[-1] - np.trapezoid(integrand, r)) < 1e-4 * np.max(np.abs(d.M_flux))
    assert np.max(np.abs(d.M_flux)) > 1e-4   # flux genuinely non-zero in this data
    np.testing.assert_allclose(d.M_int, d.M[0] + d.M_rho + d.M_flux)

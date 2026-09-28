"""Constraints (contract section 5, 9): manufactured solution with exact H = M = 0, measured
convergence orders, violated constraints, the F10 identity, error-estimate quality, norms."""
import numpy as np
import pytest

from conftest import manufactured, manufactured_slice, order
from sphdiag import Projections, SphericalSlice, diagnose

NS = [100, 200, 400]


@pytest.mark.parametrize("center", ["vertex", "cell", "none"])
def test_manufactured_convergence(center):
    res = {k: [] for k in ("ham_bulk", "mom_all", "hamM_bulk", "hamM_center", "M", "Mint", "dR")}
    for N in NS:
        s, p, d = manufactured_slice(center, N)
        D = diagnose(s, p)
        assert D.status["data"] == "OK"
        assert D.status["constraints"] == "CONSISTENT_WITH_TRUNCATION", N
        res["ham_bulk"].append(D.norms["ham"]["bulk"]["Linf"])
        res["mom_all"].append(D.norms["mom"]["all"]["Linf"])
        res["hamM_bulk"].append(D.norms["hamM"]["bulk"]["Linf"])
        res["hamM_center"].append(D.norms["hamM"]["center"]["Linf"])
        res["M"].append(np.max(np.abs(D.M - d["M"])))
        res["Mint"].append(np.max(np.abs(D.M_int - D.M)))
        res["dR"].append(np.max(np.abs(D.dR - d["dR"])))
    for k, e in res.items():
        if center == "none" and k == "hamM_center":
            # documented limitation: M' = D(M) with M built from one-sided R' -> O(h^3) at the edges
            assert np.all(order(e, NS) > 2.8), (center, k, order(e, NS))
            continue
        assert np.all(order(e, NS) > 3.6), (center, k, order(e, NS))


@pytest.mark.parametrize("center", ["vertex", "cell"])
def test_ham_centre_is_2nd_order_hamM_is_regular(center):
    """Documented property: H contains 2(1-Gamma^2)/R^2, so its L_inf near a regular centre
    converges only as h^2; the mass form H_M (regular) keeps 4th order there."""
    eh, em = [], []
    for N in NS:
        s, p, _ = manufactured_slice(center, N)
        D = diagnose(s, p)
        eh.append(D.norms["ham"]["center"]["Linf"])
        em.append(D.norms["hamM"]["center"]["Linf"])
    assert np.all(np.abs(order(eh, NS) - 2) < 0.3)
    assert np.all(order(em, NS) > 3.7)


def test_analytic_derivatives_path():
    s, p, d = manufactured_slice("vertex", 200, analytic=True)
    D = diagnose(s, p)
    assert np.all(D.sigma_dR == 0) and np.all(D.sigma_Gamma == 0)
    assert np.all(D.sigma_M == 0) and np.all(D.sigma_Rtheta_plus == 0)
    assert np.array_equal(D.dR, d["dR"])
    assert D.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    assert np.allclose(D.M, d["M"], rtol=1e-12, atol=1e-15)
    # the only remaining numerical derivative is Gamma' = (D(dR) A - dR A')/A^2 (and M')
    s2, p2, _ = manufactured_slice("vertex", 200, analytic=False)
    D2 = diagnose(s2, p2)
    assert np.nanmax(np.abs(D.mom)) < 1e-12 < np.nanmax(np.abs(D2.mom))


@pytest.mark.parametrize("which", ["rho", "jr"])
@pytest.mark.parametrize("center", ["vertex", "none"])
def test_violated_constraints_detected(which, center):
    N = 300
    s, p, d = manufactured_slice(center, N)
    r = s.r
    bump = 1e-4 * np.exp(-((r - 1.5) / 0.2) ** 2)
    rho, jr = p.rho.copy(), p.j_r.copy()
    if which == "rho":
        rho = rho + bump
    else:
        jr = jr + bump * r
    D = diagnose(s, Projections(rho, jr, p.S_rr, p.S_thth))
    assert D.status["constraints"] == "VIOLATED"
    region = np.abs(r - 1.5) < 0.2
    name = "ham" if which == "rho" else "mom"
    assert np.all(D.constraint_point_status[name][region] == "VIOLATED")
    assert np.all(D.constraint_point_status["hamM"][region] == "VIOLATED")
    far = np.abs(r - 1.5) > 1.2
    assert np.all(D.constraint_point_status[name][far & (s.R > 0)] == "CONSISTENT")
    # the measured residual equals the injected violation
    if which == "rho":
        assert np.allclose(D.ham[region], -16 * np.pi * bump[region], rtol=1e-3)
    else:
        assert np.allclose(D.mom[region], 4 * np.pi * bump[region] * r[region], rtol=1e-3)


def test_mass_form_identity_F10_numerically():
    """H_M = +(R^2 R'/4) H + R^3 K_B M  (the '+' sign; the draft '-' sign fails) on violated data."""
    errs, errs_wrong = [], []
    for N in NS:
        s, p, d = manufactured_slice("none", N)
        r = s.r
        rho = p.rho + 1e-2 * np.exp(-((r - 1.5) / 0.4) ** 2)
        jr = p.j_r + 3e-3 * np.exp(-((r - 2.0) / 0.4) ** 2)
        D = diagnose(s, Projections(rho, jr, p.S_rr, p.S_thth))
        rhs = s.R ** 2 * D.dR / 4 * D.ham + s.R ** 3 * s.KB * D.mom
        rhs_wrong = -s.R ** 2 * D.dR / 4 * D.ham + s.R ** 3 * s.KB * D.mom
        inner = slice(4, -4)
        errs.append(np.max(np.abs(D.hamM - rhs)[inner]))
        errs_wrong.append(np.max(np.abs(D.hamM - rhs_wrong)[inner]))
    assert np.max(np.abs(D.hamM)) > 1e-2
    assert errs[-1] < 1e-7 and np.all(order(errs, NS) > 3.5)
    assert errs_wrong[-1] > 1e-3


def test_sigma_heuristic_quality():
    """sigma(f') = |D4 f - D6 f| estimates the actual error of D4 f (contract section 9)."""
    for center in ("vertex", "cell", "none"):
        for N in (100, 400):
            s, p, d = manufactured_slice(center, N)
            D = diagnose(s, p)
            err = np.abs(D.dR - d["dR"])
            sig = D.sigma_dR
            big = err > 1e3 * np.finfo(float).eps
            ratio = sig[big] / err[big]
            assert 0.8 < np.median(ratio) < 1.25, (center, N, np.median(ratio))
            assert np.mean(sig[big] >= 0.5 * err[big]) > 0.9
            # Gamma' and K_B' estimates
            for est, sg in ((D.dGamma, D.sigma_dGamma), (D.dKB, D.sigma_dKB)):
                ex = d["dGamma"] if est is D.dGamma else d["dKB"]
                e = np.abs(est - ex)
                big = e > 1e5 * np.finfo(float).eps
                rt = sg[big] / e[big]
                assert 0.5 < np.median(rt) < 2.0, (center, N, np.median(rt))
                assert np.mean(sg[big] >= 0.3 * e[big]) > 0.85


def test_sigma_ham_bounds_true_residual():
    """The residual of exact data is pure truncation error: |H| <= kappa_c sigma(H) everywhere."""
    for center in ("vertex", "cell", "none"):
        for N in (60, 200, 800):
            s, p, _ = manufactured_slice(center, N)
            D = diagnose(s, p)
            pos = s.R > 0
            for nm, X, S in (("ham", D.ham, D.sigma_ham), ("mom", D.mom, D.sigma_mom), ("hamM", D.hamM, D.sigma_hamM)):
                sel = pos if nm != "hamM" else np.ones_like(pos)
                assert np.all(np.abs(X[sel]) <= 10 * S[sel] + 1e3 * np.finfo(float).eps * 1e3), (center, N, nm)


def test_relative_residuals_and_norm_structure():
    s, p, d = manufactured_slice("vertex", 200)
    D = diagnose(s, p)
    for nm in ("ham", "mom", "hamM"):
        assert set(D.norms[nm]) == {"all", "center", "bulk", "outer"}
        for reg in D.norms[nm]:
            assert set(D.norms[nm][reg]) == {"Linf", "L2vol", "Linf_rel"}
            for v in D.norms[nm][reg].values():
                assert np.isfinite(v)
        assert D.norms[nm]["all"]["Linf"] == pytest.approx(
            max(D.norms[nm][g]["Linf"] for g in ("center", "bulk", "outer")))
    rel = D.ham_rel[1:]
    assert np.all((rel >= 0) & (rel <= 1))
    # ham_rel = |H| / sum |T_k|  recomputed by hand
    G, R, A = D.Gamma, s.R, s.A
    with np.errstate(all="ignore"):
        T = [2 * (1 - G ** 2) / R ** 2, -4 * D.dGamma / (A * R), 4 * s.KA * s.KB, 2 * s.KB ** 2, -16 * np.pi * p.rho]
        S = np.sum(np.abs(T), axis=0)
    assert np.allclose(D.ham_rel[1:], (np.abs(D.ham) / S)[1:], rtol=1e-6)
    # L2 over proper volume, normalised by the volume (documented choice)
    w = 4 * np.pi * R ** 2 * A
    X = np.nan_to_num(D.ham)
    L2 = np.sqrt(np.trapezoid(X ** 2 * w, s.r) / np.trapezoid(w, s.r))
    assert D.norms["ham"]["all"]["L2vol"] == pytest.approx(L2, rel=1e-12)


def test_rel_undefined_when_all_terms_vanish():
    """Flat space, K = 0, rho = 0: every term of M vanishes exactly -> relative residual undefined (NaN, flagged)."""
    N = 40
    r = np.linspace(0, 2, N)
    z = np.zeros(N)
    s = SphericalSlice(r, np.ones(N), r.copy(), z, z)
    D = diagnose(s, Projections(z, z, z, z))
    assert D.rel_undefined["mom"][1:].all()
    assert np.all(np.isnan(D.mom_rel))
    assert D.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    assert np.isnan(D.norms["mom"]["all"]["Linf_rel"])


@pytest.mark.parametrize("center", ["vertex", "cell", "none"])
def test_noise_guard_is_harmless_on_smooth_data(center):
    for N in (100, 400):
        s, p, _ = manufactured_slice(center, N)
        D0, D1 = diagnose(s, p), diagnose(s, p, noise_guard=True)
        assert D1.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
        assert D1.status["geometry"] == D0.status["geometry"]
        assert np.all(D1.sigma_Rtheta_plus >= D0.sigma_Rtheta_plus)
        assert [rt["type"] for rt in D1.roots] == [rt["type"] for rt in D0.roots]


@pytest.mark.parametrize("center", ["vertex", "none"])
def test_nonuniform_grid_full_pipeline(center):
    """Smoothly stretched grid r = x + 0.25 x^3/(1+x^2) (odd map -> smooth parity ghosts)."""
    errs = {"ham": [], "mom": [], "hamM": [], "M": []}
    for N in NS:
        x = np.linspace(0.0, 3.0, N) if center == "vertex" else np.linspace(0.4, 3.0, N)
        r = x + 0.25 * x ** 3 / (1 + x ** 2)
        d = manufactured(r)
        s = SphericalSlice(r, d["A"], d["R"], d["KA"], d["KB"], center)
        z = np.zeros(N)
        D = diagnose(s, Projections(d["rho"], d["jr"], z, z))
        assert D.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
        for k in ("ham", "mom", "hamM"):
            errs[k].append(D.norms[k]["bulk"]["Linf"])
        errs["M"].append(np.max(np.abs(D.M - d["M"])))
    for k, e in errs.items():
        assert np.all(order(e, NS) > 3.5), (k, order(e, NS))


def test_sigma_propagation_formulas_and_gamma_error():
    """Section 9 propagation: sigma(Gamma) = sigma(R')/A, sigma(R theta) = 2 sigma(Gamma),
    sigma(M) = R |Gamma| sigma(Gamma); and sigma(Gamma) matches the true error of Gamma for A >> 1."""
    s, p, d = manufactured_slice("vertex", 300, a=3.0)          # A(0) = 4
    D = diagnose(s, p)
    assert np.allclose(D.sigma_Gamma, D.sigma_dR / s.A, rtol=1e-14, atol=0)
    assert np.allclose(D.sigma_Rtheta_plus, 2 * D.sigma_Gamma, rtol=1e-14, atol=0)
    assert np.allclose(D.sigma_Rtheta_minus, 2 * D.sigma_Gamma, rtol=1e-14, atol=0)
    assert np.allclose(D.sigma_M, s.R * np.abs(D.Gamma) * D.sigma_Gamma, rtol=1e-14, atol=0)
    err = np.abs(D.Gamma - d["Gamma"])
    big = err > 1e-12
    ratio = D.sigma_Gamma[big] / err[big]
    assert 0.7 < np.median(ratio) < 1.4, np.median(ratio)

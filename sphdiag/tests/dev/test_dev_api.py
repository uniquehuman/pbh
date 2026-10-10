"""API completeness (contract §10) and matter adapters (§4)."""
import numpy as np
from scipy.integrate import solve_ivp

import sphdiag
from sphdiag import (DIAGNOSTICS_FIELDS, Projections, ScalarField, SphericalSlice, VlasovMoments,
                     diagnose, pi_from_dtphi)
import devbg as bg

CONTRACT_FIELDS = [
    "Gamma", "U", "dR", "M", "sigma_M", "compactness", "M_over_R3", "Rtheta_plus",
    "Rtheta_minus", "sigma_Rtheta_plus", "sigma_Rtheta_minus", "theta_plus", "theta_minus",
    "sphere_class", "roots", "outer_ah_candidate", "projections", "ham", "ham_rel", "mom",
    "mom_rel", "hamM", "hamM_rel", "sigma_ham", "sigma_mom", "sigma_hamM", "norms", "M_rho",
    "M_flux", "M_int", "M_prop", "center_checks", "status", "M_over_R3_extrapolated"]
ROOT_KEYS = {"which", "r", "R", "R_lo", "R_hi", "M", "type", "ah_candidate", "other_sign",
             "inner_sign", "outer_sign"}


def test_exports():
    for name in ("SphericalSlice", "ScalarField", "VlasovMoments", "Projections",
                 "pi_from_dtphi", "diagnose"):
        assert hasattr(sphdiag, name)
    assert set(CONTRACT_FIELDS) <= set(DIAGNOSTICS_FIELDS)


def test_all_fields_present_and_shaped():
    r = bg.stretched(300, 0.4, 10.0)
    kw, ex = bg.schwarzschild_pg(r)
    d = diagnose(SphericalSlice(center="none", **kw), Projections(0 * r, 0 * r, 0 * r, 0 * r))
    N = r.size
    for f in CONTRACT_FIELDS:
        assert hasattr(d, f), f
    for f in ("Gamma", "U", "dR", "M", "sigma_M", "compactness", "M_over_R3", "Rtheta_plus",
              "Rtheta_minus", "sigma_Rtheta_plus", "sigma_Rtheta_minus", "theta_plus",
              "theta_minus", "sphere_class", "ham", "ham_rel", "mom", "mom_rel", "hamM",
              "hamM_rel", "sigma_ham", "sigma_mom", "sigma_hamM", "M_rho", "M_flux", "M_int",
              "M_prop"):
        assert np.shape(getattr(d, f)) == (N,), f
    assert d.roots and ROOT_KEYS <= set(d.roots[0])
    assert d.roots[0]["which"] in ("+", "-")
    assert set(d.norms) == {"ham", "mom", "hamM"}
    for nm in d.norms:
        assert set(d.norms[nm]) == {"all", "inner", "bulk", "outer"}   # center='none' (v1.1)
        for rg in d.norms[nm]:
            assert set(d.norms[nm][rg]) == {"Linf", "L2vol", "Linf_rel"}
            assert np.isfinite(d.norms[nm][rg]["Linf"])
    assert {"Gamma0", "KA_minus_KB0"} <= set(d.center_checks)
    assert {"data", "constraints", "geometry", "marginal_present"} <= set(d.status)
    assert isinstance(d.projections, Projections)


def test_sign_objects_compare_to_ints_and_strings():
    r = bg.stretched(300, 0.4, 10.0)
    kw, ex = bg.schwarzschild_pg(r)
    root = diagnose(SphericalSlice(center="none", **kw)).roots[0]
    assert root["other_sign"] == -1 and root["other_sign"] == "-" and root["other_sign"] < 0
    assert root["outer_sign"] == 1 and root["outer_sign"] == "+"


def test_norm_definitions():
    r = bg.stretched(120, 0.4, 10.0)
    kw, ex = bg.schwarzschild_pg(r)
    d = diagnose(SphericalSlice(center="none", **kw), Projections(0 * r, 0 * r, 0 * r, 0 * r))
    X = d.hamM
    assert d.norms["hamM"]["all"]["Linf"] == np.max(np.abs(X))
    L2 = np.sqrt(np.trapezoid(X ** 2 * 4 * np.pi * d.R ** 2 * kw["A"], r))
    assert abs(d.norms["hamM"]["all"]["L2vol"] - L2) <= 1e-12 * L2
    assert d.norms["hamM"]["outer"]["Linf"] == np.max(np.abs(X[-4:]))
    assert d.norms["hamM"]["inner"]["Linf"] == np.max(np.abs(X[:6]))  # max(4, 120//20)


def test_pi_from_dtphi():
    assert pi_from_dtphi(1.0, 2.0, 2.0, 0.5) == (1.0 - 0.5 * 2.0) / 2.0
    a = pi_from_dtphi(np.array([1.0, 2.0]), np.array([0.0, 1.0]), np.array([1.0, 2.0]),
                      np.array([0.3, 0.3]))
    np.testing.assert_allclose(a, [1.0, (2.0 - 0.3) / 2.0])


def test_scalar_field_projections_formulae_and_order():
    errs = []
    for N in (60, 120, 240):
        r = bg.stretched(N, 0, 4)
        A = 1 + 0.2 * r ** 2 * np.exp(-r ** 2)
        kw = dict(r=r, A=A, R=r * (1 + 0.1 * r ** 2 * np.exp(-r ** 2)), KA=0 * r, KB=0 * r)
        phi = 0.3 * np.exp(-r ** 2)
        dphi = -0.6 * r * np.exp(-r ** 2)
        Pi = 0.1 * np.cos(r)
        m = 0.7
        p = ScalarField(phi, Pi, m).projections(SphericalSlice(center="vertex", **kw))
        Phi = dphi / A
        V = 0.5 * m ** 2 * phi ** 2
        np.testing.assert_allclose(p.rho, 0.5 * Pi ** 2 + 0.5 * Phi ** 2 + V, atol=1e-4)
        np.testing.assert_allclose(p.S_rr, 0.5 * Pi ** 2 + 0.5 * Phi ** 2 - V, atol=1e-4)
        np.testing.assert_allclose(p.S_thth, 0.5 * Pi ** 2 - 0.5 * Phi ** 2 - V, atol=1e-4)
        errs.append(np.max(np.abs(p.j_r - (-Pi * dphi))))
        assert abs(p.j_r[0]) < 1e-14
    o = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(o > 3.5), o


def test_vlasov_adapter():
    r = np.linspace(0, 1, 20)
    A = 1 + r ** 2
    kw = dict(r=r, A=A, R=r, KA=0 * r, KB=0 * r)
    p = VlasovMoments(1 + r, r, 2 + r, 3 + r).projections(SphericalSlice(center="vertex", **kw))
    np.testing.assert_allclose(p.j_r, A * r)
    np.testing.assert_allclose(p.rho, 1 + r)
    np.testing.assert_allclose(p.S_rr, 2 + r)
    np.testing.assert_allclose(p.S_thth, 3 + r)


def test_fkg_homogeneous_scalar_slice():
    """Flat FLRW slice of a homogeneous massive scalar: H^2 = 8 pi rho / 3 (F6)."""
    m, phi0, dphi0 = 2.0, 0.2, -0.15
    rho = 0.5 * dphi0 ** 2 + 0.5 * m ** 2 * phi0 ** 2
    H = np.sqrt(8 * np.pi / 3 * rho)
    r = np.linspace(0, 3 / H, 200)
    kw, ex = bg.flrw(r, H, a=1.7)
    sf = ScalarField(phi0 + 0 * r, dphi0 + 0 * r, m)
    d = diagnose(SphericalSlice(center="vertex", **kw), sf)
    np.testing.assert_allclose(d.projections.rho, rho, rtol=1e-14)
    np.testing.assert_allclose(d.projections.S_rr, 0.5 * dphi0 ** 2 - 0.5 * m ** 2 * phi0 ** 2)
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    assert d.norms["ham"]["all"]["Linf_rel"] < 1e-10
    assert d.status["geometry"] == "PAST_TRAPPED_ONLY"
    assert abs(d.roots[0]["R"] - 1 / H) < 1e-8


def test_scalar_field_time_symmetric_star_converges():
    """K = 0, Pi = 0, areal R = r: M' = 4 pi r^2 rho(M) solved to 1e-13; H must converge."""
    m = 1.0
    amp = 0.05

    def phi(r):
        return amp * np.exp(-r ** 2)

    def dphi(r):
        return -2 * r * amp * np.exp(-r ** 2)

    def rhs(r, M):
        return 4 * np.pi * r ** 2 * (0.5 * (1 - 2 * M / r) * dphi(r) ** 2
                                     + 0.5 * m ** 2 * phi(r) ** 2) if r > 0 else [0.0]

    errs = []
    for N in (50, 100, 200):
        r = np.linspace(0, 5, N)
        sol = solve_ivp(rhs, (0, 5), [0.0], t_eval=r, rtol=1e-13, atol=1e-16, method="DOP853")
        Mr = sol.y[0]
        with np.errstate(all="ignore"):
            A = np.where(r > 0, 1 / np.sqrt(1 - 2 * Mr / np.where(r > 0, r, 1)), 1.0)
        kw = dict(r=r, A=A, R=r.copy(), KA=0 * r, KB=0 * r)
        d = diagnose(SphericalSlice(center="vertex", **kw), ScalarField(phi(r), 0 * r, m))
        assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
        assert d.status["geometry"] == "NO_TRAPPED_SPHERES"
        errs.append(np.nanmax(np.abs(d.ham[(r > 1.0) & (r < 4.5)])))
        np.testing.assert_allclose(d.M, Mr, atol=1e-5)
        np.testing.assert_allclose(d.M_int, Mr, atol=1e-5)
    o = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print("scalar star ham bulk errors", errs, "orders", o)
    assert np.all(o > 3.0)

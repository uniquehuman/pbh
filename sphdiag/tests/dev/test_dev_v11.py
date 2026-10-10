"""Contract v1.1 features and the tests requested by the audit (A6-A13) for impl-B v0.2."""
import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from sphdiag import Projections, ScalarField, SphericalSlice, diagnose
from sphdiag.diagnose import _find_roots
import devbg as bg


def vac(r):
    return Projections(0 * r, 0 * r, 0 * r, 0 * r)


# ---- A13 mutation survivors -------------------------------------------------------------------
def test_M_prop_uses_A():
    r = np.linspace(0, 3.0, 200)
    kw, ex = bg.generic_slice(r)
    d = diagnose(SphericalSlice(center="vertex", **kw), Projections(ex["rho"], ex["j_r"], 0 * r, 0 * r))

    def integrand(x):
        k, e = bg.generic_slice(np.array([x]))
        return 4 * np.pi * k["R"][0] ** 2 * k["A"][0] * e["rho"][0]

    ref = quad(integrand, 0, 3.0, epsabs=1e-13, limit=200)[0]
    assert abs(d.M_prop[-1] - ref) < 1e-7 * abs(ref)
    assert abs(kw["A"] - 1).max() > 0.05     # A != 1 genuinely matters here


def test_L2vol_uses_A():
    r = np.linspace(0, 3.0, 120)
    kw, ex = bg.generic_slice(r)
    d = diagnose(SphericalSlice(center="vertex", **kw),
                 Projections(1.01 * ex["rho"], ex["j_r"], 0 * r, 0 * r))
    X = d.hamM
    L2 = np.sqrt(np.trapezoid(X ** 2 * 4 * np.pi * d.R ** 2 * kw["A"], r))
    assert abs(d.norms["hamM"]["all"]["L2vol"] - L2) <= 1e-12 * L2
    Lwrong = np.sqrt(np.trapezoid(X ** 2 * 4 * np.pi * d.R ** 2, r))
    assert abs(Lwrong - L2) > 1e-3 * L2


def test_sigma_M_formula_with_Gamma_not_one():
    r = bg.stretched(200, 0.5, 10.0)
    kw, ex = bg.schwarzschild_ks(r)
    d = diagnose(SphericalSlice(center="none", **kw))
    assert np.max(np.abs(d.Gamma - 1)) > 0.3
    np.testing.assert_allclose(d.sigma_M, d.R * np.abs(d.Gamma) * d.sigma_Gamma, rtol=1e-14)


def test_unresolved_past_marginal_is_uncertain_not_past_trapped():
    # flat FLRW, Hubble sphere R = 1/H = 2 exactly on the last node: (+,0) with no resolved side
    H = 0.5
    r = np.linspace(0, 2.0, 101)
    kw, ex = bg.flrw(r, H)
    d = diagnose(SphericalSlice(center="vertex", **kw), Projections(ex["rho"], 0 * r, 0 * r, 0 * r))
    assert d.sphere_class[-1] == "PAST_MARGINAL"
    assert not np.any(d.sphere_class == "PAST_TRAPPED")
    assert d.status["geometry"] == "UNCERTAIN"


def test_other_sign_checked_over_uncertainty_interval():
    r = np.linspace(0.0, 2.0, 21)
    v = r - 1.0 + 1e-3                          # root near r = 1
    tau = np.full_like(r, 0.45)                 # wide band -> r interval about [0.55, 1.45]
    vo = r - 1.2                                # other: -0.2 at the root, +0.25 at r_hi
    tauo = np.full_like(r, 0.1)
    s = np.where(np.abs(v) > tau, np.sign(v), 0).astype(int)
    one = np.ones_like(r)
    roots = _find_roots("+", r, r, one, one, v, tau, s, vo, tauo, np.ones(r.size, bool))
    assert len(roots) == 1
    assert roots[0]["r_hi"] > 1.2
    assert roots[0]["other_sign"] == 0 and roots[0]["type"] == "DEGENERATE"
    # control: narrow band -> the other sign is resolved (-)
    tau2 = np.full_like(r, 0.01)
    s2 = np.where(np.abs(v) > tau2, np.sign(v), 0).astype(int)
    roots2 = _find_roots("+", r, r, one, one, v, tau2, s2, vo, tauo, np.ones(r.size, bool))
    assert roots2[0]["other_sign"] == -1 and roots2[0]["type"] == "FUTURE_MOTS"


def test_cell_center_gamma_is_extrapolated():
    N = 40
    r = 1.0 * (np.arange(N) + 0.5) / N          # r0 = 0.0125, Gamma(r0) - 1 ~ (3b - a) r0^2
    kw, ex = bg.generic_slice(r, a=0.3, b=0.3)
    d = diagnose(SphericalSlice(center="cell", **kw))
    assert abs(d.Gamma[0] - 1) > 5e-5
    assert abs(d.center_checks["Gamma0"] - 1) < 0.05 * abs(d.Gamma[0] - 1)


def test_nonfinite_integral_masses_are_numerical_failure():
    r = np.linspace(0, 10.0, 60)
    kw, ex = bg.minkowski_flat(r)
    d = diagnose(SphericalSlice(center="vertex", **kw),
                 Projections(1e305 + 0 * r, 0 * r, 0 * r, 0 * r))
    assert np.all(np.isfinite(d.ham[1:])) and np.all(np.isfinite(d.hamM))
    assert not np.all(np.isfinite(d.M_rho))
    assert d.status["data"] == "NUMERICAL_FAILURE"
    assert {"M_rho", "M_int", "M_prop"} <= set(d.status["numerical_failures"])
    assert d.status["geometry"] == "UNDETERMINED"


def test_fkg_analytic_values():
    m, phi0, dphi0, a = 2.0, 0.2, -0.15, 1.7
    rho = 0.5 * dphi0 ** 2 + 0.5 * m ** 2 * phi0 ** 2
    H = np.sqrt(8 * np.pi / 3 * rho)
    r = np.linspace(0, 3 / (a * H), 200)
    kw, ex = bg.flrw(r, H, a=a)
    d = diagnose(SphericalSlice(center="vertex", **kw), ScalarField(phi0 + 0 * r, dphi0 + 0 * r, m))
    np.testing.assert_allclose(d.projections.rho, rho, rtol=1e-14)
    np.testing.assert_allclose(d.projections.S_thth, 0.5 * dphi0 ** 2 - 0.5 * m ** 2 * phi0 ** 2)
    assert np.max(np.abs(d.projections.j_r)) < 1e-14
    R = a * r
    np.testing.assert_allclose(d.M, 0.5 * H ** 2 * R ** 3, atol=1e-12 * (H ** 2 * R[-1] ** 3))
    np.testing.assert_allclose(d.M, 4 * np.pi / 3 * rho * R ** 3, rtol=1e-12, atol=1e-15)
    assert abs(d.M_over_R3[0] - 4 * np.pi / 3 * rho) < 1e-10
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"


# ---- A6 / C1: horizon with Gamma < 0 (R decreasing with the label) ------------------------------
def test_ah_candidate_with_decreasing_R():
    Mb = 1.0
    x = np.linspace(0.4, 12.0, 300)
    R = 12.4 - x
    kw = dict(r=x, A=np.ones_like(x), R=R, KA=-0.5 * np.sqrt(2 * Mb / R ** 3),
              KB=np.sqrt(2 * Mb / R ** 3))
    d = diagnose(SphericalSlice(center="none", **kw), vac(x))
    assert d.status["geometry"] == "AH_CANDIDATE"
    ah = d.outer_ah_candidate
    assert ah["which"] == "-" and ah["Gamma"] < 0 and ah["type"] == "FUTURE_MOTS"
    assert ah["smallR_sign"] == "-" and ah["largeR_sign"] == "+"
    assert ah["R_lo"] <= 2 * Mb <= ah["R_hi"]
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    # the same physics with increasing label gives the same verdict and horizon
    r2 = 12.4 - x[::-1]
    kw2, _ = bg.schwarzschild_pg(r2, Mb)
    d2 = diagnose(SphericalSlice(center="none", **kw2), vac(r2))
    assert d2.status["geometry"] == "AH_CANDIDATE"
    assert abs(d2.outer_ah_candidate["R"] - ah["R"]) < 1e-6


# ---- scalar field with flux and K != 0 satisfying both constraints -----------------------------
def scalar_flux_data(N, rmax=5.0, m=1.0):
    """Areal gauge R = r. phi, Pi, K_B chosen; M from M' = 4 pi r^2 (rho - r K_B j_r) (H = 0 via
    F10 with R' = 1), A = (1 - 2M/r + r^2 K_B^2)^(-1/2); K_A from the momentum constraint."""
    phi = lambda x: 0.1 * np.exp(-x ** 2)
    dphi = lambda x: -0.2 * x * np.exp(-x ** 2)
    Pi = lambda x: 0.08 * (1 + x ** 2) * np.exp(-x ** 2)
    KB = lambda x: 0.3 * np.exp(-x ** 2)
    dKB = lambda x: -0.6 * x * np.exp(-x ** 2)

    def rhs(x, y):
        if x == 0:
            return [0.0]
        M = y[0]
        A2 = 1.0 / (1 - 2 * M / x + x ** 2 * KB(x) ** 2)
        rho = 0.5 * Pi(x) ** 2 + 0.5 * dphi(x) ** 2 / A2 + 0.5 * m ** 2 * phi(x) ** 2
        jr = -Pi(x) * dphi(x)
        return [4 * np.pi * x ** 2 * (rho - x * KB(x) * jr)]

    r = np.linspace(0, rmax, N)
    sol = solve_ivp(rhs, (0, rmax), [0.0], t_eval=r, rtol=1e-13, atol=1e-16, method="DOP853")
    M = sol.y[0]
    with np.errstate(all="ignore"):
        A = np.where(r > 0, 1 / np.sqrt(1 - 2 * M / np.where(r > 0, r, 1) + r ** 2 * KB(r) ** 2), 1.0)
    jr = -Pi(r) * dphi(r)
    KA = KB(r) + r * (dKB(r) + 4 * np.pi * jr)
    kw = dict(r=r, A=A, R=r.copy(), KA=KA, KB=KB(r))
    return kw, ScalarField(phi(r), Pi(r), m), M


def test_scalar_field_with_flux_and_extrinsic_curvature():
    e_ham, e_mom = [], []
    for N in (100, 200, 400):
        kw, sf, Mref = scalar_flux_data(N)
        d = diagnose(SphericalSlice(center="vertex", **kw), sf)
        assert d.status["data"] == "OK"
        assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION", d.status
        assert np.max(np.abs(d.projections.j_r)) > 1e-3 and np.max(np.abs(kw["KA"])) > 0.1
        assert np.max(np.abs(d.M_flux)) > 1e-5
        np.testing.assert_allclose(d.M_int, Mref, atol=1e-6)
        m = (kw["r"] > 1.0) & (kw["r"] < 4.5)
        e_ham.append(np.max(np.abs(d.ham[m])))
        e_mom.append(np.max(np.abs(d.mom[m])))
    o_h = np.log2(np.array(e_ham[:-1]) / np.array(e_ham[1:]))
    o_m = np.log2(np.array(e_mom[:-1]) / np.array(e_mom[1:]))
    print("scalar flux orders ham", o_h, "mom", o_m)
    assert np.all(o_h > 3.5) and np.all(o_m > 3.5)
    # a wrong flux sign is detected
    kw, sf, _ = scalar_flux_data(200)
    d = diagnose(SphericalSlice(center="vertex", **kw), ScalarField(sf.phi, -sf.Pi, 1.0))
    assert d.status["constraints"] == "VIOLATED" and d.status["n_violated"]["mom"] > 0


# ---- C8: R = 0 away from a vertex centre ------------------------------------------------------
def test_R_zero_not_at_vertex_is_invalid():
    chi = np.linspace(0, np.pi, 101)
    kw, ex = bg.flrw(chi, 0.0, k=1)
    kw["R"] = kw["R"].copy()
    kw["R"][-1] = 0.0                                  # antipode of a closed universe
    d = diagnose(SphericalSlice(center="vertex", **kw))
    assert d.status["data"] == "INVALID_INPUT" and d.status["geometry"] == "UNDETERMINED"
    r = np.linspace(0, 1, 20)
    kw, ex = bg.minkowski_flat(r)
    d = diagnose(SphericalSlice(center="none", **kw))
    assert d.status["data"] == "INVALID_INPUT"


# ---- C7: resolution ---------------------------------------------------------------------------
def test_glued_balls_kink_is_not_consistent():
    x = np.linspace(0.02, 2.5, 101)
    R = np.abs(x - 1.0)
    d = diagnose(SphericalSlice(r=x, A=np.ones_like(x), R=R, KA=0 * x, KB=0 * x, center="none"),
                 vac(x))
    assert d.status["constraints"] in ("UNRESOLVED", "VIOLATED")


def test_coarse_centre_is_unresolved_fine_is_consistent():
    kw, ex = bg.generic_slice(np.linspace(0, 4.0, 12))
    d = diagnose(SphericalSlice(center="vertex", **kw),
                 Projections(ex["rho"], ex["j_r"], 0 * kw["r"], 0 * kw["r"]))
    assert d.status["constraints"] == "UNRESOLVED"
    assert sum(d.status["n_unresolved"].values()) > 0
    kw, ex = bg.generic_slice(np.linspace(0, 4.0, 200))
    d = diagnose(SphericalSlice(center="vertex", **kw),
                 Projections(ex["rho"], ex["j_r"], 0 * kw["r"], 0 * kw["r"]))
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"


def test_violated_floor_on_flat_space_in_curvilinear_label():
    for N in (20, 60, 200, 1000):
        r = np.linspace(0, 3, N)
        R = r + 0.3 * r ** 3
        d = diagnose(SphericalSlice(r=r, A=1 + 0.9 * r ** 2, R=R, KA=0 * r, KB=0 * r,
                                    center="vertex"), vac(r))
        assert d.status["n_violated"] == {"ham": 0, "mom": 0, "hamM": 0}


# ---- C3 / A10: hyperboloid far out: no PAST_TRAPPED_ONLY from unresolved signs ----------------
def test_far_hyperboloid_not_past_trapped():
    r = np.linspace(0, 1e7, 4000)
    kw, ex = bg.minkowski_hyperboloid(r)
    d = diagnose(SphericalSlice(center="vertex", **kw))
    assert not np.any(d.sphere_class == "PAST_TRAPPED")
    assert d.status["geometry"] in ("NO_TRAPPED_SPHERES", "UNCERTAIN")


# ---- C6: noise in the zero band ---------------------------------------------------------------
def test_noise_widens_zero_band():
    r = np.linspace(0.4, 12.0, 400)
    kw, ex = bg.schwarzschild_pg(r)
    d0 = diagnose(SphericalSlice(center="none", **kw))
    rng = np.random.default_rng(3)
    kw2 = dict(kw, KB=kw["KB"] * (1 + 1e-4 * rng.standard_normal(r.size)))
    d = diagnose(SphericalSlice(center="none", **kw2))
    assert np.median(d.tau_plus) > 30 * np.median(d0.tau_plus)
    assert np.median(d.tau_plus) > 1e-4


# ---- C9: warnings -----------------------------------------------------------------------------
def test_warnings():
    r = np.linspace(0, 1, 50)
    d = diagnose(SphericalSlice(r=r, A=np.ones(50), R=1.05 * r, KA=0.2 + 0 * r, KB=0 * r,
                                center="vertex"))
    assert d.status["center_deviation"] is True
    assert any(w.startswith("CENTER_IRREGULAR:") for w in d.status["warnings"])
    # M_int mismatch: matter inconsistent with geometry
    kw, ex = bg.generic_slice(np.linspace(0, 3, 100))
    d = diagnose(SphericalSlice(center="vertex", **kw),
                 Projections(2 * ex["rho"], ex["j_r"], 0 * kw["r"], 0 * kw["r"]))
    assert d.status["M_int_mismatch"] > 1e-2
    assert any(w.startswith("M_INT_MISMATCH:") for w in d.status["warnings"])
    # negative mass: Gamma > 1 at rest (K = 0) -> M < 0
    r = np.linspace(0, 2, 80)
    R = r + 0.2 * r ** 3
    d = diagnose(SphericalSlice(r=r, A=np.ones_like(r), R=R, KA=0 * r, KB=0 * r, center="vertex"))
    assert d.status["n_M_negative"] > 0
    # undefined relative residuals (flat FLRW: momentum terms all zero)
    kw, ex = bg.flrw(np.linspace(0, 1, 40), 1.0)
    d = diagnose(SphericalSlice(center="vertex", **kw),
                 Projections(ex["rho"], 0 * kw["r"], 0 * kw["r"], 0 * kw["r"]))
    assert d.status["n_rel_undefined"]["mom"] > 0
    assert np.all(np.isnan(d.mom_rel[1:][d.mom[1:] == 0]))
    # clean data: no warnings
    kw, ex = bg.flrw(np.linspace(0, 1, 40), 1.0)
    d = diagnose(SphericalSlice(center="vertex", **kw),
                 Projections(ex["rho"], 1e-3 * kw["r"], 0 * kw["r"], 0 * kw["r"]))
    assert not any(w.startswith(("CENTER_IRREGULAR", "NEGATIVE_MASS")) for w in d.status["warnings"])

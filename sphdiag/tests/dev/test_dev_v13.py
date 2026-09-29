"""Contract v1.3: C12 (sigma of derivatives of module-computed fields) and C13 (warning codes)."""
import numpy as np

from sphdiag import Projections, ScalarField, SphericalSlice, diagnose
import devbg as bg

CODES = ("CENTER_IRREGULAR", "M_INT_MISMATCH", "NEGATIVE_MASS", "REL_UNDEFINED",
         "UNRESOLVED_POINTS", "UNRESOLVED_INSIDE_AH", "CELL_BALL_EXCLUDED", "ALPHA_BETA", "OTHER")


def generic(N, fac=1.0, bump=None):
    r = np.linspace(0, 4.0, N)
    kw, ex = bg.generic_slice(r)
    rho = ex["rho"] * fac + (0 if bump is None else bump(r))
    return diagnose(SphericalSlice(center="vertex", **kw), Projections(rho, ex["j_r"], 0 * r, 0 * r))


def max_margin(d, name):
    X, s = getattr(d, name), getattr(d, "sigma_" + name)
    m = np.isfinite(X) & (s > 0)
    return float(np.max(np.abs(X[m]) / s[m]))


def test_sigma_of_derived_field_derivatives_contains_nodal_term():
    d = generic(100)
    slc = SphericalSlice(center="vertex", **bg.generic_slice(np.linspace(0, 4.0, 100))[0])
    op = slc.op
    assert np.all(d.sigma_dM >= op.abs_apply(d.sigma_M, 1) * (1 - 1e-12))
    assert np.all(d.sigma_dGamma >= op.abs_apply(d.sigma_Gamma, 1) * (1 - 1e-12))


def test_consistent_smooth_data_margin_small_everywhere_incl_centre():
    # v0.3: hamM margin 6.1 at N = 50 (close to kappa_c = 10); v0.4: < 3 on all grids
    for N in (50, 100, 200):
        d = generic(N)
        assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
        for nm in ("ham", "mom", "hamM"):
            assert max_margin(d, nm) < 3.0, (N, nm, max_margin(d, nm))


def test_deliberate_violations_still_detected():
    rc = abs(bg.generic_slice(np.array([0.0]))[1]["rho"][0])
    centre_bump = lambda r: 1e-4 * rc * np.exp(-((r - 0.3) / 0.1) ** 2)
    for N in (200, 400):
        d = generic(N, fac=1 + 1e-4)                       # 1e-4 relative everywhere
        assert d.status["constraints"] == "VIOLATED" and max_margin(d, "ham") > 50
        d = generic(N, bump=centre_bump)                   # 1e-4 relative, centre region only
        assert d.status["constraints"] == "VIOLATED"
        assert d.status["n_violated"]["ham"] > 0 and d.status["n_violated"]["hamM"] > 0
    d = generic(100, fac=1 + 1e-3)                         # coarse grid: 1e-3 still detected
    assert d.status["constraints"] == "VIOLATED"


def _all_warnings():
    out = []
    r = np.linspace(0, 1, 50)
    out += diagnose(SphericalSlice(r=r, A=np.ones(50), R=1.05 * r, KA=0.2 + 0 * r, KB=0 * r,
                                   center="vertex")).status["warnings"]
    r = np.linspace(0, 2, 80)
    out += diagnose(SphericalSlice(r=r, A=np.ones_like(r), R=r + 0.2 * r ** 3, KA=0 * r, KB=0 * r,
                                   center="vertex")).status["warnings"]
    kw, ex = bg.generic_slice(np.linspace(0, 3, 100))
    out += diagnose(SphericalSlice(center="vertex", **kw),
                    Projections(2 * ex["rho"], ex["j_r"], 0 * kw["r"], 0 * kw["r"])).status["warnings"]
    kw, ex = bg.flrw(np.linspace(0, 1, 40), 1.0)
    out += diagnose(SphericalSlice(center="vertex", **kw),
                    Projections(ex["rho"], 0 * kw["r"], 0 * kw["r"], 0 * kw["r"])).status["warnings"]
    rc = (np.arange(40) + 0.5) / 40
    kw, ex = bg.flrw(rc, 1.0)
    out += diagnose(SphericalSlice(center="cell", **kw),
                    Projections(ex["rho"], 0 * rc, 0 * rc, 0 * rc)).status["warnings"]
    x = np.linspace(0.4, 12.0, 300)
    kw, ex = bg.schwarzschild_pg(x)
    kw["KB"] = kw["KB"].copy()
    kw["KB"][5] *= 3.0
    out += diagnose(SphericalSlice(center="none", **kw)).status["warnings"]
    r = np.linspace(0.2, 4, 100)
    kw, ex = bg.schwarzschild_iso(r)
    alpha = (1 - 1 / (2 * r)) / (1 + 1 / (2 * r))
    out += diagnose(SphericalSlice(center="none", alpha=alpha, beta=r[:3], **kw)).status["warnings"]
    r = np.linspace(0, 1, 30)
    r[0] = 1e-18
    kw, ex = bg.minkowski_flat(r)
    kw["R"] = r.copy()
    out += diagnose(SphericalSlice(center="vertex", **kw)).status["warnings"]
    return out


def test_warning_codes():
    ws = _all_warnings()
    seen = {w.split(":", 1)[0] for w in ws}
    for w in ws:
        code, sep, text = w.partition(":")
        assert sep == ":" and code in CODES and text.strip(), w
    assert set(CODES) - seen <= set(), sorted(set(CODES) - seen)

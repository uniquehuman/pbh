"""Regular centre: limits of derivation.md §4, centre checks, cell centre."""
import numpy as np

from sphdiag import Projections, SphericalSlice, diagnose
import devbg as bg


def test_center_values_flrw_vertex():
    H = 0.8
    r = bg.stretched(200, 0, 2.0)
    kw, ex = bg.flrw(r, H)
    d = diagnose(SphericalSlice(center="vertex", **kw), Projections(ex["rho"], 0 * r, 0 * r, 0 * r))
    assert d.sphere_class[0] == "CENTER"
    assert np.isnan(d.theta_plus[0]) and np.isnan(d.theta_minus[0])
    assert d.Rtheta_plus[0] == 2.0 and d.Rtheta_minus[0] == -2.0
    # R theta_pm = +-2 - 2 A(0) K_B(0) r + O(r^2) (here exact: 2(HR +- 1))
    np.testing.assert_allclose(d.Rtheta_plus[1:5], 2 + 2 * H * r[1:5], atol=1e-10)
    assert d.compactness[0] == 0.0
    assert d.M_over_R3_extrapolated is True
    assert abs(d.M_over_R3[0] - 4 * np.pi / 3 * ex["rho"][0]) < 1e-8
    assert np.isnan(d.ham[0]) and np.isnan(d.mom[0]) and np.isfinite(d.hamM[0])
    assert d.ham_status[0] == "CENTER_EXCLUDED"
    assert d.status["data"] == "OK"
    cc = d.center_checks
    assert abs(cc["Gamma0"] - 1) < 1e-12 and abs(cc["KA_minus_KB0"]) < 1e-15


def test_M_over_R3_limit_generic_converges():
    # uniform grid: a grid map r(x) that is not odd in x (e.g. bg.stretched) makes the mirrored
    # grid non-smooth at r = 0 and the centre limit converges only at ~3rd order there
    errs = []
    for N in (50, 100, 200):
        r = np.linspace(0, 4.0, N)
        kw, ex = bg.generic_slice(r)
        d = diagnose(SphericalSlice(center="vertex", **kw),
                     Projections(ex["rho"], ex["j_r"], 0 * r, 0 * r))
        errs.append(abs(d.M_over_R3[0] - ex["MR3_0"]))
    o = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print("M/R^3 centre errors", errs, "orders", o)
    assert errs[-1] < 1e-5 and np.all(o > 3)


def test_cell_center_extrapolation_and_conical_defect():
    N = 100
    r = (np.arange(N) + 0.5) * 0.02
    kw, ex = bg.flrw(r, 0.7)
    d = diagnose(SphericalSlice(center="cell", **kw), Projections(ex["rho"], 0 * r, 0 * r, 0 * r))
    cc = d.center_checks
    assert cc["extrapolated"] and abs(cc["Gamma0"] - 1) < 1e-10
    assert abs(cc["KA_minus_KB0"]) < 1e-14
    assert not d.M_over_R3_extrapolated          # no R = 0 point on a cell grid
    assert not np.any(d.sphere_class == "CENTER")
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    # conical defect R = 1.1 r and K_A(0) != K_B(0): reported, not imposed
    kw2 = dict(r=r, A=np.ones(N), R=1.1 * r, KA=0.3 + 0 * r, KB=0.1 + 0 * r)
    d2 = diagnose(SphericalSlice(center="cell", **kw2))
    assert abs(d2.center_checks["Gamma0"] - 1.1) < 1e-12
    assert abs(d2.center_checks["KA_minus_KB0"] - 0.2) < 1e-12


def test_vertex_center_checks_report_deviation():
    r = np.linspace(0, 1, 50)
    kw = dict(r=r, A=np.ones(50), R=1.05 * r, KA=0.2 + 0 * r, KB=0 * r)
    d = diagnose(SphericalSlice(center="vertex", **kw))
    assert abs(d.center_checks["Gamma0_minus_1"] - 0.05) < 1e-12
    assert abs(d.center_checks["KA_minus_KB0"] - 0.2) < 1e-15


def test_hamM_regular_at_centre_and_H_excluded():
    r = bg.stretched(100, 0, 3.0)
    kw, ex = bg.generic_slice(r)
    d = diagnose(SphericalSlice(center="vertex", **kw),
                 Projections(ex["rho"], ex["j_r"], 0 * r, 0 * r))
    assert np.isfinite(d.hamM[0]) and abs(d.hamM[0]) < 1e-5
    assert d.hamM_status[0] == "CONSISTENT"
    assert d.mom_status[0] == "CENTER_EXCLUDED"

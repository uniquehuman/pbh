"""NaN / inf / bad input handling: never a benign status."""
import numpy as np
import pytest

from sphdiag import Projections, ScalarField, SphericalSlice, VlasovMoments, diagnose
import devbg as bg

BENIGN_GEOMETRY = {"NO_TRAPPED_SPHERES", "PAST_TRAPPED_ONLY", "AH_CANDIDATE",
                   "FUTURE_TRAPPED_NO_AH_CANDIDATE", "UNCERTAIN"}


def base(N=60):
    r = np.linspace(0, 2, N)
    kw, ex = bg.flrw(r, 0.5)
    return kw, ex


def vac(r):
    return Projections(0 * r, 0 * r, 0 * r, 0 * r)


def assert_invalid(d, data="INVALID_INPUT"):
    assert d.status["data"] == data
    assert d.status["geometry"] == "UNDETERMINED"
    assert d.status["constraints"] != "CONSISTENT_WITH_TRUNCATION"


@pytest.mark.parametrize("field", ["A", "R", "KA", "KB"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_slice_input(field, bad):
    kw, ex = base()
    kw[field] = kw[field].copy()
    kw[field][30] = bad
    d = diagnose(SphericalSlice(center="vertex", **kw),
                 Projections(ex["rho"], 0 * kw["r"], 0 * kw["r"], 0 * kw["r"]))
    assert_invalid(d)
    assert d.sphere_class[30] == "INVALID"
    assert "INVALID" in set(d.sphere_class)


@pytest.mark.parametrize("which", ["rho", "j_r"])
def test_nonfinite_matter(which):
    kw, ex = base()
    r = kw["r"]
    m = dict(rho=ex["rho"].copy(), j_r=0 * r, S_rr=0 * r, S_thth=0 * r)
    m[which][10] = np.nan
    d = diagnose(SphericalSlice(center="vertex", **kw), Projections(**m))
    assert_invalid(d)


def test_nonfinite_scalar_field_and_vlasov():
    kw, ex = base()
    r = kw["r"]
    phi = 0.1 + 0 * r
    phi[5] = np.nan
    assert_invalid(diagnose(SphericalSlice(center="vertex", **kw), ScalarField(phi, 0 * r, 1.0)))
    E = ex["rho"].copy()
    E[7] = np.inf
    assert_invalid(diagnose(SphericalSlice(center="vertex", **kw), VlasovMoments(E, 0 * r, 0 * r, 0 * r)))


def test_A_nonpositive_and_R_negative():
    kw, ex = base()
    kw2 = dict(kw, A=kw["A"].copy())
    kw2["A"][20] = 0.0
    assert_invalid(diagnose(SphericalSlice(center="vertex", **kw2), vac(kw["r"])))
    kw3 = dict(kw, A=kw["A"].copy())
    kw3["A"][20] = -1.0
    assert_invalid(diagnose(SphericalSlice(center="vertex", **kw3), vac(kw["r"])))
    kw4 = dict(kw, R=kw["R"].copy())
    kw4["R"][20] = -0.1
    assert_invalid(diagnose(SphericalSlice(center="vertex", **kw4), vac(kw["r"])))


def test_structural_problems():
    kw, ex = base()
    r = kw["r"]
    cases = [
        dict(kw, r=r[::-1].copy()),                      # decreasing
        dict(kw, A=kw["A"][:-1]),                        # shape mismatch
        dict(kw, r=np.where(np.arange(r.size) == 5, r[4], r)),  # repeated point
        {k: v[:7] for k, v in kw.items()},               # N < 8
        dict(kw, r=r + 0.1),                             # vertex needs r[0] = 0
    ]
    for c in cases:
        d = diagnose(SphericalSlice(center="vertex", **c), vac(c["r"]))
        assert_invalid(d)
        assert np.all(d.sphere_class == "INVALID")
    d = diagnose(SphericalSlice(center="spiral", **kw))
    assert_invalid(d)
    d = diagnose(SphericalSlice(center="cell", **kw))   # cell needs r[0] > 0
    assert_invalid(d)
    d = diagnose(SphericalSlice(center="vertex", **dict(kw, A="abc")))
    assert_invalid(d)
    d = diagnose(SphericalSlice(center="vertex", **kw), Projections(r[:-2], r, r, r))
    assert_invalid(d)
    d = diagnose(SphericalSlice(center="vertex", **kw), matter="dust")
    assert_invalid(d)


def test_vertex_nonzero_R0_invalid():
    kw, ex = base()
    kw["R"] = kw["R"] + 0.01
    assert_invalid(diagnose(SphericalSlice(center="vertex", **kw)))


def test_overflow_is_numerical_failure():
    kw, ex = base()
    kw = dict(kw, KB=1e200 + 0 * kw["r"], KA=1e200 + 0 * kw["r"])
    d = diagnose(SphericalSlice(center="vertex", **kw))
    assert d.status["data"] == "NUMERICAL_FAILURE"
    assert d.status["geometry"] == "UNDETERMINED"
    assert "M" in d.status["numerical_failures"]


def test_center_nan_theta_is_documented_not_failure():
    kw, ex = base()
    d = diagnose(SphericalSlice(center="vertex", **kw), vac(kw["r"]) if False else
                 Projections(ex["rho"], 0 * kw["r"], 0 * kw["r"], 0 * kw["r"]))
    assert d.status["data"] == "OK"
    assert np.isnan(d.theta_plus[0]) and d.sphere_class[0] == "CENTER"


def test_alpha_sign_change_is_only_a_warning():
    # isotropic Schwarzschild lapse changes sign; alpha is not used by the diagnostics
    r = np.linspace(0.2, 4, 100)
    kw, ex = bg.schwarzschild_iso(r)
    alpha = (1 - 1 / (2 * r)) / (1 + 1 / (2 * r))
    d = diagnose(SphericalSlice(center="none", alpha=alpha, beta=0 * r, **kw), vac(r))
    assert d.status["data"] == "OK" and d.status["warnings"]


def test_no_matter_not_evaluated():
    kw, ex = base()
    d = diagnose(SphericalSlice(center="vertex", **kw))
    assert d.status["constraints"] == "NOT_EVALUATED"
    assert np.all(np.isnan(d.ham)) and np.all(np.isnan(d.hamM)) and np.all(np.isnan(d.M_int))
    assert d.status["data"] == "OK"

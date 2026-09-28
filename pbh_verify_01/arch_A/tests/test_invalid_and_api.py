"""Invalid inputs, NaN/inf handling (never a benign status), API shape of Diagnostics."""
import dataclasses

import numpy as np
import pytest

from conftest import ekg_slice, manufactured_slice
from sphdiag import Diagnostics, Projections, ScalarField, SphericalSlice, VlasovMoments, diagnose, Sign

BENIGN_GEOMETRY = {"NO_TRAPPED_SPHERES", "PAST_TRAPPED_ONLY"}


def base(N=40):
    r = np.linspace(0, 2, N)
    z = np.zeros(N)
    return dict(r=r, A=np.ones(N), R=r.copy(), KA=z.copy(), KB=z.copy())


def assert_not_benign(d, matter_given):
    assert d.status["data"] in ("INVALID_INPUT", "NUMERICAL_FAILURE")
    assert d.status["geometry"] == "UNDETERMINED"
    if matter_given:
        assert d.status["constraints"] != "CONSISTENT_WITH_TRUNCATION"
    assert d.outer_ah_candidate is None or d.status["geometry"] == "UNDETERMINED"


@pytest.mark.parametrize("mutate,msg", [
    (lambda b: b.update(A=np.where(np.arange(b["r"].size) == 7, np.nan, b["A"])), "A: non-finite"),
    (lambda b: b.update(KB=np.where(np.arange(b["r"].size) == 3, np.inf, b["KB"])), "KB: non-finite"),
    (lambda b: b.update(A=-b["A"]), "A: must be > 0"),
    (lambda b: b.update(A=np.where(np.arange(b["r"].size) == 5, 0.0, b["A"])), "A: must be > 0"),
    (lambda b: b.update(r=b["r"][::-1].copy()), "strictly increasing"),
    (lambda b: b.update(R=-b["R"]), "R: must be >= 0"),
    (lambda b: b.update(R=b["R"][:-1]), "shape"),
    (lambda b: b.update(r=b["r"] + 0.1), "vertex' requires r[0] == 0"),
    (lambda b: b.update(R=b["R"] + 0.1), "requires R[0] == 0"),
    (lambda b: b.update(R=np.where(np.arange(b["r"].size) == 9, 0.0, b["R"])), "R must be > 0"),
])
def test_invalid_slice_inputs(mutate, msg):
    b = base()
    mutate(b)
    s = SphericalSlice(b["r"], b["A"], b["R"], b["KA"], b["KB"], "vertex")
    z = np.zeros(40)
    for matter in (None, Projections(z, z, z, z)):
        d = diagnose(s, matter)
        assert d.status["data"] == "INVALID_INPUT"
        assert any(msg in m for m in d.status["messages"]), d.status["messages"]
        assert_not_benign(d, matter is not None)
        assert np.all(d.sphere_class == "INVALID")
        assert d.roots == []


def test_small_N_and_bad_center_and_bad_types():
    r = np.linspace(0, 1, 7)
    d = diagnose(SphericalSlice(r, np.ones(7), r, np.zeros(7), np.zeros(7)))
    assert d.status["data"] == "INVALID_INPUT" and any("N = 7 < 8" in m for m in d.status["messages"])
    b = base()
    d = diagnose(SphericalSlice(b["r"], b["A"], b["R"], b["KA"], b["KB"], "centre"))
    assert d.status["data"] == "INVALID_INPUT"
    d = diagnose(SphericalSlice(b["r"] + 1, b["A"], b["R"] + 1, b["KA"], b["KB"], "cell"),
                 matter="not matter")
    assert d.status["data"] == "INVALID_INPUT"
    d = diagnose("not a slice")
    assert d.status["data"] == "INVALID_INPUT" and d.status["geometry"] == "UNDETERMINED"
    d = diagnose(SphericalSlice(b["r"], b["A"], b["R"], b["KA"], ["a"] * 40))
    assert d.status["data"] == "INVALID_INPUT"
    d = diagnose(SphericalSlice(b["r"], b["A"], b["R"], b["KA"], b["KB"]), kappa_theta=np.nan)
    assert d.status["data"] == "INVALID_INPUT"


def test_cell_requires_positive_r0():
    b = base()
    d = diagnose(SphericalSlice(b["r"], b["A"], b["R"], b["KA"], b["KB"], "cell"))
    assert d.status["data"] == "INVALID_INPUT"


@pytest.mark.parametrize("field", ["rho", "j_r", "S_rr", "S_thth"])
def test_nan_in_matter_is_invalid(field):
    s, p, _ = manufactured_slice("cell", 50)
    kw = dict(rho=p.rho.copy(), j_r=p.j_r.copy(), S_rr=p.S_rr.copy(), S_thth=p.S_thth.copy())
    kw[field][10] = np.nan
    d = diagnose(s, Projections(**kw))
    assert d.status["data"] == "INVALID_INPUT"
    assert d.status["constraints"] == "VIOLATED"
    assert d.status["geometry"] == "UNDETERMINED"


def test_nan_in_scalar_field_and_bad_mass():
    s, f, _ = ekg_slice("vertex", 60)
    phi = f.phi.copy()
    phi[5] = np.nan
    assert diagnose(s, ScalarField(phi, f.Pi, f.m_phi)).status["data"] == "INVALID_INPUT"
    assert diagnose(s, ScalarField(f.phi, f.Pi, -1.0)).status["data"] == "INVALID_INPUT"
    assert diagnose(s, ScalarField(f.phi, f.Pi, np.inf)).status["data"] == "INVALID_INPUT"
    assert diagnose(s, VlasovMoments(f.phi, f.Pi, f.phi[:-1], f.Pi)).status["data"] == "INVALID_INPUT"


def test_overflow_in_outputs_is_numerical_failure():
    """Finite inputs whose outputs overflow (U^2 = R^2 K_B^2 -> inf) must not pass as benign."""
    b = base()
    KB = np.zeros(40)
    KB[20] = 1e200
    s = SphericalSlice(b["r"], b["A"], b["R"], b["KA"], KB)
    z = np.zeros(40)
    for matter in (None, Projections(z, z, z, z)):
        d = diagnose(s, matter)
        assert d.status["data"] == "NUMERICAL_FAILURE"
        assert d.status["geometry"] == "UNDETERMINED"
        assert d.sphere_class[20] == "INVALID"
        if matter is not None:
            assert d.status["constraints"] == "VIOLATED"


def test_overflow_in_matter_only():
    s, p, _ = manufactured_slice("cell", 50)
    rho = p.rho.copy()
    rho[10] = 1e308
    d = diagnose(s, Projections(rho, p.j_r, p.S_rr, p.S_thth))
    # 16 pi rho overflows -> inf residual; must be flagged
    assert d.status["data"] == "NUMERICAL_FAILURE"
    assert d.status["constraints"] == "VIOLATED"
    assert d.status["geometry"] == "UNDETERMINED"


def test_nan_constraint_point_is_violated_not_consistent():
    """Direct guard for the classic bug `abs(nan) > thr == False -> CONSISTENT`."""
    s, p, _ = manufactured_slice("cell", 50)
    rho = p.rho.copy()
    rho[10] = 1e308
    d = diagnose(s, Projections(rho, p.j_r, p.S_rr, p.S_thth))
    assert d.constraint_point_status["ham"][10] == "VIOLATED"


def test_diagnostics_has_every_contract_field():
    s, p, _ = manufactured_slice("vertex", 60)
    d = diagnose(s, p)
    assert isinstance(d, Diagnostics)
    names = {f.name for f in dataclasses.fields(d)}
    required = ["Gamma", "U", "dR", "M", "sigma_M", "compactness", "M_over_R3", "M_over_R3_extrapolated",
                "Rtheta_plus", "Rtheta_minus", "sigma_Rtheta_plus", "sigma_Rtheta_minus", "theta_plus",
                "theta_minus", "sphere_class", "roots", "outer_ah_candidate", "projections", "ham", "ham_rel",
                "mom", "mom_rel", "hamM", "hamM_rel", "sigma_ham", "sigma_mom", "sigma_hamM", "norms",
                "M_rho", "M_flux", "M_int", "M_prop", "center_checks", "status"]
    assert set(required) <= names
    N = s.N
    for nm in required:
        v = getattr(d, nm)
        if isinstance(v, np.ndarray):
            assert v.shape == (N,), nm
    assert set(d.status) >= {"data", "constraints", "geometry", "marginal_present"}
    assert set(d.center_checks) >= {"Gamma0", "KA_minus_KB0"}
    assert isinstance(d.projections, Projections)
    for rt in d.roots:
        assert set(rt) >= {"which", "r", "R", "R_lo", "R_hi", "M", "type", "ah_candidate", "other_sign",
                           "inner_sign", "outer_sign"}
        assert rt["which"] in ("+", "-")
        assert rt["type"] in ("FUTURE_MOTS", "PAST_MOTS", "DEGENERATE")
        assert rt["R_lo"] <= rt["R"] <= rt["R_hi"]


def test_no_matter_gives_nan_residuals_and_not_evaluated():
    s, p, _ = manufactured_slice("vertex", 60)
    d = diagnose(s)
    assert d.status["constraints"] == "NOT_EVALUATED"
    assert d.projections is None
    for nm in ("ham", "mom", "hamM", "M_rho", "M_int"):
        assert np.all(np.isnan(getattr(d, nm)))
    assert d.status["data"] == "OK"


def test_sign_type():
    assert Sign(-1) == -1 and Sign(-1) == "-" and Sign(-1) != "+" and Sign(0) == "0" and Sign(1) > 0
    assert str(Sign(1)) == "+"
    with pytest.raises(ValueError):
        Sign(2)


def test_inputs_not_modified_and_slice_copies():
    s, p, d = manufactured_slice("vertex", 60)
    A0 = s.A.copy()
    rho0 = p.rho.copy()
    diagnose(s, p)
    assert np.array_equal(s.A, A0) and np.array_equal(p.rho, rho0)
    arr = np.linspace(0, 1, 10)
    sl = SphericalSlice(arr, np.ones(10), arr, np.zeros(10), np.zeros(10))
    arr[3] = 99.0
    assert sl.r[3] != 99.0

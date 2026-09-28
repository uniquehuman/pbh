"""Exact vacuum slices: Minkowski (several radial labels), Schwarzschild in isotropic (time-symmetric,
Einstein-Rosen throat), Painleve-Gullstrand (black and white hole), Kerr-Schild, non-areal labels."""
import numpy as np
import pytest

from conftest import isotropic_slice, ks_slice, order, pg_slice, zeros_matter
from sphdiag import SphericalSlice, diagnose


@pytest.mark.parametrize("center", ["vertex", "cell"])
@pytest.mark.parametrize("label", ["areal", "stretched"])
def test_minkowski(center, label):
    N = 120
    x = np.linspace(0, 3, N) if center == "vertex" else (np.arange(N) + 0.5) * 3 / N
    if label == "areal":
        R, A = x.copy(), np.ones(N)
    else:  # R = x + 0.3 x^3 / (1 + x^2): odd, monotone; flat metric dR^2 => A = R'
        R = x + 0.3 * x ** 3 / (1 + x ** 2)
        A = 1 + 0.3 * (3 * x ** 2 + x ** 4) / (1 + x ** 2) ** 2
    s = SphericalSlice(x, A, R, np.zeros(N), np.zeros(N), center)
    d = diagnose(s, zeros_matter(N))
    assert d.status == {**d.status, "data": "OK", "constraints": "CONSISTENT_WITH_TRUNCATION",
                        "geometry": "NO_TRAPPED_SPHERES"}
    assert np.max(np.abs(d.M)) < 1e-6 * 3
    assert np.max(np.abs(d.Gamma - 1)) < 1e-5
    assert d.roots == [] and d.outer_ah_candidate is None
    ok = {"NORMAL", "CENTER"}
    assert set(d.sphere_class) <= ok
    if center == "vertex":
        assert d.sphere_class[0] == "CENTER"
        assert np.isnan(d.theta_plus[0]) and np.isnan(d.theta_minus[0])
        assert abs(d.Rtheta_plus[0] - 2.0) <= 3 * d.sigma_Rtheta_plus[0] + 1e-12
        assert abs(d.Rtheta_minus[0] + 2.0) <= 3 * d.sigma_Rtheta_minus[0] + 1e-12
        assert abs(d.Rtheta_plus[0] - 2.0) < 1e-5
        assert d.compactness[0] == 0.0
        assert np.isnan(d.ham[0]) and np.isnan(d.mom[0]) and np.isfinite(d.hamM[0])
        assert d.constraint_point_status["ham"][0] == "EXCLUDED_CENTER"
        assert d.M_over_R3_extrapolated
    assert abs(d.center_checks["Gamma0"] - 1) < 1e-6


def test_schwarzschild_isotropic_throat_is_not_a_horizon():
    """Time-symmetric slice through the Einstein-Rosen throat: 2M/R = 1 at R = 2M, but
    theta+ = -theta- (U = 0); the throat is DOUBLY_MARGINAL/DEGENERATE, never an AH candidate."""
    Mb = 1.0
    for on_grid in (True, False):
        if on_grid:   # r = M/2 (the throat) exactly on the grid
            r = np.linspace(0.1, 5.0, 491)
            r[40] = 0.5
        else:
            r = np.linspace(0.1, 5.0, 400)
        s = isotropic_slice(r, Mb)
        d = diagnose(s, zeros_matter(r.size))
        assert d.status["data"] == "OK"
        assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
        assert d.status["geometry"] in ("UNCERTAIN", "NO_TRAPPED_SPHERES")
        assert d.status["geometry"] != "AH_CANDIDATE"
        assert d.outer_ah_candidate is None
        assert all(not rt["ah_candidate"] for rt in d.roots)
        assert {rt["type"] for rt in d.roots} == {"DEGENERATE"}
        assert all(abs(rt["R"] - 2 * Mb) < 1e-4 for rt in d.roots)
        assert np.all(np.abs(d.M - Mb) <= 3 * d.sigma_M + 1e-10)
        assert np.max(np.abs(d.M - Mb)[r > 0.2]) < 1e-4
        inner = r < 0.45
        outer = r > 0.55
        assert np.all(d.sphere_class[inner] == "NORMAL_REVERSED")
        assert np.all(d.sphere_class[outer] == "NORMAL")
        if np.any(r == 0.5):
            i = int(np.flatnonzero(r == 0.5)[0])
            assert d.sphere_class[i] == "DOUBLY_MARGINAL"
            assert d.compactness[i] == pytest.approx(1.0, abs=1e-6)
            # literal contract priority (UNCERTAIN before NO_TRAPPED_SPHERES), see REPORT.md
            assert d.status["geometry"] == "UNCERTAIN"
            assert d.status["marginal_present"]


def test_isotropic_mass_converges_4th_order():
    Ns = [200, 400, 800]
    errs = []
    for N in Ns:
        r = np.linspace(0.3, 5.0, N)
        d = diagnose(isotropic_slice(r))
        errs.append(np.max(np.abs(d.M - 1.0)))
        assert np.all(np.abs(d.M - 1.0) <= 3 * d.sigma_M + 1e-12)     # sigma_M bounds the true error
    assert np.all(order(errs, Ns) > 3.5)


@pytest.mark.parametrize("which", ["PG", "KS"])
def test_black_hole_slices_have_ah_candidate_at_2M(which):
    Mb = 1.3
    r = np.linspace(0.4 * Mb, 8 * Mb, 500)
    s = pg_slice(r, Mb) if which == "PG" else ks_slice(r, Mb)
    d = diagnose(s, zeros_matter(r.size))
    assert d.status["data"] == "OK"
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    assert d.status["geometry"] == "AH_CANDIDATE"
    oc = d.outer_ah_candidate
    assert oc is not None and oc["which"] == "+" and oc["type"] == "FUTURE_MOTS"
    assert oc["R"] == pytest.approx(2 * Mb, rel=1e-7)
    assert oc["R_lo"] <= 2 * Mb <= oc["R_hi"]
    assert oc["M"] == pytest.approx(Mb, rel=1e-8)
    assert oc["other_sign"] == -1 and oc["other_sign"] == "-"
    assert oc["inner_sign"] == -1 and oc["outer_sign"] == +1
    assert np.all(d.sphere_class[r < 1.9 * Mb] == "FUTURE_TRAPPED")
    assert np.all(d.sphere_class[r > 2.1 * Mb] == "NORMAL")
    assert np.max(np.abs(d.M - Mb)) < 1e-9
    if which == "PG":   # theta+- = 2(+-1 - sqrt(2M/r))/r  (Gamma = 1, U = -sqrt(2M/r))
        assert np.allclose(d.theta_plus, 2 * (1 - np.sqrt(2 * Mb / r)) / r, atol=1e-9)
        assert np.allclose(d.theta_minus, 2 * (-1 - np.sqrt(2 * Mb / r)) / r, atol=1e-9)
    # F13: R^2 theta+ theta- / 4 = 2M/R - 1
    pos = d.R > 0
    lhs = d.R[pos] ** 2 * d.theta_plus[pos] * d.theta_minus[pos] / 4
    assert np.allclose(lhs, 2 * d.M[pos] / d.R[pos] - 1, atol=1e-10)


def test_white_hole_is_past_mots_never_pbh():
    r = np.linspace(0.5, 8, 400)
    d = diagnose(pg_slice(r, 1.0, white=True), zeros_matter(r.size))
    assert d.status["geometry"] == "PAST_TRAPPED_ONLY"
    assert len(d.roots) == 1 and d.roots[0]["type"] == "PAST_MOTS" and d.roots[0]["which"] == "-"
    assert not d.roots[0]["ah_candidate"] and d.outer_ah_candidate is None
    assert np.all(d.sphere_class[r < 1.9] == "PAST_TRAPPED")


def test_invariants_across_slicings_and_labels():
    """M_MS and theta+ theta- are slicing-invariant for spheres of the same R; theta+- themselves are not."""
    Mb = 1.0
    Rq = np.linspace(2.5, 6.0, 8)
    # isotropic: invert R = psi^2 r
    ri = np.linspace(0.6, 6.0, 800)
    di = diagnose(isotropic_slice(ri, Mb))
    ok = di.R > 2.4
    rp = np.linspace(1.0, 7.0, 800)
    dp = diagnose(pg_slice(rp, Mb))
    x = np.linspace(1.0, 2.7, 800)          # non-areal label R = x^2
    dx = diagnose(pg_slice(x, Mb, R_of_r=lambda x: x ** 2, dR_of_r=lambda x: 2 * x))
    prod = lambda d: d.R ** 2 * d.theta_plus * d.theta_minus / 4
    for d, sel in ((di, ok), (dp, np.ones(rp.size, bool)), (dx, np.ones(x.size, bool))):
        Mi = np.interp(Rq, d.R[sel], d.M[sel])
        assert np.allclose(Mi, Mb, atol=1e-6)
        assert np.allclose(np.interp(Rq, d.R[sel], prod(d)[sel]), 2 * Mb / Rq - 1, atol=1e-5)
    # theta+ differs between isotropic and PG slices (boost of the normal)
    tpi = np.interp(Rq, di.R[ok], di.theta_plus[ok])
    tpp = np.interp(Rq, dp.R, dp.theta_plus)
    assert np.all(np.abs(tpi - tpp) > 1e-2)
    # the non-areal PG label gives the same horizon
    assert dx.status["geometry"] == "AH_CANDIDATE"
    assert dx.outer_ah_candidate["R"] == pytest.approx(2.0, rel=1e-6)


def test_orientation_reversed_label_is_not_reported_as_candidate():
    """r increasing toward smaller R (s^a points to decreasing R): the BH horizon appears as a zero of theta-.
    Contract: candidate only for a theta+ zero with Gamma > 0 -> no candidate, but trapped spheres
    are still reported (never 'no collapse')."""
    Mb = 1.0
    x = np.linspace(-8.0, -0.5, 400)
    R = -x
    KB = np.sqrt(2 * Mb) * R ** -1.5
    KA = -0.5 * np.sqrt(2 * Mb) * R ** -1.5
    s = SphericalSlice(x, np.ones_like(x), R, KA, KB, "none")
    d = diagnose(s, zeros_matter(x.size))
    assert np.all(d.Gamma < 0)
    assert d.status["geometry"] == "FUTURE_TRAPPED_NO_AH_CANDIDATE"
    assert [rt["which"] for rt in d.roots] == ["-"]
    assert d.roots[0]["type"] == "FUTURE_MOTS" and not d.roots[0]["ah_candidate"]
    assert set(d.sphere_class) == {"FUTURE_TRAPPED", "NORMAL_REVERSED"}

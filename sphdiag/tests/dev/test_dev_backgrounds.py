"""Exact backgrounds of derivation.md §3: values, classes, roots, statuses."""
import numpy as np
import pytest

from sphdiag import Projections, SphericalSlice, diagnose
import devbg as bg


def run(kw, center, matter="vacuum"):
    slc = SphericalSlice(center=center, **kw)
    r = kw["r"]
    if isinstance(matter, str):
        matter = Projections(0 * r, 0 * r, 0 * r, 0 * r)
    return diagnose(slc, matter)


def assert_close_to_exact(d, ex, tol):
    for key, name in (("Gamma", "Gamma"), ("U", "U"), ("M", "M"), ("Rtp", "Rtheta_plus"),
                      ("Rtm", "Rtheta_minus")):
        err = np.max(np.abs(getattr(d, name) - ex[key]))
        assert err < tol, (name, err)


def test_minkowski_flat_vertex():
    r = np.linspace(0, 5, 101)
    kw, ex = bg.minkowski_flat(r)
    d = run(kw, "vertex")
    assert_close_to_exact(d, ex, 1e-12)
    assert d.sphere_class[0] == "CENTER"
    assert np.all(d.sphere_class[1:] == "NORMAL")
    assert np.isnan(d.theta_plus[0]) and np.isnan(d.theta_minus[0])
    np.testing.assert_allclose(d.theta_plus[1:], 2 / r[1:], rtol=1e-12)
    assert d.status == {**d.status, "data": "OK", "constraints": "CONSISTENT_WITH_TRUNCATION",
                        "geometry": "NO_TRAPPED_SPHERES"}
    assert d.roots == [] and d.outer_ah_candidate is None


def test_minkowski_hyperboloid_and_milne():
    r = np.linspace(0, 1e3, 2000)
    kw, ex = bg.minkowski_hyperboloid(r)
    d = run(kw, "vertex")
    assert np.max(np.abs(d.Rtheta_minus - ex["Rtm"])) < 1e-6
    assert d.status["geometry"] == "NO_TRAPPED_SPHERES"
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    chi = bg.stretched(200, 0, 4)
    kw, ex = bg.milne(chi)
    d = run(kw, "vertex")
    assert_close_to_exact(d, ex, 2e-4 * np.exp(4))
    assert d.status["geometry"] == "NO_TRAPPED_SPHERES"


@pytest.mark.parametrize("which", ["ks", "pg"])
def test_schwarzschild_ah_candidate(which):
    Mb = 1.3
    r = bg.stretched(400, 0.4, 12.0)
    kw, ex = (bg.schwarzschild_ks if which == "ks" else bg.schwarzschild_pg)(r, Mb)
    d = run(kw, "none")
    assert_close_to_exact(d, ex, 5e-5)
    assert d.status["geometry"] == "AH_CANDIDATE"
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    ah = d.outer_ah_candidate
    assert ah is not None and ah["which"] == "+" and ah["type"] == "FUTURE_MOTS"
    assert ah["other_sign"] < 0 and ah["inner_sign"] == "-" and ah["outer_sign"] == "+"
    assert abs(ah["R"] - 2 * Mb) < 1e-6
    assert ah["R_lo"] <= 2 * Mb <= ah["R_hi"]
    assert abs(ah["M"] - Mb) < 1e-5
    assert np.all(d.sphere_class[r < 2 * Mb * 0.99] == "FUTURE_TRAPPED")
    assert np.all(d.sphere_class[r > 2 * Mb * 1.01] == "NORMAL")
    np.testing.assert_allclose(d.M_int, d.M[0], atol=1e-5)   # vacuum: M_int = M(r0)
    assert np.all(d.M_rho == 0) and np.all(d.M_prop == 0)


def test_schwarzschild_ks_analytic_derivatives():
    Mb = 1.0
    r = bg.stretched(200, 0.5, 10.0)
    kw, ex = bg.schwarzschild_ks(r, Mb)
    kw2 = dict(kw, dA=ex["dA"], dR=ex["dR"], dKB=ex["dKB"])
    d = run(kw2, "none")
    assert np.all(d.sigma_Rtheta_plus == 0) and np.all(d.sigma_M == 0)
    assert np.max(np.abs(d.M - Mb)) < 1e-12
    assert np.max(np.abs(d.Rtheta_plus - ex["Rtp"])) < 1e-12
    assert d.status["geometry"] == "AH_CANDIDATE"
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    ah = d.outer_ah_candidate
    assert abs(ah["R"] - 2 * Mb) < 1e-6 and ah["R_lo"] <= 2 * Mb <= ah["R_hi"]


def test_schwarzschild_inside_only_is_trapped_without_candidate():
    r = np.linspace(0.3, 1.5, 60)
    kw, ex = bg.schwarzschild_pg(r, 1.0)
    d = run(kw, "none")
    assert np.all(d.sphere_class == "FUTURE_TRAPPED")
    assert d.status["geometry"] == "FUTURE_TRAPPED_NO_AH_CANDIDATE"


@pytest.mark.parametrize("node", [True, False])
def test_schwarzschild_isotropic_throat(node):
    Mb = 1.0
    # (v0.4: with C12 the steep one-sided inner edge r = 0.1 is UNRESOLVED at N = 300; N = 600
    # resolves it)
    r = np.linspace(0.1, 5.1, 501) if node else bg.stretched(600, 0.1, 5.0)
    kw, ex = bg.schwarzschild_iso(r, Mb)
    d = run(kw, "none")
    assert np.max(np.abs(d.M - Mb)) < 1e-2
    assert d.status["geometry"] == "NO_TRAPPED_SPHERES"
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    assert not np.any(np.isin(d.sphere_class, ["FUTURE_TRAPPED", "PAST_TRAPPED"]))
    assert np.all(d.sphere_class[r < 0.45] == "NORMAL_REVERSED")
    assert np.all(d.sphere_class[r > 0.55] == "NORMAL")
    assert {x["type"] for x in d.roots} == {"DEGENERATE"}
    assert {x["which"] for x in d.roots} == {"+", "-"}
    for x in d.roots:
        assert x["R_lo"] <= 2 * Mb <= x["R_hi"]
        assert not x["ah_candidate"]
    if node:
        i = np.argmin(np.abs(r - 0.5))
        assert d.sphere_class[i] == "DOUBLY_MARGINAL"
        assert d.status["marginal_present"] is True


@pytest.mark.parametrize("k", [0, 1, -1])
@pytest.mark.parametrize("H", [1.5, -1.5])
def test_flrw(k, H):
    chi = np.linspace(0, 2.9 if k == 1 else 2.0, 301)
    kw, ex = bg.flrw(chi, H, a=1.0, k=k)
    d = run(kw, "vertex", Projections(ex["rho"], 0 * chi, 0 * chi, 0 * chi))
    assert_close_to_exact(d, ex, 1e-6)
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    Rstar = (H ** 2 + k) ** -0.5
    types = {x["type"] for x in d.roots}
    assert not any(x["ah_candidate"] for x in d.roots)
    for x in d.roots:
        assert abs(x["R"] - Rstar) < 1e-6 and x["R_lo"] <= Rstar + 1e-12
        assert x["R_hi"] >= Rstar - 1e-12
        assert abs(2 * x["M"] / x["R"] - 1) < 1e-6
    if H > 0:
        assert types == {"PAST_MOTS"}
        assert d.status["geometry"] == "PAST_TRAPPED_ONLY"
    else:
        assert types == {"FUTURE_MOTS"}
        assert d.status["geometry"] == "FUTURE_TRAPPED_NO_AH_CANDIDATE"
    assert len(d.roots) == (2 if k == 1 else 1)
    # integral masses: j_r = 0 -> M_int = M; M_prop analytic
    np.testing.assert_allclose(d.M_int, ex["M"], atol=1e-8)
    S_int = {0: chi ** 3 / 3, 1: chi / 2 - np.sin(2 * chi) / 4,
             -1: np.sinh(2 * chi) / 4 - chi / 2}[k]
    np.testing.assert_allclose(d.M_prop, 4 * np.pi * ex["rho"] * S_int, atol=1e-8)


def test_closed_flrw_turnaround_doubly_marginal():
    chi = np.linspace(0, 0.75 * np.pi, 301)  # chi = pi/2 is node 200
    kw, ex = bg.flrw(chi, 0.0, a=1.0, k=1)
    d = run(kw, "vertex", Projections(ex["rho"], 0 * chi, 0 * chi, 0 * chi))
    assert d.sphere_class[200] == "DOUBLY_MARGINAL"
    assert d.status["geometry"] == "NO_TRAPPED_SPHERES"
    assert d.status["marginal_present"] is True
    assert {x["type"] for x in d.roots} == {"DEGENERATE"}
    assert abs(2 * d.M[200] / d.R[200] - 1) < 1e-10

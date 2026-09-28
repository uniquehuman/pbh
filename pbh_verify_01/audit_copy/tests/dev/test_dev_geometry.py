"""Classification, roots and status precedence on synthetic slices (R = r, A = 1 => Gamma = 1).

With K_B = -u(r)/r we get U = u, R theta_+ = 2(u + 1), R theta_- = 2(u - 1).
"""
import numpy as np

from sphdiag import Projections, SphericalSlice, diagnose


def slice_u(r, u, **kw):
    return SphericalSlice(r=r, A=np.ones_like(r), R=r.copy(), KA=0 * r, KB=-u / r,
                          center="none", **kw)


def test_ah_candidate_with_trapped_interior():
    r = np.linspace(0.2, 2.5, 200)
    d = diagnose(slice_u(r, -2 + r))
    assert d.status["geometry"] == "AH_CANDIDATE"
    ah = d.outer_ah_candidate
    assert abs(ah["r"] - 1.0) < 1e-12 and ah["ah_candidate"]
    assert np.all(d.sphere_class[r < 0.99] == "FUTURE_TRAPPED")


def test_outermost_of_several_candidates():
    r = np.linspace(0.3, 2.2, 400)
    d = diagnose(slice_u(r, -1 + 0.5 * np.sin(2 * np.pi * r)))
    plus = [x for x in d.roots if x["which"] == "+"]
    assert len(plus) == 4
    cands = [x for x in plus if x["ah_candidate"]]
    assert [round(x["r"], 6) for x in cands] == [1.0, 2.0]
    assert abs(d.outer_ah_candidate["r"] - 2.0) < 1e-6
    # roots where theta_+ goes + -> - outward are FUTURE_MOTS but not candidates
    assert all(x["type"] == "FUTURE_MOTS" for x in plus)


def test_trapped_to_grid_edge():
    r = np.linspace(0.2, 1.0, 50)
    d = diagnose(slice_u(r, -3 + 0 * r))
    assert np.all(d.sphere_class == "FUTURE_TRAPPED")
    assert d.status["geometry"] == "FUTURE_TRAPPED_NO_AH_CANDIDATE"


def test_tangential_zero_on_node_is_uncertain():
    r = np.linspace(0.5, 3, 251)
    r0 = r[100]
    d = diagnose(slice_u(r, -1 + 0.5 * (r - r0) ** 2))
    assert d.sphere_class[100] == "FUTURE_MARGINAL"
    assert d.roots and d.roots[0]["tangential"] and not d.roots[0]["ah_candidate"]
    assert d.status["geometry"] == "UNCERTAIN"


def test_reversed_orientation():
    r = np.linspace(0.0, 2.0, 60)
    slc = SphericalSlice(r=r, A=np.ones_like(r), R=3 - r, KA=0 * r, KB=0 * r, center="none")
    d = diagnose(slc)
    assert np.all(d.sphere_class == "NORMAL_REVERSED")
    assert d.status["geometry"] == "NO_TRAPPED_SPHERES"


def test_undetermined_beats_ah_candidate():
    r = np.linspace(0.2, 2.5, 200)
    rho = 0 * r
    rho[-1] = np.nan
    d = diagnose(slice_u(r, -2 + r), Projections(rho, 0 * r, 0 * r, 0 * r))
    assert any(x["ah_candidate"] for x in d.roots)
    assert d.status["geometry"] == "UNDETERMINED"


def test_past_trapped_only_and_marginal_edge():
    r = np.linspace(0.2, 2.5, 200)
    d = diagnose(slice_u(r, r))                  # theta_- root at r = 1, past trapped outside
    assert d.status["geometry"] == "PAST_TRAPPED_ONLY"
    assert d.roots[0]["type"] == "PAST_MOTS" and d.roots[0]["which"] == "-"


def test_throat_degenerate_roots_no_trapped():
    # R = 1 + (r-1)^2: R' = 2(r-1) changes sign at r = 1 with U = 0 (time-symmetric throat)
    r = np.linspace(0.2, 2.0, 181)
    R = 1 + (r - 1) ** 2
    slc = SphericalSlice(r=r, A=np.ones_like(r), R=R, KA=0 * r, KB=0 * r, center="none")
    d = diagnose(slc)
    assert {x["type"] for x in d.roots} == {"DEGENERATE"}
    assert d.status["geometry"] == "NO_TRAPPED_SPHERES"
    assert np.all(d.sphere_class[r < 0.95] == "NORMAL_REVERSED")
    assert np.all(d.sphere_class[r > 1.05] == "NORMAL")

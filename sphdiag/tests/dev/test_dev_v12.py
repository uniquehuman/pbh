"""Contract v1.2 (C11): UNRESOLVED sphere class, found on V1 Einstein-Vlasov slices."""
import os

import numpy as np
import pytest

from sphdiag import Projections, SphericalSlice, VlasovMoments, diagnose
import devbg as bg

DUMP = os.path.join(os.path.dirname(__file__), "..", "..", "results", "extension", "dump",
                    "slice_t11.842.npz")


def spike_slice(N=51, u1=-1.2):
    """Vertex slice, R = r, A = 1 (Gamma = 1); U jumps from 0 to u1 across the first cell
    (2M/R from 0 to U^2 = 1.44 in one cell), then decays smoothly and stays normal."""
    r = np.linspace(0, 1.0, N)
    U = -0.3 * r * np.exp(-r ** 2)
    U[1] = u1
    KB = np.empty_like(r)
    KB[1:] = -U[1:] / r[1:]
    KB[0] = 0.3
    return SphericalSlice(r=r, A=np.ones_like(r), R=r.copy(), KA=KB.copy(), KB=KB, center="vertex")


def test_one_cell_spike_is_unresolved_not_benign():
    d = diagnose(spike_slice())
    assert d.Rtheta_plus[1] < 0 and d.Rtheta_minus[1] < 0     # 'trapped' by the raw values
    assert d.sphere_class[1] == "UNRESOLVED"
    assert d.tau_plus[1] > 0.1
    assert d.status["n_unresolved_points"] >= 1
    assert d.status["geometry"] == "UNCERTAIN"
    assert d.sphere_class[0] == "CENTER"                        # CENTER is never UNRESOLVED
    assert any(w.startswith("UNRESOLVED_POINTS:") for w in d.status["warnings"])
    # smooth version of the same slice is resolved and benign
    r = np.linspace(0, 1.0, 51)
    U = -0.3 * r * np.exp(-r ** 2)
    KB = 0.3 * np.exp(-r ** 2)
    d0 = diagnose(SphericalSlice(r=r, A=np.ones_like(r), R=r.copy(), KA=KB, KB=KB,
                                 center="vertex"))
    assert d0.status["n_unresolved_points"] == 0
    assert d0.status["geometry"] == "NO_TRAPPED_SPHERES"


def test_unresolved_inside_ah_candidate_keeps_candidate_and_warns():
    r = np.linspace(0.4, 12.0, 300)
    kw, ex = bg.schwarzschild_pg(r)
    kw["KB"] = kw["KB"].copy()
    kw["KB"][5] *= 3.0                       # one-cell spike at R ~ 0.6, inside R = 2M
    d = diagnose(SphericalSlice(center="none", **kw))
    assert d.status["n_unresolved_points"] > 0
    assert np.all(d.R[d.sphere_class == "UNRESOLVED"] < 2.0)
    assert d.status["geometry"] == "AH_CANDIDATE"
    assert abs(d.outer_ah_candidate["R"] - 2.0) < 1e-4
    assert any(w.startswith("UNRESOLVED_INSIDE_AH:") for w in d.status["warnings"])


def test_doubly_marginal_exemption_needs_two_resolved_neighbours():
    # closed FLRW at turnaround: chi = pi/2 is DOUBLY_MARGINAL
    chi = np.linspace(0, 0.75 * np.pi, 301)          # interior node -> exempt
    kw, ex = bg.flrw(chi, 0.0, k=1)
    d = diagnose(SphericalSlice(center="vertex", **kw))
    assert d.sphere_class[200] == "DOUBLY_MARGINAL"
    assert d.status["geometry"] == "NO_TRAPPED_SPHERES" and d.status["marginal_present"]
    chi = np.linspace(0, 0.5 * np.pi, 201)           # the doubly marginal point is the last one
    kw, ex = bg.flrw(chi, 0.0, k=1)
    d = diagnose(SphericalSlice(center="vertex", **kw))
    assert d.sphere_class[-1] == "DOUBLY_MARGINAL"
    assert d.status["geometry"] == "UNCERTAIN"


@pytest.mark.skipif(not os.path.exists(DUMP), reason="V1 dump not available")
def test_v1_vlasov_first_horizon_slice_regression():
    z = np.load(DUMP)
    slc = SphericalSlice(z["x"], z["A"], z["R"], z["KA"], z["KB"], center="vertex")
    d = diagnose(slc, VlasovMoments(z["E"], z["J"], z["p"], z["q"]))
    assert d.status["data"] == "OK"
    # edge 1: R theta_+ = -0.40, R theta_- = -0.71 (V1: trapped), tau ~ 5: unresolved
    assert d.Rtheta_plus[1] < 0 and d.Rtheta_minus[1] < 0
    assert np.all(d.sphere_class[1:6] == "UNRESOLVED")
    assert d.status["n_unresolved_points"] >= 5
    assert d.status["geometry"] == "UNCERTAIN"                 # v0.2 gave PAST_TRAPPED_ONLY
    assert not np.any(d.sphere_class == "DOUBLY_MARGINAL")

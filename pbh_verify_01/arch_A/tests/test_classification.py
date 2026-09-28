"""Sphere classification, roots and the geometry status (contract section 7): PBH-like trapped core in
an expanding background, false horizons, tangent zeros, noise, exact zeros on grid points."""
import numpy as np
import pytest

from sphdiag import SphericalSlice, diagnose
from sphdiag.diagnose import _CLASS_TABLE, _geometry_status


def core_in_flrw(N=1500, rmax=15.0, Hc=-4.0, Hinf=0.1, noise=0.0, seed=0):
    """Areal data, Gamma = 1, K_A = K_B = -H(r) with a collapsing core (H < 0) inside an expanding
    background (H > 0).  R theta+ = 2(1 + r H): trapped where r H < -1."""
    r = np.linspace(0, rmax, N)
    H = Hinf - (Hinf - Hc) * np.exp(-r ** 2)
    R = r.copy()
    if noise:
        R[1:] = R[1:] * (1 + noise * np.random.default_rng(seed).standard_normal(N - 1))
    return SphericalSlice(r, np.ones(N), R, -H, -H, "vertex"), H


def test_pbh_like_core_candidate_and_cosmological_root():
    s, H = core_in_flrw()
    d = diagnose(s)
    assert d.status["data"] == "OK" and d.status["constraints"] == "NOT_EVALUATED"
    assert d.status["geometry"] == "AH_CANDIDATE"
    types = [(rt["which"], rt["type"], rt["ah_candidate"]) for rt in d.roots]
    assert types == [("+", "FUTURE_MOTS", False), ("+", "FUTURE_MOTS", True), ("-", "PAST_MOTS", False)]
    inner, outer, cosmo = d.roots
    assert inner["inner_sign"] == +1 and inner["outer_sign"] == -1        # inner boundary of trapped shell
    assert outer["inner_sign"] == -1 and outer["outer_sign"] == +1        # outer boundary -> candidate
    assert d.outer_ah_candidate is outer
    from scipy.optimize import brentq
    f = lambda x: 1 + x * (0.1 - 4.1 * np.exp(-x ** 2))
    assert outer["R"] == pytest.approx(brentq(f, 0.8, 3.0), abs=1e-7)
    assert inner["R"] == pytest.approx(brentq(f, 0.05, 0.6), abs=1e-7)
    assert cosmo["R"] == pytest.approx(brentq(lambda x: x * (0.1 - 4.1 * np.exp(-x ** 2)) - 1, 5, 14.9), abs=1e-7)
    # 2M/R = 1 on every root (F13), whatever its type
    for rt in d.roots:
        assert rt["M"] == pytest.approx(rt["R"] / 2, rel=1e-6)
    assert d.status["ah_candidate_robust"] is True
    trapped = (s.r > inner["r"] + 0.02) & (s.r < outer["r"] - 0.02)
    assert np.all(d.sphere_class[trapped] == "FUTURE_TRAPPED")
    assert np.all(d.sphere_class[s.r > cosmo["r"] + 0.05] == "PAST_TRAPPED")


def test_compactness_one_is_not_enough():
    """2M/R = 1 at the Hubble sphere of the background (PAST_MOTS) is never a candidate."""
    s, H = core_in_flrw(Hc=0.1)            # no core: pure expanding background
    d = diagnose(s)
    assert np.any(np.abs(d.compactness - 1) < 1e-2)
    assert d.status["geometry"] == "PAST_TRAPPED_ONLY"
    assert d.outer_ah_candidate is None


def test_trapped_to_grid_edge_no_candidate():
    s, _ = core_in_flrw(N=400, rmax=1.0)   # grid ends inside the trapped shell
    d = diagnose(s)
    assert d.status["geometry"] == "FUTURE_TRAPPED_NO_AH_CANDIDATE"
    assert all(not rt["ah_candidate"] for rt in d.roots)


def test_exact_zero_on_grid_point():
    """Candidate root exactly on a grid point (R theta+ == 0.0 exactly)."""
    Mb = 1.0
    r = np.linspace(0.5, 6.0, 12)          # contains exactly 2.0? build it so
    r = np.concatenate([np.linspace(0.5, 2.0, 7), np.linspace(2.0, 6.0, 9)[1:]])
    KB = np.sqrt(2 * Mb / r) / r            # = 0.5 exactly at r = 2
    s = SphericalSlice(r, np.ones_like(r), r, -0.5 * KB, KB, "none", dR=np.ones_like(r))
    d = diagnose(s)
    i = int(np.flatnonzero(r == 2.0)[0])
    assert d.Rtheta_plus[i] == 0.0
    assert d.sphere_class[i] == "FUTURE_MARGINAL"
    assert len([rt for rt in d.roots if rt["which"] == "+"]) == 1
    rt = d.roots[0]
    assert rt["r"] == 2.0 and rt["ah_candidate"] and rt["exact_grid_zero"]
    assert d.status["geometry"] == "AH_CANDIDATE"


def test_tangent_zero_is_uncertain():
    """R theta+ touches zero without changing sign (pair creation of MOTS): the contract's sign-change
    search misses it; the module must not answer NO_TRAPPED_SPHERES."""
    N = 401
    r = np.linspace(0, 4, N)
    # choose K_B so that R theta+ = 2(1 - r K_B) = 2 (r - 2)^2 / 4 >= 0, zero only at r = 2 (a grid point)
    KB = np.where(r > 0, (1 - (r - 2) ** 2 / 4) / np.where(r > 0, r, 1), 1.0)
    KB[0] = 1.0                    # limit r->0: (1 - (1 - r + r^2/4))/r -> 1
    s = SphericalSlice(r, np.ones(N), r.copy(), KB, KB, "vertex", dR=np.ones(N))
    d = diagnose(s)
    assert d.status["geometry"] == "UNCERTAIN"
    assert d.status["geometry"] != "NO_TRAPPED_SPHERES"


def test_near_tangent_with_noise():
    """Minimum of R theta+ is 2e-9 > 0 but R carries white noise of relative size 1e-8.
    FOUND IN DEVELOPMENT: sigma = |D4 R' - D6 R'| underestimates the error of D4 on white noise by ~6x,
    so the contract-literal classification turns noise into a spurious one-cell trapped region and an
    AH candidate.  The module (i) flags such candidates as not robust (always) and (ii) with
    noise_guard=True adds a grid-scale noise term to sigma, which resolves the case as UNCERTAIN."""
    N = 401
    r = np.linspace(0, 4, N)
    delta = 1e-9
    KB = np.where(r > 0, (1 - delta - (r - 2) ** 2 / 4) / np.where(r > 0, r, 1), 1.0)
    KB[0] = 1.0
    R = r * (1 + 1e-8 * np.random.default_rng(3).standard_normal(N))
    R[0] = 0.0
    outcomes = set()
    for seed in range(6):
        R = r * (1 + 1e-8 * np.random.default_rng(seed).standard_normal(N))
        R[0] = 0.0
        s = SphericalSlice(r, np.ones(N), R, KB, KB, "vertex")
        d = diagnose(s)                               # contract-literal
        outcomes.add(d.status["geometry"])
        assert d.status["noise_sensitive"]            # always warned
        if d.status["geometry"] == "AH_CANDIDATE":
            assert d.status["ah_candidate_robust"] is False
        dg = diagnose(s, noise_guard=True)
        assert dg.status["geometry"] == "UNCERTAIN"
        assert np.any(dg.sphere_class == "FUTURE_MARGINAL")
        assert dg.outer_ah_candidate is None
    # documents the contract-level weakness: literal statuses are seed-dependent on this case
    assert outcomes <= {"AH_CANDIDATE", "NO_TRAPPED_SPHERES", "UNCERTAIN"}
    # clean data: resolved as NO_TRAPPED_SPHERES because 2e-9 >> tolerance 1e3 eps
    s2 = SphericalSlice(r, np.ones(N), r.copy(), KB, KB, "vertex", dR=np.ones(N))
    d2 = diagnose(s2)
    assert d2.status["geometry"] == "NO_TRAPPED_SPHERES" and not d2.status["noise_sensitive"]


def test_noise_guard_keeps_smooth_results():
    s, _ = core_in_flrw()
    for ng in (False, True):
        d = diagnose(s, noise_guard=ng)
        assert d.status["geometry"] == "AH_CANDIDATE" and d.status["ah_candidate_robust"]
        assert not d.status["noise_sensitive"]


def test_noisy_pbh_core_still_candidate_at_right_place():
    s, _ = core_in_flrw(noise=1e-9)
    d = diagnose(s)
    assert d.status["geometry"] == "AH_CANDIDATE"
    oc = d.outer_ah_candidate
    from scipy.optimize import brentq
    Rtrue = brentq(lambda x: 1 + x * (0.1 - 4.1 * np.exp(-x ** 2)), 0.8, 3.0)
    assert oc["R_lo"] - 1e-6 <= Rtrue <= oc["R_hi"] + 1e-6
    assert abs(oc["R"] - Rtrue) < 1e-5


def test_class_table_is_complete_and_matches_contract():
    assert _CLASS_TABLE[(1, -1)] == "NORMAL"
    assert _CLASS_TABLE[(-1, 1)] == "NORMAL_REVERSED"
    assert _CLASS_TABLE[(-1, -1)] == "FUTURE_TRAPPED"
    assert _CLASS_TABLE[(1, 1)] == "PAST_TRAPPED"
    assert _CLASS_TABLE[(0, -1)] == _CLASS_TABLE[(-1, 0)] == "FUTURE_MARGINAL"
    assert _CLASS_TABLE[(0, 1)] == _CLASS_TABLE[(1, 0)] == "PAST_MARGINAL"
    assert _CLASS_TABLE[(0, 0)] == "DOUBLY_MARGINAL"
    assert len(_CLASS_TABLE) == 9


def test_signs_of_marginal_classes_require_gamma_sign():
    """(0,-),(+,0) only with Gamma > 0; (-,0),(0,+) only with Gamma < 0 (contract section 7)."""
    for G in (0.7, -0.7):
        for U in (-G, G):
            Rtp, Rtm = 2 * (U + G), 2 * (U - G)
            sp_ = 0 if abs(Rtp) < 1e-14 else int(np.sign(Rtp))
            sm_ = 0 if abs(Rtm) < 1e-14 else int(np.sign(Rtm))
            if (sp_, sm_) in ((0, -1), (1, 0)):
                assert G > 0
            if (sp_, sm_) in ((-1, 0), (0, 1)):
                assert G < 0


def test_status_priority_rules_directly():
    r = np.arange(6.0)
    s0 = np.zeros(6, int)
    # PAST_MARGINAL alone -> PAST_TRAPPED_ONLY
    cl = np.array(["CENTER", "NORMAL", "PAST_MARGINAL", "PAST_TRAPPED", "PAST_TRAPPED", "PAST_TRAPPED"])
    assert _geometry_status(cl, [], r, s0, s0, True)[0] == "PAST_TRAPPED_ONLY"
    # DEGENERATE root next to a trapped sphere -> UNCERTAIN (beats PAST_TRAPPED_ONLY)
    deg = [{"type": "DEGENERATE", "index": 2, "r": 2.5, "which": "+", "ah_candidate": False, "r_lo": 2.4, "r_hi": 2.6}]
    assert _geometry_status(cl, deg, r, s0, s0, True)[0] == "UNCERTAIN"
    # INVALID anywhere -> UNDETERMINED even with a candidate
    cand = [{"type": "FUTURE_MOTS", "ah_candidate": True, "index": 1, "r": 1.5, "which": "+", "r_lo": 1.4, "r_hi": 1.6}]
    cl2 = np.array(["CENTER", "NORMAL", "INVALID", "NORMAL", "NORMAL", "NORMAL"])
    assert _geometry_status(cl2, cand, r, s0, s0, True)[0] == "UNDETERMINED"
    # non-OK data -> UNDETERMINED
    cl3 = np.array(["CENTER"] + ["NORMAL"] * 5)
    assert _geometry_status(cl3, [], r, s0, s0, False)[0] == "UNDETERMINED"
    assert _geometry_status(cl3, [], r, s0, s0, True)[0] == "NO_TRAPPED_SPHERES"
    # FUTURE_MARGINAL not tied to any root -> UNCERTAIN
    cl4 = np.array(["CENTER", "NORMAL", "FUTURE_MARGINAL", "NORMAL", "NORMAL", "NORMAL"])
    sp_ = np.array([1, 1, 0, 1, 1, 1])
    sm_ = -np.ones(6, int)
    assert _geometry_status(cl4, [], r, sp_, sm_, True)[0] == "UNCERTAIN"
    # marginal run sandwiched between trapped spheres is not by itself uncertain
    cl5 = np.array(["CENTER", "NORMAL", "FUTURE_TRAPPED", "FUTURE_MARGINAL", "FUTURE_TRAPPED", "NORMAL"])
    assert _geometry_status(cl5, [], r, sp_, sm_, True)[0] == "FUTURE_TRAPPED_NO_AH_CANDIDATE"


def test_kappa_theta_widens_zero_band():
    s, _ = core_in_flrw(N=300)
    d1 = diagnose(s, kappa_theta=3.0)
    d2 = diagnose(s, kappa_theta=1e13)
    n1 = np.sum(np.isin(d1.sphere_class, ["FUTURE_MARGINAL", "PAST_MARGINAL", "DOUBLY_MARGINAL"]))
    n2 = np.sum(np.isin(d2.sphere_class, ["FUTURE_MARGINAL", "PAST_MARGINAL", "DOUBLY_MARGINAL"]))
    assert n2 > n1
    for rt in d2.roots:
        assert rt["R_hi"] - rt["R_lo"] >= 0


def test_large_grid_runs():
    s, _ = core_in_flrw(N=100000, rmax=15.0)
    d = diagnose(s)
    assert d.status["geometry"] == "AH_CANDIDATE" and len(d.roots) == 3


def test_root_position_converges():
    from scipy.optimize import brentq
    Rtrue = brentq(lambda x: 1 + x * (0.1 - 4.1 * np.exp(-x ** 2)), 0.8, 3.0)
    Ns = [200, 400, 800]
    errs = []
    for N in Ns:
        r = np.linspace(0, 4.0, N)
        H = 0.1 - 4.1 * np.exp(-r ** 2)
        # non-trivial Gamma: stretched areal radius R = r + 0.1 r^3 exp(-r^2) with A = R' (flat 3-metric)
        R = r + 0.1 * r ** 3 * np.exp(-r ** 2)
        dR = 1 + 0.1 * (3 * r ** 2 - 2 * r ** 4) * np.exp(-r ** 2)
        s = SphericalSlice(r, dR, R, -H, -H, "vertex")
        d = diagnose(s)
        oc = d.outer_ah_candidate
        # exact: Gamma = 1, so R theta+ = 2(1 + R H(r)) = 0
        rt = brentq(lambda x: 1 + (x + 0.1 * x ** 3 * np.exp(-x ** 2)) * (0.1 - 4.1 * np.exp(-x ** 2)), 0.8, 3.0)
        Rt = rt + 0.1 * rt ** 3 * np.exp(-rt ** 2)
        errs.append(abs(oc["R"] - Rt))
        assert oc["R_lo"] - 1e-12 <= Rt <= oc["R_hi"] + 1e-12
    o = np.log(np.array(errs[:-1]) / np.array(errs[1:])) / np.log(2)
    assert errs[-1] < 1e-8 and np.all(o > 3.0), (errs, o)

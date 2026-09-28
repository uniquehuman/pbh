"""FLRW slices (contract F6): flat / closed / open, homogeneous Klein-Gordon (FKG) from an
independent ODE integration, collapsing universe (cosmological trapping is not a PBH)."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sphdiag import Projections, ScalarField, SphericalSlice, diagnose


def flrw(k, a, H, N, rmax, center="vertex"):
    r = np.linspace(0, rmax, N) if center == "vertex" else (np.arange(N) + 0.5) * rmax / N
    S = {0: r, 1: np.sin(r), -1: np.sinh(r)}[k]
    rho = 3 * (H ** 2 + k / a ** 2) / (8 * np.pi)
    s = SphericalSlice(r, a * np.ones(N), a * S, -H * np.ones(N), -H * np.ones(N), center)
    z = np.zeros(N)
    return s, Projections(rho * np.ones(N), z, z.copy(), z.copy()), rho


@pytest.mark.parametrize("k", [0, 1, -1])
@pytest.mark.parametrize("center", ["vertex", "cell"])
def test_expanding_flrw(k, center):
    a, H = 1.7, 0.6
    rmax = 3.0 if k == -1 else 1.4          # keep the Hubble sphere on the grid, k=+1 below the equator
    s, p, rho = flrw(k, a, H, 300, rmax, center)
    d = diagnose(s, p)
    assert d.status["data"] == "OK"
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    # M_MS = (4 pi / 3) rho R^3 for every k (F6 + F11)
    Mex = 4 * np.pi / 3 * rho * s.R ** 3
    assert np.all(np.abs(d.M - Mex) <= 3 * d.sigma_M + 1e-13)        # sigma_M bounds the true error
    far = s.R > 0.3
    assert np.allclose(d.M_over_R3[far], 4 * np.pi / 3 * rho, rtol=1e-6)
    if center == "vertex" and k == 0:
        assert d.M_over_R3[0] == pytest.approx(4 * np.pi / 3 * rho, rel=1e-6)
    assert np.allclose(d.M_int, d.M, atol=1e-9)
    assert np.allclose(d.M_flux, 0.0)          # j_r = 0
    # Hubble sphere: theta- = 0 where R H = Gamma = S_k'(r)
    roots = d.roots
    assert len(roots) == 1
    rt = roots[0]
    assert rt["which"] == "-" and rt["type"] == "PAST_MOTS" and not rt["ah_candidate"]
    # exact: a H S_k(r*) = S_k'(r*)
    f = {0: lambda x: a * H * x - 1, 1: lambda x: a * H * np.sin(x) - np.cos(x),
         -1: lambda x: a * H * np.sinh(x) - np.cosh(x)}[k]
    from scipy.optimize import brentq
    rstar = brentq(f, 0.1, rmax)
    assert rt["r"] == pytest.approx(rstar, abs=1e-8)
    assert rt["R_lo"] <= a * {0: rstar, 1: np.sin(rstar), -1: np.sinh(rstar)}[k] <= rt["R_hi"]
    assert rt["M"] == pytest.approx(rt["R"] / 2, rel=1e-6)       # 2M/R = 1 on a marginal sphere (F13)
    assert d.status["geometry"] == "PAST_TRAPPED_ONLY"
    assert d.outer_ah_candidate is None
    assert np.all(d.sphere_class[s.r > rstar + 0.02] == "PAST_TRAPPED")


def test_collapsing_flrw_trapped_region_is_not_an_ah_candidate():
    """Contracting FLRW: all spheres beyond R = 1/|H| are future trapped, up to the grid edge.
    The theta+ root is the INNER boundary (+ inside, - outside) -> not a candidate."""
    s, p, _ = flrw(0, 1.0, -0.8, 200, 3.0)
    d = diagnose(s, p)
    assert d.status["geometry"] == "FUTURE_TRAPPED_NO_AH_CANDIDATE"
    assert len(d.roots) == 1
    rt = d.roots[0]
    assert rt["which"] == "+" and rt["type"] == "FUTURE_MOTS"
    assert rt["inner_sign"] == +1 and rt["outer_sign"] == -1
    assert not rt["ah_candidate"]
    assert rt["R"] == pytest.approx(1 / 0.8, rel=1e-8)


def test_closed_flrw_equator_at_maximal_expansion():
    """k = +1, H = 0: the equator r = pi/2 has U = Gamma = 0 -> DOUBLY_MARGINAL; beyond it NORMAL_REVERSED.
    Not a horizon."""
    a = 2.0
    N = 201
    r = np.linspace(0, 0.9 * np.pi, N)
    i_eq = np.argmin(np.abs(r - np.pi / 2))
    r[i_eq] = np.pi / 2
    R = a * np.sin(r)
    R[0] = 0.0
    s = SphericalSlice(r, a * np.ones(N), R, np.zeros(N), np.zeros(N), "vertex")
    rho = 3 / (8 * np.pi * a ** 2)
    z = np.zeros(N)
    d = diagnose(s, Projections(rho * np.ones(N), z, z, z))
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    assert d.sphere_class[i_eq] == "DOUBLY_MARGINAL"
    assert np.all(d.sphere_class[1:i_eq - 1] == "NORMAL")
    assert np.all(d.sphere_class[i_eq + 2:] == "NORMAL_REVERSED")
    assert d.status["geometry"] == "UNCERTAIN"         # never AH_CANDIDATE / never "no collapse" by default
    assert all(rt["type"] == "DEGENERATE" for rt in d.roots)
    assert d.outer_ah_candidate is None
    assert d.compactness[i_eq] == pytest.approx(1.0, abs=1e-8)


def fkg_solution(m=1.0, phi0=1.0, t1=6.0):
    """Independent homogeneous Klein-Gordon + Friedmann integration (k = 0)."""
    def rhs(t, y):
        a, phi, dphi = y
        H = np.sqrt(8 * np.pi / 3 * (0.5 * dphi ** 2 + 0.5 * m ** 2 * phi ** 2))
        return [a * H, dphi, -3 * H * dphi - m ** 2 * phi]
    sol = solve_ivp(rhs, (0, t1), [1.0, phi0, 0.0], method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
    return sol


@pytest.mark.parametrize("t", [0.7, 3.3, 6.0])
def test_fkg_slice_scalar_field(t):
    m = 1.0
    sol = fkg_solution(m)
    a, phi, dphi = sol.sol(t)
    H = np.sqrt(8 * np.pi / 3 * (0.5 * dphi ** 2 + 0.5 * m ** 2 * phi ** 2))
    N = 200
    rmax = 2.5 / (a * H)
    r = np.linspace(0, rmax, N)
    s = SphericalSlice(r, a * np.ones(N), a * r, -H * np.ones(N), -H * np.ones(N), "vertex")
    d = diagnose(s, ScalarField(phi * np.ones(N), dphi * np.ones(N), m))
    pr = d.projections
    V = 0.5 * m ** 2 * phi ** 2
    assert np.allclose(pr.rho, 0.5 * dphi ** 2 + V, rtol=1e-12)
    assert np.allclose(pr.S_rr, 0.5 * dphi ** 2 - V, rtol=1e-12, atol=1e-15)   # instantaneous p, not zero
    assert np.allclose(pr.S_thth, pr.S_rr)
    assert np.allclose(pr.j_r, 0.0)
    assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    assert d.norms["ham"]["all"]["Linf_rel"] < 1e-9
    rt = d.roots[0]
    assert rt["type"] == "PAST_MOTS" and rt["R"] == pytest.approx(1 / H, rel=1e-9)
    assert d.status["geometry"] == "PAST_TRAPPED_ONLY"
    # the ODE solution itself obeys Hdot = -4 pi phidot^2 (F6), independent of the module
    eps = 1e-4
    Hf = lambda tt: np.sqrt(8 * np.pi / 3 * (0.5 * sol.sol(tt)[2] ** 2 + 0.5 * m ** 2 * sol.sol(tt)[1] ** 2))
    Hdot = (Hf(t - eps) - Hf(t + eps)) / (-2 * eps) if t + eps <= 6.0 else (Hf(t) - Hf(t - eps)) / eps
    assert Hdot == pytest.approx(-4 * np.pi * dphi ** 2, rel=1e-3, abs=1e-6)


def test_fkg_wrong_hubble_rate_is_violated():
    m = 1.0
    sol = fkg_solution(m)
    a, phi, dphi = sol.sol(2.0)
    H = np.sqrt(8 * np.pi / 3 * (0.5 * dphi ** 2 + 0.5 * m ** 2 * phi ** 2)) * 1.001
    N = 100
    r = np.linspace(0, 2 / (a * H), N)
    s = SphericalSlice(r, a * np.ones(N), a * r, -H * np.ones(N), -H * np.ones(N), "vertex")
    d = diagnose(s, ScalarField(phi * np.ones(N), dphi * np.ones(N), m))
    assert d.status["constraints"] == "VIOLATED"
    assert np.all(d.constraint_point_status["ham"][1:] == "VIOLATED")

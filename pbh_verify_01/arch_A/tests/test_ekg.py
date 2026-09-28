"""Einstein-Klein-Gordon data with radial momentum flux (ODE-constructed, both constraints satisfied),
ScalarField projections, integral masses and the flux term, vertex vs cell centre, Vlasov adapter."""
import numpy as np
import pytest

from conftest import ekg_slice, manufactured_slice, order
from sphdiag import Projections, ScalarField, SphericalSlice, VlasovMoments, diagnose, pi_from_dtphi

NS = [100, 200, 400]


@pytest.mark.parametrize("center", ["vertex", "cell"])
def test_ekg_constraints_converge(center):
    e = {k: [] for k in ("ham_bulk", "mom", "hamM", "Mint")}
    for N in NS:
        s, f, d = ekg_slice(center, N)
        D = diagnose(s, f)
        assert D.status == {**D.status, "data": "OK", "constraints": "CONSISTENT_WITH_TRUNCATION"}
        e["ham_bulk"].append(D.norms["ham"]["bulk"]["Linf"])
        e["mom"].append(D.norms["mom"]["all"]["Linf"])
        e["hamM"].append(D.norms["hamM"]["bulk"]["Linf"])
        e["Mint"].append(np.max(np.abs(D.M_int - D.M)))
    for k, v in e.items():
        assert np.all(order(v, NS) > 3.6), (center, k, order(v, NS))


def test_ekg_projections_formulas():
    s, f, d = ekg_slice("vertex", 300)
    D = diagnose(s, f)
    p = D.projections
    Phi = d["dphi"] / s.A
    V = 0.5 * d["m"] ** 2 * d["phi"] ** 2
    tol = dict(rtol=1e-6, atol=1e-9)
    assert np.allclose(p.rho, 0.5 * d["Pi"] ** 2 + 0.5 * Phi ** 2 + V, **tol)                 # F3
    assert np.allclose(p.j_r, -d["Pi"] * d["dphi"], **tol)                                      # F4
    assert np.allclose(p.S_rr, 0.5 * d["Pi"] ** 2 + 0.5 * Phi ** 2 - V, **tol)                 # F5
    assert np.allclose(p.S_thth, 0.5 * d["Pi"] ** 2 - 0.5 * Phi ** 2 - V, **tol)
    assert p.j_r[0] == pytest.approx(0.0, abs=1e-14)          # j_r = O(r) at a regular centre
    # scalar-field derivative error enters sigma (documented extension of section 9)
    assert np.all(p.sigma_rho >= 0) and np.any(p.sigma_rho > 0)


def test_flux_term_is_essential():
    """With U j_r != 0 the naive dM = 4 pi R^2 rho dR fails; M_int (with flux) reproduces M_MS."""
    s, f, d = ekg_slice("vertex", 400)
    D = diagnose(s, f)
    naive = D.M[0] + D.M_rho
    assert np.max(np.abs(naive - D.M)) > 1e-3
    assert np.max(np.abs(D.M_int - D.M)) < 1e-8
    assert np.max(np.abs(D.M_flux)) > 1e-3
    # M_prop differs from M_MS (different quantity)
    assert abs(D.M_prop[-1] - D.M[-1]) > 1e-3


def test_time_symmetric_scalar_mprop_exceeds_m():
    """Pi = 0, K = 0: M_MS = M_rho (no flux); M_prop = int 4 pi R^2 A rho > M_MS because A > 1."""
    s, f, d = ekg_slice("vertex", 300, pi0=0.0, k0=0.0, k1=0.0)
    D = diagnose(s, f)
    assert D.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    assert np.allclose(D.M_flux, 0.0, atol=1e-15)
    assert np.max(np.abs(D.M_rho - D.M)) < 1e-8
    assert D.M_prop[-1] > D.M[-1] * (1 + 1e-4)
    assert D.status["geometry"] == "NO_TRAPPED_SPHERES"


def test_vertex_and_cell_agree():
    sv, fv, dv = ekg_slice("vertex", 400)
    sc, fc, dc = ekg_slice("cell", 400)
    Dv, Dc = diagnose(sv, fv), diagnose(sc, fc)
    Mc = np.interp(sv.r[5:-5], sc.r, Dc.M)
    assert np.max(np.abs(Mc - Dv.M[5:-5])) < 1e-5
    assert Dc.center_checks["extrapolated"] and not Dv.center_checks["extrapolated"]
    assert abs(Dc.center_checks["Gamma0"] - 1) < 1e-6 and abs(Dv.center_checks["Gamma0"] - 1) < 1e-12
    assert abs(Dc.center_checks["KA_minus_KB0"]) < 1e-6


def test_irregular_centre_is_reported():
    """A conical defect (Gamma(0) != 1) or K_A(0) != K_B(0) is reported, not repaired."""
    N = 100
    r = np.linspace(0, 2, N)
    s = SphericalSlice(r, 1.1 * np.ones(N), r.copy(), np.full(N, 0.3), np.full(N, 0.1), "vertex")
    D = diagnose(s)
    assert D.center_checks["Gamma0"] == pytest.approx(1 / 1.1)
    assert D.center_checks["KA_minus_KB0"] == pytest.approx(0.2)
    assert D.center_checks["Gamma0_minus_1"] == pytest.approx(1 / 1.1 - 1)


def test_vlasov_adapter():
    s, p, d = manufactured_slice("cell", 200)
    A = s.A
    vm = VlasovMoments(p.rho, p.j_r / A, np.full(s.N, 0.1), np.full(s.N, 0.05))
    pr = vm.projections(s)
    assert np.allclose(pr.j_r, p.j_r) and np.array_equal(pr.rho, p.rho)
    assert np.array_equal(pr.S_rr, np.full(s.N, 0.1)) and np.array_equal(pr.S_thth, np.full(s.N, 0.05))
    D1, D2 = diagnose(s, vm), diagnose(s, p)
    assert np.allclose(D1.mom, D2.mom, equal_nan=True)
    assert D1.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
    # forgetting the factor A (j_r = J instead of A J) is caught as a violation
    D3 = diagnose(s, Projections(p.rho, p.j_r / A, p.S_rr, p.S_thth))
    assert D3.status["constraints"] == "VIOLATED"


def test_pi_from_dtphi():
    dt, dr, al, be = np.array([1.0, 2.0]), np.array([0.5, -1.0]), np.array([2.0, 0.5]), np.array([0.2, 0.3])
    assert np.allclose(pi_from_dtphi(dt, dr, al, be), (dt - be * dr) / al)
    with pytest.raises(ValueError):
        pi_from_dtphi(dt, dr, np.array([1.0, 0.0]), be)
    with pytest.raises(ValueError):
        pi_from_dtphi(dt, dr, al, np.array([np.nan, 0.0]))


def test_pi_from_dtphi_in_moving_gauge():
    """Pi is gauge-covariant: the same n-frame data with different (alpha, beta) give the same Pi."""
    N = 50
    r = np.linspace(0, 2, N)
    phi_r = -r * np.exp(-r ** 2)
    Pi_true = 0.3 * np.exp(-r ** 2)
    for alpha, beta in ((np.ones(N), np.zeros(N)), (1.5 + 0.1 * r ** 2, 0.2 * r)):
        dtphi = alpha * Pi_true + beta * phi_r
        assert np.allclose(pi_from_dtphi(dtphi, phi_r, alpha, beta), Pi_true)
    s = SphericalSlice(r, np.ones(N), r, np.zeros(N), np.zeros(N), "vertex", alpha=1.5 + 0.1 * r ** 2, beta=0.2 * r)
    phi = 0.5 * np.exp(-r ** 2)
    Pi2 = s.pi_from_dtphi(s.alpha * Pi_true + s.beta * phi_r, phi)
    assert np.allclose(Pi2, Pi_true, atol=1e-5)

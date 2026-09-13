"""Schwarzschild vacuum regression: excised interior mass, zero-weight test particles."""
import numpy as np
from pbhgr.spherical_ev import Grid, Particles, RunConfig, Run, solve_metric
from pbhgr.analytic import circular_orbit_L, schwarzschild_radial_infall, eta_of_t

M = 1.0


def _grid(K=600):
    return Grid(r_out=40.0, K=K, r_in=2.5 * M, m_in=M)


def test_exterior_metric_is_schwarzschild():
    g = _grid()
    P = Particles(np.array([10.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
    met = solve_metric(g, P)
    assert np.allclose(met.m_edge, M)
    exact = 0.5 * np.log(1 - 2 * M / g.r_edge)
    e1 = np.max(np.abs(met.mu_edge - exact))
    g2 = _grid(K=1200)
    e2 = np.max(np.abs(solve_metric(g2, P).mu_edge - 0.5 * np.log(1 - 2 * M / g2.r_edge)))
    assert e1 < 1e-3                       # midpoint-rule error, steepest near the excision
    assert 3.0 < e1 / e2 < 5.0             # second-order convergence


def test_circular_orbit_stays_circular():
    g = _grid()
    r0 = 8.0 * M
    P = Particles(np.array([r0]), np.array([0.0]), np.array([circular_orbit_L(M, r0)]),
                  np.array([0.0]), np.array([1.0]))
    errs = []
    for K in (600, 1200):
        P = Particles(np.array([r0]), np.array([0.0]), np.array([circular_orbit_L(M, r0)]),
                      np.array([0.0]), np.array([1.0]))
        run = Run(_grid(K), P, RunConfig(t_end=2 * np.pi * np.sqrt(r0**3 / M) * 3, cfl=0.4, diag_every=100000))
        run.run()
        errs.append(abs(run.P.r[0] - r0))
        assert abs(run.P.w[0]) < 1e-4
    assert errs[0] < 1e-3 * r0
    assert 3.0 < errs[0] / errs[1] < 5.0   # second-order convergence of the orbit radius


def test_radial_infall_coordinate_time():
    g = _grid(K=1200)
    R0 = 15.0 * M
    P = Particles(np.array([R0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
    t_end = 60.0
    run = Run(g, P, RunConfig(t_end=t_end, cfl=0.4, diag_every=100000))
    run.run()
    eta = eta_of_t(M, R0, t_end)
    r_exact, tau_exact, _ = schwarzschild_radial_infall(M, R0, eta)
    assert abs(run.P.r[0] - r_exact) < 2e-3 * R0
    assert abs(run.P.tau[0] - tau_exact) < 2e-3 * tau_exact

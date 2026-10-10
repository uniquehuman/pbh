import numpy as np
from pbhgr.spherical_ev import Grid, Particles, solve_metric
from pbhgr.analytic import circular_orbit_L


def test_exterior_of_matter_shell_is_schwarzschild():
    rng = np.random.default_rng(0)
    g = Grid(r_out=40.0, K=800)
    n = 5000
    r = rng.uniform(8.0, 10.0, n)
    P = Particles(r, np.zeros(n), circular_orbit_L(1.0, r), np.full(n, 1.0 / n), np.ones(n))
    met = solve_metric(g, P)
    M = met.M
    ext = g.r_edge > 10.5
    exact = 0.5 * np.log(1 - 2 * M / g.r_edge[ext])
    assert np.max(np.abs(met.mu_edge[ext] - exact)) < 1e-4
    assert np.allclose(met.m_edge[ext], M)

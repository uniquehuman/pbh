"""Known Einstein-Vlasov benchmark: an isotropic polytropic steady state (f = (E0 - e^mu E)_+)
sampled into macroparticles must reproduce the analytic metric at t = 0 and remain static."""
import numpy as np
import pytest
from pbhgr.steady_state import build, sample
from pbhgr.spherical_ev import Grid, Run, RunConfig, solve_metric


def test_sampled_state_matches_analytic_metric():
    st = build(1.3)
    g = Grid(r_out=4 * st["R"], K=400)
    P = sample(st, 40000, np.random.default_rng(5))
    met = solve_metric(g, P)
    m_an = np.interp(g.r_edge, st["r"], st["m"], right=st["M"])
    assert np.max(np.abs(met.m_edge - m_an)) < 2e-3 * st["M"]
    inside = g.r_edge < st["R"]
    mu_an = np.interp(g.r_edge[inside], st["r"], st["mu"])
    assert np.max(np.abs(met.mu_edge[inside] - mu_an)) < 3e-3


@pytest.mark.slow
def test_steady_state_stays_steady():
    st = build(1.3); R, M = st["R"], st["M"]; t_dyn = np.sqrt(R**3 / M)
    g = Grid(r_out=4 * R, K=400)
    P = sample(st, 20000, np.random.default_rng(5))
    comp0 = 2 * solve_metric(g, P).m_edge[1:] / g.r_edge[1:]
    run = Run(g, P, RunConfig(t_end=2 * t_dyn, cfl=0.4, diag_every=25))
    run.run()
    cm = np.array([d["max_2m_over_r"] for d in run.diag])
    assert abs(cm.mean() - comp0.max()) < 0.01 * comp0.max()
    assert cm.std() < 0.02 * comp0.max()
    assert run.ledger.escaped_rest_mass == 0.0

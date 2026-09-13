"""Proper-time decay of macroparticles, daughter emission, energy bookkeeping, Gamma -> 0."""
import numpy as np
from pbhgr.spherical_ev import Grid, Particles, RunConfig, Run, solve_metric
from pbhgr.initial_data import shell_family


def test_time_dilation_of_decay_in_weak_field():
    # negligible mass -> flat space; particles moving with gamma = 2 decay slower by 1/gamma
    g = Grid(r_out=200.0, K=400)
    n = 2000
    gam = 2.0
    w = np.sqrt(gam**2 - 1) * np.ones(n)
    P = Particles(np.full(n, 10.0), w, np.zeros(n), np.full(n, 1e-9), np.ones(n))
    G = 0.5
    run = Run(g, P, RunConfig(Gamma=G, t_end=4.0, cfl=0.5, emit_fraction=0.3, diag_every=1000000))
    run.run()
    N_expected = 1e-9 * np.exp(-G * 4.0 / gam)
    Xmask = run.P.m > 0
    assert np.allclose(run.P.N[Xmask], N_expected, rtol=1e-6)
    # rest-mass ledger closes: remaining + pending + decayed(emitted) = initial
    tot = run.P.N[Xmask].sum() + run.P.pending.sum() + run.ledger.decayed_rest_mass
    assert abs(tot - n * 1e-9) < 1e-12 * n
    # daughters carry the parent energy: sum N E of daughters == decayed rest mass * gamma
    E = run.P.energy()
    Ed = np.sum((run.P.N * E)[~Xmask])
    assert abs(Ed - run.ledger.decayed_rest_mass * gam) < 1e-6 * Ed


def test_adm_mass_conserved_through_decay_and_escape():
    rng = np.random.default_rng(5)
    g = Grid(r_out=30.0, K=300)
    P = shell_family(M=1.0, R=5.0, n_part=4000, rng=rng, flatness=2.0, L_rms=0.3)
    P.N *= 1.0 / solve_metric(g, P).M
    run = Run(g, P, RunConfig(Gamma=0.3, t_end=40.0, cfl=0.4, emit_fraction=0.25, diag_every=50))
    run.run()
    Mt = np.array([d["M_total"] for d in run.diag])
    assert np.max(np.abs(Mt - Mt[0])) < 3e-3          # daughters escape carrying Killing energy
    assert run.ledger.escaped_daughters > 0
    assert run.diag[-1]["n_daughters"] >= 0


def test_gamma_to_zero_limit_is_continuous():
    rng = np.random.default_rng(7)
    res = {}
    for G in (0.0, 1e-4):
        g = Grid(r_out=30.0, K=300)
        P = shell_family(M=1.0, R=5.0, n_part=3000, rng=np.random.default_rng(7), flatness=2.0, L_rms=0.3)
        P.N *= 1.0 / solve_metric(g, P).M
        run = Run(g, P, RunConfig(Gamma=G, t_end=15.0, cfl=0.4, diag_every=50, seed=1))
        run.run()
        res[G] = run.diag[-1]["max_2m_over_r"]
    assert abs(res[0.0] - res[1e-4]) < 5e-3

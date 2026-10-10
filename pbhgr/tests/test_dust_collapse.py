"""Cold dust collapse (LTB / Oppenheimer-Snyder): every shell is a radial Schwarzschild geodesic
of the mass m(r0) enclosed initially, in its own proper time; the outermost shell also follows
the exterior coordinate-time law."""
import numpy as np
import pytest
from pbhgr.spherical_ev import Grid, RunConfig, Run, solve_metric, metric_at
from pbhgr.initial_data import dust_ball, normalise_mass
from pbhgr.analytic import schwarzschild_radial_infall, eta_of_t
from scipy.optimize import brentq

M, R0 = 1.0, 10.0


def _run(n_part, K, t_end=25.0, seed=3):
    rng = np.random.default_rng(seed)
    g = Grid(r_out=20.0, K=K)
    P = normalise_mass(g, dust_ball(M, R0, n_part, rng), M)
    met0 = solve_metric(g, P)
    m0 = metric_at(g, met0, P.r)[0]
    r0 = P.r.copy()
    run = Run(g, P, RunConfig(t_end=t_end, cfl=0.4, diag_every=50))
    run.run()
    return run, r0, m0


def _r_of_tau(m, r0, tau):
    if m <= 0:
        return r0
    f = lambda e: np.sqrt(r0**3 / (8 * m)) * (e + np.sin(e)) - tau
    e = brentq(f, 0.0, np.pi, xtol=1e-12)
    return 0.5 * r0 * (1 + np.cos(e))


def test_interior_shells_follow_ltb_geodesics():
    run, r0, m0 = _run(20000, 400)
    P = run.P
    # all particles still inside (no escape) and ADM mass conserved
    assert run.diag[-1]["escaped_energy"] == 0.0
    assert abs(run.diag[-1]["M_total"] - M) < 1e-4
    sel = r0 > 0.2 * R0            # skip innermost shells where 1/r^3 resolution is poor
    r_pred = np.array([_r_of_tau(m, a, tau) for m, a, tau in zip(m0[sel], r0[sel], P.tau[sel])])
    rel = np.abs(P.r[sel] - r_pred) / r0[sel]
    assert np.median(rel) < 5e-3
    assert np.percentile(rel, 95) < 2e-2


def test_surface_follows_exterior_coordinate_time_law():
    run, r0, m0 = _run(20000, 400)
    i = int(np.argmax(r0))
    eta = eta_of_t(M, r0[i], run.t)
    r_exact = schwarzschild_radial_infall(m0[i], r0[i], eta)[0]
    assert abs(run.P.r[i] - r_exact) / r0[i] < 1e-2


@pytest.mark.slow
def test_surface_error_converges_with_resolution():
    errs = []
    for n, K in ((5000, 100), (20000, 200), (80000, 400)):
        run, r0, m0 = _run(n, K, t_end=25.0)
        i = int(np.argmax(r0))
        eta = eta_of_t(M, r0[i], run.t)
        r_exact = schwarzschild_radial_infall(m0[i], r0[i], eta)[0]
        errs.append(abs(run.P.r[i] - r_exact) / r0[i])
    assert errs[2] < errs[0]

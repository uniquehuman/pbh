"""Ladder step 4: an Einstein-Vlasov steady state (isotropic polytrope from pbhgr.steady_state) must stay static in the
areal gauge with maximal slicing (K = 0).  Radial orbits cross the centre: this exercises the multi-stream machinery."""
import sys, time, numpy as np
sys.path.insert(0, '.')
from pbhgr.steady_state import build, sample
from pbhgr.cosmo_ev import StaticBackground, CosmoGrid, CParticles, CosmoRun, CosmoRunConfig, solve_metric
E0c = float(sys.argv[1]) if len(sys.argv) > 1 else 1.3
n_part = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
K = int(sys.argv[3]) if len(sys.argv) > 3 else 800
st = build(E0c); R, M = st["R"], st["M"]; t_dyn = np.sqrt(R**3 / M)
rng = np.random.default_rng(3)
Pp = sample(st, n_part, rng)
part = CParticles(Pp.r.copy(), Pp.w.copy(), Pp.L.copy(), Pp.N.copy())
bg = StaticBackground(); grid = CosmoGrid(x_out=3.0 * R, K=K)
met0 = solve_metric(grid, bg, part, 0.0)
for _ in range(4):
    part.N *= M / met0.m_edge[-1]; met0 = solve_metric(grid, bg, part, 0.0)
comp0 = 2 * met0.m_edge[1:] / met0.R_edge[1:]
mex = np.interp(met0.R_edge, st["r"], st["m"]); ins = (met0.R_edge > 0.2 * R) & (met0.R_edge < R)
print(f"initial data: renormalisation factor of N = {part.N[0] / Pp.N[0]:.4f}; exact max 2m/r = {np.max(2 * st['m'][1:] / st['r'][1:]):.4f}; max |m_edge/m_exact - 1| on 0.2R<R<R = {np.max(np.abs(met0.m_edge[ins] / mex[ins] - 1)):.2e}")
print(f"polytrope E0c={E0c}: R={R:.3f}, M={M:.4f}, 2M/R={2*M/R:.4f}, t_dyn={t_dyn:.2f}; grid K={K} dx={grid.dx:.4f}; particles {n_part}")
print(f"t=0: max 2m/R={comp0.max():.4f} at R={met0.R_edge[1:][np.argmax(comp0)]:.3f}; alpha(0)={met0.alpha_c[0]:.4f} (exact e^mu(0)={np.exp(st['mu'][0]):.4f}); alpha_out={met0.alpha_edge[-1]:.4f}")
alpha_exact = np.exp(np.interp(met0.R_c, st["r"], st["mu"]))
inside = met0.R_c < R
print(f"     alpha_c vs exact e^mu inside the star: max rel diff={np.max(np.abs(met0.alpha_c[inside]/alpha_exact[inside]-1)):.2e}")
def killing(met):
    return np.sqrt(1 + part.P**2 + part.Lsq / np.maximum(part.x, 1e-12)**2) * np.interp(part.x, met.R_edge, met.alpha_edge)
ek0 = killing(met0); E_kill0 = np.sum(part.N * ek0)
hist = []
def cb(run, d):
    met = solve_metric(grid, bg, part, run.t); comp = 2 * met.m_edge[1:] / met.R_edge[1:]
    ek = killing(met); Ek = np.sum(part.N * ek); rms = np.sqrt(np.mean((ek / ek0 - 1) ** 2))
    hist.append((run.t / t_dyn, comp.max(), met.alpha_c[0], Ek / E_kill0 - 1, np.max(np.abs(met.alpha_c[inside] / alpha_exact[inside] - 1)), rms))
    if len(hist) % 4 == 0:
        print(f"t/t_dyn={hist[-1][0]:5.2f}: max 2m/R={hist[-1][1]:.4f} alpha(0)={hist[-1][2]:.4f} Killing energy drift={hist[-1][3]:+.2e} per-particle rms={rms:.2e} max|alpha/exact-1|={hist[-1][4]:.2e} bad={met.n_bad} n(x<0.05R)={(part.x<0.05*R).sum()}", flush=True)
cfg = CosmoRunConfig(cfl=0.3, t_end=3.0 * t_dyn, diag_every=50, max_dt_frac=1e9)
run = CosmoRun(grid, bg, part, cfg, 1e-9)
t0 = time.time(); run.run(cb)
h = np.array(hist)
print(f"done {run.step} steps, {time.time()-t0:.0f}s: max 2m/R mean over run={h[:,1].mean():.4f} (t=0: {comp0.max():.4f}), std={h[:,1].std():.4f}; alpha(0) drift={h[-1,2]/h[0,2]-1:+.2e}; Killing energy drift={h[-1,3]:+.2e}")

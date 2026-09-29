#!/usr/bin/env python3
"""Extension (contract §3 of the brief): feed Einstein-Vlasov slices of pbhgr.cosmo_ev (areal-CMC gauge, V1 code)
into sphdiag and compare with V1's own horizon criterion (2m/R >= 1 with K^theta_theta > 0, outermost trapped edge).

Setup = pbhgr scripts/v1_ladder.py "ltb" test at mu = 0.36, K = 2000, stretch 4, 4 shells/cell, cold (the run in
pbhgr/runs/v1/mu036fix, first AH at 12.19 t_H).  Slice handed to sphdiag in the comoving label x (not R):
    r = x_edge, R = a x, A_x = a A_R (gamma_xx = a^2 A_R^2), K^r_r = K - 2 Kth, K^theta_theta = Kth.
Matter (n-frame moments per cell of proper volume 4 pi R^2 A dR, pbhgr deposition), moved to edges by averaging the
two neighbouring cells (edge 0: E, p, q from cell 0; J = 0 by parity):
    E = sE/vol, J = sJ/vol (orthonormal radial momentum density), p = sp/vol, q = S^theta_theta = sq/(2 vol).
What is circular here: pbhgr builds A from m and Kth (1/A^2 = 1 - 2m/R + R^2 Kth^2), so M_MS(sphdiag) = m(pbhgr)
identically wherever pbhgr did not clip u = 1/A^2 at 1e-14.  Not circular: the classification/roots (sphdiag computes
theta_pm from (A, R, Kth) and locates roots itself), the momentum constraint (Kth' by FD against deposited J) and the
Hamiltonian constraint (A' by FD against deposited E).
Usage: PYTHONPATH=/root/PBH/pbhgr:src python3 extension/vlasov_snapshots.py [--tend 13] [--diag 20]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

from pbhgr.cosmo_ev import Background, CosmoGrid, CParticles, CosmoRun, CosmoRunConfig, solve_metric
from pbhgr.cosmo_id import yoo_spherical_cmc, sample_shells_quiet
from pbhgr.ltb import LTBGaussian
import sphdiag

ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, default=0.36); ap.add_argument("--K", type=int, default=2000)
ap.add_argument("--n", type=int, default=4); ap.add_argument("--stretch", type=float, default=4.0)
ap.add_argument("--xout", type=float, default=5.0); ap.add_argument("--tend", type=float, default=13.0)
ap.add_argument("--dR", type=float, default=0.0, help="dR_target (re-stretch), as V1 --dR")
ap.add_argument("--tstart", type=float, default=6.0, help="first diagnosed time (t_H)")
ap.add_argument("--diag", type=int, default=20); ap.add_argument("--out", default="results/extension")
ap.add_argument("--dump", default="", help="t_H range lo,hi: save full slice arrays (npz) inside it")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

bg = Background(H_i=5.0); rm = np.sqrt(6.0); t_H = bg.t_i * (np.sqrt(6) / 0.2) ** 3
x_out = args.xout * rm
grid = CosmoGrid(x_out=x_out, K=args.K, stretch=args.stretch)
prof = yoo_spherical_cmc(args.mu, r_max=x_out * 1.05)
x, P, Lsq, N = sample_shells_quiet(prof, grid.x_edge, args.n)
part = CParticles(x, P, Lsq, N)
L = LTBGaussian(args.mu)
print(f"mu={args.mu} K={args.K} t_H={t_H:.3f}; LTB t_C(0)={L.tC(1e-6)/t_H:.3f} t_H", flush=True)


def edge_from_cells(c, parity):
    e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[1:] + c[:-1])
    e[0] = c[0] if parity > 0 else 0.0
    e[-1] = c[-1]
    return e


records = []


def analyse(run):
    met = solve_metric(run.grid, run.bg, run.P, run.t)
    a, K = met.a, run.bg.K(run.t)
    xe = met.R_edge / a
    Ax = a * met.A_edge
    KB = met.Kth_edge
    KA = K - 2 * KB
    V3 = (met.R_edge[1:] ** 3 - met.R_edge[:-1] ** 3) / 3.0
    Ac = 0.5 * (met.A_edge[1:] + met.A_edge[:-1])
    vol = 4 * np.pi * V3 * Ac
    S = met.S
    E = edge_from_cells(S["sE"] / vol, +1); J = edge_from_cells(S["sJ"] / vol, -1)
    p = edge_from_cells(S["sp"] / vol, +1); q = edge_from_cells(S["sq"] / (2 * vol), +1)
    slc = sphdiag.SphericalSlice(xe, Ax, met.R_edge, KA, KB, center="vertex")
    d = sphdiag.diagnose(slc, sphdiag.VlasovMoments(E, J, p, q))
    # V1 criterion on edges
    R = met.R_edge[1:]
    trapped = (KB[1:] > 0) & (2 * met.m_edge[1:] / R >= 1.0)
    v1 = None
    if trapped.any():
        j = int(np.max(np.nonzero(trapped)[0]))
        v1 = dict(R_AH=float(R[j]), M_AH=float(met.m_edge[j + 1]), x_AH=float(R[j] / a))
    oah = d.outer_ah_candidate
    clipped = met.u_edge <= 1.5e-14
    ok = ~clipped
    rel_m = np.abs(d.M[ok] - met.m_edge[ok]) / np.maximum(np.abs(met.m_edge[ok]), 1e-300)
    cls, cnt = np.unique(d.sphere_class, return_counts=True)
    rec = dict(t_over_tH=run.t / t_H, a=a, n_clipped=int(clipped.sum()),
               x_clipped_max_over_rm=float(xe[clipped].max() / rm) if clipped.any() else 0.0,
               v1_AH=v1, sphdiag_status=dict(data=d.status["data"], constraints=d.status["constraints"],
                                             geometry=d.status["geometry"]),
               sphdiag_outer_ah=(dict(R=oah["R"], R_lo=oah["R_lo"], R_hi=oah["R_hi"], M=oah["M"], x=oah["r"])
                                 if oah else None),
               n_ah_candidates=int(sum(1 for rr in d.roots if rr["ah_candidate"])),
               classes={str(c): int(n) for c, n in zip(cls, cnt)},
               max_rel_M_vs_m_unclipped=float(np.max(rel_m[1:])) if ok.sum() > 1 else None,
               M_at_clipped=[(float(xe[i] / rm), float(met.m_edge[i]), float(d.M[i])) for i in np.flatnonzero(clipped)[:5]],
               norms={k: {rg: d.norms[k][rg]["Linf_rel"] for rg in ("center", "bulk", "outer")} for k in ("ham", "mom", "hamM")},
               n_violated=d.status.get("n_violated"), warnings=[str(w)[:160] for w in d.status["warnings"]][:6])
    records.append(rec)
    if args.dump:
        lo, hi = map(float, args.dump.split(","))
        if lo <= run.t / t_H <= hi:
            np.savez(out / f"slice_t{run.t / t_H:.3f}.npz", x=xe, R=met.R_edge, A=Ax, KA=KA, KB=KB, E=E, J=J, p=p, q=q,
                     m=met.m_edge, u=met.u_edge, Rtp=d.Rtheta_plus, Rtm=d.Rtheta_minus, taup=d.tau_plus, taum=d.tau_minus,
                     sig=d.sigma_Rtheta_plus, M=d.M, cls=d.sphere_class, a=a, t=run.t / t_H)
    oa = rec["sphdiag_outer_ah"]
    print(f"t={rec['t_over_tH']:7.3f} t_H clipped={rec['n_clipped']:2d} V1 AH: "
          + (f"R={v1['R_AH']:.4g} M={v1['M_AH']:.4g}" if v1 else "none")
          + f" | sphdiag {rec['sphdiag_status']['geometry']}/{rec['sphdiag_status']['constraints']}/{rec['sphdiag_status']['data']} "
          + (f"outer AH R={oa['R']:.4g} [{oa['R_lo']:.4g},{oa['R_hi']:.4g}] M={oa['M']:.4g}" if oa else "no candidate")
          + f" | max|M-m|/m (unclipped) {rec['max_rel_M_vs_m_unclipped']:.1e}"
          + f" | Linf_rel ham/mom/hamM bulk {rec['norms']['ham']['bulk']:.1e}/{rec['norms']['mom']['bulk']:.1e}/{rec['norms']['hamM']['bulk']:.1e}", flush=True)


def cb(run, d):
    if run.t >= args.tstart * t_H:
        analyse(run)


cfg = CosmoRunConfig(cfl=0.4, t_end=args.tend * t_H, diag_every=args.diag, dR_target=args.dR)
run = CosmoRun(grid, bg, part, cfg, bg.t_i)
t0 = time.time()
run.run(cb)
print(f"done: steps={run.step} wall={time.time() - t0:.1f}s; V1 first AH: {run.ah_first}", flush=True)
json.dump(dict(args=vars(args), t_H=t_H, v1_first_AH={k: float(v) for k, v in (run.ah_first or {}).items()},
               records=records), open(out / f"vlasov_mu{args.mu:.3f}_K{args.K}_dR{args.dR:g}.json", "w"), indent=1)

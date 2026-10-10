#!/usr/bin/env python3
"""V1 validation ladder, steps 1-3 (areal-CMC gauge):
  flrw     homogeneous FLRW to a chosen t/t_H: alpha = 1, A = 1, Kth = -H, particles static in x, P ∝ 1/a
  idcheck  Yoo spherical CMC initial data vs the exact LTB functions (mass and energy per shell)
  ltb      cold collapse for the Yoo profile compared shell-by-shell with the LTB cycloid in proper time
  ah       apparent-horizon detection for mu = 0.5 (LTB: central singularity at 7.5 t_H)
Usage: python3 scripts/v1_ladder.py {flrw,idcheck,ltb,ah} [--mu 0.3] [--tend 6] [--K 2000] [--n 40000] [--xout 8]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbhgr.cosmo_ev import Background, CosmoGrid, CParticles, CosmoRun, CosmoRunConfig, solve_metric, metric_at
from pbhgr.cosmo_id import yoo_spherical_cmc, flrw_profile, sample_shells, sample_shells_quiet
from pbhgr.ltb import LTBGaussian

ap = argparse.ArgumentParser()
ap.add_argument("test"); ap.add_argument("--mu", type=float, default=0.3); ap.add_argument("--tend", type=float, default=6.0)
ap.add_argument("--K", type=int, default=2000); ap.add_argument("--n", type=int, default=4); ap.add_argument("--xout", type=float, default=5.0)
ap.add_argument("--cfl", type=float, default=0.4); ap.add_argument("--out", default="runs/v1"); ap.add_argument("--stretch", type=float, default=0.0); ap.add_argument("--sigma", type=float, default=0.0); ap.add_argument("--tinj", type=float, default=1.0); ap.add_argument("--xdisp", type=float, default=2.0); ap.add_argument("--diag", type=int, default=100); ap.add_argument("--stop_after_ah", type=float, default=0.0); ap.add_argument("--dR", type=float, default=0.0); ap.add_argument("--release", default=""); ap.add_argument("--rel_k", type=float, default=10.0); ap.add_argument("--rel_delta", type=float, default=0.05); ap.add_argument("--rel_h", type=float, default=0.6); ap.add_argument("--rel_xmax", type=float, default=2.0); ap.add_argument("--rel_bnl", type=float, default=0.0); ap.add_argument("--profile", default="yoo"); ap.add_argument("--n_inner", type=int, default=0); ap.add_argument("--x_inner", type=float, default=0.5)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
bg = Background(H_i=5.0); rm = np.sqrt(6.0); t_H = bg.t_i * (np.sqrt(6) / 0.2) ** 3
from pbhgr.ltb import LTBProfile
from pbhgr.profiles import profile_table
_table = None
if args.profile != "yoo":
    _table = profile_table(args.profile); _table["name"] = args.profile
    _Lref = LTBProfile(0.5 if args.test == "ah" else args.mu, _table); rm = _Lref.r_m; t_H = _Lref.t_H
def make_L(mu_):
    return LTBGaussian(mu_) if _table is None else LTBProfile(mu_, _table)
def make_prof(mu_, r_max):
    L_ = make_L(mu_)
    return yoo_spherical_cmc(mu_, r_max=r_max) if _table is None else yoo_spherical_cmc(mu_, r_max=r_max, zeta_fns=(L_.zeta, L_.dzeta, L_.ddzeta))
x_out = args.xout * rm
grid = CosmoGrid(x_out=x_out, K=args.K, stretch=args.stretch)
log = open(out / (f"{args.test}_mu{args.mu:.3f}_K{args.K}_n{args.n}" + (f"i{args.n_inner}" if args.n_inner else "") + f"_s{args.stretch:.0f}_sig{args.sigma:.0e}" + (f"_dR{args.dR:g}" if args.dR > 0 else "") + (f"_rel{args.release}_k{args.rel_k:g}_d{args.rel_delta:g}" + (f"_b{args.rel_bnl:g}" if args.rel_bnl > 0 else "") if args.release else "") + (f"_{args.profile}" if args.profile != "yoo" else "") + ".log"), "w")
def P_(*a):
    print(*a); print(*a, file=log); log.flush()
P_(f"t_i={bg.t_i:.5f} t_H={t_H:.3f} ({t_H/bg.t_i:.1f} t_i)  x_out={x_out:.3f} (H_i x_out={5*x_out:.0f})  K={args.K} stretch={args.stretch} dx_min={grid.dx:.2e} ({grid.dx/rm:.1e} r_m) dx_max={grid.dxc.max():.2e}  particles/cell={args.n}")

def cb(run, d):
    if args.stop_after_ah > 0 and run.ah_first is not None and run.t > run.ah_first["t"] + args.stop_after_ah * t_H:
        run.cfg.t_end = run.t          # classification done: stop shortly after the first apparent horizon
    ah = d["AH"]
    P_(f"step {d['step']:6d} t/t_H={d['t']/t_H:8.4f} a={d['a']:9.3f} s={d.get('stretch',0):.2f} dRmin={d.get('dR_min',0):.2f} max2m/R(coll)={d['max_2m_over_R_collapsing']:.4f} at x/rm={d['x_at_max']/rm:.3f} "
       f"min_alpha={d['min_alpha']:.4f} max_A={d['max_A']:.4f} m_out/flrw-1={d['m_out_over_flrw']-1:+.2e} bad={d['n_bad']} "
       f"ueq={d['unused_eq_residual']:.2e} min_x={d['min_x']:.2e} xbad/rm={d['x_bad_max']/rm:.4f}" + (f" AH: x/rm={ah['x_AH']/rm:.4f} M={ah['M_AH']:.4g}" if ah else ""))

if args.test == "flrw":
    prof = flrw_profile(H_i=5.0, r_max=x_out * 1.01)
    x, P, Lsq, N = sample_shells_quiet(prof, grid.x_edge, args.n, n_inner=args.n_inner, x_inner=args.x_inner * rm)      # quiet start, P = 0
    pec = np.zeros(len(x), bool)
    part = CParticles(x, P, Lsq, N)
    met0 = solve_metric(grid, bg, part, bg.t_i)
    P_(f"t_i: max|alpha-1|={np.max(np.abs(met0.alpha_c-1)):.2e} max|A-1|={np.max(np.abs(met0.A_edge-1)):.2e} max|Kth+H|/H={np.max(np.abs(met0.Kth_edge+5))/5:.2e} m_out/flrw-1={met0.m_edge[-1]/(0.5*25*x_out**3)-1:+.2e}")
    cfg = CosmoRunConfig(cfl=args.cfl, t_end=args.tend * t_H, diag_every=200)
    run = CosmoRun(grid, bg, part, cfg, bg.t_i); x0 = part.x.copy(); P0 = part.P.copy()
    t0 = time.time(); met = run.run(cb)
    a = met.a
    P_(f"done: steps={run.step} wall={time.time()-t0:.1f}s  max|alpha-1|={np.max(np.abs(met.alpha_c-1)):.2e} max|A-1|={np.max(np.abs(met.A_edge-1)):.2e} "
       f"max|Kth+H|/H={np.max(np.abs(met.Kth_edge+bg.H(run.t)))/bg.H(run.t):.2e}  max|x-x0|/x_out={np.max(np.abs(part.x-x0))/x_out:.2e}  "
       f"max|P| (rest)={np.max(np.abs(part.P)):.2e}  m_out/flrw-1={met.m_edge[-1]/(0.5*bg.H(run.t)**2*met.R_edge[-1]**3)-1:+.2e}")

elif args.test == "idcheck":
    prof = make_prof(args.mu, x_out * 1.05)
    L = make_L(args.mu)
    # enclosed rest mass as the shell label: LTB  M_rest(r) = int dm/sqrt(1+2E);  ID  M_rest(R) = int dN/dR dR
    r = prof["r"]; R = prof["R"]
    Mrest_id = np.concatenate([[0], np.cumsum(0.5 * (prof["dN_dR"][1:] + prof["dN_dR"][:-1]) * np.diff(R))])
    rL = np.linspace(1e-6, x_out * 1.05, 80001); mL = L.m(rL); EL = L.E(rL)
    Mrest_L = np.concatenate([[0], np.cumsum(np.diff(mL) / np.sqrt(1 + 2 * 0.5 * (EL[1:] + EL[:-1])))])
    # shell energy from the ID state: dR/dtau = P/A - R Kth W with P = Gamma v, W = Gamma
    Gam = prof["Gamma"]; Pn = Gam * prof["v"]
    dRdtau = Pn / prof["A"] - R * prof["Kth"] * Gam
    E_id = 0.5 * dRdtau**2 - prof["m"] / np.maximum(R, 1e-300)
    tau0 = LTBGaussian.shell_tau(np.maximum(prof["m"], 1e-300), np.minimum(E_id, -1e-300), R, dRdtau)
    P_("Yoo CMC data vs exact LTB at equal enclosed rest mass (eps_i = 0.2):")
    P_("  r_iso    R       m_id/m_LTB-1   E_id/E_LTB-1   delta=E/rho_bar-1   v      tau0/t_i-1")
    rho_bar = 3 * 25 / (8 * np.pi)
    for rr in (0.25, 0.5, 1.0, 1.5, 2.0, 2.449, 3.0, 4.0, 6.0, 8.0, 12.0):
        i = np.searchsorted(r, rr)
        Mi = Mrest_id[i]
        rl = np.interp(Mi, Mrest_L, rL)
        P_(f"  {r[i]:6.3f} {R[i]:7.4f}  {prof['m'][i]/L.m(rl)-1:+.3e}   {E_id[i]/L.E(rl)-1:+.3e}   {prof['E'][i]/rho_bar-1:+.4e}  {prof['v'][i]:+.2e}  {tau0[i]/bg.t_i-1:+.3e}")
    P_(f"  total mass excess inside x_out relative to FLRW: {(prof['m'][np.searchsorted(r, x_out)]/(0.5*25*R[np.searchsorted(r, x_out)]**3)-1):+.3e}")
    P_(f"  central delta = {prof['E'][0]/rho_bar-1:.5f}  (linear growing mode: (2/5) eps^2 * 6/... = 2.4 mu eps_i^2/6*...; expected ~ (2/5)(k^2/(aH)^2)*mu = {0.4*0.04*args.mu:.5f})")

elif args.test in ("ltb", "ah"):
    mu = 0.5 if args.test == "ah" else args.mu
    prof = make_prof(mu, x_out * 1.05)
    x, P, Lsq, N = sample_shells_quiet(prof, grid.x_edge, args.n, n_inner=args.n_inner, x_inner=args.x_inner * rm)
    part = CParticles(x, P, Lsq, N)
    L = make_L(mu)
    tC0 = L.tC(1e-6)
    if not np.isfinite(tC0) or tC0 <= 0: tC0 = L.tC(rm)          # flat-core profiles: no central growing mode, use the r_m shell
    P_(f"mu={mu}: LTB t_C(0)={tC0/t_H:.3f} t_H, t_C(r_m)={L.tC(rm)/t_H:.2f} t_H; run to {args.tend} t_H")
    met0 = solve_metric(grid, bg, part, bg.t_i)
    R0 = met0.a * part.x
    # shell labels: conserved rest mass M_i = sum_{j<i} N_j + N_i/2 (sorted); LTB invariants from M_rest(r) = int dm/sqrt(1+2E)
    order = np.argsort(x); Mi = np.empty(len(x)); cs = np.cumsum(N[order]); Mi[order] = cs - 0.5 * N[order]
    rL = np.linspace(1e-6, x_out * 1.05, 200001); mL_all = L.m(rL); EL_all = L.E(rL)
    MrestL = np.concatenate([[0], np.cumsum(np.diff(mL_all) / np.sqrt(1 + 2 * 0.5 * (EL_all[1:] + EL_all[:-1])))])
    r_lab = np.interp(Mi, MrestL, rL); m0 = L.m(r_lab); E0 = L.E(r_lab)
    # initial proper time since the bang of each shell from its areal radius on the CMC slice (expanding branch)
    tau_off = LTBGaussian.shell_tau(m0, E0, R0, np.ones_like(R0))
    P_(f"t_i solve: max|alpha-1|={np.max(np.abs(met0.alpha_c-1)):.2e} max|A-1|={np.max(np.abs(met0.A_edge-1)):.2e} m_out/flrw-1={met0.m_edge[-1]/(0.5*25*x_out**3)-1:+.2e} "
       f"median tau0/t_i-1={np.median(tau_off)/bg.t_i-1:+.2e}  p_m/m_LTB-1 median={np.median(met0.p_m/m0-1):+.2e} p95={np.percentile(np.abs(met0.p_m/m0-1),95):.2e}")
    snaps = tuple(np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]) * min(tC0, args.tend * t_H))
    cfg = CosmoRunConfig(cfl=args.cfl, t_end=args.tend * t_H, diag_every=args.diag, snapshot_times=snaps,
                         inject_t=args.tinj * t_H if args.sigma > 0 else 0.0, inject_sigma=args.sigma, inject_xmax=args.xdisp * rm, dR_target=args.dR, release_mode=args.release, release_k_ratio=args.rel_k, release_delta_ent=args.rel_delta, release_h=args.rel_h, release_t_ent=t_H, release_xmax=args.rel_xmax * rm, release_b_nl=args.rel_bnl)
    run = CosmoRun(grid, bg, part, cfg, bg.t_i)
    t0 = time.time(); met = run.run(cb)
    if args.release: P_(f"  release mode {args.release}: k ratio {args.rel_k}, delta/zeta {args.rel_delta}, h {args.rel_h}; shells released {run.n_released}, first release at t/t_H={run.t_first_release/t_H:.2f}")
    if run.injected: P_(f"  dispersion injected at t/t_H={run.injected['t']/t_H:.2f}: sigma={run.injected['sigma']:.1e} on {run.injected['n']} shells (x < {args.xdisp} r_m)")
    P_(f"done: steps={run.step} wall={time.time()-t0:.1f}s; first AH: {run.ah_first}")
    tr = ~np.isnan(run.tau_trap)
    if tr.any():
        tt = tau_off + run.tau_trap
        tL = np.array([L.t_AH(r) for r in r_lab])
        P_(f"  per-shell trapping (proper time): {tr.sum()} shells trapped; labels r/r_m in [{r_lab[tr].min()/rm:.3f},{r_lab[tr].max()/rm:.3f}]; "
           f"earliest tau_trap={np.nanmin(tt[tr])/t_H:.2f} t_H (label r/r_m={r_lab[tr][np.argmin(tt[tr])]/rm:.3f}); median tau_trap/LTB t_AH = {np.median(tt[tr]/tL[tr]):.4f}, p5-p95 [{np.percentile(tt[tr]/tL[tr],5):.4f},{np.percentile(tt[tr]/tL[tr],95):.4f}]; "
           f"median coordinate lag t_detect - tau_trap = {np.median(run.t_trap[tr]-tt[tr])/t_H:.2f} t_H")
        for lo, hi in ((0,0.1),(0.1,0.3),(0.3,0.5),(0.5,0.8),(0.8,1.2)):
            sel = tr & (r_lab >= lo*rm) & (r_lab < hi*rm)
            if sel.any(): P_(f"      labels [{lo},{hi}) r_m: n={sel.sum()}, tau_trap/t_AH^LTB median={np.median(tt[sel]/tL[sel]):.4f}, LTB t_AH range [{tL[sel].min()/t_H:.2f},{tL[sel].max()/t_H:.2f}] t_H")
    if run.ah_first is not None:
        sA = run.ah_snapshot; xA = run.ah_first["x_AH"]
        near = np.argsort(np.abs(sA["R"] / sA["a"] - xA))[:8]
        tauA = (tau_off + sA["tau"])[near]
        lab_AH = np.median(r_lab[near]); inside = r_lab < lab_AH; tauI = tau_off + sA["tau"]
        P_(f"  shells inside the first AH (labels < {lab_AH/rm:.3f} r_m): n={inside.sum()}, proper time median={np.median(tauI[inside])/t_H:.2f} t_H, min={tauI[inside].min()/t_H:.2f}, max={tauI[inside].max()/t_H:.2f}; "
           f"fraction re-expanded beyond x_AH: {np.mean(sA['R'][inside]/sA['a'] > xA):.3f}; LTB caustic t_C(0)={L.tC(1e-6)/t_H:.2f}")
        P_(f"  AH invariant check: coordinate t={run.ah_first['t']/t_H:.2f} t_H, M_AH={run.ah_first['M_AH']:.2f} (= {run.ah_first['M_AH']/(0.75*t_H):.3f} M_H(t_k)); shells at the AH: "
           f"label r/r_m={np.median(r_lab[near])/rm:.3f}, LTB m={np.median(m0[near]):.2f}, shell proper time={np.median(tauA)/t_H:.2f} t_H vs LTB t_AH(shell)={np.median([L.t_AH(r) for r in r_lab[near]])/t_H:.2f} t_H, LTB t_C(shell)={np.median([L.tC(r) for r in r_lab[near]])/t_H:.2f}")
    # LTB comparison at snapshots: predicted R(tau) from the shell invariants (m0, E0) fixed at t_i
    inner = (x < 1.5 * rm) & (E0 < -1e-10) & (m0 > 0) & (r_lab > 0.02)
    rows = []
    for s in run.snapshots:
        tau = (tau_off + s["tau"])[inner]
        Rpred = LTBGaussian.shell_R_of_tau(m0[inner], E0[inner], tau)
        err = np.abs(s["R"][inner] / Rpred - 1)
        mdrift = np.abs(s["m"][inner] / m0[inner] - 1)
        inner_ = np.ones(inner.sum(), bool)
        rows.append((s["t"] / t_H, s["t"] / tC0, np.median(err), np.percentile(err, 95), np.median(mdrift), np.percentile(mdrift, 95)))
        P_(f"  snapshot t/t_H={rows[-1][0]:7.3f} (t/t_C(0)={rows[-1][1]:.2f}): |R/R_LTB-1| median={rows[-1][2]:.2e} p95={rows[-1][3]:.2e}; enclosed-mass drift median={rows[-1][4]:.2e} p95={rows[-1][5]:.2e} (shells x<1.5 r_m)")
    s = run.snapshots[-1]; tau = (tau_off + s["tau"])[inner]
    Rpred = LTBGaussian.shell_R_of_tau(m0[inner], E0[inner], tau); err = s["R"][inner] / Rpred - 1; xi = x[inner]
    P_("  last snapshot, error vs radius (bins in x/r_m): median |R/R_LTB-1|, p95, median m-drift")
    for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.5)):
        sel = (xi >= lo * rm) & (xi < hi * rm)
        if sel.any():
            P_(f"    [{lo:.2f},{hi:.2f}) n={sel.sum():5d}: {np.median(np.abs(err[sel])):.2e}  {np.percentile(np.abs(err[sel]),95):.2e}  {np.median(np.abs(s['m'][inner][sel]/m0[inner][sel]-1)):.2e}  E0 in [{E0[inner][sel].min():.3f},{E0[inner][sel].max():.3f}]")
    if hasattr(run, "ueq_profile"):
        up = np.abs(run.ueq_profile); xe = grid.x_edge
        P_("  unused-equation residual profile (|res|/scale) by x/r_m: " + ", ".join(f"[{lo},{hi}):{np.max(up[(xe>=lo*rm)&(xe<hi*rm)]):.1e}" for lo,hi in ((0,0.02),(0.02,0.2),(0.2,1),(1,2),(2,4),(4,5.01))))
    tag = f"{args.test}_mu{mu:.3f}_K{args.K}_n{args.n}" + (f"i{args.n_inner}" if args.n_inner else "") + f"_s{args.stretch:.0f}_sig{args.sigma:.0e}" + (f"_dR{args.dR:g}" if args.dR > 0 else "") + (f"_rel{args.release}_k{args.rel_k:g}_d{args.rel_delta:g}" + (f"_b{args.rel_bnl:g}" if args.rel_bnl > 0 else "") if args.release else "") + (f"_{args.profile}" if args.profile != "yoo" else "")
    json.dump({"diag": run.diag, "ltb_rows": rows, "ah_first": run.ah_first}, open(out / f"{tag}.json", "w"), default=float)
    np.savez(out / f"{tag}_shells.npz", x0=x, N=N, r_lab=r_lab, m0=m0, E0=E0, tau_off=tau_off, tau_trap=run.tau_trap, t_trap=run.t_trap,
             tau_end=part.tau, x_end=part.x, P_end=part.P, t_H=t_H, mu=mu)

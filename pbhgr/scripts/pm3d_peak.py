"""Item 3b: Newtonian 3D PM run of one triaxial peak.  Usage:
  python3 scripts/pm3d_peak.py --mu 0.1 --e 0 --p 0 --N 128 --Np 64 --L 8 --tend 40 [--tail 0.05 --k1 2 --k2 10] [--profile yoo] [--out runs/pm3d]"""
import argparse, json, sys, time, numpy as np
sys.path.insert(0, '.')
from pbhgr.pm3d import PM3D, analyse, H_I, T_I
from pbhgr.profiles import profile_table
from pbhgr.ltb import LTBGaussian, LTBProfile
ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, default=0.1); ap.add_argument("--e", type=float, default=0.0); ap.add_argument("--p", type=float, default=0.0)
ap.add_argument("--N", type=int, default=128); ap.add_argument("--Np", type=int, default=64); ap.add_argument("--L", type=float, default=8.0)
ap.add_argument("--tend", type=float, default=40.0); ap.add_argument("--tail", type=float, default=0.0); ap.add_argument("--k1", type=float, default=2.0); ap.add_argument("--k2", type=float, default=10.0)
ap.add_argument("--profile", default="yoo"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--da_frac", type=float, default=0.02); ap.add_argument("--out", default="runs/pm3d")
args = ap.parse_args()
from pathlib import Path; Path(args.out).mkdir(parents=True, exist_ok=True)
table = profile_table(args.profile); table["name"] = args.profile
L_ref = LTBGaussian(args.mu) if args.profile == "yoo" else LTBProfile(args.mu, table)
rm, t_H = L_ref.r_m, L_ref.t_H
tC0 = L_ref.tC(1e-6)
if not np.isfinite(tC0): tC0 = L_ref.tC(rm)
pm = PM3D(args.N, args.L * rm, args.Np, seed=args.seed).make_ic(args.mu, table["spline"], args.e, args.p, args.tail, (args.k1, args.k2))
tag = f"mu{args.mu:g}_e{args.e:g}_p{args.p:g}_N{args.N}_Np{args.Np}_L{args.L:g}" + (f"_tail{args.tail:g}_k{args.k1:g}-{args.k2:g}_s{args.seed}" if args.tail > 0 else "") + (f"_{args.profile}" if args.profile != "yoo" else "")
print(f"PM3D {tag}: box {args.L} r_m = {args.L*rm:.2f}/k, dx = {pm.dx:.4f}/k ({pm.dx/rm:.4f} r_m), {args.Np**3} particles, m_part = {pm.m_part:.3e}; axes A = {np.round(pm.A,3)}; "
      f"t_C(0) = {tC0/t_H:.2f} t_H, run to {args.tend} t_H; central delta_lin(a=1) = {pm.delta_lin[args.N//2, args.N//2, args.N//2]:.4f} (expected 0.4 mu/25 = {0.4*args.mu/25:.4f}); tail rms on grid = {pm.tail_rms_grid:.4f}", flush=True)
a = 1.0; t = T_I; step = 0; rows = []; t0 = time.time(); next_diag = 0.0
a_end = (args.tend * t_H / T_I) ** (2.0 / 3.0)
while a < a_end:
    da = min(args.da_frac * a, a_end - a)
    a, delta = pm.step(a, da); t = T_I * a**1.5; step += 1
    if t / t_H >= next_diag or a >= a_end:
        d = analyse(pm, a); d.update(t=t, t_over_tH=t / t_H, t_over_tC0=t / tC0, step=step, delta_max=float(delta.max()))
        rows.append(d)
        print(f"step {step:5d} t/t_H={t/t_H:7.3f} (t/tC0={t/tC0:.3f}) a={a:8.2f} delta_max={delta.max():8.1f} | comoving axes(0.1/0.2/0.3 r_m)={np.round(d['axes_com_0.1'],3)} {np.round(d['axes_com_0.2'],3)} {np.round(d['axes_com_0.3'],3)} hoop(0.2)={d['hoop_0.2']:.4f} | m(<1,2,4,8 cells)={d['m_in_1c']:.2f} {d['m_in_2c']:.2f} {d['m_in_4c']:.2f} {d['m_in_8c']:.2f} C(<2c)={d['C_2c']:.4f} | {time.time()-t0:.0f}s", flush=True)
        next_diag = t / t_H + max(0.5, 0.02 * tC0 / t_H)
json.dump(dict(args=vars(args), rows=rows, rm=rm, t_H=t_H, tC0=tC0, dx=pm.dx, m_part=pm.m_part), open(f"{args.out}/{tag}.json", "w"))
print("done", f"{time.time()-t0:.0f}s")

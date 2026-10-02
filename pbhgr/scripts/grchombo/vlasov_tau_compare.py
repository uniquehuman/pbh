#!/usr/bin/env python3
"""Invariant comparison of the 3D particle run with the 1D Einstein-Vlasov references and LTB:
horizon mass M_AH versus the proper time of the matter at the horizon.

3D: pbh_ah_rays.dat (ray horizon: <R_AH>, M = R/2) + pbh_ah_tau.dat (mass-weighted tau of the particles within
    |r - r_AH| < max(dx_f, 0.05 r_AH)); 1D: *_shells.npz (tau_trap, t_trap per label) + json diag AH history
    (M_AH at the coordinate time the label is trapped); LTB: t_AH(r) of the shell with m(r) = M (no shell crossing).
Usage: vlasov_tau_compare.py RUN3D_DIR [--ref1d runs/v1/ref3d] [--mu 0.3] [--M 0.1 0.15 ...]"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pbhgr.ltb import LTBGaussian  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run3d")
ap.add_argument("--ref1d", default="runs/v1/ref3d")
ap.add_argument("--mu", type=float, default=0.3)
ap.add_argument("--M", type=float, nargs="*", default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
a = ap.parse_args()

setup = json.load(open(os.path.join(a.run3d, "setup.json")))
MH, tC0, t0 = setup["MH"], setup["tC0"], setup["t0"]
rays = np.loadtxt(os.path.join(a.run3d, "pbh_ah_rays.dat"))
tau = np.loadtxt(os.path.join(a.run3d, "pbh_ah_tau.dat"))
ok = rays[:, 1] >= 3
t3 = (rays[ok, 0] + t0) / tC0
M3 = rays[ok, 4] / MH
tau3 = np.interp(rays[ok, 0], tau[:, 0], tau[:, 3]) / tC0
tauc3 = np.interp(rays[ok, 0], tau[:, 0], tau[:, 11]) / tC0
print(f"3D run {a.run3d}: {ok.sum()} horizon rows, far time {t3.min():.4f}-{t3.max():.4f}, M_AH {M3.min():.3f}-{M3.max():.3f} M_H")
print(f"   first detection: t_far {t3[0]:.4f}, M {M3[0]:.3f}, r_AH {rays[ok, 2][0]:.2f} = {rays[ok, 2][0] / (setup['dx'] / 2 ** 6):.1f} finest cells, tau_AH {tau3[0]:.4f}, tau_core {tauc3[0]:.4f}")

L = LTBGaussian(a.mu)
def ltb_tau(M):
    r = brentq(lambda x: L.m(x) - M * MH, 1e-6, 10.0)
    return L.t_AH(r) / tC0

ref = {}
for tag, pat in [("1D cold", "sig0e+00"), ("1D sig", "sig[1-9]e-0[0-9]")]:
    fz = sorted(glob.glob(os.path.join(a.ref1d, f"ltb_mu{a.mu:.3f}_*_{pat}_*_shells.npz")))
    if not fz:
        continue
    z = np.load(fz[0]); d = json.load(open(fz[0].replace("_shells.npz", ".json")))
    ah = [(e["t"], e["AH"]["M_AH"]) for e in d["diag"] if e.get("AH")]
    t_ah = np.array([x[0] for x in ah]) / tC0; M_ah = np.array([x[1] for x in ah]) / MH
    sel = np.isfinite(z["tau_trap"]) & (z["tau_trap"] > 0) & (z["t_trap"] > 0)
    Mtrap = np.interp(z["t_trap"][sel] / tC0, t_ah, M_ah)
    o = np.argsort(Mtrap)
    ref[tag] = (Mtrap[o], z["tau_trap"][sel][o] / tC0, os.path.basename(fz[0]))
    print(f"   {tag}: {ref[tag][2]} (first AH at coordinate {t_ah.min():.4f} t_C(0), {M_ah.min():.4f} M_H)")

print("\nM_AH/M_H | 3D far time | 3D tau_AH | 1D cold tau | 3D-1D | LTB t_AH(m=M) | 1D cold - LTB" + (" | 1D sig tau" if "1D sig" in ref else ""))
for M in a.M:
    if M < M3.min() or M > M3.max():
        continue
    tf = np.interp(M, M3, t3); ta = np.interp(M, M3, tau3)
    row = f"{M:5.3f}    | {tf:.4f}      | {ta:.4f}    |"
    if "1D cold" in ref:
        Mc, tc, _ = ref["1D cold"]; t1 = np.interp(M, Mc, tc)
        row += f" {t1:.4f}      | {ta - t1:+.4f} | {ltb_tau(M):.4f}        | {t1 - ltb_tau(M):+.4f}"
    else:
        row += f" -           | -       | {ltb_tau(M):.4f}        | -"
    if "1D sig" in ref:
        Ms, ts, _ = ref["1D sig"]; row += f" | {np.interp(M, Ms, ts):.4f}"
    print(row)

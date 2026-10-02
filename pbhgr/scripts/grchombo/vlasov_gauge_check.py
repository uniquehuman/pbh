#!/usr/bin/env python3
"""Gauge check of the particle push: central density versus the proper time of the core, compared with LTB.
Input: run directories with extraction_data/rho_line_extraction_*.dat and pbh_ah_tau.dat (column 12 = tau_core).
Usage: vlasov_gauge_check.py RUN_DIR [RUN_DIR ...] [--mu 0.3] [--t0 1079.891010] [--dt 24.19691913]"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pbhgr.ltb import LTBGaussian  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--mu", type=float, default=0.3)
ap.add_argument("--t0", type=float, default=1079.891010)
ap.add_argument("--dt", type=float, default=24.19691913)
ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()
L = LTBGaussian(a.mu)
tC0 = 0.589 * np.exp(3 * a.mu) * a.mu ** -1.5 * L.t_H  # same formula as the generator

def rho_ltb(t, r=2e-3, h=1e-4):
    dR = (L.R(t, r + h) - L.R(t, r - h)) / (2 * h)
    return L.mp(r) / (4 * np.pi * L.R(t, r) ** 2 * dR)

cols = {}
for run in a.runs:
    tau = np.loadtxt(os.path.join(run, "pbh_ah_tau.dat"))
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "extraction_data", "rho_line*_*.dat"))):
        s = int(f.split("_")[-1].split(".")[0])
        rho_c = float(open(f).read().split("\n")[1].split()[-1])
        tf = s * a.dt
        tc = np.interp(tf, tau[:, 0], tau[:, 11])
        rows.append((s, (tf + a.t0) / tC0, tc / tC0, rho_c))
    cols[run] = rows
steps = sorted(set(r[0] for rows in cols.values() for r in rows))
print("step | " + " | ".join(f"{os.path.basename(r.rstrip('/'))}: t_far tau_core rho_c/rho_LTB(tau_core)" for r in a.runs))
for s in steps:
    if s % a.every:
        continue
    line = f"{s:4d} |"
    for run in a.runs:
        m = [r for r in cols[run] if r[0] == s]
        if not m:
            line += "   -   "; continue
        _, tf, tc, rc = m[0]
        rl = rho_ltb(tc * tC0) if tc < 0.995 else np.nan
        line += f" {tf:.4f} {tc:.4f} {rc / rl if rl == rl else float('nan'):.4f} |"
    print(line)
for run in a.runs:
    rows = cols[run]; k = int(np.argmax([r[3] for r in rows]))
    print(f"{os.path.basename(run.rstrip('/'))}: rho_c maximum at step {rows[k][0]}, t_far {rows[k][1]:.4f}, tau_core {rows[k][2]:.4f} t_C(0)")

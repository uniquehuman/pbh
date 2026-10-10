#!/usr/bin/env python3
"""Invariant check of the 3D particle run against LTB before the caustic, shell by shell.
For each radial bin of tau_profiles/tau_<step>.dat (cumulative rest mass M_in, <tau>) the LTB shell with the same rest
mass is found (M_rest(r) = int dm / sqrt(1 + 2E)); its areal radius R(tau) and Misner-Sharp mass m(r) are compared with
the 3D values from theta_profiles/theta_<step>.dat (ray 0: R(r), M_MS(r)) interpolated at the bin centre.
Usage: vlasov_shell_check.py RUN_DIR --steps 20 40 [--mu 0.3] [--bins 2 4 8 16 32]"""
import argparse, os, sys
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pbhgr.ltb import LTBGaussian  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--steps", type=int, nargs="+", required=True)
ap.add_argument("--mu", type=float, default=0.3)
ap.add_argument("--bins", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
a = ap.parse_args()
L = LTBGaussian(a.mu)
tC0 = 0.589 * np.exp(3 * a.mu) * a.mu ** -1.5 * L.t_H
# LTB rest mass as a function of the label
rl = np.linspace(1e-5, 6.0, 60001)
Mrest = cumulative_trapezoid(L.mp(rl) / np.sqrt(1 + 2 * L.E(rl)), rl, initial=0.0)
print(f"mu = {a.mu}: t_C(0) = {tC0:.3f}; LTB rest mass inside r_m = {np.interp(L.r_m, rl, Mrest):.2f}, m(r_m) = {L.m(L.r_m):.2f}")
for step in a.steps:
    tp = os.path.join(a.run, "tau_profiles", f"tau_{step}.dat"); th = os.path.join(a.run, "theta_profiles", f"theta_{step}.dat")
    if not (os.path.exists(tp) and os.path.exists(th)):
        print(f"step {step}: profiles missing"); continue
    T = np.loadtxt(tp); time = float(open(tp).readline().split()[-1])
    H = np.loadtxt(th); H = H[H[:, 0] == 0]  # ray +x: r R Theta M_MS
    print(f"\nstep {step}: t_far = {(time + 0.5 * tC0) / tC0:.4f} t_C(0)" if False else f"\nstep {step}: run time {time:.1f}")
    print("  bin    r_hi    M_in(rest)  <tau>/tC0  label r/r_m   R_3D     R_LTB    R ratio   M_MS,3D   m_LTB   M ratio")
    for k in a.bins:
        if k >= len(T): continue
        r_mid = T[k, 2]; Min = T[k, 5]; tau = T[k, 6]  # cumulative mass is inside the outer edge r_hi
        if Min <= 0 or tau <= 0: continue
        r_lab = np.interp(Min, Mrest, rl)
        R3 = np.interp(r_mid, H[:, 1], H[:, 2]); M3 = np.interp(r_mid, H[:, 1], H[:, 4])
        try:
            RL = L.R(tau, r_lab); mL = L.m(r_lab)
        except Exception:
            RL, mL = np.nan, np.nan
        print(f"  {k:3d}  {r_mid:8.2f}  {Min:9.4f}  {tau / tC0:.4f}     {r_lab / L.r_m:.4f}   {R3:8.3f} {RL:8.3f}  {R3 / RL:.4f}    {M3:8.4f} {mL:8.4f}  {M3 / mL:.4f}")

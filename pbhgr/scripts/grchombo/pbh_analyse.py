#!/usr/bin/env python3
"""Analyse a PBHCosmo (GRChombo) run directory: data/data_out.dat, data/*_lineout.dat, AH stats.
Compares the far field with FLRW (K = -2/t), the central density with the LTB prediction (until collapse) and
reports the first apparent horizon (time in t_C(0) units, mass in M_H = 0.75 t_H).  usage: pbh_analyse.py RUNDIR"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pbhgr.ltb import LTBGaussian


def load_small(fn):
    rows = [l.split() for l in open(fn) if l.strip() and not l.startswith("#")]
    return np.array([[float(v) for v in r] for r in rows])


def main(run):
    S = json.load(open(os.path.join(run, "setup.json")))
    t0, tC0, tH, MH = S["t0"], S["tC0"], S["tH"], S["MH"]
    ltb = LTBGaussian(S["mu"])
    d = load_small(os.path.join(run, "data", "data_out.dat"))
    t = d[:, 0] + t0
    print(f"mu = {S['mu']}, t_0 = {S['f0']} t_C(0), m t_H = {S['mtH']} (q = {S['q']:.0f}, sigma_ent = {S['sigma_ent']:.3f}); "
          f"{len(t)} coarse steps, t = {t[-1]/tC0:.4f} t_C(0)")
    K_flrw = -2.0 / t
    print("far field: K_corner/K_FLRW - 1: max |.| = %.2e ; lapse_corner - 1: max |.| = %.2e" %
          (np.abs(d[:, 10] / K_flrw - 1).max(), np.abs(d[:, 11] - 1).max()))
    # central density from the rho lineout (mid point) vs LTB central shell rho_c(t) = rho_c(t0) (R0/R)^3
    fn = os.path.join(run, "data", "rho_lineout.dat")
    if os.path.exists(fn):
        r = load_small(fn); n = (r.shape[1] - 1); ic = 1 + n // 2
        rho_c = r[:, ic]; tt = r[:, 0] + t0
        rr = 1e-4 * ltb.r_m
        R0 = ltb.R(t0, rr); rho0 = rho_c[0]
        pred = np.array([rho0 * (R0 / ltb.R(x, rr)) ** 3 if x < 0.999 * tC0 else np.nan for x in tt])
        sel = ~np.isnan(pred)
        print("centre: rho_c(t)/rho_LTB(t): " + ", ".join(f"{x:.3f}" for x in (rho_c[sel] / pred[sel])[:: max(1, sel.sum() // 8)]))
    print("lapse_min: %.4f -> %.4f ; chi_min: %.4f -> %.4g ; L2 Ham: %.2e -> %.2e" %
          (d[0, 6], d[-1, 6], d[0, 5], d[-1, 5], d[0, 1], d[-1, 1]))
    for fn in sorted(glob.glob(os.path.join(run, "data", "stats_AH*.dat"))):
        rows = [l for l in open(fn) if l.strip()]
        hdr = [l for l in rows if l.startswith("#")]
        vals = load_small(fn)
        print(f"AH file {os.path.basename(fn)}: {len(vals)} rows; header: {hdr[-1].strip()[:150] if hdr else '-'}")
        if len(vals):
            ok = ~np.isnan(vals[:, 1])
            if ok.any():
                i = np.argmax(ok); tAH = vals[i, 0] + t0
                print(f"  first converged AH at t = {tAH:.2f} = {tAH/tC0:.4f} t_C(0) = {tAH/tH:.3f} t_H; row: " +
                      " ".join(f"{v:.5g}" for v in vals[i, :6]))
                print(f"  last row: t = {(vals[-1,0]+t0)/tC0:.4f} t_C(0): " + " ".join(f"{v:.5g}" for v in vals[-1, :6]))
                # horizon growth: time at which M_AH (column 2 = mass) reaches given fractions of M_H
                tt = vals[ok, 0] + t0; mm = vals[ok, 2] / MH
                for thr in (0.03, 0.05, 0.1, 0.2, 0.3, 0.5):
                    if mm.max() >= thr:
                        j = np.argmax(mm >= thr)
                        print(f"  M_AH = {thr} M_H reached at t = {tt[j]/tC0:.4f} t_C(0)")
                refs = {0.1: "cold M_AH = 0.1/0.3 M_H at 1.428/1.922 t_C(0); sigma_ent = 0.054: 1.548/1.959; first tiny AH 1.120/1.212",
                        0.3: "cold M_AH = 0.1/0.3 M_H at 1.691/2.034 t_C(0); sigma_ent = 0.027: 1.669/2.078; first tiny AH 1.352/1.451"}
                print("  1D references (runs/v1/ref3d): " + refs.get(round(S["mu"], 2), "none tabulated for this mu"))


if __name__ == "__main__":
    main(sys.argv[1])

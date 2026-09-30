#!/usr/bin/env python3
"""Compare a PBHVlasov (GRTeclyn) LTB e = 0 particle run in geodesic slicing with the analytic LTB solution.
Geodesic slicing + zero shift keeps the coordinate ϱ attached to the LTB label r through the t_0 map, so
rho_LTB(r(ϱ), t), K_LTB, chi_LTB = [gamma_ϱϱ (R/ϱ)^4]^(-1/3) are exact references at any later time.
usage: vlasov_ltb_compare.py OUTDIR (extraction_data files + pbh_vlasov_out.dat + setup.json)"""
import glob, json, os, re, sys
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pbhgr.ltb import LTBGaussian

out = sys.argv[1]
S = json.load(open(os.path.join(out, "setup.json")))
mu, t0, tC0, L, N = S["mu"], S["t0"], S["tC0"], S["L"], S["N"]
ltb = LTBGaussian(mu); rm = ltb.r_m; a0 = (t0 / ltb.t_i) ** (2.0 / 3.0)

# t_0 isotropic map ϱ(r) exactly as in the generator
r = np.linspace(1e-6, 8 * rm, 20000)
R0 = np.array([ltb.R(t0, x) for x in r]); E = ltb.E(r)
Rp0 = CubicSpline(r, R0)(r, 1)
f = Rp0 / (R0 * np.sqrt(1 + 2 * E)) - 1.0 / r
tail = cumulative_trapezoid(f[::-1], r[::-1], initial=0.0)[::-1]
varrho = a0 * r * np.exp(tail)
r_of_rho = CubicSpline(varrho, r)
drho_dr = CubicSpline(r, varrho)(r, 1)

def ltb_profiles(t, rho_pts):
    rr = r_of_rho(rho_pts)
    R = np.array([ltb.R(t, x) for x in rr]); Rd = np.array([ltb.Rdot(t, x) for x in rr])
    # derivatives w.r.t. r from splines on a local fine grid
    rf = np.linspace(rr.min() * 0.9 + 1e-7, rr.max() * 1.1, 4000)
    Rf = np.array([ltb.R(t, x) for x in rf]); Rdf = np.array([ltb.Rdot(t, x) for x in rf])
    Rp = CubicSpline(rf, Rf)(rr, 1); Rdp = CubicSpline(rf, Rdf)(rr, 1)
    Ef = ltb.E(rr); mp = ltb.mp(rr)
    rho = mp / (4 * np.pi * R**2 * Rp)
    K = -(Rdp / Rp + 2 * Rd / R)
    drhodr = np.interp(rr, r, drho_dr)
    g_rr = Rp**2 / (1 + 2 * Ef) / drhodr**2
    chi = (g_rr * (R / rho_pts) ** 4) ** (-1.0 / 3.0)
    return rho, K, chi

def read_line(fn):
    rows = []
    for l in open(fn):
        if l.startswith("#") or not l.strip():
            continue
        # fixed-width coordinates (12 chars each, no separator) followed by the value
        rows.append([float(l[0:12]), float(l[12:24]), float(l[24:36]), float(l[36:].split()[0])])
    return np.array(rows)

d = np.loadtxt(os.path.join(out, "pbh_vlasov_out.dat"))
files = sorted(glob.glob(os.path.join(out, "rho_line_*.dat")))
print(f"mu = {mu}, t_0 = {t0/tC0:.2f} t_C(0), N = {N}, dx = {L/N:.1f}; {len(files)} line files, {len(d)} steps")
picks = [0, len(files) // 4, len(files) // 2, 3 * len(files) // 4, len(files) - 1]
cells = [2, 3, 4, 6, 8, 12, 16, 24]
for k in picks:
    t = d[k, 0] + t0
    rho3 = read_line(files[k]); x = rho3[:, 0] - L / 2; rho3 = rho3[:, 3]
    chi3 = read_line(files[k].replace("rho_line", "chi_line"))[:, 3]
    K3 = read_line(files[k].replace("rho_line", "K_line"))[:, 3]
    dx = L / N
    idx = [int(np.argmin(np.abs(x - c * dx))) for c in cells]
    pts = x[idx]
    rl, Kl, cl = ltb_profiles(t, pts)
    print(f"t = {t/tC0:.3f} t_C(0):  cells " + " ".join(f"{c:>6d}" for c in cells))
    print("   rho3D/rhoLTB  " + " ".join(f"{rho3[i]/rl[j]:6.3f}" for j, i in enumerate(idx)))
    print("   K3D/KLTB      " + " ".join(f"{K3[i]/Kl[j]:6.3f}" for j, i in enumerate(idx)))
    print("   chi3D/chiLTB  " + " ".join(f"{chi3[i]/cl[j]:6.3f}" for j, i in enumerate(idx)))
    print(f"   rho(0)/rho_far: 3D {rho3[0]/rho3[-5]:.2f}, LTB(2 cells) {rl[0]/ltb_profiles(t, np.array([x[-5]]))[0][0]:.2f}")

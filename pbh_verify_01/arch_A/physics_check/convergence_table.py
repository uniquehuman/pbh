"""Measured convergence orders quoted in REPORT.md.  Run: PYTHONPATH=src python3 physics_check/convergence_table.py"""
import os, sys, time, resource
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from conftest import manufactured_slice, ekg_slice, isotropic_slice, pg_slice, zeros_matter  # noqa
from sphdiag import diagnose

Ns = [100, 200, 400, 800]


def orders(e):
    e = np.asarray(e)
    return " ".join(f"{x:5.2f}" for x in np.log2(e[:-1] / e[1:]))


def row(label, e):
    print(f"| {label:<42} | {e[0]:.2e} | {e[-1]:.2e} | {orders(e)} |")


print("| quantity | err N=100 | err N=800 | orders (100->200->400->800) |")
print("|---|---|---|---|")
for c in ("vertex", "cell", "none"):
    res = {}
    for N in Ns:
        s, p, d = manufactured_slice(c, N)
        D = diagnose(s, p)
        for nm in ("ham", "mom", "hamM"):
            for reg in ("bulk", "center", "outer"):
                res.setdefault(f"{nm} Linf {reg}", []).append(D.norms[nm][reg]["Linf"])
        res.setdefault("M_MS max err", []).append(np.max(np.abs(D.M - d["M"])))
        res.setdefault("M_int - M_MS max", []).append(np.max(np.abs(D.M_int - D.M)))
        res.setdefault("R' max err", []).append(np.max(np.abs(D.dR - d["dR"])))
    for k, v in res.items():
        row(f"manufactured/{c}: {k}", v)
for c in ("vertex", "cell"):
    res = {}
    for N in Ns:
        s, f, d = ekg_slice(c, N)
        D = diagnose(s, f)
        res.setdefault("ham Linf bulk", []).append(D.norms["ham"]["bulk"]["Linf"])
        res.setdefault("mom Linf all", []).append(D.norms["mom"]["all"]["Linf"])
        res.setdefault("hamM Linf bulk", []).append(D.norms["hamM"]["bulk"]["Linf"])
        res.setdefault("M_int - M_MS max", []).append(np.max(np.abs(D.M_int - D.M)))
    for k, v in res.items():
        row(f"EKG+flux/{c}: {k}", v)
e = []
for N in Ns:
    D = diagnose(isotropic_slice(np.linspace(0.3, 5.0, N)))
    e.append(np.max(np.abs(D.M - 1)))
row("Schwarzschild isotropic: max|M-1|", e)
e = []
for N in Ns:
    D = diagnose(pg_slice(np.linspace(0.5, 8.0, N)), zeros_matter(N))
    e.append(D.norms["mom"]["all"]["Linf"])
row("Painleve-Gullstrand: mom Linf", e)
t = time.time()
s, f, d = ekg_slice("vertex", 100000)
D = diagnose(s, f)
print(f"\nN=1e5 EKG diagnose: {time.time() - t:.2f} s (incl. ODE data), status {D.status['constraints']}, "
      f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")

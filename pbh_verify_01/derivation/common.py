"""Shared helpers for the PBH-VERIFY-01 physics derivation (sympy only).

Nothing here encodes a formula from the contract: only generic differential
geometry (Christoffel symbols, Riemann/Ricci/Einstein, pull-backs)."""
import sympy as sp

RESULTS = []


def check(label, expr, simp=sp.simplify):
    """Record PASS if expr simplifies to 0 (expr may be a Matrix/list)."""
    if isinstance(expr, (list, tuple)):
        vals = [simp(e) for e in expr]
        ok = all(v == 0 for v in vals)
        rem = vals
    elif isinstance(expr, sp.MatrixBase):
        vals = expr.applyfunc(simp)
        ok = all(v == 0 for v in vals)
        rem = vals
    else:
        rem = simp(expr)
        ok = rem == 0
    tag = "PASS" if ok else "FAIL"
    RESULTS.append((label, tag))
    print(f"[{tag}] {label}" + ("" if ok else f"\n        remainder: {rem}"), flush=True)
    return ok


def info(label, expr):
    print(f"   {label}: {expr}", flush=True)


def christoffel(g, ginv, X):
    n = len(X)
    dg = [[[sp.diff(g[a, b], X[c]) for c in range(n)] for b in range(n)] for a in range(n)]
    Gam = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(b, n):
                s = 0
                for d in range(n):
                    if ginv[a, d] != 0:
                        s += ginv[a, d] * (dg[d][b][c] + dg[d][c][b] - dg[b][c][d])
                s = sp.simplify(s / 2)
                Gam[a][b][c] = s
                Gam[a][c][b] = s
    return Gam


def ricci(Gam, X):
    """R_bd = d_a Gam^a_bd - d_d Gam^a_ba + Gam^a_ae Gam^e_bd - Gam^a_de Gam^e_ba"""
    n = len(X)
    Ric = sp.zeros(n, n)
    for b in range(n):
        for d in range(b, n):
            s = 0
            for a in range(n):
                s += sp.diff(Gam[a][b][d], X[a]) - sp.diff(Gam[a][b][a], X[d])
                for e in range(n):
                    s += Gam[a][a][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][a]
            s = sp.simplify(s)
            Ric[b, d] = s
            Ric[d, b] = s
    return Ric


def pullback(g_old, Xold_of_new, Xnew):
    """g'_{mn} = (dX^a/dx'^m)(dX^b/dx'^n) g_ab, with g_old already expressed in new coords."""
    J = sp.Matrix([[sp.diff(Xa, xn) for xn in Xnew] for Xa in Xold_of_new])
    return (J.T * g_old * J).applyfunc(sp.simplify)


def summary():
    npass = sum(1 for _, t in RESULTS if t == "PASS")
    nfail = len(RESULTS) - npass
    print(f"\n=== {npass} PASS, {nfail} FAIL ===")
    for l, t in RESULTS:
        if t != "PASS":
            print(f"   FAIL: {l}")
    return nfail

"""RS1/RS2: symbolic first-principles checks (independent of sphdiag and of the contract formulas).

Everything is computed from the 4-metric
    ds^2 = -alpha^2 dt^2 + A^2 (dr + beta dt)^2 + R^2 dOmega^2
with generic tensor code (Christoffel symbols, covariant derivatives), and then compared with
the closed forms used by the independent tests (verification_plan.md section 2).

Run:  PYTHONPATH=/root/PBH/pbh_verify_01/.pydeps python3 derive_sympy.py
Returns a dict of check-name -> bool (also printed).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYDEPS = os.path.join(os.path.dirname(_HERE), ".pydeps")
if _PYDEPS not in sys.path:
    sys.path.append(_PYDEPS)

import sympy as sp  # noqa: E402

t, r, th, ph = sp.symbols("t r theta phi", real=True)
X4 = [t, r, th, ph]
X3 = [r, th, ph]


def christoffel(g, X):
    n = len(X)
    gi = sp.simplify(g.inv())
    G = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                G[a][b][c] = sp.simplify(sum(gi[a, d] * (sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b])
                                                          - sp.diff(g[b, c], X[d])) for d in range(n)) / 2)
    return G, gi


def ricci_scalar(g, X):
    n = len(X)
    G, gi = christoffel(g, X)
    Ric = sp.zeros(n, n)
    for b in range(n):
        for c in range(n):
            e = 0
            for a in range(n):
                e += sp.diff(G[a][b][c], X[a]) - sp.diff(G[a][b][a], X[c])
                for d in range(n):
                    e += G[a][a][d] * G[d][b][c] - G[a][c][d] * G[d][b][a]
            Ric[b, c] = e
    return sp.simplify(sum(gi[i, j] * Ric[i, j] for i in range(n) for j in range(n)))


def adm_objects(alpha, beta, A, R):
    """Return dict with K_ij (from the ADM definition), mixed KA, KB, 3-Ricci scalar, momentum
    constraint r-component D_j(K^j_r - delta^j_r K), and the 4D quantity -nabla_a n^a."""
    gam = sp.diag(A**2, R**2, R**2 * sp.sin(th)**2)
    G3, gi3 = christoffel(gam, X3)
    beta_up = [beta, 0, 0]
    beta_dn = [sum(gam[i, j] * beta_up[j] for j in range(3)) for i in range(3)]
    # D_i beta_j = d_i beta_j - Gamma^k_ij beta_k
    Db = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            Db[i, j] = sp.diff(beta_dn[j], X3[i]) - sum(G3[k][i][j] * beta_dn[k] for k in range(3))
    K = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            K[i, j] = sp.simplify(-(sp.diff(gam[i, j], t) - Db[i, j] - Db[j, i]) / (2 * alpha))
    Kmix = sp.simplify(gi3 * K)  # K^i_j
    KA = sp.simplify(Kmix[0, 0])
    KB = sp.simplify(Kmix[1, 1])
    trK = sp.simplify(Kmix.trace())
    R3 = ricci_scalar(gam, X3)
    # momentum: D_j P^j_i with P^j_i = K^j_i - delta^j_i K, i = r
    P = Kmix - sp.eye(3) * trK
    i = 0
    div = 0
    for j in range(3):
        div += sp.diff(P[j, i], X3[j])
        for k in range(3):
            div += G3[j][j][k] * P[k, i] - G3[k][j][i] * P[j, k]
    div = sp.simplify(div)
    # 4-metric and -nabla_a n^a
    g4 = sp.zeros(4, 4)
    g4[0, 0] = -alpha**2 + A**2 * beta**2
    g4[0, 1] = g4[1, 0] = A**2 * beta
    g4[1, 1] = A**2
    g4[2, 2] = R**2
    g4[3, 3] = R**2 * sp.sin(th)**2
    sqrtg = alpha * A * R**2 * sp.sin(th)
    n_up = [1 / alpha, -beta / alpha, 0, 0]
    divn = sp.simplify(sum(sp.diff(sqrtg * n_up[a], X4[a]) for a in range(4)) / sqrtg)
    return dict(K=K, KA=KA, KB=KB, trK=trK, R3=R3, momdiv=div, minus_div_n=sp.simplify(-divn), g4=g4,
                n_up=n_up, gam=gam)


def is_zero(expr, subs_trials=40, tol=1e-12, syms=None):
    e = sp.simplify(expr)
    if e == 0:
        return True
    # numerical fallback at random points
    import random
    random.seed(1)
    fs = list(e.free_symbols)
    for _ in range(subs_trials):
        vals = {s: random.uniform(0.3, 2.5) for s in fs}
        v = complex(e.subs(vals).evalf())
        if abs(v) > tol:
            return False
    return True


def rs1_general():
    """RS1: generic spherical ADM metric, arbitrary functions of (t, r)."""
    al = sp.Function("alpha", positive=True)(t, r)
    be = sp.Function("beta", real=True)(t, r)
    A = sp.Function("A", positive=True)(t, r)
    R = sp.Function("R", positive=True)(t, r)
    o = adm_objects(al, be, A, R)
    Rp = sp.diff(R, r)
    Gam = Rp / A
    U_def = (sp.diff(R, t) - be * Rp) / al          # n^a d_a R, n^a = (1, -beta)/alpha
    res = {}
    res["F1_U_equals_minus_R_KB"] = is_zero(U_def + R * o["KB"])
    res["K_trace_equals_minus_div_n"] = is_zero(o["trK"] - o["minus_div_n"])
    R3_contract = 2 * (1 - Gam**2) / R**2 - 4 * sp.diff(Gam, r) / (A * R)
    res["F7_3Ricci"] = is_zero(o["R3"] - R3_contract)
    KA, KB = o["KA"], o["KB"]
    res["F8_K2_minus_KK"] = is_zero((KA + 2 * KB)**2 - (KA**2 + 2 * KB**2) - (4 * KA * KB + 2 * KB**2))
    # momentum: D_j(K^j_r - delta K) = 8 pi j_r  <=>  (-1/2)*div = -4 pi j_r
    Mcal_geom = sp.diff(KB, r) - (Rp / R) * (KA - KB)
    res["F9_momentum_form"] = is_zero(-o["momdiv"] / 2 - Mcal_geom)
    # scalar field stress tensor projections
    phi = sp.Function("phi", real=True)(t, r)
    m = sp.symbols("m", positive=True)
    g4 = o["g4"]
    gi4 = sp.simplify(g4.inv())
    dphi = [sp.diff(phi, x) for x in X4]
    kin = sum(gi4[a, b] * dphi[a] * dphi[b] for a in range(4) for b in range(4))
    V = m**2 * phi**2 / 2
    T = sp.zeros(4, 4)
    for a in range(4):
        for b in range(4):
            T[a, b] = dphi[a] * dphi[b] - g4[a, b] * (kin / 2 + V)
    n_up = o["n_up"]
    Pi = sum(n_up[a] * dphi[a] for a in range(4))
    Phi = sp.diff(phi, r) / A
    rho = sp.simplify(sum(T[a, b] * n_up[a] * n_up[b] for a in range(4) for b in range(4)))
    jr = sp.simplify(-sum(n_up[b] * T[1, b] for b in range(4)))
    Srr = sp.simplify(T[1, 1] / A**2)
    Sthth = sp.simplify(T[2, 2] / R**2)
    res["F3_rho"] = is_zero(rho - (Pi**2 / 2 + Phi**2 / 2 + V))
    res["F4_jr"] = is_zero(jr + Pi * sp.diff(phi, r))
    res["F5_Srr"] = is_zero(Srr - (Pi**2 / 2 + Phi**2 / 2 - V))
    res["F5_Sthth"] = is_zero(Sthth - (Pi**2 / 2 - Phi**2 / 2 - V))
    # Misner-Sharp from 4-metric: M = R/2 (1 - g^{ab} d_a R d_b R)
    dR = [sp.diff(R, x) for x in X4]
    gradR2 = sum(gi4[a, b] * dR[a] * dR[b] for a in range(4) for b in range(4))
    M4 = R / 2 * (1 - gradR2)
    M_slice = R / 2 * (1 - (Rp / A)**2 + R**2 * KB**2)
    res["F11_MS_mass"] = is_zero(M4 - M_slice)
    # F10 identity with independent symbols (algebraic check of the sign)
    Rs, Rps, As, Gps, KAs, KBs, KBps, rhos, jrs = sp.symbols("R Rp A Gp KA KB KBp rho jr")
    Gs = Rps / As
    Ham = 2 * (1 - Gs**2) / Rs**2 - 4 * Gps / (As * Rs) + 4 * KAs * KBs + 2 * KBs**2 - 16 * sp.pi * rhos
    Mom = KBps - (Rps / Rs) * (KAs - KBs) + 4 * sp.pi * jrs
    Us = -Rs * KBs
    # M' by chain rule: M = R/2 (1 - G^2 + U^2); dG/dr = Gp; dU/dr = -Rp KB - R KBp
    Mp = Rps / 2 * (1 - Gs**2 + Us**2) + Rs / 2 * (-2 * Gs * Gps + 2 * Us * (-Rps * KBs - Rs * KBps))
    HamM = Mp - 4 * sp.pi * Rs**2 * (rhos * Rps + Us * jrs)
    res["F10_identity_plus_sign(mine)"] = sp.simplify(HamM - (+Rs**2 * Rps / 4 * Ham + Rs**3 * KBs * Mom)) == 0
    res["F10_identity_minus_sign(contract_1.0-draft)_is_FALSE"] = sp.simplify(
        HamM - (-Rs**2 * Rps / 4 * Ham + Rs**3 * KBs * Mom)) != 0
    # F13
    th_p = 2 / Rs * (Us + Gs)
    th_m = 2 / Rs * (Us - Gs)
    Mms = Rs / 2 * (1 - Gs**2 + Us**2)
    res["F13_product"] = sp.simplify(Rs**2 * th_p * th_m / 4 - (2 * Mms / Rs - 1)) == 0
    return res


def rs2_exact_solutions():
    """RS2: exact slices; K computed from the ADM definition; compare with the closed forms in
    independent_reference/analytic.py and check vacuum/dust constraints and M_MS."""
    M, tau, H0 = sp.symbols("M tau H0", positive=True)
    out = {}

    def check(name, alpha, beta, A, R, KA_closed, KB_closed, M_expected, rho=0, jr=0):
        o = adm_objects(alpha, beta, A, R)
        KA, KB = o["KA"], o["KB"]
        Rp = sp.diff(R, r)
        Gam = Rp / A
        ham = o["R3"] + 4 * KA * KB + 2 * KB**2 - 16 * sp.pi * rho
        mom = sp.diff(KB, r) - Rp / R * (KA - KB) + 4 * sp.pi * jr
        Mms = R / 2 * (1 - Gam**2 + R**2 * KB**2)
        out[name + ":KA"] = is_zero(KA - KA_closed)
        out[name + ":KB"] = is_zero(KB - KB_closed)
        out[name + ":ham=0"] = is_zero(ham)
        out[name + ":mom=0"] = is_zero(mom)
        out[name + ":M_MS"] = is_zero(Mms - M_expected)

    # Painleve-Gullstrand
    check("PG", 1, sp.sqrt(2 * M / r), 1, r,
          -sp.sqrt(2 * M / r) / (2 * r), sp.sqrt(2 * M / r) / r, M)
    # ingoing Eddington-Finkelstein / Kerr-Schild
    Aks = sp.sqrt(1 + 2 * M / r)
    check("KS", 1 / Aks, (2 * M / r) / (1 + 2 * M / r), Aks, r,
          -2 * M * (r + M) / (r**3 * (1 + 2 * M / r)**sp.Rational(3, 2)), 2 * M / (r**2 * Aks), M)
    # isotropic, K = 0
    psi = 1 + M / (2 * r)
    check("ISO", (1 - M / (2 * r)) / psi, 0, psi**2, psi**2 * r, 0, 0, M)
    # Minkowski height function t = T(r); ds^2 = -(dt'+T'dr)^2 + dr^2 + r^2 dOmega^2
    T = sp.Function("T", real=True)(r)
    Tp = sp.diff(T, r)
    N = 1 / sp.sqrt(1 - Tp**2)
    check("HEIGHT", N, -Tp / (1 - Tp**2), 1 / N, r, -sp.diff(N * Tp, r), -N * Tp / r, 0)
    # Milne in chi coordinate: tau is time here
    Ra = t * sp.sinh(r)
    check("MILNE", 1, 0, t, Ra, -1 / t, -1 / t, 0)
    # FLRW dust: a(t) arbitrary, H = a'/a, rho = 3H^2/(8 pi) (Friedmann) => ham=0
    a = sp.Function("a", positive=True)(t)
    Hs = sp.diff(a, t) / a
    check("FLRW", 1, 0, a, a * r, -Hs, -Hs, Hs**2 * (a * r)**3 / 2, rho=3 * Hs**2 / (8 * sp.pi))
    return out


def run_all(verbose=True):
    res = {}
    res.update({"RS1:" + k: bool(v) for k, v in rs1_general().items()})
    res.update({"RS2:" + k: bool(v) for k, v in rs2_exact_solutions().items()})
    if verbose:
        for k, v in res.items():
            print(("OK   " if v else "FAIL ") + k)
    return res


if __name__ == "__main__":
    r_ = run_all()
    sys.exit(0 if all(r_.values()) else 1)

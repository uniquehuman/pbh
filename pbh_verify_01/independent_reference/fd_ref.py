"""Own 4th-order finite-difference reference pipeline (independent of sphdiag).

USED ONLY TO SET THE SCALE OF TRUNCATION TOLERANCES (E_ref in verification_plan.md, section 1.3),
never as an expected value.

- Fornberg (1988) weights, algorithm from B. Fornberg, Math. Comp. 51 (1988) 699, written from the
  published recursion.
- regular centre: ghost points by parity (vertex: r_{-k} = -r_k; cell: r_{-1-k} = -r_k).
- 'none' and outer boundary: one-sided stencils of the same width.
- Gamma' = D(Gamma), M' = D(M) (composition), as the simplest consistent choice.
"""
import numpy as np


def fornberg(x0, x, m):
    """Weights c[k, j] for the k-th derivative (k <= m) at x0 from nodes x[j]."""
    n = len(x)
    c = np.zeros((m + 1, n))
    c1 = 1.0
    c4 = x[0] - x0
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, i] = c1 * (k * c[k - 1, i - 1] - c5 * c[k, i - 1]) / c2
                c[0, i] = -c1 * c5 * c[0, i - 1] / c2
            for k in range(mn, 0, -1):
                c[k, j] = (c4 * c[k, j] - k * c[k - 1, j]) / c3
            c[0, j] = c4 * c[0, j] / c3
        c1 = c2
    return c


def _extend(r, f, center, parity, ng):
    if center == "vertex":
        rg = -r[ng:0:-1]
        fg = parity * f[ng:0:-1]
    elif center == "cell":
        rg = -r[ng - 1::-1]
        fg = parity * f[ng - 1::-1]
    else:
        return r, f, 0
    return np.concatenate([rg, r]), np.concatenate([fg, f]), ng


def deriv(f, r, center="vertex", parity=1, order=4):
    """First derivative, stencil width order+1."""
    w = order + 1
    half = order // 2
    re, fe, off = _extend(np.asarray(r, float), np.asarray(f, float), center, parity, half)
    n = len(re)
    out = np.empty(len(r))
    for i in range(len(r)):
        ie = i + off
        s = min(max(ie - half, 0), n - w)
        c = fornberg(re[ie], re[s:s + w], 1)[1]
        out[i] = np.dot(c, fe[s:s + w])
    return out


def pipeline(r, A, R, KA, KB, center="vertex", rho=None, jr=None, order=4):
    """Own evaluation of Gamma, U, M, R theta_pm and residuals (formulas derived in
    verification_plan.md section 2)."""
    dR = deriv(R, r, center, -1, order)
    G = dR / A
    U = -R * KB
    M = 0.5 * R * (1 - G**2 + U**2)
    out = dict(dR=dR, Gamma=G, U=U, M=M, Rtp=2 * (U + G), Rtm=2 * (U - G))
    if rho is not None:
        dG = deriv(G, r, center, 1, order)
        dKB = deriv(KB, r, center, 1, order)
        dM = deriv(M, r, center, -1, order)
        with np.errstate(divide="ignore", invalid="ignore"):
            ham = 2 * (1 - G**2) / R**2 - 4 * dG / (A * R) + 4 * KA * KB + 2 * KB**2 - 16 * np.pi * rho
            mom = dKB - dR / R * (KA - KB) + 4 * np.pi * jr
        hamM = dM - 4 * np.pi * R**2 * (rho * dR + U * jr)
        ham[R == 0] = np.nan
        mom[R == 0] = np.nan
        out.update(ham=ham, mom=mom, hamM=hamM)
    return out


def scalar_projections(r, A, phi, Pi, m, center="vertex", order=4):
    """Own FD evaluation of the scalar-field projections (for E_ref of V02/X50 only)."""
    dphi = deriv(phi, r, center, 1, order)
    Phi = dphi / A
    V = 0.5 * m**2 * phi**2
    return dict(rho=0.5 * Pi**2 + 0.5 * Phi**2 + V, jr=-Pi * dphi,
                Srr=0.5 * Pi**2 + 0.5 * Phi**2 - V, Sthth=0.5 * Pi**2 - 0.5 * Phi**2 - V)

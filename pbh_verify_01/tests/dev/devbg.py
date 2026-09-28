"""Exact reference backgrounds (derivation.md §3, §4) for the developer tests.

All closed forms are copied from derivation.md §3 (conventions of the contract: K = -3H for
expansion, k_pm = n pm s). Each function returns (slice_kwargs, exact) where ``exact`` holds
closed-form Gamma, U, M, Rtheta_pm, rho, j_r where available.
"""
import numpy as np

PI = np.pi


def stretched(N, r0, r1, c=0.3, cell=False):
    """Smooth non-uniform grid on [r0, r1] (finer near r0)."""
    x = (np.arange(N) + 0.5) / N if cell else np.linspace(0.0, 1.0, N)
    y = (x + c * x ** 2) / (1.0 + c)
    return r0 + (r1 - r0) * y


def minkowski_flat(r):
    one = np.ones_like(r)
    kw = dict(r=r, A=one, R=r.copy(), KA=0 * r, KB=0 * r)
    ex = dict(Gamma=one, U=0 * r, M=0 * r, Rtp=2 * one, Rtm=-2 * one, rho=0 * r, j_r=0 * r)
    return kw, ex


def minkowski_hyperboloid(r, tau=1.0):
    s = np.sqrt(tau ** 2 + r ** 2)
    kw = dict(r=r, A=tau / s, R=r.copy(), KA=-np.ones_like(r) / tau, KB=-np.ones_like(r) / tau,
              alpha=np.ones_like(r), beta=-r / tau)
    ex = dict(Gamma=s / tau, U=r / tau, M=0 * r, Rtp=2 * (r + s) / tau, Rtm=2 * (r - s) / tau,
              rho=0 * r, j_r=0 * r)
    return kw, ex


def milne(chi, tau=1.3):
    kw = dict(r=chi, A=tau * np.ones_like(chi), R=tau * np.sinh(chi),
              KA=-np.ones_like(chi) / tau, KB=-np.ones_like(chi) / tau)
    ex = dict(Gamma=np.cosh(chi), U=np.sinh(chi), M=0 * chi, Rtp=2 * np.exp(chi),
              Rtm=-2 * np.exp(-chi), rho=0 * chi, j_r=0 * chi)
    return kw, ex


def schwarzschild_ks(r, M=1.0):
    kw = dict(r=r, A=np.sqrt(1 + 2 * M / r), R=r.copy(),
              KA=-2 * M * (r + M) / (r ** 1.5 * (r + 2 * M) ** 1.5),
              KB=2 * M / (r ** 1.5 * np.sqrt(r + 2 * M)),
              alpha=1 / np.sqrt(1 + 2 * M / r), beta=2 * M / (r + 2 * M))
    ex = dict(Gamma=np.sqrt(r / (r + 2 * M)), U=-2 * M / np.sqrt(r * (r + 2 * M)),
              M=M * np.ones_like(r), Rtp=2 * (r - 2 * M) / np.sqrt(r * (r + 2 * M)),
              Rtm=-2 * np.sqrt((r + 2 * M) / r), rho=0 * r, j_r=0 * r)
    # analytic derivatives for the dA/dR/dKB path
    dA = -M / (r ** 2 * np.sqrt(1 + 2 * M / r))
    KB = kw["KB"]
    dKB = KB * (-1.5 / r - 0.5 / (r + 2 * M))
    ex.update(dA=dA, dR=np.ones_like(r), dKB=dKB)
    return kw, ex


def schwarzschild_pg(r, M=1.0):
    q = np.sqrt(2 * M / r)
    kw = dict(r=r, A=np.ones_like(r), R=r.copy(), KA=-0.5 * np.sqrt(2 * M / r ** 3),
              KB=np.sqrt(2 * M / r ** 3), alpha=np.ones_like(r), beta=q)
    ex = dict(Gamma=np.ones_like(r), U=-q, M=M * np.ones_like(r), Rtp=2 * (1 - q),
              Rtm=-2 * (1 + q), rho=0 * r, j_r=0 * r)
    return kw, ex


def schwarzschild_iso(r, M=1.0):
    psi = 1 + M / (2 * r)
    kw = dict(r=r, A=psi ** 2, R=psi ** 2 * r, KA=0 * r, KB=0 * r)
    G = (2 * r - M) / (2 * r + M)
    ex = dict(Gamma=G, U=0 * r, M=M * np.ones_like(r), Rtp=2 * G, Rtm=-2 * G, rho=0 * r,
              j_r=0 * r)
    return kw, ex


def flrw(r, H, a=1.0, k=0):
    """FLRW slice, comoving radius r (k = 0: R = a r; k = +1: a sin r; k = -1: a sinh r)."""
    S, dS = {0: (r, np.ones_like(r)), 1: (np.sin(r), np.cos(r)),
             -1: (np.sinh(r), np.cosh(r))}[k]
    R = a * S
    one = np.ones_like(r)
    kw = dict(r=r, A=a * one, R=R, KA=-H * one, KB=-H * one)
    rho = 3 * (H ** 2 + k / a ** 2) / (8 * PI)
    ex = dict(Gamma=dS, U=H * R, M=0.5 * R ** 3 * (H ** 2 + k / a ** 2), Rtp=2 * (H * R + dS),
              Rtm=2 * (H * R - dS), rho=rho * one, j_r=0 * r)
    return kw, ex


# --- a generic, non-areal, non-vacuum slice with flux: matter defined from the constraints ----
def generic_slice(r, a=0.3, b=0.05, k=0.4, c=0.7):
    """A = 1 + a r^2 g, R = r + b r^3 g, K_B = k g, K_A = k (1 + c r^2) g, g = exp(-r^2).

    rho and j_r are *defined* by H = 0 and M = 0 (test data only), with hand-derived
    derivatives; the r = 0 limit uses Gamma = 1 + (3b - a) r^2 + ... (derivation.md §4).
    """
    g = np.exp(-r ** 2)
    A = 1 + a * r ** 2 * g
    dA = a * g * (2 * r - 2 * r ** 3)
    R = r + b * r ** 3 * g
    dR = 1 + b * g * (3 * r ** 2 - 2 * r ** 4)
    ddR = b * g * (6 * r - 14 * r ** 3 + 4 * r ** 5)
    KB = k * g
    KA = k * (1 + c * r ** 2) * g
    dKB = -2 * k * r * g
    G = dR / A
    dG = (ddR * A - dR * dA) / A ** 2
    with np.errstate(all="ignore"):
        R3 = 2 * (1 - G ** 2) / R ** 2 - 4 * dG / (A * R)
        jr = -(dKB - (dR / R) * (KA - KB)) / (4 * PI)
    z = r == 0
    R3 = np.where(z, -12 * (3 * b - a), R3)
    jr = np.where(z, 0.0, jr)
    rho = (R3 + 4 * KA * KB + 2 * KB ** 2) / (16 * PI)
    M = 0.5 * R * (1 - G ** 2 + R ** 2 * KB ** 2)
    kw = dict(r=r, A=A, R=R, KA=KA, KB=KB)
    ex = dict(Gamma=G, U=-R * KB, M=M, rho=rho, j_r=jr, dA=dA, dR=dR, dKB=dKB,
              Rtp=2 * (-R * KB + G), Rtm=2 * (-R * KB - G),
              MR3_0=(4 * PI / 3) * rho[0] if z[0] else None)
    return kw, ex

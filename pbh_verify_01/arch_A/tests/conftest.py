"""Exact / manufactured slices used by the tests.  All in G = c = 1."""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from sphdiag import Projections, ScalarField, SphericalSlice  # noqa: E402


def grid(center, N, rmax, rmin=None):
    if center == "vertex":
        return np.linspace(0.0, rmax, N)
    if center == "cell":
        h = rmax / N
        return (np.arange(N) + 0.5) * h
    return np.linspace(rmin, rmax, N)


def zeros_matter(N):
    z = np.zeros(N)
    return Projections(z, z.copy(), z.copy(), z.copy())


def order(errs, Ns):
    errs, Ns = np.asarray(errs, float), np.asarray(Ns, float)
    return np.log(errs[:-1] / errs[1:]) / np.log(Ns[1:] / Ns[:-1])


# ------------------------------------------------------------------ manufactured smooth slice
def manufactured(r, a=0.3, c=0.2, k0=-0.4, k1=0.5, k2=0.3):
    """Regular smooth non-trivial slice (all functions even/odd as required, Gamma(0)=1, KA(0)=KB(0)).
    rho and j_r are chosen so that H = M = 0 exactly (analytic derivatives)."""
    E = np.exp(-r ** 2)
    A = 1 + a * E
    dA = -2 * a * r * E
    R = r + a * r * E + c * r ** 3 * E
    dR = 1 + a * E * (1 - 2 * r ** 2) + c * E * (3 * r ** 2 - 2 * r ** 4)
    d2R = a * E * (4 * r ** 3 - 6 * r) + c * E * (4 * r ** 5 - 14 * r ** 3 + 6 * r)
    KB = k0 + k1 * E
    dKB = -2 * k1 * r * E
    KA = KB + k2 * r ** 2 * E
    G = dR / A
    dG = (d2R - G * dA) / A
    with np.errstate(all="ignore"):
        rho = (2 * (1 - G ** 2) / R ** 2 - 4 * dG / (A * R) + 4 * KA * KB + 2 * KB ** 2) / (16 * np.pi)
        jr = -(dKB - (dR / R) * (KA - KB)) / (4 * np.pi)
    if r[0] == 0.0:
        # rho(0) is irrelevant for every output (weights R^2 vanish, H/M excluded at R=0),
        # but it must be finite; take the even extrapolation from the next points.
        rho[0] = (4 * rho[1] - rho[2]) / 3 if r[2] == 2 * r[1] else rho[1]
        jr[0] = 0.0
    return dict(r=r, A=A, dA=dA, R=R, dR=dR, KA=KA, KB=KB, dKB=dKB, rho=rho, jr=jr, Gamma=G, dGamma=dG,
                M=0.5 * R * (1 - G ** 2 + (R * KB) ** 2))


def manufactured_slice(center, N, rmax=4.0, analytic=False, **kw):
    r = grid(center, N, rmax, rmin=0.3)
    d = manufactured(r, **kw)
    extra = dict(dA=d["dA"], dR=d["dR"], dKB=d["dKB"]) if analytic else {}
    s = SphericalSlice(r, d["A"], d["R"], d["KA"], d["KB"], center, **extra)
    z = np.zeros(N)
    return s, Projections(d["rho"], d["jr"], z, z.copy()), d


# ------------------------------------------------------------------ EKG data with flux (areal gauge)
def ekg_data(r, phi0=0.08, w=1.0, pi0=0.06, m=1.5, k0=-0.3, k1=0.4):
    """Areal gauge R = r.  phi, Pi, K_B prescribed; M from the mass-form ODE
    M' = 4 pi r^2 (rho + U j_r) with rho depending on A via Gamma^2 = 1 - 2M/r + r^2 K_B^2,
    A = 1/Gamma, K_A from the momentum constraint.  Both constraints then hold to ODE accuracy."""
    phi = lambda x: phi0 * np.exp(-x ** 2 / w ** 2)
    dphi = lambda x: -2 * x / w ** 2 * phi(x)
    Pi = lambda x: pi0 * (1 + x ** 2) * np.exp(-x ** 2 / w ** 2)
    KB = lambda x: k0 + k1 * np.exp(-x ** 2)
    dKB = lambda x: -2 * k1 * x * np.exp(-x ** 2)

    def rhs(x, y):
        M = y[0]
        twoMr = 2 * M / x if x > 0 else 0.0
        G2 = 1 - twoMr + x ** 2 * KB(x) ** 2
        rho = 0.5 * Pi(x) ** 2 + 0.5 * dphi(x) ** 2 * G2 + 0.5 * m ** 2 * phi(x) ** 2
        Uj = (-x * KB(x)) * (-Pi(x) * dphi(x))
        return [4 * np.pi * x ** 2 * (rho + Uj)]

    rr = np.asarray(r, float)
    sol = solve_ivp(rhs, (0.0, rr[-1]), [0.0], method="DOP853", rtol=1e-13, atol=1e-18, t_eval=rr)
    assert sol.success
    M = sol.y[0]
    with np.errstate(all="ignore"):
        G2 = 1 - np.where(rr > 0, 2 * M / np.where(rr > 0, rr, 1), 0.0) + rr ** 2 * KB(rr) ** 2
    A = 1 / np.sqrt(G2)
    jr = -Pi(rr) * dphi(rr)
    KA = KB(rr) + rr * (dKB(rr) + 4 * np.pi * jr)
    return dict(r=rr, A=A, R=rr.copy(), KA=KA, KB=KB(rr), phi=phi(rr), Pi=Pi(rr), m=m, M=M, jr=jr,
                dphi=dphi(rr))


def ekg_slice(center, N, rmax=5.0, **kw):
    r = grid(center, N, rmax, rmin=0.5)
    d = ekg_data(r, **kw)
    s = SphericalSlice(r, d["A"], d["R"], d["KA"], d["KB"], center)
    return s, ScalarField(d["phi"], d["Pi"], d["m"]), d


# ------------------------------------------------------------------ Schwarzschild slices
def pg_slice(r, M=1.0, white=False, R_of_r=None, dR_of_r=None):
    """Painleve-Gullstrand slice (flat, alpha=1). R_of_r: optional non-areal radial label."""
    if R_of_r is None:
        R, dRr = r.copy(), np.ones_like(r)
    else:
        R, dRr = R_of_r(r), dR_of_r(r)
    sgn = -1.0 if white else 1.0
    KB = sgn * np.sqrt(2 * M) * R ** -1.5
    KA = -0.5 * sgn * np.sqrt(2 * M) * R ** -1.5
    return SphericalSlice(r, dRr, R, KA, KB, "none", alpha=np.ones_like(r), beta=sgn * np.sqrt(2 * M / R) / dRr)


def ks_slice(r, M=1.0):
    """Kerr-Schild slice of Schwarzschild, R = r, gamma_rr = 1 + 2M/r."""
    s = 1 + 2 * M / r
    KB = 2 * M / r ** 2 / np.sqrt(s)
    KA = -(2 * M / r ** 2) * (1 + M / r) * s ** -1.5
    return SphericalSlice(r, np.sqrt(s), r.copy(), KA, KB, "none")


def isotropic_slice(r, M=1.0):
    psi = 1 + M / (2 * r)
    return SphericalSlice(r, psi ** 2, psi ** 2 * r, np.zeros_like(r), np.zeros_like(r), "none")


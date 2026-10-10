"""Curvature profiles zeta(r) beyond the Yoo Gaussian: mean peak profiles of log-normal spectra (BBKS conditioning
on the peak height nu and the mean curvature x_bar(nu, gamma)) and a broad near-top-hat control.

Conventions: comoving wavenumber in units of the spectrum peak k_p (= the "k" of the runs), dimensionless power
P(k) = exp(-ln^2(k/k_p)/(2 Delta^2)); moments sigma_j^2 = int P k^{2j} dk/k; psi(r) = xi(r)/xi(0).
Mean profile of a peak of height nu (BBKS 1986, eqs 7.8-7.10 written as a Gaussian conditional mean):
    <F(r) | F(0) = nu s0, lap F(0) = -x s2> = c(r)^T C^{-1} (nu s0, -x s2),
    c(r) = s0^2 (psi(r), lap psi(r)),  C = [[s0^2, -s1^2], [-s1^2, s2^2]],
with x = x_bar = gamma nu + theta(gamma, gamma nu) (BBKS eq. 6.14).  The profile is returned normalised to 1 at r = 0.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def _theta_bbks(gamma, gnu):
    return (3 * (1 - gamma**2) + (1.216 - 0.9 * gamma**4) * np.exp(-(gamma / 2) * (gnu / 2) ** 2)) / (
        np.sqrt(3 * (1 - gamma**2) + 0.45 + (gnu / 2) ** 2) + gnu / 2)


def lognormal_peak_profile(Delta: float, nu: float = 4.0, r_max: float = 40.0, n_r: int = 4001):
    lnk = np.linspace(-8 * Delta - 2, 8 * Delta + 2, 20001)
    k = np.exp(lnk); P = np.exp(-lnk**2 / (2 * Delta**2)); dlnk = lnk[1] - lnk[0]
    s0 = np.sqrt(np.sum(P) * dlnk); s1 = np.sqrt(np.sum(P * k**2) * dlnk); s2 = np.sqrt(np.sum(P * k**4) * dlnk)
    gamma = s1**2 / (s0 * s2); R_star = np.sqrt(3) * s1 / s2
    r = np.linspace(0.0, r_max, n_r)
    psi = np.empty(n_r); lap_psi = np.empty(n_r)
    for i0 in range(0, n_r, 200):                        # chunked: the full outer product would take ~2 GB
        kr = np.outer(r[i0:i0 + 200], k)
        j0 = np.where(kr > 1e-8, np.sin(kr) / np.maximum(kr, 1e-300), 1.0)
        psi[i0:i0 + 200] = (j0 @ P) * dlnk / s0**2
        lap_psi[i0:i0 + 200] = -((j0 * k[None, :]**2) @ P) * dlnk / s0**2
    x = gamma * nu + _theta_bbks(gamma, gamma * nu)
    C = np.array([[s0**2, -s1**2], [-s1**2, s2**2]])
    u = np.array([nu * s0, -x * s2])
    w = np.linalg.solve(C, u)
    F = s0**2 * (psi * w[0] + lap_psi * w[1])
    Psi = F / F[0]
    return dict(r=r, Psi=Psi, psi=psi, gamma=gamma, R_star=R_star, x_bar=x, nu=nu, Delta=Delta, s0=s0)


def tophat_profile(r0: float = 2.5, n: float = 4.0, r_max: float = 40.0, n_r: int = 4001):
    """Flat-core, steep-edge control: super-Gaussian exp(-(r/r0)^n) (smooth at r = 0, unlike a shifted tanh)."""
    r = np.linspace(0.0, r_max, n_r)
    Psi = np.exp(-(r / r0) ** n)
    return dict(r=r, Psi=Psi, r0=r0, n=n)


def profile_table(name: str):
    """name: 'yoo' | 'ln0.1' | 'ln0.5' | 'tophat'  ->  dict with r, Psi (normalised), spline s(r) and derivatives."""
    if name == "yoo":
        r = np.linspace(0.0, 40.0, 4001); d = dict(r=r, Psi=np.exp(-r**2 / 6))
    elif name.startswith("ln"):
        d = lognormal_peak_profile(float(name[2:]))
    elif name == "tophat":
        d = tophat_profile()
    else:
        raise ValueError(name)
    s = CubicSpline(d["r"], d["Psi"], bc_type=((1, 0.0), (1, 0.0)))
    d["spline"] = s
    return d

"""Initial data families for the spherical solver.

All configurations are isolated (asymptotically flat) systems.  The cosmological
(expanding radiation background) initial-data path is a later Stage-A step.
"""
from __future__ import annotations

import numpy as np
from .spherical_ev import Particles


def _sample_radii(rng, n, pdf_r, r_max, oversample=8):
    """Rejection-free inverse-CDF sampling of r from pdf_r(r) ~ n(r) r^2 on [0, r_max]."""
    rr = np.linspace(0, r_max, 20001)
    p = pdf_r(rr) * rr**2
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    u = rng.uniform(0, 1, n)
    return np.interp(u, cdf, rr)


def dust_ball(M: float, R: float, n_part: int, rng, stratified=True):
    """Homogeneous cold dust ball at rest (Oppenheimer-Snyder). Weights set so that the
    Misner-Sharp mass is exactly M (the rest mass then exceeds M by the binding energy)."""
    # radii with uniform proper *rest-mass* density: need n(r) e^lambda uniform in coordinate volume
    # -> use uniform coordinate density; the O(2M/R) difference is absorbed by the weight
    # normalisation via the exact Hamiltonian constraint (m_edge matched to M below).
    if stratified:
        u = (np.arange(n_part) + rng.uniform(0, 1, n_part)) / n_part
    else:
        u = rng.uniform(0, 1, n_part)
    r = R * u ** (1 / 3)
    N = np.full(n_part, M / n_part)
    P = Particles(r, np.zeros(n_part), np.zeros(n_part), N, np.ones(n_part))
    return P


def normalise_mass(grid, P: Particles, M_target: float, iters=6):
    """Rescale weights so that the Misner-Sharp mass inside r_out equals M_target."""
    from .spherical_ev import solve_metric
    for _ in range(iters):
        met = solve_metric(grid, P)
        P.N *= M_target / met.M
    return P


def shell_family(M: float, R: float, n_part: int, rng, flatness: float = 1.0,
                 sigma_w: float = 0.0, L_rms: float = 0.0, w_mean: float = 0.0):
    """Generic isolated X cloud used in the outcome scan.

    rest-mass density n(r) ~ exp(-(r/R)^(2 flatness))  (flatness=1: Gaussian, large: top-hat)
    w ~ Normal(w_mean, sigma_w)      (radial velocity dispersion / infall)
    L = r^2 v_perp^2, v_perp ~ Rayleigh with rms L_rms * sqrt(M/R)  (fraction of circular speed)
    """
    pdf = lambda r: np.exp(-((r / R) ** (2 * flatness)))
    r_max = R * (25.0) ** (1 / (2 * flatness))
    r = _sample_radii(rng, n_part, pdf, r_max)
    r = np.maximum(r, 1e-6 * R)
    w = w_mean + sigma_w * rng.standard_normal(n_part)
    vperp = L_rms * np.sqrt(M / R) * np.sqrt(0.5) * np.sqrt(rng.standard_normal(n_part) ** 2 + rng.standard_normal(n_part) ** 2)
    L = (r * vperp) ** 2
    N = np.full(n_part, M / n_part)
    return Particles(r, w, L, N, np.ones(n_part))

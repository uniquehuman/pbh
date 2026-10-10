"""Analytic references for the mandatory Stage-A tests."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def schwarzschild_radial_infall(M: float, R0: float, eta: np.ndarray):
    """Radial geodesic from rest at areal radius R0 in Schwarzschild (mass M), cycloid parameter eta.

    Returns (r, tau, t): areal radius, proper time and Schwarzschild coordinate time
    (MTW eqs. 25.38 / 31.10).  This is also the surface trajectory of an Oppenheimer-Snyder
    dust ball of total mass M and initial radius R0, and, in proper time, the trajectory of
    every interior shell with m -> m(r0) and R0 -> r0.
    """
    eta = np.asarray(eta, dtype=float)
    r = 0.5 * R0 * (1 + np.cos(eta))
    tau = np.sqrt(R0**3 / (8 * M)) * (eta + np.sin(eta))
    q = np.sqrt(R0 / (2 * M) - 1)
    tn = np.tan(eta / 2)
    t = 2 * M * np.log(np.abs((q + tn) / (q - tn))) + 2 * M * q * (eta + (R0 / (4 * M)) * (eta + np.sin(eta)))
    return r, tau, t


def eta_of_t(M, R0, t_target):
    """Invert t(eta) for the exterior radial geodesic (eta in [0, eta_h) where r(eta_h) = 2M)."""
    eta_h = np.arccos(4 * M / R0 - 1)
    f = lambda e: schwarzschild_radial_infall(M, R0, e)[2] - t_target
    if f(eta_h * (1 - 1e-9)) < 0:
        return np.nan
    return brentq(f, 0.0, eta_h * (1 - 1e-9), xtol=1e-13)


def circular_orbit_L(M, r):
    """Squared specific angular momentum L = (r p_perp)^2 of a circular timelike geodesic."""
    return M * r**2 / (r - 3 * M)


def circular_orbit_Omega(M, r):
    return np.sqrt(M / r**3)

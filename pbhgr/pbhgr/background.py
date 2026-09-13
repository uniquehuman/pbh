"""Homogeneous flat FLRW with cold X + radiation + X -> radiation decay.

    rho_X' = -3 H rho_X - Gamma rho_X
    rho_R' = -4 H rho_R + Gamma rho_X
    H^2    = (8 pi / 3) (rho_X + rho_R)          (G = c = 1)

This is the homogeneous control limit required by the specification
(Stage A, mandatory tests: "homogeneous FLRW X + radiation + decay").
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def hubble(rho_X: float, rho_R: float) -> float:
    return np.sqrt(8.0 * np.pi / 3.0 * (rho_X + rho_R))


def rhs(t, y, Gamma):
    a, rho_X, rho_R = y
    H = hubble(rho_X, rho_R)
    return [a * H, -3.0 * H * rho_X - Gamma * rho_X, -4.0 * H * rho_R + Gamma * rho_X]


def evolve(rho_X0: float, rho_R0: float, Gamma: float, t_end: float, t0: float = 0.0,
           n_out: int = 400, rtol: float = 1e-10, atol: float = 1e-14):
    """Integrate from t0 to t_end. Returns dict of arrays t, a, rho_X, rho_R, H.

    a(t0) = 1. t0 must be > 0 if the initial state is given at a finite Hubble time.
    """
    t_eval = np.linspace(t0, t_end, n_out)
    sol = solve_ivp(rhs, (t0, t_end), [1.0, rho_X0, rho_R0], args=(Gamma,), t_eval=t_eval,
                    rtol=rtol, atol=atol, method="DOP853")
    if not sol.success:
        raise RuntimeError(sol.message)
    a, rX, rR = sol.y
    return {"t": sol.t, "a": a, "rho_X": rX, "rho_R": rR, "H": hubble(rX, rR)}


def conservation_residual(res: dict, Gamma: float) -> np.ndarray:
    """Residual of the total covariant conservation law d(rho a^3)/dt = -3 H p a^3 with
    p = rho_R/3, evaluated by finite differences on the output grid. Should converge to 0."""
    t, a, rX, rR, H = res["t"], res["a"], res["rho_X"], res["rho_R"], res["H"]
    tot = (rX + rR) * a**3
    d = np.gradient(tot, t)
    return d + H * rR * a**3

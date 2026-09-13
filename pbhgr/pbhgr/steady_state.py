"""Static spherically symmetric Einstein-Vlasov solutions (isotropic polytropes, Rein-Rendall
type ansatz f = (E0 - e^mu E)_+^k) and particle sampling from them.

A static solution evolved by the dynamical solver must stay static: this is the
"known Einstein-Vlasov benchmark" test that exercises the full coupled system with
L > 0 (angular momentum), the lambda_dot term and the pressure term in mu'.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from .spherical_ev import Particles

_v = np.linspace(0, 1, 401)  # quadrature nodes on [0, 1] rescaled to [0, v_max]


def _moments(eps, k):
    """Return (n, rho, p) for the local shape (eps - E)_+^k, E = sqrt(1+v^2), isotropic in v.
    The full ansatz f = (E0 - e^mu E)_+^k = e^{k mu} (eps - E)_+^k with eps = E0 e^{-mu};
    the caller multiplies by e^{k mu}."""
    if eps <= 1.0:
        return 0.0, 0.0, 0.0
    vmax = np.sqrt(eps**2 - 1)
    v = _v * vmax
    E = np.sqrt(1 + v**2)
    f = np.maximum(eps - E, 0) ** k
    n = 4 * np.pi * np.trapezoid(f * v**2, v)
    rho = 4 * np.pi * np.trapezoid(f * E * v**2, v)
    p = 4 * np.pi / 3 * np.trapezoid(f * v**4 / E, v)
    return n, rho, p


def build(E0_central: float = 1.2, k: float = 1.0, r_max: float = 200.0):
    """Integrate the static equations from a regular centre with mu(0) = 0 and local cutoff
    energy eps(0) = E0_central; then shift mu so that mu(R) matches Schwarzschild.

    Returns dict with r, m, mu, n (coordinate number density), rho, p, R, M, E0 (shifted), k.
    """
    def rhs(r, y):
        m, mu = y
        eps = E0_central * np.exp(-mu)
        _, rho, p = _moments(eps, k)
        rho *= np.exp(k * mu); p *= np.exp(k * mu)
        if r < 1e-9:
            return [0.0, 0.0]
        a = max(1 - 2 * m / r, 1e-12)
        return [4 * np.pi * r**2 * rho, (m / r**2 + 4 * np.pi * r * p) / a]

    def surface(r, y):
        return E0_central * np.exp(-y[1]) - 1.0
    surface.terminal = True
    sol = solve_ivp(rhs, (1e-9, r_max), [0.0, 0.0], events=surface, max_step=0.02, rtol=1e-9, atol=1e-12, dense_output=True)
    R = float(sol.t_events[0][0]); M = float(sol.y_events[0][0][0])
    c = 0.5 * np.log(1 - 2 * M / R) - float(sol.y_events[0][0][1])
    r = np.linspace(1e-6, R, 2000)
    m, mu = sol.sol(r); mu = mu + c
    E0 = E0_central * np.exp(c)
    eps = E0 * np.exp(-mu)
    mom = np.array([_moments(e, k) for e in eps]) * np.exp(k * (mu - c))[:, None]   # e^{k mu_num}: the ansatz is fixed by mu_num, E0_central
    return dict(r=r, m=m, mu=mu, n=mom[:, 0], rho=mom[:, 1], p=mom[:, 2], R=R, M=M, E0=E0, k=k)


def sample(st: dict, n_part: int, rng: np.random.Generator) -> Particles:
    """Sample macroparticles from the static distribution. Weights carry the total rest mass
    (number) N_tot = int 4 pi r^2 e^lambda n dr; the caller may renormalise to M."""
    r, m, mu, n = st["r"], st["m"], st["mu"], st["n"]
    elam = 1 / np.sqrt(np.maximum(1 - 2 * m / r, 1e-12))
    dNdr = 4 * np.pi * r**2 * elam * n
    cdf = np.concatenate([[0], np.cumsum(0.5 * (dNdr[1:] + dNdr[:-1]) * np.diff(r))])
    N_tot = cdf[-1]; cdf /= N_tot
    u = (np.arange(n_part) + rng.uniform(0, 1, n_part)) / n_part
    rs = np.interp(u, cdf, r)
    eps = np.interp(rs, r, st["E0"] * np.exp(-mu))
    vmax = np.sqrt(np.maximum(eps**2 - 1, 0))
    # rejection sampling of v from (eps - E)^k v^2 on [0, vmax]
    v = np.empty(n_part); todo = np.ones(n_part, bool)
    while np.any(todo):
        idx = np.nonzero(todo)[0]
        vt = rng.uniform(0, 1, len(idx)) * vmax[idx]
        Et = np.sqrt(1 + vt**2)
        f = np.maximum(eps[idx] - Et, 0) ** st["k"] * vt**2
        fmax = (eps[idx] - 1) ** st["k"] * vmax[idx] ** 2   # crude envelope (f <= this)
        acc = rng.uniform(0, 1, len(idx)) * fmax < f
        v[idx[acc]] = vt[acc]; todo[idx[acc]] = False
    cos = rng.uniform(-1, 1, n_part)
    w = v * cos
    L = (rs * v) ** 2 * (1 - cos**2)
    return Particles(rs, w, L, np.full(n_part, N_tot / n_part), np.ones(n_part))

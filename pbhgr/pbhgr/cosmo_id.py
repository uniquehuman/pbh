"""Cosmological initial data for the areal-CMC spherical Einstein-Vlasov solver (Stage A step 2).

Long-wavelength constant-mean-curvature data of Yoo, Escrivà, Harada & Kohri 2026 (arXiv:2609.14218,
eqs. 3.4-3.11) restricted to spherical symmetry, transformed to the areal gauge and converted into cold
dust shells.  Conventions: G = c = 1, isotropic radius r, a_i = 1, H_i = 5k (eps_i = k/(a_i H_i) = 0.2).

  Psi = exp(zeta/2),  zeta = mu exp(-k^2 r^2/6)
  q   = -(4/3) lap(Psi)/Psi^5
  p_ij = P(r) (n_i n_j - delta_ij/3),  P = Psi^-4 [ -(2/Psi)(Psi'' - Psi'/r) + 6 Psi'^2/Psi^2 ]
  psi = Psi [1 - q/(6 (aH)^2)],  gamma~_rr = 1 - (8/15) P/(aH)^2,  gamma~_thth/r^2 = 1 + (4/15) P/(aH)^2
  K = -3H,  A~^th_th = -(2/15) P H/(aH)^2   ->  K^th_th = -H - (2/15) P H/(aH)^2
  areal radius R = a psi^2 r sqrt(gamma~_thth/r^2),  A = a psi^2 sqrt(gamma~_rr) / (dR/dr)
  E and J from the constraints in the areal gauge (Appendix B of the plan):
      4 pi A J = K/R - 3 K_th/R - dK_th/dR,   m = (R/2)(1 - 1/A^2 + R^2 K_th^2),   E = m'/(4 pi R^2) + R K_th A J
  dust: E = Gamma^2 rho, J = Gamma^2 rho v  ->  v = J/E,  particle number on the slice dN = 4 pi R^2 A Gamma rho dR.
"""
from __future__ import annotations

import numpy as np


def yoo_spherical_cmc(mu: float, k: float = 1.0, H_i: float = 5.0, r_max: float = 20.0, n: int = 80001, zeta_fns=None):
    """zeta_fns: optional (zeta, dzeta, ddzeta) callables of r replacing the Gaussian profile."""
    r = np.linspace(0.0, r_max, n)
    aH = H_i                                          # a_i = 1
    if zeta_fns is None:
        zeta = mu * np.exp(-k**2 * r**2 / 6)
        dz = -(k**2 * r / 3) * zeta
        ddz = (-k**2 / 3 + k**4 * r**2 / 9) * zeta
    else:
        zeta, dz, ddz = (np.asarray(f(r), dtype=float) for f in zeta_fns)
    Psi = np.exp(zeta / 2)
    Psi1 = Psi * dz / 2
    Psi2 = Psi * (ddz / 2 + dz**2 / 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        Psi1_over_r = np.where(r > 0, Psi1 / r, Psi2)                     # Psi'/r -> Psi''(0)
    lap = Psi2 + 2 * Psi1_over_r
    q = -(4.0 / 3.0) * lap / Psi**5
    P = Psi**-4 * (-(2.0 / Psi) * (Psi2 - Psi1_over_r) + 6.0 * Psi1**2 / Psi**2)
    psi = Psi * (1 - q / (6 * aH**2))
    g_rr = 1 - (8.0 / 15.0) * P / aH**2
    g_tt = 1 + (4.0 / 15.0) * P / aH**2                                   # gamma~_thth / r^2
    R = psi**2 * r * np.sqrt(g_tt)
    dRdr = np.gradient(R, r, edge_order=2)
    A = psi**2 * np.sqrt(g_rr) / dRdr
    K = -3 * H_i
    Kth = -H_i - (2.0 / 15.0) * P * H_i / aH**2
    dKth_dR = np.gradient(Kth, r, edge_order=2) / dRdr
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(R > 0, (K / R - 3 * Kth / R - dKth_dR) / (4 * np.pi * A), 0.0)
    m = 0.5 * R * (1 - 1 / A**2 + R**2 * Kth**2)
    dm_dR = np.gradient(m, r, edge_order=2) / dRdr
    with np.errstate(divide="ignore", invalid="ignore"):
        E = np.where(R > 0, dm_dR / (4 * np.pi * R**2) + R * Kth * A * J, 3 * H_i**2 / (8 * np.pi))
    # the two numerical derivatives are inaccurate within a few points of r = 0: replace by a quadratic fit in r^2
    i0, i1 = 8, 60
    cfit = np.polyfit(r[i0:i1] ** 2, E[i0:i1], 2)
    E[:i0] = np.polyval(cfit, r[:i0] ** 2)
    v = J / E
    Gamma = 1 / np.sqrt(1 - v**2)
    rho = E / Gamma**2
    dN_dR = 4 * np.pi * R**2 * A * Gamma * rho                           # particles per unit areal radius
    return dict(r=r, R=R, A=A, Kth=Kth, J=J, E=E, v=v, Gamma=Gamma, rho=rho, m=m, dN_dR=dN_dR,
                P=P, q=q, psi=psi, mu=mu, k=k, H_i=H_i)


def flrw_profile(H_i: float = 5.0, r_max: float = 20.0, n: int = 80001):
    r = np.linspace(0.0, r_max, n)
    rho = 3 * H_i**2 / (8 * np.pi)
    return dict(r=r, R=r, A=np.ones_like(r), Kth=np.full_like(r, -H_i), J=np.zeros_like(r), E=np.full_like(r, rho),
                v=np.zeros_like(r), Gamma=np.ones_like(r), rho=np.full_like(r, rho), m=0.5 * H_i**2 * r**3,
                dN_dR=4 * np.pi * r**2 * rho, mu=0.0, k=1.0, H_i=H_i)


def sample_shells(prof: dict, x_out: float, n_part: int, rng=None, stratified: bool = True, sigma_v: float = 0.0):
    """Cold (or warm) dust shells: uniform sampling in areal radius on [0, x_out] (equal count per cell),
    weights N_i = dN/dR * dx, normal-frame radial momentum P = Gamma v (+ Maxwellian dispersion sigma_v in
    the shell rest frame, one-dimensional width, units of c; tangential components go into Lsq)."""
    rng = rng or np.random.default_rng(0)
    if stratified:
        u = (np.arange(n_part) + rng.uniform(0, 1, n_part)) / n_part
    else:
        u = rng.uniform(0, 1, n_part)
    x = u * x_out
    dx = x_out / n_part
    N = np.interp(x, prof["R"], prof["dN_dR"]) * dx
    v = np.interp(x, prof["R"], prof["v"])
    P = v / np.sqrt(1 - v**2)
    Lsq = np.zeros(n_part)
    if sigma_v > 0:
        # isotropic Maxwellian in the (non-relativistic) shell frame, added to the bulk momentum
        wr = sigma_v * rng.standard_normal(n_part)
        wt1 = sigma_v * rng.standard_normal(n_part)
        wt2 = sigma_v * rng.standard_normal(n_part)
        P = P + wr
        Lsq = (x * np.sqrt(wt1**2 + wt2**2)) ** 2
    return x, P, Lsq, N


def sample_shells_quiet(prof: dict, x_edge: np.ndarray, n_per_cell=4, n_inner: int = 0, x_inner: float = 0.0):
    """Quiet start: n_per_cell equally spaced shells in every grid cell (n_inner in the cells with x_edge < x_inner),
    weights from the smooth profile.  Needed for super-horizon starts: in the areal-CMC gauge a density error d at
    H R >> 1 changes 1/A^2 by ~ d (H R)^2."""
    dx = np.diff(x_edge)
    from scipy.interpolate import CubicSpline
    Rp, dN = prof["R"], prof["dN_dR"]
    Mcum = CubicSpline(Rp, dN).antiderivative()              # spline-exact cumulative particle number
    n_cell = np.full(len(dx), int(n_per_cell))
    if n_inner > 0 and x_inner > 0:
        n_cell[x_edge[1:] <= x_inner] = n_inner
    xs, Ns = [], []
    for n in np.unique(n_cell):
        sel = n_cell == n
        off = (np.arange(n) + 0.5) / n
        # sub-intervals [xl, xr]; each shell sits at the volume midpoint ((xl^3 + xr^3)/2)^(1/3) so that the cumulative
        # deposition (linear in x^3 between shells) is exact for uniform density
        xl = (x_edge[:-1][sel, None] + (off[None, :] - 0.5 / n) * dx[sel, None]).ravel()
        xr = (x_edge[:-1][sel, None] + (off[None, :] + 0.5 / n) * dx[sel, None]).ravel()
        xs.append((0.5 * (xl**3 + xr**3)) ** (1.0 / 3.0)); Ns.append(Mcum(xr) - Mcum(xl))
    x = np.concatenate(xs); N = np.concatenate(Ns)
    order = np.argsort(x); x, N = x[order], N[order]
    v = np.interp(x, prof["R"], prof["v"])
    P = v / np.sqrt(1 - v**2)
    return x, P, np.zeros_like(x), N

"""Exact growing-mode LTB solution for a Gaussian curvature profile (Yoo et al. 2026, arXiv:2609.14218).

Profile   ln Psi = (mu/2) exp(-k^2 r^2/6),  zeta = 2 ln Psi,  isotropic radius r.
Solution  t_B = 0 (pure growing mode),  Misner-Sharp mass m(r) = (H_i^2/2)(Psi^2 r)^3  [= m_Yoo r^3/6, eq. 4.10],
          energy E(r) = r zeta'(1 + r zeta'/2)  [= -k~ r^2/2, eq. 4.12],
          cycloid R = (m/(-2E))(1 - cos eta),  t = T (eta - sin eta),  T = m/(-2E)^{3/2},  t_C = 2 pi T.
Units     G = c = 1, k = 1 unless given; a_i = 1, H_i = 5k (eps_i = 0.2) as in the paper; t_i = 2/(3 H_i).
Entry     k/(aH) = sqrt6  ->  t_H = t_i (sqrt6/eps_i)^3 = 100 sqrt6 / k.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


class LTBGaussian:
    def __init__(self, mu: float, k: float = 1.0, H_i: float = 5.0):
        self.mu, self.k, self.H_i = mu, k, H_i
        self.t_i = 2.0 / (3.0 * H_i)
        self.eps_i = k / H_i
        self.t_H = self.t_i * (np.sqrt(6.0) / self.eps_i) ** 3
        self.r_m = np.sqrt(6.0) / k

    # --- profile
    def zeta(self, r):   return self.mu * np.exp(-self.k**2 * r**2 / 6)
    def dzeta(self, r):  return -(self.k**2 * r / 3) * self.zeta(r)
    def ddzeta(self, r): return (-self.k**2 / 3 + self.k**4 * r**2 / 9) * self.zeta(r)

    # --- LTB functions of the isotropic coordinate r
    def m(self, r):  return 0.5 * self.H_i**2 * (np.exp(self.zeta(r)) * r) ** 3
    def mp(self, r): return 1.5 * self.H_i**2 * np.exp(3 * self.zeta(r)) * r**2 * (1 + r * self.dzeta(r))
    def E(self, r):
        rz = r * self.dzeta(r); return rz * (1 + 0.5 * rz)
    def T(self, r):  return self.m(r) / (-2 * self.E(r)) ** 1.5
    def tC(self, r): return 2 * np.pi * self.T(r)

    # --- cycloid utilities (scalar or array in eta)
    @staticmethod
    def _f(e):  # eta - sin eta with series for small eta
        e = np.asarray(e, dtype=float)
        return np.where(e < 1e-2, e**3 / 6 - e**5 / 120 + e**7 / 5040, e - np.sin(e))

    @staticmethod
    def _g(e):  # 1 - cos eta with series
        e = np.asarray(e, dtype=float)
        return np.where(e < 1e-2, e**2 / 2 - e**4 / 24 + e**6 / 720, 1 - np.cos(e))

    def eta_of_t(self, t, r):
        Tr = self.T(r); x = t / Tr
        if x < 1e-6:
            e = (6 * x) ** (1 / 3); return e * (1 + e**2 / 60)
        return brentq(lambda e: Tr * (e - np.sin(e)) - t, 0.0, 2 * np.pi, xtol=1e-14, rtol=1e-14, maxiter=200)

    def R(self, t, r):
        e = self.eta_of_t(t, r); return self.m(r) / (-2 * self.E(r)) * float(self._g(e))

    def Rdot(self, t, r):
        e = self.eta_of_t(t, r); return np.sqrt(-2 * self.E(r)) * np.sin(e) / float(self._g(e))

    def t_AH(self, r):
        """time at which shell r is trapped in the collapsing phase (R = 2m, eta > pi)"""
        c = 1 + 4 * self.E(r)
        if c < -1: return np.nan
        e = 2 * np.pi - np.arccos(c); return self.T(r) * (e - np.sin(e))

    # --- shell from an invariant state (m, E, R, dR/dtau): proper time since the bang and prediction
    @staticmethod
    def shell_eta(m, E, R, Rdot):
        """cycloid parameter of a bound shell (E<0) from (R, dR/dtau); eta in (0, 2pi)"""
        c = 1 - R * (-2 * E) / m                      # cos eta = 1 - R/B, B = m/(-2E)
        c = np.clip(c, -1.0, 1.0)
        e = np.arccos(c)
        return np.where(Rdot >= 0, e, 2 * np.pi - e)

    @classmethod
    def shell_tau(cls, m, E, R, Rdot):
        """proper time since the big bang of a bound shell with invariants (m, E) at state (R, Rdot)"""
        T = m / (-2 * E) ** 1.5
        return T * cls._f(cls.shell_eta(m, E, R, Rdot))

    @classmethod
    def shell_R_of_tau(cls, m, E, tau):
        """areal radius of a bound shell at proper time tau since the bang (vectorised over shells)"""
        m, E, tau = np.broadcast_arrays(np.asarray(m, float), np.asarray(E, float), np.asarray(tau, float))
        T = m / (-2 * E) ** 1.5
        x = tau / T
        eta = np.empty_like(x)
        for i in np.ndindex(x.shape):
            xi = x[i]
            if xi < 1e-6:
                e = (6 * xi) ** (1 / 3); eta[i] = e * (1 + e**2 / 60)
            elif xi >= 2 * np.pi:
                eta[i] = 2 * np.pi
            else:
                eta[i] = brentq(lambda e: e - np.sin(e) - xi, 0.0, 2 * np.pi, xtol=1e-13, rtol=1e-13, maxiter=200)
        return m / (-2 * E) * cls._g(eta)


class LTBProfile(LTBGaussian):
    """Growing-mode LTB solution for a tabulated normalised profile Psi(r): zeta = mu Psi(r) (spline derivatives).
    r_m = maximum of the turnaround compaction -2E(r); t_H = entry time of the r_m scale, t_i (r_m H_i)^3."""

    def __init__(self, mu: float, table: dict, k: float = 1.0, H_i: float = 5.0):
        super().__init__(mu, k, H_i)
        self._s = table["spline"]; self._s1 = self._s.derivative(1); self._s2 = self._s.derivative(2)
        self.name = table.get("name", "table")
        r = np.linspace(1e-3, table["r"][-1] * 0.9, 20000)
        self.r_m = float(r[np.argmax(-self.E(r))])
        self.t_H = self.t_i * (self.r_m * H_i) ** 3

    def zeta(self, r):   return self.mu * self._s(np.asarray(r, dtype=float) * self.k)
    def dzeta(self, r):  return self.mu * self.k * self._s1(np.asarray(r, dtype=float) * self.k)
    def ddzeta(self, r): return self.mu * self.k**2 * self._s2(np.asarray(r, dtype=float) * self.k)

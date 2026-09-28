"""Matter adapters (contract section 4): scalar field (EKG) and Vlasov moments -> n-frame projections."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fd import FDGrid


def _arr(x):
    try:
        return np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None


def _check_arrays(names_vals, N):
    """Return list of problems for 1-D, length-N, finite arrays."""
    probs = []
    for name, v in names_vals:
        a = _arr(v)
        if a is None:
            probs.append(f"{name}: not convertible to float array")
        elif a.ndim != 1 or a.size != N:
            probs.append(f"{name}: shape {a.shape} != ({N},)")
        elif not np.all(np.isfinite(a)):
            probs.append(f"{name}: non-finite values at indices {np.flatnonzero(~np.isfinite(a))[:10].tolist()}")
    return probs


@dataclass
class Projections:
    """n-frame projections: rho = T_nn, j_r covariant (= -gamma_r^a n^b T_ab),
    S_rr = S^r_r, S_thth = S^theta_theta (mixed)."""
    rho: np.ndarray
    j_r: np.ndarray
    S_rr: np.ndarray
    S_thth: np.ndarray
    # rounding/truncation error estimates of rho and j_r (zero unless derived by finite differences)
    sigma_rho: np.ndarray | None = field(default=None, repr=False)
    sigma_j_r: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        for nm in ('rho', 'j_r', 'S_rr', 'S_thth'):
            a = _arr(getattr(self, nm))
            setattr(self, nm, getattr(self, nm) if a is None else a)

    @property
    def S(self):
        return self.S_rr + 2.0 * self.S_thth

    def validate(self, slc) -> list[str]:
        return _check_arrays([('rho', self.rho), ('j_r', self.j_r),
                              ('S_rr', self.S_rr), ('S_thth', self.S_thth)], slc.N)

    def projections(self, slc=None, grid=None) -> "Projections":
        if slc is not None:
            probs = self.validate(slc)
            if probs:
                raise ValueError("; ".join(probs))
        n = np.asarray(self.rho).size
        return Projections(self.rho, self.j_r, self.S_rr, self.S_thth,
                           np.zeros(n) if self.sigma_rho is None else self.sigma_rho,
                           np.zeros(n) if self.sigma_j_r is None else self.sigma_j_r)


@dataclass
class ScalarField:
    """Minimally coupled scalar, V = m_phi^2 phi^2 / 2; Pi = n^a d_a phi.  (G = c = 1, phi dimensionless.)"""
    phi: np.ndarray
    Pi: np.ndarray
    m_phi: float

    def validate(self, slc) -> list[str]:
        probs = _check_arrays([('phi', self.phi), ('Pi', self.Pi)], slc.N)
        try:
            m = float(self.m_phi)
            if not np.isfinite(m) or m < 0:
                probs.append(f"m_phi: must be finite and >= 0, got {self.m_phi!r}")
        except (TypeError, ValueError):
            probs.append(f"m_phi: not a number: {self.m_phi!r}")
        return probs

    def projections(self, slc, grid: FDGrid | None = None, noise_guard: bool = False) -> Projections:
        probs = self.validate(slc) + slc.validate()
        if probs:
            raise ValueError("; ".join(probs))
        g = grid if grid is not None else FDGrid(slc.r, slc.center)
        phi = np.asarray(self.phi, dtype=float)
        Pi = np.asarray(self.Pi, dtype=float)
        A = slc.A
        with np.errstate(all='ignore'):
            d4, rnd = g.deriv(phi, +1, p=4, d=1)
            d6, _ = g.deriv(phi, +1, p=6, d=1)
            sig_dphi = np.abs(d4 - d6) + rnd
            if noise_guard:
                sig_dphi = sig_dphi + g.noise_gain(4, 1) * g.noise_amplitude(phi, +1)
            Phi = d4 / A
            V = 0.5 * float(self.m_phi) ** 2 * phi ** 2
            rho = 0.5 * Pi ** 2 + 0.5 * Phi ** 2 + V
            j_r = -Pi * d4
            S_rr = 0.5 * Pi ** 2 + 0.5 * Phi ** 2 - V
            S_thth = 0.5 * Pi ** 2 - 0.5 * Phi ** 2 - V
            sigma_rho = np.abs(Phi) * sig_dphi / A
            sigma_j = np.abs(Pi) * sig_dphi
        return Projections(rho, j_r, S_rr, S_thth, sigma_rho, sigma_j)


@dataclass
class VlasovMoments:
    """n-frame moments as in pbhgr.cosmo_ev: E (energy density), J (orthonormal radial flux),
    p (radial pressure S^r_r), q (tangential pressure S^theta_theta).  j_r = A J."""
    E: np.ndarray
    J: np.ndarray
    p: np.ndarray
    q: np.ndarray

    def validate(self, slc) -> list[str]:
        return _check_arrays([('E', self.E), ('J', self.J), ('p', self.p), ('q', self.q)], slc.N)

    def projections(self, slc, grid=None) -> Projections:
        probs = self.validate(slc) + slc.validate()
        if probs:
            raise ValueError("; ".join(probs))
        E = np.asarray(self.E, dtype=float)
        n = E.size
        return Projections(E.copy(), slc.A * np.asarray(self.J, dtype=float),
                           np.asarray(self.p, dtype=float).copy(), np.asarray(self.q, dtype=float).copy(),
                           np.zeros(n), np.zeros(n))


def pi_from_dtphi(dtphi, dphi_dr, alpha, beta):
    """Pi = n^a d_a phi = (d_t phi - beta^r d_r phi) / alpha.  Raises ValueError for alpha <= 0 or non-finite input."""
    dtphi, dphi_dr, alpha, beta = (np.asarray(x, dtype=float) for x in (dtphi, dphi_dr, alpha, beta))
    for nm, a in (('dtphi', dtphi), ('dphi_dr', dphi_dr), ('alpha', alpha), ('beta', beta)):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"pi_from_dtphi: non-finite {nm}")
    if np.any(alpha <= 0):
        raise ValueError("pi_from_dtphi: lapse alpha must be > 0")
    return (dtphi - beta * dphi_dr) / alpha

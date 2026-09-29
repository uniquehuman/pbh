"""Matter adapters (contract §4): n-frame projections rho, j_r, S^r_r, S^theta_theta."""
from __future__ import annotations

import numpy as np


def _arr(x):
    try:
        return np.array(x, dtype=float)
    except (TypeError, ValueError):
        return None


class Projections:
    """n-frame projections of T_ab.

    rho = T(n, n); j_r = -T(n, e_r) (covariant, radial); S_rr = S^r_r; S_thth = S^theta_theta.
    Optional sigma_rho, sigma_j_r: numerical error estimates of rho and j_r (non-zero only
    when they were computed with finite differences, e.g. by ScalarField).
    """

    def __init__(self, rho, j_r, S_rr, S_thth, sigma_rho=None, sigma_j_r=None):
        self.rho = _arr(rho)
        self.j_r = _arr(j_r)
        self.S_rr = _arr(S_rr)
        self.S_thth = _arr(S_thth)
        self.sigma_rho = None if sigma_rho is None else _arr(sigma_rho)
        self.sigma_j_r = None if sigma_j_r is None else _arr(sigma_j_r)

    @property
    def S(self):
        """Trace S = S^r_r + 2 S^theta_theta."""
        return self.S_rr + 2.0 * self.S_thth

    def projections(self, slc):  # uniform interface with the other matter types
        return self

    def check(self, N):
        """List of problems with respect to a grid of N points."""
        errs = []
        for name in ("rho", "j_r", "S_rr", "S_thth", "sigma_rho", "sigma_j_r"):
            a = getattr(self, name)
            if a is None:
                if name.startswith("sigma"):
                    continue
                errs.append(f"matter {name}: not numeric")
                continue
            if a.ndim != 1 or a.size != N:
                errs.append(f"matter {name}: shape {a.shape} does not match N = {N}")
        return errs

    def __repr__(self):
        n = None if self.rho is None else self.rho.shape
        return f"Projections(shape={n})"


class ScalarField:
    """Minimally coupled scalar field, V = m_phi^2 phi^2 / 2 (contract §1, §4).

    phi (dimensionless, G = c = 1 normalisation), Pi = n^a d_a phi, m_phi as inverse length.
    """

    def __init__(self, phi, Pi, m_phi):
        self.phi = _arr(phi)
        self.Pi = _arr(Pi)
        try:
            self.m_phi = float(m_phi)
        except (TypeError, ValueError):
            self.m_phi = float("nan")

    def potential(self):
        return 0.5 * self.m_phi ** 2 * self.phi ** 2

    def check(self, N):
        errs = []
        for name in ("phi", "Pi"):
            a = getattr(self, name)
            if a is None:
                errs.append(f"ScalarField {name}: not numeric")
            elif a.ndim != 1 or a.size != N:
                errs.append(f"ScalarField {name}: shape {a.shape} does not match N = {N}")
        if not np.isfinite(self.m_phi):
            errs.append("ScalarField m_phi is not a finite number")
        return errs

    def projections(self, slc):
        """rho, j_r, S^r_r, S^theta_theta on the slice [F3-F5]; phi' by order-4 FD (phi even)."""
        N = slc.N
        errs = self.check(N)
        if errs or slc.op is None:
            nan = np.full(N, np.nan)
            return Projections(nan, nan, nan, nan, nan, nan)
        with np.errstate(all="ignore"):
            dphi, sig_dphi = slc.op.d_err(self.phi, self.phi, +1)
            A = slc.A
            Phi = dphi / A
            Pi = self.Pi
            V = self.potential()
            kin = 0.5 * Pi ** 2
            grad = 0.5 * Phi ** 2
            rho = kin + grad + V
            j_r = -Pi * dphi
            S_rr = kin + grad - V
            S_thth = kin - grad - V
            sig_rho = np.abs(Phi) * sig_dphi / A
            sig_j = np.abs(Pi) * sig_dphi
        p = Projections(rho, j_r, S_rr, S_thth, sig_rho, sig_j)
        p.dphi = dphi
        p.sigma_dphi = sig_dphi
        return p


class VlasovMoments:
    """Vlasov adapter: n-frame moments E, J (orthonormal radial flux), p, q (contract §4).

    rho = E, j_r = A J, S^r_r = p, S^theta_theta = q.
    """

    def __init__(self, E, J, p, q):
        self.E = _arr(E)
        self.J = _arr(J)
        self.p = _arr(p)
        self.q = _arr(q)

    def check(self, N):
        errs = []
        for name in ("E", "J", "p", "q"):
            a = getattr(self, name)
            if a is None:
                errs.append(f"VlasovMoments {name}: not numeric")
            elif a.ndim != 1 or a.size != N:
                errs.append(f"VlasovMoments {name}: shape {a.shape} does not match N = {N}")
        return errs

    def projections(self, slc):
        N = slc.N
        if self.check(N) or slc.A is None or np.shape(slc.A) != (N,):
            nan = np.full(N, np.nan)
            return Projections(nan, nan, nan, nan)
        return Projections(self.E, slc.A * self.J, self.p, self.q)


def pi_from_dtphi(dtphi, dphi_dr, alpha, beta):
    """Pi = n^a d_a phi = (d_t phi - beta^r d_r phi) / alpha (contract §2)."""
    dtphi = np.asarray(dtphi, dtype=float)
    dphi_dr = np.asarray(dphi_dr, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    with np.errstate(all="ignore"):
        out = (dtphi - beta * dphi_dr) / alpha
    return out if out.ndim else float(out)

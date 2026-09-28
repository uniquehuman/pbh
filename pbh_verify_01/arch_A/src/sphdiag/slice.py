"""Spherical 3+1 slice (contract section 2):
ds^2 = -alpha^2 dt^2 + A^2 (dr + beta dt)^2 + R^2 dOmega^2, input K_A = K^r_r, K_B = K^theta_theta."""
from __future__ import annotations

import numpy as np

CENTERS = ("vertex", "cell", "none")


def _to_arr(x):
    try:
        return np.array(x, dtype=float)       # copy: the slice owns its data
    except (TypeError, ValueError):
        return None


class SphericalSlice:
    """Container for slice data.  Never raises on bad content: problems are collected by
    validate() and reported by diagnose() as status data = INVALID_INPUT."""

    def __init__(self, r, A, R, KA, KB, center="vertex", alpha=None, beta=None,
                 dA=None, dR=None, dKB=None):
        self._raw = dict(r=r, A=A, R=R, KA=KA, KB=KB, alpha=alpha, beta=beta, dA=dA, dR=dR, dKB=dKB)
        self.r = _to_arr(r)
        self.A = _to_arr(A)
        self.R = _to_arr(R)
        self.KA = _to_arr(KA)
        self.KB = _to_arr(KB)
        self.center = center
        self.alpha = None if alpha is None else _to_arr(alpha)
        self.beta = None if beta is None else _to_arr(beta)
        self.dA = None if dA is None else _to_arr(dA)
        self.dR = None if dR is None else _to_arr(dR)
        self.dKB = None if dKB is None else _to_arr(dKB)

    @property
    def N(self) -> int:
        if self.r is None or self.r.ndim != 1:
            return 0
        return int(self.r.size)

    def validate(self) -> list[str]:
        p: list[str] = []
        if not isinstance(self.center, str) or self.center not in CENTERS:
            p.append(f"center must be one of {CENTERS}, got {self.center!r}")
        if self.r is None or self.r.ndim != 1:
            p.append("r: must be a 1-D float array")
            return p
        N = self.r.size
        if N < 8:
            p.append(f"N = {N} < 8 (contract section 8)")
        for nm in ("A", "R", "KA", "KB"):
            a = getattr(self, nm)
            if a is None or self._raw[nm] is None:
                p.append(f"{nm}: missing or not convertible to float")
            elif a.shape != (N,):
                p.append(f"{nm}: shape {a.shape} != ({N},)")
        for nm in ("alpha", "beta", "dA", "dR", "dKB"):
            if self._raw[nm] is not None:
                a = getattr(self, nm)
                if a is None:
                    p.append(f"{nm}: not convertible to float")
                elif a.shape != (N,):
                    p.append(f"{nm}: shape {a.shape} != ({N},)")
        if p:
            return p
        for nm in ("r", "A", "R", "KA", "KB", "alpha", "beta", "dA", "dR", "dKB"):
            a = getattr(self, nm)
            if a is not None and not np.all(np.isfinite(a)):
                bad = np.flatnonzero(~np.isfinite(a))[:10].tolist()
                p.append(f"{nm}: non-finite values at indices {bad}")
        if p:
            return p
        if np.any(np.diff(self.r) <= 0):
            p.append("r: must be strictly increasing")
        if np.any(self.A <= 0):
            p.append("A: must be > 0 everywhere")
        if np.any(self.R < 0):
            p.append("R: must be >= 0 (areal radius)")
        if self.alpha is not None and np.any(self.alpha <= 0):
            p.append("alpha: lapse must be > 0")
        c = self.center
        if c == "vertex":
            if self.r[0] != 0.0:
                p.append("center='vertex' requires r[0] == 0")
            if self.R[0] != 0.0:
                p.append("center='vertex' requires R[0] == 0 (regular centre)")
            if np.any(self.R[1:] <= 0):
                p.append("R must be > 0 away from the centre")
        elif c == "cell":
            if not self.r[0] > 0:
                p.append("center='cell' requires r[0] > 0")
            if np.any(self.R <= 0):
                p.append("R must be > 0 on a 'cell' grid")
        elif c == "none":
            if np.any(self.R <= 0):
                p.append("R must be > 0 when the inner edge is not a centre")
        return p

    def pi_from_dtphi(self, dtphi, phi):
        """Pi from d_t phi using this slice's alpha, beta and a 4th-order d_r phi."""
        from .fd import FDGrid
        from .matter import pi_from_dtphi
        if self.alpha is None or self.beta is None:
            raise ValueError("slice has no alpha/beta")
        g = FDGrid(self.r, self.center)
        dphi, _ = g.deriv(np.asarray(phi, dtype=float), +1, p=4, d=1)
        return pi_from_dtphi(dtphi, dphi, self.alpha, self.beta)

"""SphericalSlice: container + validation of the 3+1 slice data (contract §2, §3, §8, §9).

The constructor never raises on bad data: problems are recorded in ``errors`` (make the data
status INVALID_INPUT) and ``warnings`` (informational). ``diagnose`` reports them.
"""
from __future__ import annotations

import numpy as np

from .fd import EPS, RadialOperator

CENTERS = ("vertex", "cell", "none")
N_MIN = 8


def _as_float_array(x):
    """Convert to a float ndarray; return (array or None, error message or None)."""
    try:
        a = np.array(x, dtype=float)
    except (TypeError, ValueError) as exc:  # not numeric
        return None, f"cannot convert to float array ({exc})"
    return a, None


class SphericalSlice:
    """Spherically symmetric slice ds^2 = -a^2dt^2 + A^2(dr + b dt)^2 + R^2 dOmega^2.

    Parameters (contract §10): r, A, R, KA = K^r_r, KB = K^theta_theta, center, alpha, beta
    (only for pi_from_dtphi; not used by the diagnostics), optional analytic dA, dR, dKB.
    """

    def __init__(self, r, A, R, KA, KB, center="vertex", alpha=None, beta=None,
                 dA=None, dR=None, dKB=None):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.center = center
        self.structure_ok = True

        arrays = {}
        for name, val, required in (("r", r, True), ("A", A, True), ("R", R, True),
                                    ("KA", KA, True), ("KB", KB, True),
                                    ("alpha", alpha, False), ("beta", beta, False),
                                    ("dA", dA, False), ("dR", dR, False), ("dKB", dKB, False)):
            if val is None:
                arrays[name] = None
                continue
            a, err = _as_float_array(val)
            if err is not None:
                self.errors.append(f"{name}: {err}")
                self.structure_ok = False
                arrays[name] = None
                continue
            if name in ("alpha", "beta") and a.ndim == 0:
                a = np.full(np.shape(arrays["r"]) if arrays.get("r") is not None else (), float(a))
            arrays[name] = a

        rr = arrays["r"]
        if rr is None or rr.ndim != 1:
            self.errors.append("r must be a 1-D array")
            self.structure_ok = False
            N = int(rr.size) if (rr is not None and rr.ndim == 1) else 0
        else:
            N = rr.size
        self.N = N
        for name in ("A", "R", "KA", "KB", "dA", "dR", "dKB", "alpha", "beta"):
            a = arrays[name]
            if a is None:
                continue
            if a.ndim != 1 or a.size != N:
                msg = f"{name}: shape {a.shape} does not match r ({N},)"
                if name in ("alpha", "beta"):
                    # not used by the diagnostics: a warning, not an input error
                    self.warnings.append(msg + " (ignored)")
                    arrays[name] = None
                    continue
                self.errors.append(msg)
                self.structure_ok = False

        if center not in CENTERS:
            self.errors.append(f"center must be one of {CENTERS}, got {center!r}")
            self.structure_ok = False

        if self.structure_ok:
            if N < N_MIN:
                self.errors.append(f"N = {N} < {N_MIN} grid points")
                self.structure_ok = False
            elif not np.all(np.isfinite(rr)):
                self.errors.append("r contains non-finite values")
                self.structure_ok = False
            elif not np.all(np.diff(rr) > 0):
                self.errors.append("r is not strictly increasing")
                self.structure_ok = False

        if self.structure_ok and center == "vertex":
            scale = max(abs(rr[-1]), abs(rr[1]))
            if rr[0] != 0.0:
                if abs(rr[0]) <= 64 * EPS * scale:
                    self.warnings.append(f"vertex centre: r[0] = {rr[0]:.3e} snapped to 0")
                    rr = rr.copy()
                    rr[0] = 0.0
                else:
                    self.errors.append("center='vertex' requires r[0] = 0")
                    self.structure_ok = False
        if self.structure_ok and center == "cell" and not rr[0] > 0:
            self.errors.append("center='cell' requires r[0] > 0")
            self.structure_ok = False
        if self.structure_ok and center in ("vertex", "cell") and rr[0] < 0:
            self.structure_ok = False

        self.r = rr
        self.A = arrays["A"]
        self.R = arrays["R"]
        self.KA = arrays["KA"]
        self.KB = arrays["KB"]
        self.alpha = arrays["alpha"]
        self.beta = arrays["beta"]
        self.dA = arrays["dA"]
        self.dR = arrays["dR"]
        self.dKB = arrays["dKB"]

        # point-wise validity of the inputs used by the diagnostics
        self.bad_points = np.zeros(N, dtype=bool)
        if self.structure_ok:
            for name in ("A", "R", "KA", "KB", "dA", "dR", "dKB"):
                a = getattr(self, name)
                if a is None:
                    continue
                nf = ~np.isfinite(a)
                if nf.any():
                    self.errors.append(f"{name}: {int(nf.sum())} non-finite value(s)")
                    self.bad_points |= nf
            with np.errstate(invalid="ignore"):
                badA = np.isfinite(self.A) & ~(self.A > 0)
                badR = np.isfinite(self.R) & (self.R < 0)
            if badA.any():
                self.errors.append(f"A <= 0 at {int(badA.sum())} point(s)")
                self.bad_points |= badA
            if badR.any():
                self.errors.append(f"R < 0 at {int(badR.sum())} point(s)")
                self.bad_points |= badR
            if center == "vertex" and np.isfinite(self.R[0]) and self.R[0] != 0.0:
                rscale = np.nanmax(np.abs(self.R)) if np.isfinite(self.R).any() else 0.0
                if abs(self.R[0]) <= 64 * EPS * rscale:
                    self.warnings.append(f"vertex centre: R[0] = {self.R[0]:.3e} snapped to 0")
                    self.R = self.R.copy()
                    self.R[0] = 0.0
                else:
                    self.errors.append("center='vertex' requires R[0] = 0 (odd parity of R)")
                    self.bad_points[0] = True
            # alpha / beta: not used by the diagnostics -> warnings only
            if self.alpha is not None:
                if not np.all(np.isfinite(self.alpha)):
                    self.warnings.append("alpha has non-finite values (not used by diagnostics)")
                elif not np.all(self.alpha > 0):
                    self.warnings.append("alpha <= 0 somewhere (not used by diagnostics)")
            if self.beta is not None and not np.all(np.isfinite(self.beta)):
                self.warnings.append("beta has non-finite values (not used by diagnostics)")

        self.op = RadialOperator(self.r, center) if self.structure_ok else None

    @property
    def valid(self) -> bool:
        return self.structure_ok and not self.errors

    def __repr__(self):
        return (f"SphericalSlice(N={self.N}, center={self.center!r}, valid={self.valid}, "
                f"errors={len(self.errors)})")

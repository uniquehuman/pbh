"""Finite-difference machinery for sphdiag (contract §8).

* Fornberg (1988/1998) weights on arbitrary nodes, vectorised over target points.
* ``RadialOperator``: first derivatives of order 4 (5 points, primary) and 6 (7 points, error
  estimate only) on a possibly non-uniform grid r, with parity ghosts at a regular centre
  (``center="vertex"``: r[0] = 0; ``center="cell"``: r[0] > 0, centre at r = 0 off-grid) and
  one-sided (fully off-centred) stencils of the same width at the outer edge and for
  ``center="none"``.
* Cumulative integration from r[0] with the exact integral of the local cubic interpolant
  (order 4 on smooth data), using the same parity ghosts.

Parity convention: +1 = even in r, -1 = odd in r. Ghost values are f(-x) = parity * f(x).
"""
from __future__ import annotations

import numpy as np

EPS = float(np.finfo(float).eps)
N_GHOST = 3            # enough for the 7-point (order-6) centred stencil
ROUNDOFF_FACTOR = 1.0  # multiplier of the stencil round-off bound eps*sum|w_k f_k| added to sigma(f')


def fornberg_weights(z, x, m):
    """Fornberg weights.

    Parameters
    ----------
    z : (P,) evaluation points
    x : (P, n) stencil nodes for each evaluation point (distinct within a row)
    m : highest derivative order

    Returns
    -------
    c : (P, n, m+1); c[p, j, k] is the weight of node x[p, j] in the k-th derivative at z[p].
    """
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    P, n = x.shape
    c = np.zeros((P, n, m + 1))
    c1 = np.ones(P)
    c4 = x[:, 0] - z
    c[:, 0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2 = np.ones(P)
        c5 = c4
        c4 = x[:, i] - z
        for j in range(i):
            c3 = x[:, i] - x[:, j]
            c2 = c2 * c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[:, i, k] = c1 * (k * c[:, i - 1, k - 1] - c5 * c[:, i - 1, k]) / c2
                c[:, i, 0] = -c1 * c5 * c[:, i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[:, j, k] = (c4 * c[:, j, k] - k * c[:, j, k - 1]) / c3
            c[:, j, 0] = c4 * c[:, j, 0] / c3
        c1 = c2
    return c


class RadialOperator:
    """Derivative / integration operator on the grid r for a given centre type."""

    def __init__(self, r, center):
        r = np.asarray(r, dtype=float)
        self.r = r
        self.N = N = r.size
        self.center = center
        if center == "vertex":
            # ghosts at -r3, -r2, -r1 (r0 = 0 is its own mirror)
            self.ghost_src = np.array([3, 2, 1])
        elif center == "cell":
            # ghosts at -r2, -r1, -r0
            self.ghost_src = np.array([2, 1, 0])
        elif center == "none":
            self.ghost_src = np.array([], dtype=int)
        else:
            raise ValueError(f"unknown center {center!r}")
        self.g = self.ghost_src.size
        self.xe = np.concatenate([-r[self.ghost_src], r])
        self.Ne = self.xe.size
        self._stencils = {}
        for order in (4, 6):
            self._stencils[order] = self._build(3 if order == 4 else order + 1, deriv=1)
        self._int_idx, self._int_w = self._build_integration()

    # -- construction ------------------------------------------------------------------------
    def _window_starts(self, w, targets_ext):
        return np.clip(targets_ext - w // 2, 0, self.Ne - w)

    def _build(self, w, deriv):
        e = np.arange(self.N) + self.g
        starts = self._window_starts(w, e)
        idx = starts[:, None] + np.arange(w)[None, :]
        nodes = self.xe[idx]
        W = fornberg_weights(self.r, nodes, deriv)[:, :, deriv]
        return idx, W

    def _build_integration(self):
        """Weights for int_{r_j}^{r_{j+1}} f dr from the cubic through 4 neighbouring nodes."""
        N = self.N
        e = np.arange(N - 1) + self.g              # extended index of left end of interval
        starts = np.clip(e - 1, 0, self.Ne - 4)
        idx = starts[:, None] + np.arange(4)[None, :]
        x0 = self.xe[e]
        h = self.xe[e + 1] - x0
        t = (self.xe[idx] - x0[:, None]) / h[:, None]          # (N-1, 4) local coordinates
        V = t[:, None, :] ** np.arange(4)[None, :, None]       # V[p, m, k] = t_k^m
        mom = 1.0 / (np.arange(4) + 1.0)                       # int_0^1 t^m dt
        rhs = np.broadcast_to(mom, (N - 1, 4))[..., None]
        w = np.linalg.solve(V, rhs)[..., 0] * h[:, None]
        return idx, w

    # -- application -------------------------------------------------------------------------
    def extend(self, f, parity):
        f = np.asarray(f, dtype=float)
        if self.g == 0:
            return f
        if parity not in (1, -1):
            raise ValueError("parity (+1 even / -1 odd) is required with a regular centre")
        return np.concatenate([parity * f[self.ghost_src], f])

    def d(self, f, parity, order=4):
        """First derivative of order 4 or 6."""
        idx, W = self._stencils[order]
        fe = self.extend(f, parity)
        return np.sum(W * fe[idx], axis=1)

    def abs_apply(self, f, parity, order=4):
        """sum_k |w_k| |f_k| -- magnitude scale of the stencil sum (round-off floor)."""
        idx, W = self._stencils[order]
        fe = np.abs(self.extend(f, parity))
        return np.sum(np.abs(W) * fe[idx], axis=1)

    def d_err(self, f4, f6, parity, ef=None):
        """Derivative D4(f4) and error estimate |D4 f4 - D6 f6| + round-off floor.

        f4, f6 : the field as computed at order 4 / order 6 (identical for input fields)
        ef     : pointwise round-off magnitude of f4 (default: eps*|f4|)
        """
        d4 = self.d(f4, parity, 4)
        d6 = self.d(f6, parity, 6)
        mag = EPS * np.abs(np.asarray(f4, dtype=float))
        if ef is not None:
            mag = mag + np.abs(ef)
        rnd = ROUNDOFF_FACTOR * self.abs_apply(mag, 1, 4)
        return d4, np.abs(d4 - d6) + rnd

    def cumint(self, g, parity=1):
        """Cumulative integral int_{r[0]}^{r_i} g dr (array of length N, first entry 0)."""
        ge = self.extend(g, parity)
        seg = np.sum(self._int_w * ge[self._int_idx], axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

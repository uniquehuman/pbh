"""Finite-difference machinery: vectorised Fornberg weights, parity ghosts, one-sided
boundary stencils, 4th-order cumulative quadrature (contract section 8)."""
from __future__ import annotations

import numpy as np

EPS = np.finfo(float).eps
#: multiplier of the rounding-error estimate  C*eps*sum_j |w_j| |f_j|  added to sigma(f')
ROUND_C = 8.0
N_GHOST = 4
CHUNK = 16384          # rows per Fornberg batch (memory bound for N ~ 1e5)


def fornberg_weights(z: np.ndarray, x: np.ndarray, m: int) -> np.ndarray:
    """Fornberg (1988/1998) weights, vectorised over rows.

    z: (N,) evaluation points; x: (N, n) stencil nodes (row-wise).
    Returns c: (m+1, N, n); c[k, i, j] is the weight of f(x[i, j]) in d^k f/dx^k (z[i]).
    """
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    N, n = x.shape
    c = np.zeros((m + 1, N, n))
    c1 = np.ones(N)
    c4 = x[:, 0] - z
    c[0, :, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2 = np.ones(N)
        c5 = c4
        c4 = x[:, i] - z
        for j in range(i):
            c3 = x[:, i] - x[:, j]
            c2 = c2 * c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, :, i] = c1 * (k * c[k - 1, :, i - 1] - c5 * c[k, :, i - 1]) / c2
                c[0, :, i] = -c1 * c5 * c[0, :, i - 1] / c2
            for k in range(mn, 0, -1):
                c[k, :, j] = (c4 * c[k, :, j] - k * c[k - 1, :, j]) / c3
            c[0, :, j] = c4 * c[0, :, j] / c3
        c1 = c2
    return c


class FDGrid:
    """Radial grid with optional reflection ghosts.

    center = 'vertex' : r[0] = 0, ghosts at -r[1..G]
    center = 'cell'   : r[0] > 0, ghosts at -r[0..G-1]
    center = 'none'   : no ghosts, one-sided stencils at both edges
    A field is extended with a parity (+1 even, -1 odd); parity is ignored for 'none'.
    """

    def __init__(self, r: np.ndarray, center: str):
        self.r = np.asarray(r, dtype=float)
        self.N = self.r.size
        self.center = center
        if center == 'vertex':
            self.src = np.arange(N_GHOST, 0, -1)
        elif center == 'cell':
            self.src = np.arange(N_GHOST - 1, -1, -1)
        else:
            self.src = np.zeros(0, dtype=int)
        self.ng = self.src.size
        self.re = np.concatenate([-self.r[self.src], self.r])
        self._ops: dict = {}
        self._quad = None

    # ------------------------------------------------------------------ helpers
    def extend(self, f: np.ndarray, parity: int) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        if self.ng == 0:
            return f
        return np.concatenate([parity * f[self.src], f])

    def _windows(self, size_c: int, size_s: int):
        """Index windows (N, size) into the extended grid for every real point.

        Symmetric window of size_c (odd) where it fits, otherwise a shifted window of size_s.
        Returned arrays are padded to max(size_c, size_s) with weight-0 duplicates."""
        Ne = self.re.size
        e = np.arange(self.N) + self.ng
        half = size_c // 2
        fits = (e - half >= 0) & (e + half <= Ne - 1)
        width = max(size_c, size_s)
        idx = np.zeros((self.N, width), dtype=int)
        used = np.zeros(self.N, dtype=int)
        # centred
        base_c = e - half
        start_s = np.clip(e - size_s // 2, 0, Ne - size_s)
        for k in range(width):
            ic = np.where(k < size_c, base_c + min(k, size_c - 1), 0)
            is_ = np.where(k < size_s, start_s + min(k, size_s - 1), 0)
            idx[:, k] = np.where(fits, ic, is_)
        used[:] = np.where(fits, size_c, size_s)
        return idx, used

    def op(self, p: int, d: int):
        key = (p, d)
        if key not in self._ops:
            size_c = p + 1                    # symmetric: order p for d = 1, 2 (smooth grids)
            size_s = p + d                    # one-sided / shifted: formal order p
            if self.re.size < size_s:
                raise ValueError(f"grid too small for order {p} derivative {d}")
            idx, used = self._windows(size_c, size_s)
            width = idx.shape[1]
            W = np.zeros((self.N, width))
            z = self.r
            for u in np.unique(used):
                rows = np.flatnonzero(used == u)
                for ch in range(0, rows.size, CHUNK):          # bound temporary memory for large N
                    rr = rows[ch:ch + CHUNK]
                    W[rr, :u] = fornberg_weights(z[rr], self.re[idx[rr, :u]], d)[d]
            self._ops[key] = (idx, W)
        return self._ops[key]

    def deriv(self, f: np.ndarray, parity: int, p: int = 4, d: int = 1, abs_err=None):
        """d-th derivative with accuracy order p. Returns (value, rounding estimate).

        abs_err: per-point estimate of the absolute rounding error of f
        (default ROUND_C*eps*|f|); rounding estimate of the result = sum_j |w_j| abs_err_j
        (+ ROUND_C*eps*sum_j |w_j f_j| for the summation itself)."""
        idx, W = self.op(p, d)
        fe = self.extend(f, parity)
        terms = W * fe[idx]
        val = np.sum(terms, axis=1)
        rnd = ROUND_C * EPS * np.sum(np.abs(terms), axis=1)
        if abs_err is not None:
            ee = self.extend(np.abs(abs_err), 1)
            rnd = rnd + np.sum(np.abs(W) * ee[idx], axis=1)
        return val, rnd

    # ------------------------------------------------------------------ grid-scale noise
    def noise_amplitude(self, f: np.ndarray, parity: int) -> np.ndarray:
        """Per-point amplitude of non-smooth (grid-scale) content of f.

        Uses the 7-point undivided 6th difference h^6 D^(6) f (annihilates polynomials of degree <= 5
        on any grid; for white noise of std s its std is sqrt(924) s on a uniform grid), normalised by
        sqrt(924) and maximised over +-3 neighbours.  For smooth f it is O(h^6 f^(6)) -- negligible."""
        if "noise" not in self._ops:
            idx, _ = self._windows(7, 7)
            c = np.empty(idx.shape)
            for ch in range(0, self.N, CHUNK):
                sl = slice(ch, ch + CHUNK)
                c[sl] = fornberg_weights(self.r[sl], self.re[idx[sl]], 6)[6]
            h = np.abs(self.re[idx[:, -1]] - self.re[idx[:, 0]]) / 6.0
            self._ops["noise"] = (idx, c * h[:, None] ** 6 / np.sqrt(924.0))
        idx, W = self._ops["noise"]
        a = np.abs(np.sum(W * self.extend(f, parity)[idx], axis=1))
        pad = np.concatenate([np.full(3, a[0]), a, np.full(3, a[-1])])
        return np.max(np.lib.stride_tricks.sliding_window_view(pad, 7), axis=1)

    def noise_gain(self, p: int, d: int) -> np.ndarray:
        """sqrt(sum_j w_j^2): amplification of uncorrelated point errors by the (p, d) stencil."""
        _, W = self.op(p, d)
        return np.sqrt(np.sum(W ** 2, axis=1))

    # ------------------------------------------------------------------ quadrature
    def _quad_weights(self):
        """Weights for integral over [r_i, r_{i+1}] of the cubic through 4 neighbouring
        (extended-grid) points: 4th-order accurate on any smooth grid."""
        if self._quad is None:
            Ne = self.re.size
            e = np.arange(self.N - 1) + self.ng
            start = np.clip(e - 1, 0, Ne - 4)
            idx = start[:, None] + np.arange(4)[None, :]
            h = self.re[e + 1] - self.re[e]
            u = (self.re[idx] - self.re[e][:, None]) / h[:, None]      # nodes in units of h
            V = np.stack([u**k for k in range(4)], axis=1)              # (M, 4, 4): V[k, j] = u_j^k
            mom = np.array([1.0 / (k + 1) for k in range(4)])
            w = np.linalg.solve(V, np.broadcast_to(mom, (V.shape[0], 4))[..., None])[..., 0]
            self._quad = (idx, w * h[:, None])
        return self._quad

    def cumint(self, f: np.ndarray, parity: int = 1) -> np.ndarray:
        """Cumulative integral from r[0]: I[0] = 0, I[i] = int_{r0}^{r_i} f dr."""
        idx, w = self._quad_weights()
        fe = self.extend(f, parity)
        inc = np.sum(w * fe[idx], axis=1)
        return np.concatenate([[0.0], np.cumsum(inc)])


def lagrange_eval(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    """Evaluate the interpolating polynomial through (xs, ys) at x."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    tot = 0.0
    for j in range(xs.size):
        lj = 1.0
        for k in range(xs.size):
            if k != j:
                lj *= (x - xs[k]) / (xs[j] - xs[k])
        tot += ys[j] * lj
    return float(tot)


def even_extrapolate_to_zero(r: np.ndarray, f: np.ndarray) -> float:
    """Extrapolate an even function of r to r = 0 by a polynomial in r^2 through the given points."""
    return lagrange_eval(np.asarray(r, dtype=float) ** 2, f, 0.0)

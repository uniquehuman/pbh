"""Error norms (own implementation): L_inf and L2 with proper-volume measure dV = 4 pi R^2 A dr
(trapezoid rule in r), per region. NaN entries (e.g. R = 0 for H, M) are excluded."""
import numpy as np

from .grids import regions


def l2vol(err, r, R, A, mask, normalized=False):
    m = mask & np.isfinite(err)
    idx = np.where(m)[0]
    if idx.size < 2:
        return float("nan") if idx.size == 0 else float(abs(err[idx[0]]))
    # trapezoid on the (possibly non-contiguous) selected points: use contiguous runs
    w = 4 * np.pi * R**2 * A
    tot = 0.0
    vol = 0.0
    for k in range(idx.size - 1):
        i, j = idx[k], idx[k + 1]
        if j != i + 1:
            continue
        dr = r[j] - r[i]
        tot += 0.5 * dr * (err[i]**2 * w[i] + err[j]**2 * w[j])
        vol += 0.5 * dr * (w[i] + w[j])
    if normalized:
        return float(np.sqrt(tot / vol)) if vol > 0 else float("nan")
    return float(np.sqrt(tot))


def norms(err, r, R, A, center):
    reg = regions(len(r), R, center)
    out = {}
    for name, mask in reg.items():
        e = err[mask]
        e = e[np.isfinite(e)]
        out[name] = {
            "Linf": float(np.max(np.abs(e))) if e.size else float("nan"),
            "L2vol": l2vol(err, r, R, A, mask),
        }
    return out


def observed_order(e_coarse, e_fine, h_coarse, h_fine):
    if not (e_coarse > 0 and e_fine > 0):
        return float("nan")
    return float(np.log(e_coarse / e_fine) / np.log(h_coarse / h_fine))


def cum_cubic4(f, r):
    """Cumulative integral with the contract-1.1 rule 'cubic through 4 neighbouring nodes' (own code):
    on [r_i, r_{i+1}] the Lagrange cubic through nodes i-1..i+2 (shifted at the ends) is integrated
    exactly by 2-point Gauss-Legendre.  Used only for the quadrature tolerance envelope (plan v2)."""
    n = len(r)
    out = np.zeros(n)
    g = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    for i in range(n - 1):
        s = min(max(i - 1, 0), n - 4)
        xs, fs = r[s:s + 4], f[s:s + 4]
        a, b = r[i], r[i + 1]
        tot = 0.0
        for gx in g:
            x = 0.5 * (a + b) + 0.5 * (b - a) * gx
            L = [np.prod([(x - xs[m]) / (xs[k] - xs[m]) for m in range(4) if m != k]) for k in range(4)]
            tot += 0.5 * (b - a) * np.dot(L, fs)
        out[i + 1] = out[i] + tot
    return out

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

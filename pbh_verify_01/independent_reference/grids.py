"""Radial grids used by the independent tests (own code).

All grids are strictly increasing arrays r.  Stretched grids use r = f(x) with x uniform; the
maps are odd in x for 'vertex'/'cell' grids so that the reflection r -> -r used for parity ghost
points at a regular centre corresponds to x -> -x.
"""
import numpy as np


def x_nodes(N, center):
    if center == "cell":
        return (np.arange(N) + 0.5) / N
    return np.linspace(0.0, 1.0, N)


def sinh_map(L, beta):
    """odd map r = L sinh(beta x)/sinh(beta); returns (f, fprime)."""
    s = np.sinh(beta)
    return (lambda x: L * np.sinh(beta * x) / s), (lambda x: L * beta * np.cosh(beta * x) / s)


def exp_map(r0, r1, beta):
    """map for 'none' grids: r = r0 + (r1-r0)(e^{beta x}-1)/(e^beta-1)."""
    d = np.expm1(beta)
    return (lambda x: r0 + (r1 - r0) * np.expm1(beta * x) / d), (lambda x: (r1 - r0) * beta * np.exp(beta * x) / d)


def make_grid(kind, N, L=None, r0=None, r1=None, beta=2.0, center="vertex"):
    """kind in {'uni','str'}; center in {'vertex','cell','none'}.
    vertex/cell: domain [0, L]; none: [r0, r1]."""
    x = x_nodes(N, center)
    if center in ("vertex", "cell"):
        if kind == "uni":
            r = L * x
        else:
            f, _ = sinh_map(L, beta)
            r = f(x)
        if center == "vertex":
            r[0] = 0.0
    else:
        if kind == "uni":
            r = r0 + (r1 - r0) * x
        else:
            f, _ = exp_map(r0, r1, beta)
            r = f(x)
        r[0], r[-1] = r0, r1
    return r


def h_local(r):
    d = np.diff(r)
    h = np.empty_like(r)
    h[0], h[-1] = d[0], d[-1]
    h[1:-1] = np.maximum(d[:-1], d[1:])
    return h


def regions(N, R, center):
    """Index masks per contract section 9 region definitions (own implementation)."""
    idx = np.arange(N)
    pos = np.where(R > 0)[0] if center == "vertex" else idx
    nc = max(4, N // 20)
    c = np.zeros(N, bool)
    c[pos[:nc]] = True
    o = np.zeros(N, bool)
    o[-4:] = True
    b = ~(c | o)
    if center == "vertex":
        b[R <= 0] = False
    a = np.ones(N, bool)
    return {"all": a, "center": c, "bulk": b, "outer": o}

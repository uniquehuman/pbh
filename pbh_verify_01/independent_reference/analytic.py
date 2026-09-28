"""Closed-form slices (own derivations; K_ij of the physical slices verified from the 4-metric in
derive_sympy.py RS2).  Each slice is a SymSlice built from sympy expressions in r; values and exact
analytic derivatives are obtained by lambdify, with sympy limits used where an expression is 0/0 at
r = 0.

Nothing here imports sphdiag.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYDEPS = os.path.join(os.path.dirname(_HERE), ".pydeps")
if _PYDEPS not in sys.path:
    sys.path.append(_PYDEPS)

import sympy as sp  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from .constraints_ref import exact_diag  # noqa: E402

r = sp.symbols("r", real=True)
x = sp.symbols("x", real=True)


class _Fn:
    """lambdified expression with limit fallback at r = 0."""

    def __init__(self, expr):
        self.expr = sp.sympify(expr)
        self.f = sp.lambdify(r, self.expr, "numpy")
        self._lim0 = None

    def lim0(self):
        if self._lim0 is None:
            try:
                v = float(sp.limit(self.expr, r, 0, "+"))
            except Exception:
                v = float("nan")
            self._lim0 = v
        return self._lim0

    def __call__(self, rr):
        rr = np.asarray(rr, float)
        with np.errstate(all="ignore"):
            v = np.broadcast_to(np.asarray(self.f(rr), float), rr.shape).copy()
        z = rr == 0
        if np.any(z):
            v[z] = self.lim0()
        return v


class SymSlice:
    def __init__(self, name, A, R, KA, KB, rho=0, jr=0, Srr=0, Sthth=0, phi=None, Pi=None, m=None,
                 note=""):
        self.name = name
        self.note = note
        self.e = dict(A=sp.sympify(A), R=sp.sympify(R), KA=sp.sympify(KA), KB=sp.sympify(KB))
        self.scalar = phi is not None
        if self.scalar:
            phi, Pi, m = sp.sympify(phi), sp.sympify(Pi), sp.sympify(m)
            dphi = sp.diff(phi, r)
            Phi = dphi / self.e["A"]
            V = m**2 * phi**2 / 2
            rho = Pi**2 / 2 + Phi**2 / 2 + V
            jr = -Pi * dphi
            Srr = Pi**2 / 2 + Phi**2 / 2 - V
            Sthth = Pi**2 / 2 - Phi**2 / 2 - V
            self.e.update(phi=phi, Pi=Pi, dphi=dphi)
            self.m = float(m)
        self.e.update(rho=sp.sympify(rho), jr=sp.sympify(jr), Srr=sp.sympify(Srr), Sthth=sp.sympify(Sthth))
        e = self.e
        e["dA"] = sp.diff(e["A"], r)
        e["dR"] = sp.diff(e["R"], r)
        e["d2R"] = sp.diff(e["R"], r, 2)
        e["dKB"] = sp.diff(e["KB"], r)
        G = e["dR"] / e["A"]
        U = -e["R"] * e["KB"]
        e["M"] = e["R"] / 2 * (1 - G**2 + U**2)
        self.fn = {k: _Fn(v) for k, v in e.items()}
        self._MR3 = _Fn(e["M"] / e["R"]**3)

    def fields(self, rr):
        return {k: f(rr) for k, f in self.fn.items()}

    def exact(self, rr, with_matter=True):
        v = self.fields(rr)
        d = exact_diag(v["A"], v["dA"], v["R"], v["dR"], v["d2R"], v["KA"], v["KB"], v["dKB"],
                       v["rho"] if with_matter else None, v["jr"] if with_matter else None)
        z = v["R"] == 0
        if np.any(z):
            d["M"][z] = 0.0
            d["M_over_R3"][z] = self._MR3.lim0()
        d.update({k: v[k] for k in ("A", "R", "KA", "KB", "rho", "jr", "Srr", "Sthth", "dKB", "dA", "d2R")})
        if self.scalar:
            d.update(phi=v["phi"], Pi=v["Pi"], dphi=v["dphi"])
        return d

    def rtheta_fn(self, which, k=0):
        """k-th r-derivative of the exact R theta_pm as a callable (sympy)."""
        key = (which, k)
        if not hasattr(self, "_rt"):
            self._rt = {}
        if key not in self._rt:
            e = self.e
            ex = 2 * (-e["R"] * e["KB"] + (1 if which == "+" else -1) * e["dR"] / e["A"])
            self._rt[key] = _Fn(sp.diff(ex, r, k) if k else ex)
        return self._rt[key]

    def R_fn(self, k=0):
        return _Fn(sp.diff(self.e["R"], r, k) if k else self.e["R"])

    def M_over_R3_at0(self):
        return self._MR3.lim0()

    def roots(self, which, rmin, rmax, n=20001):
        """Exact roots of R theta_pm in (rmin, rmax) by sign scan + brentq."""
        key = "Rtp" if which == "+" else "Rtm"

        def f(q):
            return float(self.exact(np.array([q]), with_matter=False)[key][0])

        rr = np.linspace(rmin, rmax, n)
        vals = self.exact(rr, with_matter=False)[key]
        out = []
        for i in range(n - 1):
            if vals[i] == 0:
                out.append(rr[i])
            elif vals[i] * vals[i + 1] < 0:
                out.append(brentq(f, rr[i], rr[i + 1], xtol=1e-15, rtol=4e-16, maxiter=200))
        return out

    def transformed(self, fmap, name=None):
        """Same slice in coordinate x with r = fmap(x) (sympy expr in x); returns SymSlice in variable r."""
        fx = sp.sympify(fmap)
        fpx = sp.diff(fx, x)
        sub = lambda ex: ex.subs(r, fx).subs(x, r)  # noqa: E731
        kw = {}
        if self.scalar:
            kw = dict(phi=sub(self.e["phi"]), Pi=sub(self.e["Pi"]), m=self.m)
        else:
            kw = dict(rho=sub(self.e["rho"]), jr=sub(self.e["jr"] * fpx), Srr=sub(self.e["Srr"]),
                      Sthth=sub(self.e["Sthth"]))
        return SymSlice(name or self.name + "_x", sub(self.e["A"] * fpx), sub(self.e["R"]), sub(self.e["KA"]),
                        sub(self.e["KB"]), **kw)


# ---------------------------------------------------------------- catalogue of exact slices

def minkowski_trivial():
    return SymSlice("minkowski", 1, r, 0, 0)


def minkowski_height(b=0.8, w=1.0):
    b, w = sp.nsimplify(b), sp.nsimplify(w)
    T = b * w * sp.exp(-r**2 / w**2)
    Tp = sp.diff(T, r)
    N = 1 / sp.sqrt(1 - Tp**2)
    return SymSlice("minkowski_height", 1 / N, r, -sp.diff(N * Tp, r), -N * Tp / r,
                    note="t = T(r) = b w exp(-r^2/w^2) in Minkowski")


def milne_chi(tau=2.0):
    tau = sp.nsimplify(tau)
    return SymSlice("milne_chi", tau, tau * sp.sinh(r), -1 / tau, -1 / tau)


def milne_areal(tau=2.0):
    tau = sp.nsimplify(tau)
    return SymSlice("milne_areal", tau / sp.sqrt(r**2 + tau**2), r, -1 / tau, -1 / tau)


def schwarzschild_pg(M=1.0):
    M = sp.nsimplify(M)
    return SymSlice("schw_PG", 1, r, -sp.sqrt(2 * M / r) / (2 * r), sp.sqrt(2 * M / r) / r)


def schwarzschild_ks(M=1.0):
    M = sp.nsimplify(M)
    A = sp.sqrt(1 + 2 * M / r)
    return SymSlice("schw_KS", A, r, -2 * M * (r + M) / (r**3 * (1 + 2 * M / r)**sp.Rational(3, 2)),
                    2 * M / (r**2 * A))


def schwarzschild_iso(M=1.0):
    M = sp.nsimplify(M)
    psi = 1 + M / (2 * r)
    return SymSlice("schw_iso", psi**2, psi**2 * r, 0, 0)


def flrw(a=1.3, H=0.4, p=0.0):
    a, H = sp.nsimplify(a), sp.nsimplify(H)
    rho = 3 * H**2 / (8 * sp.pi)
    return SymSlice("flrw", a, a * r, -H, -H, rho=rho, jr=0, Srr=p, Sthth=p)


def synthetic_pbh_flrw(shells=((1.0, 3.0),), s=0.7, h=0.05):
    """Geometry-only snapshot: A = 1 + 0.1 e^{-r^2}, R = r, K_B = sum k[e^{-(r-c)^2/s^2}+e^{-(r+c)^2/s^2}] - h."""
    s, h = sp.nsimplify(s), sp.nsimplify(h)
    KB = -h
    for k, c in shells:
        k, c = sp.nsimplify(k), sp.nsimplify(c)
        KB = KB + k * (sp.exp(-(r - c)**2 / s**2) + sp.exp(-(r + c)**2 / s**2))
    KA = sp.Rational(1, 20) * sp.cos(r / 2) - h
    return SymSlice("synthetic_pbh_flrw", 1 + sp.Rational(1, 10) * sp.exp(-r**2), r, KA, KB)


def polynomial_slice(c3=0.05, a2=0.1, a4=-0.004, k0=0.07, k2=-0.01, k4=0.0005, irregular=0.0):
    """Degree <= 4 data: 5-point FD exact for R', A', K_B' (V03/Z80). K_A = K_B + 0.03 r^2 (+ irregular e^{-r^2})."""
    c3, a2, a4, k0, k2, k4 = (sp.nsimplify(v) for v in (c3, a2, a4, k0, k2, k4))
    R = r + c3 * r**3
    A = 1 + a2 * r**2 + a4 * r**4
    KB = k0 + k2 * r**2 + k4 * r**4
    KA = KB + sp.Rational(3, 100) * r**2 + sp.nsimplify(irregular) * sp.exp(-r**2)
    return SymSlice("poly", A, R, KA, KB, rho=sp.Rational(1, 50) * (1 + r**2), jr=sp.Rational(1, 100) * r,
                    Srr=0, Sthth=0)


def polynomial_scalar(A_coef=(1, 0.1, -0.004), phi_coef=(0.05, -0.01, 0.0007), Pi_coef=(0.03, 0.002, -0.0001),
                      c3=0.05, m=1.3):
    """Scalar field with polynomial (deg<=4, even) phi, Pi; A even poly, R = r + c3 r^3 (V02)."""
    q = [sp.nsimplify(v) for v in A_coef]
    A = q[0] + q[1] * r**2 + q[2] * r**4
    p = [sp.nsimplify(v) for v in phi_coef]
    P = [sp.nsimplify(v) for v in Pi_coef]
    phi = p[0] + p[1] * r**2 + p[2] * r**4
    Pi = P[0] + P[1] * r**2 + P[2] * r**4
    return SymSlice("poly_scalar", A, r + sp.nsimplify(c3) * r**3, sp.Rational(1, 20), sp.Rational(1, 20),
                    phi=phi, Pi=Pi, m=sp.nsimplify(m))


def manufactured():
    """D90 transcendental manufactured slice with arbitrary matter projections."""
    A = sp.Rational(6, 5) + sp.Rational(3, 10) * sp.cos(sp.Rational(13, 10) * r) * sp.exp(-r**2 / 10)
    R = r * (1 + sp.Rational(3, 20) * sp.sin(sp.Rational(7, 10) * r)**2)
    KA = sp.Rational(1, 20) + sp.Rational(1, 10) * sp.cos(sp.Rational(9, 10) * r)
    KB = sp.Rational(2, 25) * sp.exp(-r**2 / 5) - sp.Rational(1, 50) * sp.cos(r / 2)
    rho = sp.Rational(1, 50) * sp.exp(-r**2 / 3)
    jr = sp.Rational(1, 100) * r * sp.exp(-r**2 / 4)
    return SymSlice("manufactured", A, R, KA, KB, rho=rho, jr=jr,
                    Srr=sp.Rational(1, 100) * sp.exp(-r**2), Sthth=sp.Rational(1, 200))


SINH_MAP = lambda L, beta: L * sp.sinh(beta * x) / sp.sinh(beta)  # noqa: E731

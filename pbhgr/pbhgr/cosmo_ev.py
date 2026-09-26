"""Spherically symmetric Einstein-Vlasov solver in the areal-CMC gauge with an expanding FLRW exterior
(Stage A step 2; equations in docs/areal_cmc_gauge.md).

Metric   ds^2 = -alpha^2 dt^2 + A^2 (dR + beta dt)^2 + R^2 dOmega^2,   slicing K = -3 H(t) (CMC).
FLRW     alpha = 1, A = 1, beta = -H R.   Horizon-penetrating: 1/A^2 = 1 - 2m/R + R^2 Kth^2 > 0 also inside 2m/R > 1.
Solves per time step (fully constrained, no evolved metric variable):
    Kth(R)  = K/3 - (1/R^3) sum_{R_i<R} N_i P_i R_i                       (momentum constraint + CMC)
    dm/dR   = 4 pi R^2 (E - R Kth A J) = sE/dR sqrt(u) - R Kth sJ/dR      (Hamiltonian constraint), u = 1/A^2
    1/A^2   = 1 - 2m/R + R^2 Kth^2
    (1/(A R^2)) d/dR (R^2 alpha'/A) = alpha [KR^2 + 2 Kth^2 + 4 pi (E + S)] + 3 Hdot,  alpha'(0) = 0, alpha(R_out) = 1
    beta    = alpha R Kth,   KR = K - 2 Kth
Particles (unit rest mass, normal-frame radial momentum P, angular momentum squared Lsq, weight N, proper time tau):
    W = sqrt(1 + P^2 + Lsq/R^2)
    dR/dt   = alpha P/(A W) - beta,          dx/dt = (dR/dt - H R)/a   (comoving-areal grid coordinate x = R/a)
    dP/dt   = -(W/A) alpha' + alpha P KR + alpha Lsq/(W A R^3)
    dtau/dt = alpha/W
Sources per cell of proper volume 4 pi R^2 A dR:  E = sum N W,  J = sum N P,  S = p + 2q = sum N P^2/W + sum N Lsq/(R^2 W).
Apparent horizon: 2m/R = 1 with Kth > 0 (theta_+ = 0, collapsing); the cosmological anti-trapped surface has Kth < 0.
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

FOURPI = 4.0 * np.pi

_lib = None
_so = Path(__file__).with_name("_kernels.so")
if _so.exists():
    _lib = ctypes.CDLL(str(_so))
    if hasattr(_lib, "mass_integrate_cmc") and hasattr(_lib, "shell_constraints"):
        _lib.mass_integrate_cmc.restype = ctypes.c_int
        _lib.mass_integrate_cmc.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_double)] * 5 + [ctypes.c_int]
        _lib.stiff_substeps.restype = ctypes.c_int
        _lib.stiff_substeps.argtypes = ([ctypes.c_int, ctypes.c_int] + [ctypes.POINTER(ctypes.c_double)] * 5 + [ctypes.c_int, ctypes.c_double]
                                        + [ctypes.POINTER(ctypes.c_double)] * 4 + [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
                                        + [ctypes.POINTER(ctypes.c_double)] * 5)
        _lib.shell_constraints.restype = ctypes.c_int
        _lib.shell_constraints.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_double)] * 3 + [ctypes.c_double] + [ctypes.POINTER(ctypes.c_double)] * 4
    else:
        _lib = None


# ----------------------------------------------------------------------------- background
@dataclass
class Background:
    """Matter-dominated FLRW exterior in cosmic time: a = (t/t_i)^{2/3}, H = 2/(3t)."""
    H_i: float = 5.0

    def __post_init__(self):
        self.t_i = 2.0 / (3.0 * self.H_i)

    def a(self, t):    return (t / self.t_i) ** (2.0 / 3.0)
    def H(self, t):    return 2.0 / (3.0 * t)
    def K(self, t):    return -3.0 * self.H(t)
    def Hdot(self, t): return -2.0 / (3.0 * t**2)
    def alpha_out(self, m_out, R_out): return 1.0          # FLRW exterior


@dataclass
class StaticBackground:
    """Asymptotically flat, maximal slicing K = 0 (isolated benchmarks): a = 1, H = 0, alpha(R_out) = sqrt(1 - 2M/R_out)."""
    t_i: float = 0.0

    def a(self, t):    return 1.0
    def H(self, t):    return 0.0
    def K(self, t):    return 0.0
    def Hdot(self, t): return 0.0
    def alpha_out(self, m_out, R_out): return float(np.sqrt(max(1.0 - 2.0 * m_out / R_out, 1e-14)))


# ----------------------------------------------------------------------------- grid & particles
@dataclass
class CosmoGrid:
    """Grid in the comoving-areal coordinate x = R/a.  stretch = 0: uniform; stretch = s > 0: sinh stretching
    x_j = x_out sinh(s j/K)/sinh(s), smallest cell x_out s/(K sinh s) at the centre, largest ~ x_out s/K outside."""
    x_out: float
    K: int
    stretch: float = 0.0

    def __post_init__(self):
        if self.stretch > 0:
            j = np.arange(self.K + 1) / self.K
            self.x_edge = self.x_out * np.sinh(self.stretch * j) / np.sinh(self.stretch)
            self.x_edge[0] = 0.0; self.x_edge[-1] = self.x_out
        else:
            self.x_edge = np.linspace(0.0, self.x_out, self.K + 1)
        self.dxc = np.diff(self.x_edge)
        self.dx = float(self.dxc.min())
        self.x_c = 0.5 * (self.x_edge[1:] + self.x_edge[:-1])
        self.dcc = np.empty(self.K + 1)                       # centre-to-centre spacing at edges (BVP fluxes)
        self.dcc[1:-1] = self.x_c[1:] - self.x_c[:-1]
        self.dcc[0] = self.x_c[0]; self.dcc[-1] = self.x_edge[-1] - self.x_c[-1]

    def cell_of(self, x):
        return np.clip(np.searchsorted(self.x_edge, x, side="right") - 1, 0, self.K - 1)


@dataclass
class CParticles:
    x: np.ndarray
    P: np.ndarray
    Lsq: np.ndarray
    N: np.ndarray
    tau: np.ndarray = None

    def __post_init__(self):
        if self.tau is None:
            self.tau = np.zeros(len(self.x))

    def __len__(self):
        return len(self.x)


@dataclass
class CMetric:
    t: float
    a: float
    R_edge: np.ndarray
    R_c: np.ndarray
    dR: float
    m_edge: np.ndarray
    Kth_edge: np.ndarray
    u_edge: np.ndarray
    A_edge: np.ndarray
    alpha_c: np.ndarray
    alpha_edge: np.ndarray
    dalpha_edge: np.ndarray
    dA_edge: np.ndarray
    E_c: np.ndarray
    S_c: np.ndarray
    n_bad: int
    S: dict
    p_m: np.ndarray = None       # per-particle Misner-Sharp mass at the shell (m_in + dm/2), original particle order
    p_A: np.ndarray = None       # per-particle A = 1/sqrt(u)
    p_Kth: np.ndarray = None     # per-particle K^theta_theta


# ----------------------------------------------------------------------------- deposition
def deposit(grid: CosmoGrid, x, vals, order=None):
    """Cumulative deposition: the running sum of each quantity over sorted shells (half of a shell's own value at
    its position) is interpolated linearly in the volume variable v = x^3 and differenced at the cell edges.
    Exact for uniform density with shells at volume midpoints, continuous as shells move (no jumps when a shell
    crosses a cell edge), and unbiased at the regular centre.  Returns the per-cell values and the cumulative
    function at the edges (C(edge_0) = 0)."""
    if order is None:
        order = np.argsort(x, kind="stable")
    v = x[order] ** 3
    v_edge = grid.x_edge ** 3
    vv = np.concatenate([[0.0], v, [max(v_edge[-1], v[-1]) * (1 + 1e-12)]])
    out_cells, out_edges = [], []
    for val in vals:
        vs = val[order]
        c = np.cumsum(vs) - 0.5 * vs
        cc = np.concatenate([[0.0], c, [c[-1] + 0.5 * vs[-1]]])
        Ce = np.interp(v_edge, vv, cc)
        out_cells.append(np.diff(Ce))
        out_edges.append(Ce)
    return out_cells, out_edges


def mass_integrate_cmc(R_edge, sE, sJ, Kth_edge, nsub=4):
    K = len(sE)
    m_edge = np.zeros(K + 1)
    if _lib is not None:
        c = lambda arr: np.ascontiguousarray(arr, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        Rc, sEc, sJc, Kc = [np.ascontiguousarray(a, dtype=np.float64) for a in (R_edge, sE, sJ, Kth_edge)]
        bad = _lib.mass_integrate_cmc(K, c(Rc), c(sEc), c(sJc), c(Kc), m_edge.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nsub)
        return m_edge, bad
    bad = 0
    for k in range(K):
        r0, r1 = R_edge[k], R_edge[k + 1]
        dR = r1 - r0; h = dR / nsub; m = m_edge[k]
        v3 = (r1**3 - r0**3) / 3.0
        se, sj = sE[k] / v3, sJ[k] / v3
        ns = max(nsub, int(dR * abs(se) * r1 / 0.3) + 1); h = dR / ns
        k0, k1 = Kth_edge[k], Kth_edge[k + 1]

        def f(r, mm):
            nonlocal bad
            kth = k0 + (k1 - k0) * (r - r0) / dR
            u = 1 - 2 * mm / r + r * r * kth * kth if r > 0 else 1.0
            if u <= 0: bad += 1; u = 0.0
            return r * r * (se * math.sqrt(u) - r * kth * sj)
        for j in range(ns):
            r = r0 + j * h
            f1 = f(r, m); f2 = f(r + h / 2, m + h / 2 * f1); f3 = f(r + h / 2, m + h / 2 * f2); f4 = f(r + h, m + h * f3)
            m += h * (f1 + 2 * f2 + 2 * f3 + f4) / 6
        m_edge[k + 1] = m
    return m_edge, bad


# ----------------------------------------------------------------------------- metric solve
def solve_metric(grid: CosmoGrid, bg: Background, P: CParticles, t: float, x=None, Pm=None, nsub: int = 8) -> CMetric:
    """Fully constrained solve at time t for particle state (x, Pm) (defaults: P.x, P.P)."""
    x = P.x if x is None else x
    Pm = P.P if Pm is None else Pm
    a, K, Hd = bg.a(t), bg.K(t), bg.Hdot(t)
    R_edge = a * grid.x_edge
    R_c = a * grid.x_c
    dR = a * grid.dx
    dcc = a * grid.dcc
    Rp = a * np.maximum(np.abs(x), 1e-300)
    W = np.sqrt(1.0 + Pm**2 + P.Lsq / Rp**2)
    Po = np.where(x >= 0, Pm, -Pm)                 # physical outward momentum (mirror convention for x < 0)
    xa = np.clip(np.abs(x), 0.0, grid.x_out)
    order = np.argsort(xa, kind="stable")
    (sE, sJ, sJR, sp, sq), (cE, cJ, cJR, _, _) = deposit(grid, xa, [P.N * W, P.N * Po, P.N * Po * Rp, P.N * Pm**2 / W, P.N * P.Lsq / (Rp**2 * W)], order)
    # K^theta_theta on edges from the momentum constraint (cumulative momentum), Misner-Sharp mass by the
    # R^2-weighted cell integration (exact for uniform density; no discreteness error in the super-horizon region)
    Kth_edge = np.empty(grid.K + 1)
    Kth_edge[0] = K / 3
    Kth_edge[1:] = K / 3 - cJR[1:] / R_edge[1:] ** 3
    m_edge, n_bad = mass_integrate_cmc(R_edge, sE, sJ, Kth_edge, nsub)
    u_edge = np.empty(grid.K + 1)
    u_edge[0] = 1.0
    u_edge[1:] = 1.0 - 2.0 * m_edge[1:] / R_edge[1:] + R_edge[1:] ** 2 * Kth_edge[1:] ** 2
    u_edge = np.maximum(u_edge, 1e-14)
    A_edge = 1.0 / np.sqrt(u_edge)
    # per-particle values by smooth within-cell interpolation: m ~ R^3 inside the cell (exact for uniform density),
    # Kth linear; then u and A at the particle.  This is the continuum (fluid-like) enclosed mass, not a half-shell sum.
    kk = grid.cell_of(xa)
    R0e, R1e = R_edge[kk], R_edge[kk + 1]
    frac3 = (Rp**3 - R0e**3) / (R1e**3 - R0e**3)
    p_m = m_edge[kk] + (m_edge[kk + 1] - m_edge[kk]) * np.clip(frac3, 0.0, 1.0)
    p_Kth = np.interp(Rp, R_edge, Kth_edge)
    p_u = np.maximum(1.0 - 2.0 * p_m / Rp + Rp**2 * p_Kth**2, 1e-14)
    p_A = 1.0 / np.sqrt(p_u)
    A_c = 0.5 * (A_edge[1:] + A_edge[:-1])
    Kth_c = 0.5 * (Kth_edge[1:] + Kth_edge[:-1])
    KR_c = K - 2 * Kth_c
    V3_c = (R_edge[1:] ** 3 - R_edge[:-1] ** 3) / 3.0          # exact shell volume / (4 pi)
    vol_c = FOURPI * V3_c * A_c
    E_c = sE / vol_c
    S_c = (sp + sq) / vol_c                    # S = S_R + S_th + S_ph; sq already holds both tangential components
    Q_c = KR_c**2 + 2 * Kth_c**2 + FOURPI * (E_c + S_c)
    # lapse: tridiagonal BVP on cell centres, alpha'(0) = 0, alpha(R_out) = 1
    e = np.zeros(grid.K + 1)
    e[1:] = R_edge[1:] ** 2 / (dcc[1:] * A_edge[1:])
    upper = e[1:].copy()                       # u_c = e_{c+1}; the last one already uses the half spacing
    lower = e[:-1].copy()                      # e_c
    diag = -(upper + lower) - V3_c * A_c * Q_c
    rhs = 3.0 * Hd * V3_c * A_c
    a_out = bg.alpha_out(m_edge[-1], R_edge[-1])
    rhs[-1] -= upper[-1] * a_out
    ab = np.zeros((3, grid.K))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    alpha_c = solve_banded((1, 1), ab, rhs)
    alpha_edge = np.empty(grid.K + 1)
    alpha_edge[0] = alpha_c[0]
    alpha_edge[1:-1] = 0.5 * (alpha_c[1:] + alpha_c[:-1])
    alpha_edge[-1] = a_out
    dalpha_edge = np.zeros(grid.K + 1)
    dalpha_edge[1:-1] = (alpha_c[1:] - alpha_c[:-1]) / dcc[1:-1]
    dalpha_edge[-1] = (a_out - alpha_c[-1]) / dcc[-1]
    dA_edge = np.gradient(A_edge, R_edge)
    return CMetric(t, a, R_edge, R_c, dR, m_edge, Kth_edge, u_edge, A_edge, alpha_c, alpha_edge, dalpha_edge, dA_edge,
                   E_c, S_c, int(n_bad), {"sE": sE, "sJ": sJ, "sp": sp, "sq": sq}, p_m, p_A, p_Kth)


def metric_at(met: CMetric, R):
    """alpha, alpha', A, Kth, m at areal radii R (regular-centre treatment inside the first cell)."""
    Re = met.R_edge
    alpha = np.interp(R, Re, met.alpha_edge)
    dalpha = np.interp(R, Re, met.dalpha_edge)
    A = np.interp(R, Re, met.A_edge)
    Kth = np.interp(R, Re, met.Kth_edge)
    m = np.interp(R, Re, met.m_edge)
    inner = R < Re[1]
    if np.any(inner):
        m[inner] = met.m_edge[1] * (R[inner] / Re[1]) ** 3
        dalpha[inner] = met.dalpha_edge[1] * R[inner] / Re[1]
    return alpha, dalpha, A, Kth, m


def rhs(grid: CosmoGrid, bg: Background, P: CParticles, x, Pm, t):
    met = solve_metric(grid, bg, P, t, x, Pm)
    a, H, K = met.a, bg.H(t), bg.K(t)
    R = a * np.abs(x)
    sgn = np.where(x >= 0, 1.0, -1.0)
    alpha, dalpha, A_g, Kth_g, m = metric_at(met, R)
    A, Kth = met.p_A, met.p_Kth
    Rs = np.maximum(R, 1e-300)
    W = np.sqrt(1.0 + Pm**2 + P.Lsq / Rs**2)
    beta = alpha * R * Kth
    # mirror convention: x may be negative during a stage; Pm is the momentum along +x, sgn*Pm the outward momentum
    dRdt = alpha * sgn * Pm / (A * W) - beta                     # physical dR/dt
    dxdt = sgn * (dRdt - H * R) / a                              # = (alpha Pm/(A W) - sgn (beta + H R))/a
    dPdt = sgn * (-(W / A) * dalpha + alpha * P.Lsq / (W * A * Rs**3)) + alpha * Pm * (K - 2 * Kth)
    dtaudt = alpha / W
    return dxdt, dPdt, dtaudt, met


# ----------------------------------------------------------------------------- driver
@dataclass
class CosmoRunConfig:
    cfl: float = 0.4
    t_end: float = 1.0
    max_dt_frac: float = 0.02       # dt <= max_dt_frac * t
    diag_every: int = 20
    max_steps: int = 5_000_000
    nsub: int = 4
    snapshot_times: tuple = ()      # times at which particle snapshots are stored
    inject_t: float = 0.0           # if > 0: inject an isotropic velocity dispersion at this time
    inject_sigma: float = 0.0       # one-dimensional dispersion (units of c) in the shell rest frame
    inject_xmax: float = 0.0        # only for shells with x < inject_xmax
    inject_seed: int = 0
    stiff_omega_dt: float = 0.2     # warm shells with L dt/(R^2 W) above this are substepped under the frozen metric
    dR_target: float = 0.0          # > 0: re-stretch the comoving grid as a grows so that a*dx_min stays <= dR_target (physical units)
    # dispersion released during the collapse by small-scale substructure (per shell, when its own interior reaches the trigger):
    #   "harada": trigger 3 m_i/(4 pi R_i^3) >= rho_vir = 8 rho_ent(k~) (k/k~)^6 delta_ent^3, sigma_v = sqrt(h delta_ent)   [Harada+2023 eqs 42-44]
    #   "ebrahimian": trigger R_i/R_max,i <= b_nl = [zeta_k (k/k0)^2 a_max/a_ent]^(2/3), sigma_v = sqrt(m_i/R_i) (k0/k)   [Ebrahimian+2025 eqs 38-39]
    release_mode: str = ""
    release_k_ratio: float = 10.0   # k/k~ (harada) or k/k0 (ebrahimian)
    release_delta_ent: float = 0.05 # delta_ent(k) (harada) or zeta_k (ebrahimian)
    release_h: float = 0.6
    release_t_ent: float = 0.0      # entry time of the r_m scale (t_H); a_ent = a(release_t_ent)
    release_xmax: float = 0.0       # only shells with initial |x| < release_xmax (0 = all)
    release_b_nl: float = 0.0       # ebrahimian: if > 0, fixed release radius R = b_nl R_max for every shell (timing scan)
    release_seed: int = 0
    restretch_every: int = 20
    stretch_max: float = 10.0
    max_substeps: int = 4000


class CosmoRun:
    def __init__(self, grid: CosmoGrid, bg: Background, P: CParticles, cfg: CosmoRunConfig, t0: float):
        self.grid, self.bg, self.P, self.cfg = grid, bg, P, cfg
        self.t = t0
        self.step = 0
        self.diag = []
        self.snapshots = []
        self._snap_idx = 0
        self.ah_first = None
        self.n_stiff = 0
        self.n_sub_last = 0
        self.n_restretch = 0
        n = len(P.x)
        self.released = np.zeros(n, bool)
        self.release_eligible = np.ones(n, bool) if cfg.release_xmax <= 0 else (np.abs(P.x) < cfg.release_xmax)
        self.R_max = np.zeros(n); self.a_at_Rmax = np.ones(n)
        self.n_released = 0; self.t_first_release = np.nan
        self._release_rng = np.random.default_rng(cfg.release_seed)
        self.tau_trap = np.full(len(P), np.nan)      # proper time of each shell when first found trapped (2m/R >= 1, Kth > 0)
        self.t_trap = np.full(len(P), np.nan)        # coordinate time of that detection

    def dt(self, met: CMetric):
        """Particle-based step: no shell may cross more than cfl of its cell per step (the metric is solved by
        constraints, so there is no grid light-speed condition), plus dt <= max_dt_frac * t for the background
        and dt <= 0.05/|alpha K_R| for the momentum equation."""
        g, cfg, P = self.grid, self.cfg, self.P
        a, H = met.a, self.bg.H(self.t)
        xa = np.clip(np.abs(P.x), 0.0, g.x_out)
        R = a * xa
        W = np.sqrt(1.0 + P.P**2 + P.Lsq / np.maximum(R, 1e-300) ** 2)
        alpha = np.interp(R, met.R_edge, met.alpha_edge)
        dRdt = alpha * P.P / (met.p_A * W) - alpha * R * met.p_Kth
        vx = np.abs(dRdt - H * R) / a
        dxc = g.dxc[g.cell_of(xa)]
        dt_p = cfg.cfl * np.min(dxc / np.maximum(vx, 1e-300))
        KR = self.bg.K(self.t) - 2 * met.Kth_edge
        dt_k = 0.05 / (np.max(np.abs(met.alpha_edge * KR)) + 1e-300)
        return min(dt_p, dt_k, cfg.max_dt_frac * self.t if self.t > 0 else np.inf, cfg.t_end - self.t)

    def _stiff_substeps(self, met, Lsq, x0, P0, tau0, t0, dt, n_sub):
        """Integrate warm shells whose pericentre passage is not resolved by the global step: RK4 substeps under
        the metric frozen at the start of the step (alpha, alpha', A, K_theta as functions of areal radius).
        Returns the state at t0 + dt/2 (for the RK4 stage sources) and at t0 + dt."""
        bg = self.bg
        def f(x, Pm, t):
            a, H, K = bg.a(t), bg.H(t), bg.K(t)
            R = a * np.abs(x); sgn = np.where(x >= 0, 1.0, -1.0); Rs = np.maximum(R, 1e-300)
            alpha, dalpha, A, Kth, _ = metric_at(met, R)
            W = np.sqrt(1.0 + Pm**2 + Lsq / Rs**2)
            dRdt = alpha * sgn * Pm / (A * W) - alpha * R * Kth
            return (sgn * (dRdt - H * R) / a,
                    sgn * (-(W / A) * dalpha + alpha * Lsq / (W * A * Rs**3)) + alpha * Pm * (K - 2 * Kth),
                    alpha / W)
        x, Pm, tau, t = x0.copy(), P0.copy(), tau0.copy(), t0
        h = dt / n_sub; half = None
        for i in range(n_sub):
            k1 = f(x, Pm, t); k2 = f(x + 0.5 * h * k1[0], Pm + 0.5 * h * k1[1], t + 0.5 * h)
            k3 = f(x + 0.5 * h * k2[0], Pm + 0.5 * h * k2[1], t + 0.5 * h); k4 = f(x + h * k3[0], Pm + h * k3[1], t + h)
            x = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            Pm = Pm + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            tau = tau + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            t += h
            if half is None and t >= t0 + 0.5 * dt - 1e-12 * dt:
                half = (x.copy(), Pm.copy())
        return half, (x, Pm, tau)

    def _stiff_substeps_c(self, met, Lsq, x0, P0, tau0, t0, dt, nsub):
        """Compiled version of _stiff_substeps with a per-shell substep count."""
        n = len(x0); K = len(met.R_edge) - 1
        c = lambda arr: np.ascontiguousarray(arr, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out = [np.empty(n) for _ in range(5)]
        cosmo = 0 if isinstance(self.bg, StaticBackground) else 1
        nsub_i = np.ascontiguousarray(nsub, dtype=np.int32)
        self.n_center_hits = _lib.stiff_substeps(n, K, c(met.R_edge), c(met.alpha_edge), c(met.dalpha_edge), c(met.A_edge), c(met.Kth_edge),
                                                 cosmo, float(getattr(self.bg, "t_i", 1.0)), c(Lsq), c(x0), c(P0), c(tau0), float(t0), float(dt),
                                                 nsub_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                 *[o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for o in out])
        return (out[0], out[1]), (out[2], out[3], out[4])

    def _step(self, dt):
        P, g, bg = self.P, self.grid, self.bg
        x0, P0, tau0, t0 = P.x, P.P, P.tau, self.t
        k1 = rhs(g, bg, P, x0, P0, t0)
        # stiff warm shells: angular frequency omega = L/(R^2 W) not resolved by dt -> substeps under the frozen metric
        a0 = bg.a(t0); R0 = a0 * np.maximum(np.abs(x0), 1e-300)
        warm = P.Lsq > 0
        # pericentre the shell can reach: angular momentum conservation with the momentum it would have there
        # (current momentum plus a unit of infall gain as safety); flag the shell if it can get within 4 R_p during
        # the step and its angular frequency there is not resolved
        Pp2 = P0**2 + P.Lsq / R0**2 + 1.0
        Rp = np.sqrt(P.Lsq / Pp2)
        travel = 2.0 * a0 * np.abs(k1[0]) * dt + np.abs(bg.H(t0)) * R0 * dt
        omega_p = Pp2 / (np.sqrt(np.maximum(P.Lsq, 1e-300)) * np.sqrt(1.0 + Pp2))
        stiff = warm & (R0 - travel < 4.0 * Rp) & (omega_p * dt > self.cfg.stiff_omega_dt)
        self.n_stiff = int(stiff.sum())
        if self.n_stiff:
            # bucket the stiff shells by the substep count they need (powers of two) so that the few shells with the
            # smallest pericentres do not set the cost for all of them
            need = np.ceil(omega_p[stiff] * dt / (0.25 * self.cfg.stiff_omega_dt))
            need = np.minimum(np.maximum(need, 2), self.cfg.max_substeps)
            if _lib is not None and hasattr(_lib, "stiff_substeps"):
                self.n_sub_last = int(need.max())
                half, end = self._stiff_substeps_c(k1[3], P.Lsq[stiff], x0[stiff], P0[stiff], tau0[stiff], t0, dt, need)
                need = None
            lev = np.ceil(np.log2(need)).astype(int) if need is not None else None
            idx = np.flatnonzero(stiff)
            hx = np.empty(self.n_stiff); hP = np.empty(self.n_stiff); ex = np.empty(self.n_stiff); eP = np.empty(self.n_stiff); et = np.empty(self.n_stiff)
            if lev is not None: self.n_sub_last = 0
            for l in (np.unique(lev) if lev is not None else []):
                sel = lev == l; n_sub = int(min(2 ** l, self.cfg.max_substeps)); self.n_sub_last = max(self.n_sub_last, n_sub)
                ii = idx[sel]
                h_, e_ = self._stiff_substeps(k1[3], P.Lsq[ii], x0[ii], P0[ii], tau0[ii], t0, dt, n_sub)
                hx[sel], hP[sel] = h_[0], h_[1]; ex[sel], eP[sel], et[sel] = e_
            if lev is not None: half, end = (hx, hP), (ex, eP, et)
        def stage(xs, Ps, which):
            if self.n_stiff:
                xs = xs.copy(); Ps = Ps.copy()
                src = half if which == "half" else end
                xs[stiff], Ps[stiff] = src[0], src[1]
            return xs, Ps
        x2, P2 = stage(x0 + 0.5 * dt * k1[0], P0 + 0.5 * dt * k1[1], "half")
        k2 = rhs(g, bg, P, x2, P2, t0 + 0.5 * dt)
        x3, P3 = stage(x0 + 0.5 * dt * k2[0], P0 + 0.5 * dt * k2[1], "half")
        k3 = rhs(g, bg, P, x3, P3, t0 + 0.5 * dt)
        x4, P4 = stage(x0 + dt * k3[0], P0 + dt * k3[1], "end")
        k4 = rhs(g, bg, P, x4, P4, t0 + dt)
        x = x0 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Pm = P0 + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        tau = tau0 + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        if self.n_stiff:
            x[stiff], Pm[stiff], tau[stiff] = end
        neg = x < 0
        x[neg] = -x[neg]; Pm[neg] = -Pm[neg]                     # regular centre: reflection of the radial momentum
        x = np.minimum(x, g.x_out)
        P.x, P.P, P.tau = x, Pm, tau
        self.t += dt
        self.step += 1
        return k1[3]

    def unused_equation_check(self, met0: CMetric, met1: CMetric, dt):
        """Residual of the (unused) evolution equation of A: dA/dt = beta A' + A beta' - alpha A K^R_R, on edges."""
        R = met0.R_edge
        res = []
        for met in (met0, met1):
            K = self.bg.K(met.t)
            beta = met.alpha_edge * R * met.Kth_edge
            dbeta = np.gradient(beta, met.R_edge)
            KR = K - 2 * met.Kth_edge
            res.append(beta * met.dA_edge + met.A_edge * dbeta - met.alpha_edge * met.A_edge * KR)
        pred = 0.5 * (res[0] + res[1])
        # A is compared at fixed x (the grid moves with a): dA/dt|_x = dA/dt|_R + H R A'
        fd = (met1.A_edge - met0.A_edge) / dt - 0.5 * (self.bg.H(met0.t) * R * met0.dA_edge + self.bg.H(met1.t) * met1.R_edge * met1.dA_edge)
        scale = np.max(np.abs(met1.alpha_edge * met1.A_edge * (self.bg.K(met1.t) - 2 * met1.Kth_edge))) + 1e-300
        self.ueq_profile = (fd - pred) / scale
        return float(np.sqrt(np.mean((fd - pred) ** 2)) / scale)

    def diagnostics(self, met: CMetric, ueq=np.nan):
        g, bg = self.grid, self.bg
        # per-shell trapping record (slicing-independent: 2m/R is a scalar at the shell's event)
        Rp = met.a * np.abs(self.P.x)
        trapped_now = (2.0 * met.p_m / np.maximum(Rp, 1e-300) >= 1.0) & (met.p_Kth > 0) & np.isnan(self.tau_trap)
        if np.any(trapped_now):
            self.tau_trap[trapped_now] = self.P.tau[trapped_now]
            self.t_trap[trapped_now] = self.t
        R = met.R_edge[1:]
        comp = 2.0 * met.m_edge[1:] / R
        collapsing = met.Kth_edge[1:] > 0
        comp_c = np.where(collapsing, comp, -np.inf)
        k = int(np.argmax(comp_c)) if np.any(collapsing) else -1
        ah = None
        trapped = collapsing & (comp >= 1.0)
        if np.any(trapped):
            j = int(np.max(np.nonzero(trapped)[0]))       # outermost trapped edge
            ah = dict(R_AH=float(R[j]), M_AH=float(met.m_edge[j + 1]), x_AH=float(R[j] / met.a))
            if self.ah_first is None:
                self.ah_first = dict(t=self.t, **ah)
                self.snapshot(met)                      # invariant record of the shells at first detection
                self.ah_snapshot = self.snapshots.pop()
        H = bg.H(self.t)
        m_flrw_out = 0.5 * H**2 * met.R_edge[-1] ** 3
        d = {
            "t": self.t, "step": self.step, "a": met.a,
            "max_2m_over_R_collapsing": float(comp_c[k]) if k >= 0 else np.nan,
            "x_at_max": float(R[k] / met.a) if k >= 0 else np.nan,
            "min_alpha": float(met.alpha_c.min()), "max_A": float(met.A_edge.max()),
            "m_out_over_flrw": float(met.m_edge[-1] / m_flrw_out),
            "n_stiff": self.n_stiff, "n_sub": self.n_sub_last, "stretch": float(self.grid.stretch), "dR_min": float(met.a * self.grid.dx),
            "n_released": int(self.n_released), "t_first_release": float(self.t_first_release),
            "n_bad": met.n_bad, "x_bad_max": float(np.max(met.R_edge[met.u_edge <= 1.5e-14]) / met.a) if np.any(met.u_edge <= 1.5e-14) else 0.0,
            "unused_eq_residual": ueq, "n_particles": len(self.P),
            "m_in_R": [float(np.interp(Rf, met.R_edge, met.m_edge)) for Rf in (1.0, 2.5, 5.0, 10.0, 25.0)],
            "tau_core_median": float(np.median(self.P.tau[met.a * np.abs(self.P.x) < 5.0])) if np.any(met.a * np.abs(self.P.x) < 5.0) else np.nan,
            "min_x": float(self.P.x.min()), "AH": ah,
        }
        self.diag.append(d)
        return d

    def snapshot(self, met: CMetric):
        P = self.P
        R = met.a * P.x
        A, Kth, m = met.p_A, met.p_Kth, met.p_m
        W = np.sqrt(1 + P.P**2 + P.Lsq / np.maximum(R, 1e-300) ** 2)
        dRdtau = P.P / A - R * Kth * W          # dR/dtau = P/A - (beta/alpha) W
        self.snapshots.append(dict(t=self.t, a=met.a, R=R.copy(), tau=P.tau.copy(), m=m.copy(), P=P.P.copy(),
                                   dRdtau=dRdtau, N=P.N.copy(), Lsq=P.Lsq.copy()))

    def inject_dispersion(self, met: CMetric):
        """Isotropic Maxwellian (non-relativistic addition) in the shell frame: radial part into P, tangential into L^2."""
        cfg, P = self.cfg, self.P
        rng = np.random.default_rng(cfg.inject_seed)
        sel = np.abs(P.x) < cfg.inject_xmax
        n = int(sel.sum())
        R = met.a * np.abs(P.x[sel])
        P.P[sel] += cfg.inject_sigma * rng.standard_normal(n)
        vt = cfg.inject_sigma * np.sqrt(rng.standard_normal(n) ** 2 + rng.standard_normal(n) ** 2)
        P.Lsq[sel] = (R * vt) ** 2
        self.injected = dict(t=self.t, n=n, sigma=cfg.inject_sigma)

    def required_stretch(self, t):
        """Smallest sinh stretch s with a(t) x_out s/(K sinh s) <= dR_target (0 if the uniform grid already satisfies it)."""
        cfg, g = self.cfg, self.grid
        q = self.bg.a(t) * g.x_out / (g.K * cfg.dR_target)      # need sinh(s)/s >= q
        if q <= 1.0:
            return 0.0
        lo, hi = 0.0, cfg.stretch_max
        if np.sinh(hi) / hi < q:
            return hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.sinh(mid) / mid < q: lo = mid
            else: hi = mid
        return hi

    def maybe_restretch(self):
        """Replace the grid by a more strongly stretched one when the physical spacing at the centre has outgrown
        dR_target.  The scheme is fully constrained, so the particles carry the whole state: nothing to remap."""
        cfg = self.cfg
        if cfg.dR_target <= 0 or self.step % cfg.restretch_every:
            return False
        s_req = self.required_stretch(self.t)
        if s_req > self.grid.stretch + 0.05:
            self.grid = CosmoGrid(self.grid.x_out, self.grid.K, stretch=min(s_req, cfg.stretch_max))
            self.n_restretch += 1
            return True
        return False

    def maybe_release(self, met):
        """Per-shell release of a substructure velocity dispersion (see CosmoRunConfig.release_mode)."""
        cfg, P = self.cfg, self.P
        if not cfg.release_mode:
            return
        R = met.a * np.abs(P.x)
        grow = R > self.R_max
        self.R_max[grow] = R[grow]; self.a_at_Rmax[grow] = met.a
        # only shells past their turnaround (the mean interior density rises again); P < 0 holds for every overdense
        # shell in this gauge (P is the peculiar momentum) and is not a collapse criterion
        cand = self.release_eligible & ~self.released & (R < 0.95 * self.R_max) & (R > 0)
        if not np.any(cand):
            return
        m = met.p_m
        if cfg.release_mode == "harada":
            H_ent = self.bg.H(cfg.release_t_ent); rho_ent = 3.0 * H_ent**2 / (8.0 * np.pi)
            rho_vir = 8.0 * rho_ent * cfg.release_k_ratio**6 * cfg.release_delta_ent**3
            rho_bar = 3.0 * m / (4.0 * np.pi * np.maximum(R, 1e-300)**3)
            sel = cand & (rho_bar >= rho_vir)
            sigma = np.full(int(sel.sum()), np.sqrt(cfg.release_h * cfg.release_delta_ent))
        elif cfg.release_mode == "ebrahimian":
            a_ent = self.bg.a(cfg.release_t_ent)
            b_nl = np.minimum((cfg.release_delta_ent * cfg.release_k_ratio**2 * self.a_at_Rmax / a_ent) ** (2.0 / 3.0), 0.95)
            if cfg.release_b_nl > 0:
                b_nl = np.full_like(b_nl, min(cfg.release_b_nl, 0.95))
            sel = cand & (R <= b_nl * self.R_max)
            sigma = np.sqrt(np.maximum(m[sel] / np.maximum(R[sel], 1e-300), 0.0)) / cfg.release_k_ratio
        else:
            raise ValueError(cfg.release_mode)
        n = int(sel.sum())
        if n == 0:
            return
        rng = self._release_rng
        P.P[sel] += sigma * rng.standard_normal(n)
        vt = sigma * np.sqrt(rng.standard_normal(n) ** 2 + rng.standard_normal(n) ** 2)
        P.Lsq[sel] = np.maximum(P.Lsq[sel], (R[sel] * vt) ** 2)
        self.released[sel] = True
        self.n_released += n
        if np.isnan(self.t_first_release):
            self.t_first_release = self.t

    def run(self, callback=None):
        cfg = self.cfg
        met = solve_metric(self.grid, self.bg, self.P, self.t)
        self.diagnostics(met)
        self.injected = None
        while self.t < cfg.t_end - 1e-15 * cfg.t_end and self.step < cfg.max_steps:
            if self.maybe_restretch():
                met = solve_metric(self.grid, self.bg, self.P, self.t)
            dt = self.dt(met)
            if cfg.inject_t > 0 and self.injected is None and self.t < cfg.inject_t <= self.t + dt:
                dt = cfg.inject_t - self.t
            if self._snap_idx < len(cfg.snapshot_times):
                ts = cfg.snapshot_times[self._snap_idx]
                if self.t < ts <= self.t + dt:
                    dt = ts - self.t
            do_check = (self.step % cfg.diag_every == 0)
            met0 = met
            self._step(dt)
            met = solve_metric(self.grid, self.bg, self.P, self.t)
            if cfg.inject_t > 0 and self.injected is None and abs(self.t - cfg.inject_t) <= 1e-12 * self.t:
                self.inject_dispersion(met)
                met = solve_metric(self.grid, self.bg, self.P, self.t)
            if cfg.release_mode:
                n0 = self.n_released
                self.maybe_release(met)
                if self.n_released > n0:
                    met = solve_metric(self.grid, self.bg, self.P, self.t)
            if do_check:
                ueq = self.unused_equation_check(met0, met, dt)
                d = self.diagnostics(met, ueq)
                if callback: callback(self, d)
            if self._snap_idx < len(cfg.snapshot_times) and abs(self.t - cfg.snapshot_times[self._snap_idx]) <= 1e-12 * self.t:
                self.snapshot(met)
                self._snap_idx += 1
        self.diagnostics(met)
        return met

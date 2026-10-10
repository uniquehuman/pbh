"""Spherically symmetric Einstein-Vlasov system with X decay (Stage A prototype).

Gauge: polar-areal (Schwarzschild) coordinates, zero shift,
    ds^2 = -e^{2 mu} dt^2 + e^{2 lambda} dr^2 + r^2 dOmega^2,
with mu(t, r_out) matched to the exterior Schwarzschild solution and lambda(t, r_in)
fixed by the inner mass.  This is a fully constrained scheme (Rein-Rendall-Schaeffer 1998,
Andreasson-Rein 2006): lambda follows from the Hamiltonian constraint, mu from the
polar slicing condition, and lambda_dot from the momentum constraint.

Matter: macroparticles i with areal radius r_i, radial momentum w_i = p^{r-hat} and
squared angular momentum L_i = r^2 |p_perp|^2 measured in the static orthonormal frame,
weight N_i (number of physical particles) and rest mass m_i in {1 (X), 0 (massless
daughter)}.  With E_i = sqrt(m_i^2 + w_i^2 + L_i / r_i^2):

    dr/dt = e^{mu - lambda} w / E
    dw/dt = -lambda_dot w - e^{mu - lambda} mu' E + e^{mu - lambda} L / (r^3 E)
    dL/dt = 0
    dN/dt = -Gamma (e^{mu} / E) N          (X only: proper-time decay, alpha = e^mu, gamma = E)

Sources (proper densities measured by the static observer; the proper volume of a
coordinate shell is 4 pi r^2 e^{lambda} dr):

    rho = e^{-lambda} S_E / (4 pi r^2 dr),   S_E = sum_{i in cell} N_i E_i
    p   = e^{-lambda} S_p / (4 pi r^2 dr),   S_p = sum N_i w_i^2 / E_i
    j   = e^{-lambda} S_j / (4 pi r^2 dr),   S_j = sum N_i w_i
    q   = e^{-lambda} S_q / (4 pi r^2 dr),   S_q = sum N_i L_i / (r_i^2 E_i)

Field equations used:
    m' = 4 pi r^2 rho  ->  dm/dr = (S_E / dr) sqrt(1 - 2 m / r),   e^{-2 lambda} = 1 - 2 m / r
    mu' = e^{2 lambda} ( m / r^2 + 4 pi r p )
    lambda_dot = -4 pi r e^{lambda + mu} j = -e^{mu} S_j / (r dr)
The remaining (tangential) Einstein equation is not used in the evolution and serves as a
consistency check together with ADM-mass conservation and the momentum-constraint check.

Horizon diagnostics: in this gauge the outgoing null expansion of the r = const spheres is
theta_+ = (2/r) e^{-lambda} > 0 wherever the coordinates are regular, so an apparent
horizon (theta_+ = 0) can only be *approached* as max 2m/r -> 1 with lapse collapse.
That approach is reported as outcome "BH_approach" and is NOT a horizon-penetrating
continuation (see the specification, section 5 of the feasibility README).

Decay: the decayed weight of each X macroparticle accumulates in `pending`; once it exceeds
`emit_fraction` of the particle's initial weight, a decay event emits two massless daughters
back-to-back with isotropic random direction in the X rest frame, boosted to the static
frame, so that energy and momentum are conserved event by event.
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FOURPI = 4.0 * np.pi

# ----------------------------------------------------------------------------- C kernel
_lib = None
_so = Path(__file__).with_name("_kernels.so")
if _so.exists():
    _lib = ctypes.CDLL(str(_so))
    _lib.mass_integrate.restype = ctypes.c_int
    _lib.mass_integrate.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int]


def mass_integrate(r_edge: np.ndarray, s: np.ndarray, m_in: float, nsub: int = 4):
    """Integrate dm/dr = s_k sqrt(1-2m/r) across cells. Returns (m_edge, n_bad)."""
    K = len(s)
    m_edge = np.empty(K + 1)
    m_edge[0] = m_in
    if _lib is not None:
        r_c = np.ascontiguousarray(r_edge, dtype=np.float64)
        s_c = np.ascontiguousarray(s, dtype=np.float64)
        bad = _lib.mass_integrate(K, r_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  s_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  m_edge.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nsub)
        return m_edge, bad
    # pure-Python fallback (slow): same RK4 per cell
    bad = 0

    def f(r, m, sk):
        if r <= 0:
            return sk
        a = 1 - 2 * m / r
        return sk * math.sqrt(a) if a > 0 else 0.0

    for k in range(K):
        r0, h, m, sk = r_edge[k], (r_edge[k + 1] - r_edge[k]) / nsub, m_edge[k], s[k]
        for j in range(nsub):
            r = r0 + j * h
            k1 = f(r, m, sk)
            k2 = f(r + h / 2, m + h / 2 * k1, sk)
            k3 = f(r + h / 2, m + h / 2 * k2, sk)
            k4 = f(r + h, m + h * k3, sk)
            m += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        re = r_edge[k + 1]
        if re > 0 and 1 - 2 * m / re <= 0:
            bad += 1
            m = 0.5 * re * (1 - 1e-12)
        m_edge[k + 1] = m
    return m_edge, bad


# ----------------------------------------------------------------------------- grid
@dataclass
class Grid:
    r_out: float
    K: int
    r_in: float = 0.0
    m_in: float = 0.0     # mass inside r_in (0 for a regular centre; >0 = excised BH interior)

    def __post_init__(self):
        self.r_edge = np.linspace(self.r_in, self.r_out, self.K + 1)
        self.dr = self.r_edge[1] - self.r_edge[0]
        self.r_c = 0.5 * (self.r_edge[1:] + self.r_edge[:-1])
        if self.r_in > 0 and self.r_in <= 2 * self.m_in:
            raise ValueError("inner boundary must lie outside 2 m_in")


@dataclass
class Metric:
    m_edge: np.ndarray
    mu_edge: np.ndarray
    mup_c: np.ndarray
    lamdot_c: np.ndarray
    S: dict
    n_bad: int

    @property
    def M(self):
        return self.m_edge[-1]


# ----------------------------------------------------------------------------- particles
@dataclass
class Particles:
    r: np.ndarray
    w: np.ndarray
    L: np.ndarray
    N: np.ndarray
    m: np.ndarray                                  # rest mass: 1 (X) or 0 (daughter)
    tau: np.ndarray = None                         # proper time (massive particles)
    N0: np.ndarray = None                          # initial weight (for emission threshold)
    pending: np.ndarray = None                     # decayed but not yet emitted rest mass
    thr: np.ndarray = None                         # per-particle emission threshold (drawn lazily; inf = never)

    def __post_init__(self):
        n = len(self.r)
        if self.tau is None:
            self.tau = np.zeros(n)
        if self.N0 is None:
            self.N0 = self.N.copy()
        if self.pending is None:
            self.pending = np.zeros(n)
        if self.thr is None:
            self.thr = np.full(n, np.nan)

    def __len__(self):
        return len(self.r)

    def energy(self):
        return np.sqrt(self.m**2 + self.w**2 + self.L / np.maximum(self.r**2, 1e-300))

    def select(self, mask):
        return Particles(self.r[mask], self.w[mask], self.L[mask], self.N[mask], self.m[mask],
                         self.tau[mask], self.N0[mask], self.pending[mask], self.thr[mask])

    def append(self, other: "Particles"):
        for k in ("r", "w", "L", "N", "m", "tau", "N0", "pending", "thr"):
            setattr(self, k, np.concatenate([getattr(self, k), getattr(other, k)]))


# ----------------------------------------------------------------------------- metric solve
def deposit(grid: Grid, r, vals):
    """Cloud-in-cell deposition onto cell centres. Returns per-cell sums for each array in vals."""
    x = (r - grid.r_in) / grid.dr - 0.5
    k0 = np.floor(x).astype(np.int64)
    f = x - k0
    ka = np.clip(k0, 0, grid.K - 1)          # k0 = -1 (inner half cell) folds into cell 0
    kb = np.clip(k0 + 1, 0, grid.K - 1)      # outer half cell folds into cell K-1
    out = []
    for v in vals:
        s = np.bincount(ka, weights=v * (1 - f), minlength=grid.K)
        s += np.bincount(kb, weights=v * f, minlength=grid.K)
        out.append(s)
    return out


def solve_metric(grid: Grid, P: Particles, nsub: int = 4, include_pending: bool = True) -> Metric:
    E = P.energy()
    N = P.N + P.pending if include_pending else P.N
    SE, Sp, Sj, Sq = deposit(grid, P.r, [N * E, N * P.w**2 / E, N * P.w, N * P.L / (P.r**2 * E)])
    m_edge, n_bad = mass_integrate(grid.r_edge, SE / grid.dr, grid.m_in, nsub)
    m_c = 0.5 * (m_edge[1:] + m_edge[:-1])
    r_c = grid.r_c
    a_c = np.maximum(1.0 - 2.0 * m_c / r_c, 1e-14)
    e_lam_c = 1.0 / np.sqrt(a_c)               # e^{lambda} at centres
    # mu' = e^{2 lambda} (m/r^2 + 4 pi r p),  4 pi r p = e^{-lambda} S_p /(r dr)
    mup_c = (m_c / r_c**2 + Sp / (grid.dr * r_c) / e_lam_c) / a_c
    M = m_edge[-1]
    mu_edge = np.empty(grid.K + 1)
    mu_edge[-1] = 0.5 * np.log(max(1.0 - 2.0 * M / grid.r_out, 1e-14))
    mu_edge[:-1] = mu_edge[-1] - np.cumsum((mup_c * grid.dr)[::-1])[::-1]
    mu_c = 0.5 * (mu_edge[1:] + mu_edge[:-1])
    lamdot_c = -np.exp(mu_c) * Sj / (r_c * grid.dr)
    return Metric(m_edge, mu_edge, mup_c, lamdot_c, {"SE": SE, "Sp": Sp, "Sj": Sj, "Sq": Sq}, n_bad)


def metric_at(grid: Grid, met: Metric, r):
    """Interpolate m, mu, mu', lambda_dot to particle radii."""
    m = np.interp(r, grid.r_edge, met.m_edge)
    if grid.r_in == 0.0:
        # regular centre: m ~ r^3 inside the first cell (linear interpolation would give 2m/r -> const)
        inner = r < grid.r_edge[1]
        if np.any(inner):
            m[inner] = met.m_edge[1] * (r[inner] / grid.r_edge[1]) ** 3
    mu = np.interp(r, grid.r_edge, met.mu_edge)
    mup = np.interp(r, grid.r_c, met.mup_c)
    if grid.r_in == 0.0:
        inner = r < grid.r_c[0]
        if np.any(inner):
            mup[inner] = met.mup_c[0] * r[inner] / grid.r_c[0]     # mu' ~ r at the centre
    lamdot = np.interp(r, grid.r_c, met.lamdot_c)
    return m, mu, mup, lamdot


# ----------------------------------------------------------------------------- RHS
def rhs(grid: Grid, P: Particles, r, w, N, Gamma: float):
    """Time derivatives of (r, w, N, tau) at given (r, w, N) with P.L, P.m fixed.

    Metric sources use the start-of-step weight P.N plus the decayed-but-not-yet-emitted
    mass P.pending: that energy is still physically present (it belongs to daughters that
    will be emitted at the parent's position), so it must gravitate until emission."""
    Ptmp = Particles(r, w, P.L, P.N, P.m, P.tau, P.N0, P.pending, P.thr)
    met = solve_metric(grid, Ptmp)
    m, mu, mup, lamdot = metric_at(grid, met, r)
    a = np.maximum(1.0 - 2.0 * m / r, 1e-14)
    emu = np.exp(mu)
    eml = emu * np.sqrt(a)                     # e^{mu - lambda}
    r2 = np.maximum(r**2, 1e-300)
    E = np.sqrt(P.m**2 + w**2 + P.L / r2)
    dr = eml * w / E
    dw = -lamdot * w - eml * mup * E + eml * P.L / (r * r2 * E)
    massive = P.m > 0
    dtau = np.where(massive, emu / np.maximum(E, 1e-300), 0.0)
    dN = np.where(massive, -Gamma * dtau * N, 0.0)
    return dr, dw, dN, dtau, met


# ----------------------------------------------------------------------------- decay emission
def emit_daughters(P: Particles, rng: np.random.Generator, emit_fraction: float, force: bool = False):
    """Convert pending decayed X mass into pairs of massless daughters. Returns new Particles or None."""
    massive = P.m > 0
    # per-particle randomised threshold in [0.5, 1.5] * emit_fraction * N0: identical thresholds would make all
    # macroparticles emit in synchronised bursts (visible as steps in the decayed-mass history)
    undrawn = np.isnan(P.thr)
    if np.any(undrawn):
        P.thr[undrawn] = np.where(massive[undrawn], emit_fraction * P.N0[undrawn] * rng.uniform(0.5, 1.5, int(undrawn.sum())), np.inf)
    thr = P.thr
    sel = massive & (P.pending > 0) & ((P.pending >= thr) | force)
    if not np.any(sel):
        return None
    idx = np.nonzero(sel)[0]
    n = len(idx)
    r, w, L = P.r[idx], P.w[idx], P.L[idx]
    E = np.sqrt(1.0 + w**2 + L / r**2)
    vperp = np.sqrt(L) / r                     # tangential momentum per unit mass along e_theta
    # boost velocity v = u_spatial / E, gamma = E (unit rest mass)
    vx, vy = w / E, vperp / E
    gam = E
    v2 = vx**2 + vy**2
    # daughter momentum in rest frame: k0 = 1/2, k = (1/2) n_hat, n_hat isotropic
    cos_t = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    sin_t = np.sqrt(1 - cos_t**2)
    nx, ny, nz = cos_t, sin_t * np.cos(phi), sin_t * np.sin(phi)
    out = []
    for sign in (1.0, -1.0):
        kx, ky, kz = 0.5 * sign * nx, 0.5 * sign * ny, 0.5 * sign * nz
        k0 = 0.5
        vk = vx * kx + vy * ky
        fac = np.where(v2 > 0, (gam - 1) * vk / np.maximum(v2, 1e-300), 0.0) + gam * k0
        kx2, ky2 = kx + fac * vx, ky + fac * vy
        kz2 = kz
        wd = kx2
        Ld = r**2 * (ky2**2 + kz2**2)
        out.append(Particles(r.copy(), wd, Ld, P.pending[idx].copy(), np.zeros(n),
                             np.zeros(n), P.pending[idx].copy(), np.zeros(n), np.full(n, np.inf)))
    P.pending[idx] = 0.0
    P.thr[idx] = emit_fraction * P.N0[idx] * rng.uniform(0.5, 1.5, n)
    D = out[0]
    D.append(out[1])
    return D


# ----------------------------------------------------------------------------- driver
@dataclass
class RunConfig:
    Gamma: float = 0.0
    cfl: float = 0.4
    t_end: float = 10.0
    emit_fraction: float = 0.2
    horizon_threshold: float = 0.99      # max 2m/r above which the polar-areal gauge is declared to break down
    max_steps: int = 10_000_000
    diag_every: int = 20
    seed: int = 0
    nsub: int = 4
    store_profiles: bool = False
    store_particles: int = 0            # if >0: keep this many sampled particles per diagnostic snapshot


@dataclass
class Ledger:
    escaped_energy: float = 0.0          # Killing energy of particles that crossed r_out
    escaped_rest_mass: float = 0.0       # X rest mass that crossed r_out
    escaped_daughters: float = 0.0       # number of daughters that crossed r_out
    decayed_rest_mass: float = 0.0       # X rest mass converted to daughters (outside any horizon)
    absorbed_energy: float = 0.0         # Killing energy absorbed by the inner (excision) boundary


class Run:
    def __init__(self, grid: Grid, P: Particles, cfg: RunConfig):
        self.grid, self.P, self.cfg = grid, P, cfg
        self.t = 0.0
        self.step = 0
        self.rng = np.random.default_rng(cfg.seed)
        self.ledger = Ledger()
        self.diag = []
        self.profiles = []
        self.M0 = None
        self.outcome = None

    # -- one RK4 step
    def _step(self, dt):
        P, g, cfg = self.P, self.grid, self.cfg
        r0, w0, N0, tau0 = P.r, P.w, P.N, P.tau
        k1 = rhs(g, P, r0, w0, N0, cfg.Gamma)
        k2 = rhs(g, P, r0 + 0.5 * dt * k1[0], w0 + 0.5 * dt * k1[1], N0 + 0.5 * dt * k1[2], cfg.Gamma)
        k3 = rhs(g, P, r0 + 0.5 * dt * k2[0], w0 + 0.5 * dt * k2[1], N0 + 0.5 * dt * k2[2], cfg.Gamma)
        k4 = rhs(g, P, r0 + dt * k3[0], w0 + dt * k3[1], N0 + dt * k3[2], cfg.Gamma)
        r = r0 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w = w0 + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        N = N0 + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        tau = tau0 + dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        N = np.maximum(N, 0.0)
        P.pending += N0 - N
        # reflection through the regular centre
        neg = r < g.r_in
        if g.r_in == 0.0:
            r[neg] = -r[neg]
            w[neg] = -w[neg]
        P.r, P.w, P.N, P.tau = r, w, N, tau
        self.t += dt
        self.step += 1
        return k1[4]

    def _boundaries(self, met: Metric):
        P, g = self.P, self.grid
        E = P.energy()
        m, mu, _, _ = metric_at(g, met, np.clip(P.r, g.r_in, g.r_out))
        out = P.r >= g.r_out
        Neff = P.N + P.pending
        if np.any(out):
            self.ledger.escaped_energy += float(np.sum(Neff[out] * E[out] * np.exp(mu[out])))
            self.ledger.escaped_rest_mass += float(np.sum(P.N[out] * P.m[out]))
            self.ledger.decayed_rest_mass += float(np.sum(P.pending[out]))   # decayed X leaving the box
            self.ledger.escaped_daughters += float(np.sum(P.N[out] * (P.m[out] == 0)))
        keep = ~out
        if g.r_in > 0.0:
            inn = P.r <= g.r_in
            if np.any(inn):
                self.ledger.absorbed_energy += float(np.sum(Neff[inn] * E[inn] * np.exp(mu[inn])))
                keep &= ~inn
        if not np.all(keep):
            self.P = self.P.select(keep)

    def _decay_emission(self, force=False):
        if self.cfg.Gamma <= 0 and not force:
            return
        D = emit_daughters(self.P, self.rng, self.cfg.emit_fraction, force=force)
        if D is not None:
            self.ledger.decayed_rest_mass += float(np.sum(D.N)) / 2.0   # two daughters per decay
            self.P.append(D)

    def _lambda_c(self, met: Metric):
        m_c = 0.5 * (met.m_edge[1:] + met.m_edge[:-1])
        return -0.5 * np.log(np.maximum(1 - 2 * m_c / self.grid.r_c, 1e-14))

    def momentum_constraint_check(self):
        """Advance one step and compare the finite-difference d lambda/dt with the
        constraint-derived lambda_dot (trapezoidal average).  Returns the relative L2 residual.
        This is the independent consistency check of the constrained scheme: it tests that
        the deposited momentum density j matches the actual mass-flux of the particles."""
        met0 = solve_metric(self.grid, self.P, self.cfg.nsub)
        lam0, ld0 = self._lambda_c(met0), met0.lamdot_c
        dt = self.dt(met0)
        self._step(dt)
        met1 = solve_metric(self.grid, self.P, self.cfg.nsub)
        lam1, ld1 = self._lambda_c(met1), met1.lamdot_c
        fd = (lam1 - lam0) / dt
        mid = 0.5 * (ld0 + ld1)
        num = np.sqrt(np.sum((fd - mid) ** 2))
        # normalise by the energy-density rate scale 4 pi r e^{lambda+mu} rho = e^{mu} S_E/(r dr):
        # the momentum density j is bounded by rho, so this measures the mismatch relative to
        # the largest flux the matter could carry, and stays meaningful when j ~ 0 (static states)
        mu_c = 0.5 * (met1.mu_edge[1:] + met1.mu_edge[:-1])
        den = np.sqrt(np.sum((np.exp(mu_c) * met1.S["SE"] / (self.grid.r_c * self.grid.dr)) ** 2)) + 1e-300
        return float(num / den), met1

    def _diagnostics(self, met: Metric, mc_res=np.nan):
        g, P = self.grid, self.P
        comp = 2.0 * met.m_edge[1:] / g.r_edge[1:]
        k = int(np.argmax(comp))
        emu = np.exp(met.mu_edge)
        massive = P.m > 0
        d = {
            "t": self.t, "step": self.step, "M_in": float(met.M),
            "M_total": float(met.M + self.ledger.escaped_energy + self.ledger.absorbed_energy),
            "rest_mass_X": float(np.sum(P.N[massive])),
            "decayed_rest_mass": self.ledger.decayed_rest_mass,
            "pending": float(np.sum(P.pending)),
            "escaped_energy": self.ledger.escaped_energy,
            "escaped_rest_mass": self.ledger.escaped_rest_mass,
            "max_2m_over_r": float(comp[k]), "r_at_max": float(g.r_edge[k + 1]),
            "min_lapse": float(emu.min()), "n_particles": len(P),
            "n_daughters": int(np.sum(~massive)), "n_bad_cells": met.n_bad,
            "momentum_constraint_rel_residual": mc_res,
            "r_min_particle": float(P.r.min()) if len(P) else np.nan,
        }
        self.diag.append(d)
        if self.cfg.store_profiles:
            snap = {"t": self.t, "m_edge": met.m_edge.copy(), "mu_edge": met.mu_edge.copy(),
                    "SE": met.S["SE"].copy(), "Sj": met.S["Sj"].copy(), "Sp": met.S["Sp"].copy()}
            if self.cfg.store_particles > 0 and len(P):
                ns = min(self.cfg.store_particles, len(P))
                idx = np.linspace(0, len(P) - 1, ns).astype(int)
                snap["particles"] = {k: getattr(P, k)[idx].copy() for k in ("r", "w", "L", "N", "m", "tau")}
            self.profiles.append(snap)
        return d

    def dt(self, met: Metric):
        # coordinate speed of light e^{mu-lambda} <= 1 in this gauge; cfl on the grid spacing
        return self.cfg.cfl * self.grid.dr

    def run(self, callback=None):
        cfg = self.cfg
        met = solve_metric(self.grid, self.P, cfg.nsub)
        self.M0 = met.M
        d = self._diagnostics(met)
        while self.t < cfg.t_end and self.step < cfg.max_steps:
            dt = min(self.dt(met), cfg.t_end - self.t)
            if self.step % cfg.diag_every == 0 and self.step > 0:
                mc_res, met = self.momentum_constraint_check()
            else:
                self._step(dt)
                met = solve_metric(self.grid, self.P, cfg.nsub)
                mc_res = None
            self._boundaries(met)
            self._decay_emission()
            if mc_res is not None:
                met = solve_metric(self.grid, self.P, cfg.nsub)
                d = self._diagnostics(met, mc_res)
                if callback:
                    callback(self, d)
                if d["max_2m_over_r"] >= cfg.horizon_threshold or met.n_bad > 0:
                    self.outcome = "BH_approach"
                    break
                if len(self.P) == 0:
                    self.outcome = "dispersion"
                    break
        if self.outcome is None:
            self.outcome = classify(self.diag, self.M0, cfg)
        met = solve_metric(self.grid, self.P, cfg.nsub)
        self._diagnostics(met)
        return self.outcome


def classify(diag, M0, cfg: RunConfig):
    comp = np.array([d["max_2m_over_r"] for d in diag])
    last = diag[-1]
    if comp.max() >= cfg.horizon_threshold:
        return "BH_approach"
    bound_fraction = last["M_in"] / max(M0, 1e-300)
    if bound_fraction < 0.05:
        return "dispersion"
    imax = int(np.argmax(comp))
    if imax < len(comp) - 1 and comp[-1] < 0.5 * comp.max():
        return "bounce"
    return "unresolved"

"""diagnose(): physical diagnostics of a spherically symmetric slice (contract §5-§10).

Formula tags [F..] refer to physics_contract.md v1.0.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import brentq

from .fd import EPS, ROUNDOFF_FACTOR
from .matter import Projections, ScalarField, VlasovMoments
from .slice import SphericalSlice

FOUR_PI = 4.0 * np.pi

# sphere classes (§7)
NORMAL = "NORMAL"
NORMAL_REVERSED = "NORMAL_REVERSED"
FUTURE_TRAPPED = "FUTURE_TRAPPED"
PAST_TRAPPED = "PAST_TRAPPED"
FUTURE_MARGINAL = "FUTURE_MARGINAL"
PAST_MARGINAL = "PAST_MARGINAL"
DOUBLY_MARGINAL = "DOUBLY_MARGINAL"
CENTER = "CENTER"
INVALID = "INVALID"
UNRESOLVED = "UNRESOLVED"
TAU_MAX = 0.1          # [v1.2 C11] sign of R theta not resolved if tau > TAU_MAX (scale of R theta: 2)
CLASS_DTYPE = "<U16"

_CLASS_TABLE = {
    (1, -1): NORMAL, (-1, 1): NORMAL_REVERSED,
    (-1, -1): FUTURE_TRAPPED, (1, 1): PAST_TRAPPED,
    (0, -1): FUTURE_MARGINAL, (-1, 0): FUTURE_MARGINAL,
    (0, 1): PAST_MARGINAL, (1, 0): PAST_MARGINAL,
    (0, 0): DOUBLY_MARGINAL,
}

# root types / statuses
FUTURE_MOTS, PAST_MOTS, DEGENERATE = "FUTURE_MOTS", "PAST_MOTS", "DEGENERATE"
GEOMETRY_ORDER = ("UNDETERMINED", "AH_CANDIDATE", "FUTURE_TRAPPED_NO_AH_CANDIDATE",
                  "UNCERTAIN", "PAST_TRAPPED_ONLY", "NO_TRAPPED_SPHERES")

DIAGNOSTICS_FIELDS = (
    "Gamma", "U", "dR", "M", "sigma_M", "compactness", "M_over_R3", "M_over_R3_extrapolated",
    "Rtheta_plus", "Rtheta_minus", "sigma_Rtheta_plus", "sigma_Rtheta_minus",
    "theta_plus", "theta_minus", "sphere_class", "roots", "outer_ah_candidate", "projections",
    "ham", "ham_rel", "mom", "mom_rel", "hamM", "hamM_rel",
    "sigma_ham", "sigma_mom", "sigma_hamM", "norms",
    "M_rho", "M_flux", "M_int", "M_prop", "center_checks", "status",
)
NORM_NAMES = ("ham", "mom", "hamM")
NORM_REGIONS = ("all", "center", "bulk", "outer")
NORM_KINDS = ("Linf", "L2vol", "Linf_rel")


class Sign(int):
    """Sign +1 / 0 / -1 that also compares equal to '+', '0', '-'."""

    _SYM = {1: "+", 0: "0", -1: "-"}

    def __new__(cls, v):
        v = 0 if (v is None or not np.isfinite(v)) else int(np.sign(v))
        return super().__new__(cls, v)

    def __eq__(self, other):
        if isinstance(other, str):
            return self._SYM[int(self)] == other
        return int(self) == other

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = int.__hash__

    def __repr__(self):
        return self._SYM[int(self)]

    __str__ = __repr__


class Diagnostics:
    """Result of :func:`diagnose`. All contract fields are attributes (also item access)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dict__

    def keys(self):
        return list(self.__dict__)

    def __repr__(self):
        st = self.__dict__.get("status", {})
        return (f"Diagnostics(N={np.size(self.__dict__.get('M', []))}, data={st.get('data')}, "
                f"constraints={st.get('constraints')}, geometry={st.get('geometry')}, "
                f"roots={len(self.__dict__.get('roots', []))})")


# ---------------------------------------------------------------------------------------------
# small numerical helpers
# ---------------------------------------------------------------------------------------------
def _lagrange(xs, ys, x):
    tot = 0.0
    n = len(xs)
    for k in range(n):
        lk = 1.0
        for j in range(n):
            if j != k:
                lk *= (x - xs[j]) / (xs[k] - xs[j])
        tot += ys[k] * lk
    return tot


def _nodes(i, b0, b1, n=4):
    """n consecutive indices inside [b0, b1] around the interval (i, i+1)."""
    lo = i - (n // 2 - 1)
    lo = max(b0, min(lo, b1 - n + 1))
    hi = min(b1, lo + n - 1)
    return np.arange(lo, hi + 1)


def _interval(r, x, b0, b1):
    if b1 <= b0:
        return b0
    j = int(np.searchsorted(r[b0:b1 + 1], x, side="right")) - 1 + b0
    return min(max(j, b0), b1 - 1)


def _interp(r, f, x, b0, b1, cubic=True):
    if b1 <= b0:
        return float(f[b0])
    j = _interval(r, x, b0, b1)
    if cubic:
        nd = _nodes(j, b0, b1, 4)
        return float(_lagrange(r[nd], f[nd], x))
    t = (x - r[j]) / (r[j + 1] - r[j])
    return float((1 - t) * f[j] + t * f[j + 1])


def _trapz_masked(x, y, mask):
    """Trapezoid integral over consecutive pairs of points that are both in mask."""
    pair = mask[:-1] & mask[1:]
    if not pair.any():
        return 0.0
    seg = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
    return float(np.sum(seg[pair]))


def _even_extrapolate0(r, f, n=3):
    """Value at r = 0 of the even polynomial through (r_k, f_k), k < n (Lagrange in r^2)."""
    x = r[:n] ** 2
    return float(_lagrange(x, f[:n], 0.0))


def _nan(N):
    return np.full(N, np.nan)


def _region_names(center):
    """[v1.1 C10] the inner-edge region is called 'inner' when there is no centre."""
    first = "inner" if center == "none" else "center"
    return ("all", first, "bulk", "outer")


def _empty_norms(center=None):
    return {nm: {rg: {k: np.nan for k in NORM_KINDS} for rg in _region_names(center)}
            for nm in NORM_NAMES}


# ---------------------------------------------------------------------------------------------
# result for inputs that cannot be processed at all
# ---------------------------------------------------------------------------------------------
def _structural_failure(N, errors, warnings, center):
    nan = _nan(N)
    status = {
        "data": "INVALID_INPUT", "constraints": "NOT_EVALUATED", "geometry": "UNDETERMINED",
        "marginal_present": False, "errors": list(errors), "warnings": list(warnings),
        "numerical_failures": [], "n_invalid_points": N, "n_unresolved_points": 0,
    }
    fields = {k: nan.copy() for k in (
        "Gamma", "U", "dR", "M", "sigma_M", "compactness", "M_over_R3",
        "Rtheta_plus", "Rtheta_minus", "sigma_Rtheta_plus", "sigma_Rtheta_minus",
        "theta_plus", "theta_minus", "ham", "ham_rel", "mom", "mom_rel", "hamM", "hamM_rel",
        "sigma_ham", "sigma_mom", "sigma_hamM", "M_rho", "M_flux", "M_int", "M_prop",
        "sigma_Gamma", "sigma_dR", "dGamma", "sigma_dGamma", "dKB", "sigma_dKB", "dM",
        "sigma_dM", "tau_plus", "tau_minus")}
    fields.update(
        M_over_R3_extrapolated=False,
        M_over_R3_extrapolated_mask=np.zeros(N, dtype=bool),
        sphere_class=np.full(N, INVALID, dtype=CLASS_DTYPE),
        sign_plus=np.zeros(N, dtype=int), sign_minus=np.zeros(N, dtype=int),
        roots=[], outer_ah_candidate=None, projections=None, norms=_empty_norms(center),
        center_checks={"center": center, "Gamma0": np.nan, "KA_minus_KB0": np.nan,
                       "Gamma0_minus_1": np.nan, "sigma_Gamma0": np.nan, "R0": np.nan,
                       "M_over_R3_0": np.nan, "extrapolated": False},
        ham_status=np.full(N, "INVALID", dtype=CLASS_DTYPE),
        mom_status=np.full(N, "INVALID", dtype=CLASS_DTYPE),
        hamM_status=np.full(N, "INVALID", dtype=CLASS_DTYPE),
        status=status, kappa_theta=np.nan, kappa_c=np.nan,
    )
    return Diagnostics(**fields)


# ---------------------------------------------------------------------------------------------
# roots of R theta_pm (§7)
# ---------------------------------------------------------------------------------------------
def _blocks(usable):
    """Start/end index arrays (per point) of the contiguous usable run containing it."""
    N = usable.size
    b0 = np.full(N, -1)
    b1 = np.full(N, -1)
    u = usable.astype(np.int8)
    edges = np.diff(np.concatenate([[0], u, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    for a, b in zip(starts, ends):
        b0[a:b + 1] = a
        b1[a:b + 1] = b
    return b0, b1


def _find_roots(which, r, R, M, Gamma, v, tau, s, vo, tauo, usable):
    N = r.size
    b0a, b1a = _blocks(usable)
    z = usable & (s == 0)
    starts = z & ~np.concatenate([[False], z[:-1]])
    run_id = np.where(z, np.cumsum(starts) - 1, -1)
    run_pts = defaultdict(list)
    for i in np.flatnonzero(z):
        run_pts[int(run_id[i])].append(int(i))

    events = []  # (key, kind, i)
    for i in np.flatnonzero(usable & (v == 0.0)):
        events.append((("run", int(run_id[i])), "zero", int(i)))
    pair = usable[:-1] & usable[1:]
    with np.errstate(invalid="ignore"):
        sv = np.sign(v)
        cross = np.flatnonzero(pair & (sv[:-1] * sv[1:] < 0))
    for i in cross:
        i = int(i)
        if run_id[i] >= 0:
            key = ("run", int(run_id[i]))
        elif run_id[i + 1] >= 0:
            key = ("run", int(run_id[i + 1]))
        else:
            key = ("cross", i)
        events.append((key, "cross", i))

    clusters = defaultdict(list)
    for key, kind, i in events:
        clusters[key].append((kind, i))

    roots = []
    for key, evs in clusters.items():
        pts = set()
        for kind, i in evs:
            pts.add(i)
            if kind == "cross":
                pts.add(i + 1)
        if key[0] == "run":
            pts.update(run_pts[key[1]])
        pts = sorted(pts)
        lo, hi = pts[0], pts[-1]
        b0, b1 = int(b0a[lo]), int(b1a[lo])
        inner = s[lo] if s[lo] != 0 else (s[lo - 1] if lo - 1 >= b0 else 0)
        outer = s[hi] if s[hi] != 0 else (s[hi + 1] if hi + 1 <= b1 else 0)

        locs = []  # (r_cubic, r_linear, slope)
        for kind, i in sorted(evs, key=lambda e: e[1]):
            if kind == "zero":
                il, ir = max(i - 1, b0), min(i + 1, b1)
                slope = (v[ir] - v[il]) / (r[ir] - r[il]) if ir > il else 0.0
                locs.append((float(r[i]), float(r[i]), slope))
            else:
                t = v[i] / (v[i] - v[i + 1])
                r_lin = float(r[i] + t * (r[i + 1] - r[i]))
                nd = _nodes(i, b0, b1, 4)
                xs, ys = r[nd], v[nd]
                try:
                    r_cub = float(brentq(lambda x: _lagrange(xs, ys, x), r[i], r[i + 1],
                                         xtol=1e-15 * max(1.0, abs(r[i + 1])), rtol=4 * EPS))
                except ValueError:
                    r_cub = r_lin
                slope = (v[i + 1] - v[i]) / (r[i + 1] - r[i])
                locs.append((r_cub, r_lin, slope))
        locs.sort()
        r_star, _, slope = locs[len(locs) // 2]
        tau_star = _interp(r, tau, r_star, b0, b1, cubic=False)
        band = tau_star / abs(slope) if slope != 0 else np.inf
        cand = [c for c, _, _ in locs] + [l for _, l, _ in locs] + [r_star - band, r_star + band]
        zpts = [p for p in pts if s[p] == 0]
        if zpts:
            cand += [float(r[zpts[0]]), float(r[zpts[-1]])]
        r_min = float(r[max(lo - 1, b0)])
        r_max = float(r[min(hi + 1, b1)])
        r_lo = float(min(max(min(cand), r_min), r_star))
        r_hi = float(max(min(max(cand), r_max), r_star))

        R_star = _interp(r, R, r_star, b0, b1)
        # R at r*, r_lo, r_hi from cubic AND linear interpolation (the difference bounds the
        # interpolation error of R itself, which matters where R' ~ 0, e.g. at a throat)
        R_all = [R_star] + [_interp(r, R, x, b0, b1, cubic=cu)
                            for x in (r_lo, r_star, r_hi) for cu in (True, False)]
        R_lo, R_hi = min(R_all), max(R_all)
        M_star = _interp(r, M, r_star, b0, b1)
        G_star = _interp(r, Gamma, r_star, b0, b1)

        # sign of the other expansion at the root (with interpolation uncertainty)
        vo_c = _interp(r, vo, r_star, b0, b1)
        vo_l = _interp(r, vo, r_star, b0, b1, cubic=False)
        to = _interp(r, tauo, r_star, b0, b1, cubic=False) + abs(vo_c - vo_l)
        if not (np.isfinite(vo_c) and np.isfinite(to)) or abs(vo_c) <= to:
            other = 0
        else:
            other = int(np.sign(vo_c))
            for x in (r_lo, r_hi):
                ve = _interp(r, vo, x, b0, b1)
                if not np.isfinite(ve) or ve * other < 0:
                    other = 0
        rtype = FUTURE_MOTS if other < 0 else (PAST_MOTS if other > 0 else DEGENERATE)
        # [v1.1 C1] orientation-invariant criterion: the zero must be of the OUTGOING expansion
        # theta_out (towards growing R): theta_+ if Gamma > 0, theta_- if Gamma < 0 at the root;
        # theta_out < 0 on the smaller-R side and > 0 on the larger-R side.
        if G_star > 0:
            smallR, largeR, is_out = inner, outer, which == "+"
        elif G_star < 0:
            smallR, largeR, is_out = outer, inner, which == "-"
        else:
            smallR, largeR, is_out = 0, 0, False
        ah = bool(is_out and rtype == FUTURE_MOTS and smallR == -1 and largeR == 1)
        roots.append({
            "which": which, "r": r_star, "R": R_star, "R_lo": R_lo, "R_hi": R_hi,
            "M": M_star, "type": rtype, "ah_candidate": ah,
            "other_sign": Sign(other), "inner_sign": Sign(inner), "outer_sign": Sign(outer),
            "r_lo": r_lo, "r_hi": r_hi, "Gamma": G_star, "n_crossings": len(evs),
            "tangential": bool(inner != 0 and inner == outer), "points": pts,
            "outgoing": bool(is_out), "smallR_sign": Sign(smallR), "largeR_sign": Sign(largeR),
        })
    return roots


# ---------------------------------------------------------------------------------------------
def _matter_projections(slc, matter, errors):
    """Return (Projections or None, bad-point mask)."""
    N = slc.N
    bad = np.zeros(N, dtype=bool)
    if matter is None:
        return None, bad
    if not isinstance(matter, (ScalarField, VlasovMoments, Projections)):
        errors.append(f"matter of unsupported type {type(matter).__name__}")
        return None, np.ones(N, dtype=bool)
    if isinstance(matter, (ScalarField, VlasovMoments)):
        errs = matter.check(N)
        if errs:
            errors.extend(errs)
            return None, np.ones(N, dtype=bool)
        inputs = ((matter.phi, matter.Pi) if isinstance(matter, ScalarField)
                  else (matter.E, matter.J, matter.p, matter.q))
        for a in inputs:
            bad |= ~np.isfinite(a)
        if bad.any():
            errors.append(f"matter input: non-finite values at {int(bad.sum())} point(s)")
    proj = matter.projections(slc)
    errs = proj.check(N)
    if errs:
        errors.extend(errs)
        return None, np.ones(N, dtype=bool)
    pbad = np.zeros(N, dtype=bool)
    for name in ("rho", "j_r", "S_rr", "S_thth", "sigma_rho", "sigma_j_r"):
        a = getattr(proj, name)
        if a is not None:
            pbad |= ~np.isfinite(a)
    if isinstance(matter, Projections) and pbad.any():
        errors.append(f"matter projections: non-finite values at {int(pbad.sum())} point(s)")
    # for ScalarField / Vlasov a non-finite projection caused by finite inputs is a numerical
    # failure (reported later), unless the inputs themselves were bad.
    if isinstance(matter, Projections):
        bad = bad | pbad
    return proj, bad


def diagnose(slc, matter=None, *, kappa_theta=3.0, kappa_c=10.0):
    """Diagnose a SphericalSlice (optionally with matter). See physics_contract.md §10."""
    if not isinstance(slc, SphericalSlice):
        return _structural_failure(0, [f"slc is not a SphericalSlice ({type(slc).__name__})"],
                                   [], None)
    errors = list(slc.errors)
    warnings = list(slc.warnings)
    try:
        kappa_theta = float(kappa_theta)
        kappa_c = float(kappa_c)
    except (TypeError, ValueError):
        kappa_theta = kappa_c = np.nan
    if not (np.isfinite(kappa_theta) and kappa_theta >= 0 and np.isfinite(kappa_c)
            and kappa_c >= 0):
        errors.append("kappa_theta and kappa_c must be finite and >= 0")
    if not slc.structure_ok:
        return _structural_failure(slc.N, errors, warnings, slc.center)

    N = slc.N
    op = slc.op
    r, A, R, KA, KB = slc.r, slc.A, slc.R, slc.KA, slc.KB
    center = slc.center
    proj, matter_bad = _matter_projections(slc, matter, errors)
    has_matter = matter is not None
    bad_in = slc.bad_points | matter_bad

    with np.errstate(all="ignore"):
        # ---- R', Gamma, U (F1) --------------------------------------------------------------
        if slc.dR is not None:
            dR = slc.dR.copy()
            dR6 = dR
            sig_dR = np.zeros(N)
            ef_dR = EPS * np.abs(dR)
        else:
            dR, sig_dR = op.d_err(R, R, -1)
            dR6 = op.d(R, -1, 6)
            ef_dR = ROUNDOFF_FACTOR * op.abs_apply(EPS * np.abs(R), 1)
        Gamma = dR / A
        Gamma6 = dR6 / A
        sig_G = sig_dR / A
        ef_G = ef_dR / A + EPS * np.abs(Gamma)
        U = -R * KB

        # ---- Misner-Sharp mass (F11) ----------------------------------------------------------
        M = 0.5 * R * (1.0 - Gamma ** 2 + U ** 2)
        M6 = 0.5 * R * (1.0 - Gamma6 ** 2 + U ** 2)
        sig_M = R * np.abs(Gamma) * sig_G
        ef_M = (EPS * 0.5 * R * (1.0 + Gamma ** 2 + U ** 2) + R * np.abs(Gamma) * ef_G
                + EPS * R * U ** 2)

        # ---- Gamma' -------------------------------------------------------------------------
        if slc.dA is not None:
            # Gamma' = (R'' - Gamma A') / A with R'' by FD of R' (even) and analytic A'
            ddR4, sig_ddR = op.d_err(dR, dR6, +1, ef=ef_dR)
            dG = (ddR4 - Gamma * slc.dA) / A
            sig_dG = sig_ddR / A
        else:
            dG, sig_dG = op.d_err(Gamma, Gamma6, +1, ef=ef_G)
        # ---- K_B' ---------------------------------------------------------------------------
        if slc.dKB is not None:
            dKB = slc.dKB.copy()
            sig_dKB = np.zeros(N)
        else:
            dKB, sig_dKB = op.d_err(KB, KB, +1)
        # ---- M' (numerical derivative of M itself; M is odd at a regular centre) ------------
        dM, sig_dM = op.d_err(M, M6, -1, ef=ef_M)

        # ---- expansions (F12) -----------------------------------------------------------------
        Rth_p = 2.0 * (U + Gamma)
        Rth_m = 2.0 * (U - Gamma)
        sig_Rth = 2.0 * sig_G
        is_center = (R == 0.0) & ~bad_in
        Rpos = R > 0
        theta_p = np.where(Rpos, Rth_p / np.where(Rpos, R, 1.0), np.nan)
        theta_m = np.where(Rpos, Rth_m / np.where(Rpos, R, 1.0), np.nan)
        floor = 1e3 * EPS * (np.abs(U) + np.abs(Gamma))
        # [v1.1 C6] grid noise of U (odd) and Gamma (even) enters the zero band
        noise_U = op.sigma_noise(U, -1)
        noise_G = op.sigma_noise(Gamma, +1)
        tau_p = np.maximum(kappa_theta * (sig_Rth + 2.0 * noise_U + 2.0 * noise_G), floor)
        tau_m = tau_p.copy()

        def _sign(v, tau):
            s = np.sign(v).astype(int)
            s[~(np.abs(v) > tau)] = 0
            return s

        s_p = _sign(Rth_p, tau_p)
        s_m = _sign(Rth_m, tau_m)

        # ---- compactness, M/R^3 ---------------------------------------------------------------
        compact = np.where(Rpos, 2.0 * M / np.where(Rpos, R, 1.0), 0.0)
        compact[~np.isfinite(R) | ~np.isfinite(M)] = np.nan
        MR3 = np.where(Rpos, M / np.where(Rpos, R, 1.0) ** 3, np.nan)
        extrap_mask = is_center.copy()
        if extrap_mask.any():
            # lim_{R->0} M/R^3 = -(Gamma'^2 + Gamma Gamma'')/(2 R'^2) + K_B^2/2
            # (= 3R(0)/12 + K_B(0)^2/2 at a regular centre, derivation.md §4)
            ddG = op.d(dG, -1, 4)
            lim = -(dG ** 2 + Gamma * ddG) / (2.0 * dR ** 2) + 0.5 * KB ** 2
            MR3[extrap_mask] = lim[extrap_mask]

        # ---- constraints (F7-F10) -------------------------------------------------------------
        if proj is not None:
            rho, j_r = proj.rho, proj.j_r
            sig_rho = proj.sigma_rho if proj.sigma_rho is not None else np.zeros(N)
            sig_j = proj.sigma_j_r if proj.sigma_j_r is not None else np.zeros(N)
        else:
            rho = j_r = sig_rho = sig_j = _nan(N)
        Rs = np.where(Rpos, R, np.nan)          # R = 0 excluded from H and M (0/0)
        hT = (2.0 * (1.0 - Gamma ** 2) / Rs ** 2, -4.0 * dG / (A * Rs), 4.0 * KA * KB,
              2.0 * KB ** 2, -4.0 * FOUR_PI * rho)
        ham = sum(hT)
        ham_abs = sum(np.abs(t) for t in hT)
        sig_ham = (4.0 * np.abs(Gamma) * sig_G / Rs ** 2 + 4.0 * sig_dG / (A * Rs)
                   + 4.0 * FOUR_PI * sig_rho)
        mT = (dKB, -(dR / Rs) * (KA - KB), FOUR_PI * j_r)
        mom = sum(mT)
        mom_abs = sum(np.abs(t) for t in mT)
        sig_mom = sig_dKB + np.abs(KA - KB) * sig_dR / Rs + FOUR_PI * sig_j
        HT = (dM, -FOUR_PI * R ** 2 * rho * dR, -FOUR_PI * R ** 2 * U * j_r)
        hamM = sum(HT)
        hamM_abs = sum(np.abs(t) for t in HT)
        sig_hamM = (sig_dM + FOUR_PI * R ** 2 * np.abs(rho) * sig_dR
                    + FOUR_PI * R ** 2 * (np.abs(dR) * sig_rho + np.abs(U) * sig_j))

        def _rel(X, Tabs):
            return np.where(Tabs > 0, np.abs(X) / np.where(Tabs > 0, Tabs, 1.0), np.nan)

        ham_rel, mom_rel, hamM_rel = _rel(ham, ham_abs), _rel(mom, mom_abs), _rel(hamM, hamM_abs)
        if not has_matter:
            for arr in (ham, mom, hamM, ham_rel, mom_rel, hamM_rel, sig_ham, sig_mom, sig_hamM):
                arr[:] = np.nan

        # [v1.1 C4] round-off scale S_X of the terms *before* cancellation (derivative terms:
        # stencil magnitude sum_k |w_k||f_k|), so that the VIOLATED floor never vanishes
        absG = op.abs_apply(Gamma, 1)
        S_ham = (2.0 * (1.0 + Gamma ** 2) / Rs ** 2 + 4.0 * absG / (A * Rs)
                 + 4.0 * np.abs(KA * KB) + 2.0 * KB ** 2 + 4.0 * FOUR_PI * np.abs(rho))
        S_mom = ((np.abs(dKB) if slc.dKB is not None else op.abs_apply(KB, 1))
                 + np.abs(dR / Rs) * (np.abs(KA) + np.abs(KB)) + FOUR_PI * np.abs(j_r))
        Mtilde = 0.5 * R * (1.0 + Gamma ** 2 + U ** 2)
        S_hamM = (op.abs_apply(Mtilde, 1) + FOUR_PI * R ** 2 * np.abs(rho * dR)
                  + FOUR_PI * R ** 2 * np.abs(U * j_r))

        # Resolution scale of H_M: sum|T_k| with the M' term replaced by the magnitude of its
        # un-cancelled pieces (R'/2)(1+Gamma^2+U^2) + R(|Gamma Gamma'| + |U U'|). The literal
        # sum|T_k| contains only |M'| in vacuum (M' ~ 0 there), which would make every vacuum
        # slice UNRESOLVED. Used ONLY for the resolution test, not for H_M or its rel value.
        dU = -dR * KB - R * dKB
        res_hamM = (0.5 * np.abs(dR) * (1.0 + Gamma ** 2 + U ** 2)
                    + R * (np.abs(Gamma * dG) + np.abs(U * dU))
                    + np.abs(HT[1]) + np.abs(HT[2]))

        def _pstatus(X, sig, Tabs, S, evaluated, Tres=None):
            Tres = Tabs if Tres is None else Tres
            st = np.full(N, "NOT_EVALUATED", dtype=CLASS_DTYPE)
            if not has_matter:
                return st
            fin = np.isfinite(X) & np.isfinite(sig) & np.isfinite(Tabs)
            fl = 1e3 * EPS * np.maximum(Tabs, np.where(np.isfinite(S), S, 0.0))
            viol = fin & (np.abs(X) > kappa_c * sig + fl)
            # [v1.1 C7] resolution: sigma(X) > 0.1 sum|T_k| (and above the round-off floor; at
            # R = 0 all terms vanish identically and resolution is not defined)
            unres = fin & ~viol & (sig > 0.1 * Tres) & (sig > fl) & (R != 0.0)
            st[evaluated & fin] = "CONSISTENT"
            st[evaluated & unres] = "UNRESOLVED"
            st[evaluated & viol] = "VIOLATED"
            st[evaluated & ~fin] = "INVALID"
            st[~evaluated] = "CENTER_EXCLUDED"
            return st

        allpts = np.ones(N, dtype=bool)
        ham_status = _pstatus(ham, sig_ham, ham_abs, S_ham, ~(R == 0.0))
        mom_status = _pstatus(mom, sig_mom, mom_abs, S_mom, ~(R == 0.0))
        hamM_status = _pstatus(hamM, sig_hamM, hamM_abs, S_hamM, allpts, res_hamM)

        # ---- integral masses (§6) -------------------------------------------------------------
        if has_matter and proj is not None:
            M_rho = op.cumint(FOUR_PI * R ** 2 * rho * dR, +1)
            M_flux = op.cumint(FOUR_PI * R ** 2 * U * j_r, +1)
            M_int = M[0] + M_rho + M_flux
            M_prop = op.cumint(FOUR_PI * R ** 2 * A * rho, +1)
        else:
            M_rho = M_flux = M_int = M_prop = _nan(N)

    # ---- norms by region (§9) ------------------------------------------------------------------
    pos = np.flatnonzero(Rpos)
    n_c = max(4, N // 20)
    regions = {"all": np.ones(N, dtype=bool)}
    cmask = np.zeros(N, dtype=bool)
    cmask[pos[:n_c]] = True
    omask = np.zeros(N, dtype=bool)
    omask[N - 4:] = True
    inner_name = _region_names(center)[1]
    regions[inner_name] = cmask
    regions["bulk"] = Rpos & ~cmask & ~omask
    regions["outer"] = omask
    dV = FOUR_PI * R ** 2 * A
    norms = _empty_norms(center)
    if has_matter:
        for name, X, Xr in (("ham", ham, ham_rel), ("mom", mom, mom_rel),
                            ("hamM", hamM, hamM_rel)):
            for rg, msk in regions.items():
                fin = msk & np.isfinite(X)
                if fin.any():
                    norms[name][rg]["Linf"] = float(np.max(np.abs(X[fin])))
                    with np.errstate(all="ignore"):
                        norms[name][rg]["L2vol"] = float(np.sqrt(max(
                            _trapz_masked(r, np.where(fin, X ** 2 * dV, 0.0), fin), 0.0)))
                finr = msk & np.isfinite(Xr)
                if finr.any():
                    norms[name][rg]["Linf_rel"] = float(np.max(Xr[finr]))

    # ---- centre checks (§3) --------------------------------------------------------------------
    cc = {"center": center, "Gamma0": np.nan, "KA_minus_KB0": np.nan, "Gamma0_minus_1": np.nan,
          "sigma_Gamma0": np.nan, "R0": float(R[0]), "M_over_R3_0": np.nan,
          "extrapolated": False}
    if center == "vertex":
        cc.update(Gamma0=float(Gamma[0]), KA_minus_KB0=float(KA[0] - KB[0]),
                  sigma_Gamma0=float(sig_G[0]), M_over_R3_0=float(MR3[0]))
    elif center == "cell":
        cc.update(Gamma0=_even_extrapolate0(r, Gamma), KA_minus_KB0=_even_extrapolate0(r, KA - KB),
                  sigma_Gamma0=float(np.max(sig_G[:3])),
                  M_over_R3_0=_even_extrapolate0(r, MR3), extrapolated=True)
    cc["Gamma0_minus_1"] = cc["Gamma0"] - 1.0

    # ---- sphere classes (§7) -------------------------------------------------------------------
    geo_ok = (np.isfinite(Gamma) & np.isfinite(U) & np.isfinite(Rth_p) & np.isfinite(Rth_m)
              & np.isfinite(sig_Rth) & np.isfinite(M) & np.isfinite(R))
    invalid_pt = bad_in | ~geo_ok
    lut = np.array([_CLASS_TABLE[(a, b)] for a in (-1, 0, 1) for b in (-1, 0, 1)],
                   dtype=CLASS_DTYPE)
    cls = lut[(s_p + 1) * 3 + (s_m + 1)]
    # [v1.2 C11] unresolved sign: tau_+ or tau_- > TAU_MAX (not for CENTER / INVALID points)
    unres_pt = ((tau_p > TAU_MAX) | (tau_m > TAU_MAX)) & ~(R == 0.0) & ~invalid_pt
    cls[unres_pt] = UNRESOLVED
    cls[(R == 0.0) & ~invalid_pt] = CENTER
    cls[invalid_pt] = INVALID
    s_p = np.where(invalid_pt, 0, s_p)
    s_m = np.where(invalid_pt, 0, s_m)

    # ---- roots -------------------------------------------------------------------------------
    usable = ~invalid_pt
    roots = (_find_roots("+", r, R, M, Gamma, Rth_p, tau_p, s_p, Rth_m, tau_m, usable)
             + _find_roots("-", r, R, M, Gamma, Rth_m, tau_m, s_m, Rth_p, tau_p, usable))
    roots.sort(key=lambda d: (d["r"], d["which"]))
    ah_roots = [d for d in roots if d["ah_candidate"]]
    outer_ah = max(ah_roots, key=lambda d: d["R"]) if ah_roots else None   # outermost by R

    # ---- data status (§9) ----------------------------------------------------------------------
    failures = []
    if not errors:
        expect_all = ~bad_in
        checks = [("Gamma", Gamma), ("U", U), ("dR", dR), ("M", M), ("sigma_M", sig_M),
                  ("compactness", compact), ("M_over_R3", MR3), ("Rtheta_plus", Rth_p),
                  ("Rtheta_minus", Rth_m), ("sigma_Rtheta", sig_Rth)]
        for name, X in checks:
            if (~np.isfinite(X) & expect_all).any():
                failures.append(name)
        for name, X in (("theta_plus", theta_p), ("theta_minus", theta_m)):
            if (~np.isfinite(X) & Rpos).any():
                failures.append(name)
        if has_matter:
            for name, X in (("ham", ham), ("mom", mom), ("sigma_ham", sig_ham),
                            ("sigma_mom", sig_mom)):
                if (~np.isfinite(X) & Rpos).any():
                    failures.append(name)
            for name, X in (("hamM", hamM), ("sigma_hamM", sig_hamM), ("rho", rho),
                            ("j_r", j_r), ("M_rho", M_rho), ("M_flux", M_flux),
                            ("M_int", M_int), ("M_prop", M_prop)):
                if (~np.isfinite(X)).any():
                    failures.append(name)
    if errors:
        data_status = "INVALID_INPUT"
    elif failures:
        data_status = "NUMERICAL_FAILURE"
    else:
        data_status = "OK"

    # ---- constraint status -------------------------------------------------------------------
    # ---- [v1.1 C9] warnings (do not change the data status) ------------------------------------
    extra = {"center_deviation": False, "M_int_mismatch": np.nan, "n_M_negative": 0,
             "n_rel_undefined": {"ham": 0, "mom": 0, "hamM": 0}}
    if center in ("vertex", "cell") and np.isfinite(cc["Gamma0"]):
        Kmax = float(np.nanmax(np.abs(np.concatenate([KA, KB]))))
        dG0 = abs(cc["Gamma0"] - 1.0)
        dK0 = abs(cc["KA_minus_KB0"])
        if dG0 > max(kappa_theta * cc["sigma_Gamma0"], 1e-8) or dK0 > 1e-8 * Kmax:
            extra["center_deviation"] = True
            warnings.append(f"centre deviation: |Gamma(0)-1| = {dG0:.3e}, "
                            f"|K_A(0)-K_B(0)| = {dK0:.3e} (regularity not satisfied/resolved)")
    with np.errstate(all="ignore"):
        nM = int(np.sum(M < -(kappa_theta * sig_M + 1e3 * EPS * Mtilde)))
    if nM:
        extra["n_M_negative"] = nM
        warnings.append(f"M_MS < -kappa sigma_M at {nM} point(s)")
    if has_matter and proj is not None:
        with np.errstate(all="ignore"):
            den = np.nanmax(np.abs(M)) + np.nanmax(sig_M)
            mism = float(np.nanmax(np.abs(M_int - M)) / den) if den > 0 else \
                float(np.nanmax(np.abs(M_int - M)))
        extra["M_int_mismatch"] = mism
        if not np.isfinite(mism) or mism > 1e-6:
            warnings.append(f"M_int_mismatch: max|M_int - M_MS|/(max|M_MS| + max sigma_M) = "
                            f"{mism:.3e}")
        for name, Tabs, ev in (("ham", ham_abs, R != 0.0), ("mom", mom_abs, R != 0.0),
                               ("hamM", hamM_abs, allpts)):
            n = int(np.sum(ev & np.isfinite(Tabs) & (Tabs == 0)))
            extra["n_rel_undefined"][name] = n
        if any(extra["n_rel_undefined"].values()):
            warnings.append(f"n_rel_undefined: relative residuals undefined (all terms zero) at "
                            f"{extra['n_rel_undefined']} point(s)")
    if unres_pt.any():
        Ru = R[unres_pt]
        warnings.append(f"{int(unres_pt.sum())} UNRESOLVED sphere(s) (tau > {TAU_MAX}) at "
                        f"R in [{Ru.min():.4g}, {Ru.max():.4g}]: geometry at best UNCERTAIN")
        inside = [d for d in ah_roots if np.any(Ru < d["R"])]
        if inside:
            warnings.append(f"UNRESOLVED sphere(s) lie inside (in R) the AH candidate(s) at R = "
                            f"{[round(d['R'], 6) for d in inside]}: the trapped interior is "
                            "not resolved")
    if center == "cell" and has_matter:
        warnings.append("center='cell': M_rho, M_flux, M_prop start at r[0] > 0; the ball "
                        "[0, r[0]] is not included (M_int starts from M_MS(r[0]))")

    # [v1.1 C7] priority NOT_EVALUATED (no matter or data != OK) > VIOLATED > UNRESOLVED > CONSISTENT
    # the 5 % rule is applied per residual X (fraction of the points where X was evaluated)
    sts = (ham_status, mom_status, hamM_status)
    frac_unres = []
    for st in sts:
        n_eval = int(np.isin(st, ("CONSISTENT", "VIOLATED", "UNRESOLVED")).sum())
        frac_unres.append(int((st == "UNRESOLVED").sum()) / n_eval if n_eval else 0.0)
    unres_inner = any(bool(((st == "UNRESOLVED") & cmask).any()) for st in sts)
    if not has_matter or data_status != "OK":
        c_status = "NOT_EVALUATED"
    elif any((st == "VIOLATED").any() for st in sts):
        c_status = "VIOLATED"
    elif max(frac_unres) > 0.05 or unres_inner:
        c_status = "UNRESOLVED"
    else:
        c_status = "CONSISTENT_WITH_TRUNCATION"

    # ---- geometry status (§7, precedence) ------------------------------------------------------
    trapped = np.isin(cls, (FUTURE_TRAPPED, PAST_TRAPPED))
    point_roots = defaultdict(list)
    for d in roots:
        for p in d["points"]:
            point_roots[p].append(d)
    uncertain = False
    unexplained = []
    def _resolved(d):
        # root of known type whose two sides are resolved and differ (a tangential zero --
        # MOTS pair birth -- or a root touching the grid edge does not count)
        return (d["type"] in (FUTURE_MOTS, PAST_MOTS) and d["inner_sign"] != 0
                and d["outer_sign"] != 0 and not d["tangential"])

    # [v1.1 C2/C3] marginal points of ANY type must be explained by a resolved root of known
    # type; exception (priority over UNCERTAIN): isolated DOUBLY_MARGINAL / DEGENERATE point
    # without trapped neighbours -> NO_TRAPPED_SPHERES + marginal_present
    exempt = np.zeros(N, dtype=bool)
    for i in np.flatnonzero(np.isin(cls, (FUTURE_MARGINAL, PAST_MARGINAL, DOUBLY_MARGINAL))):
        rel = [d for d in point_roots.get(int(i), [])
               if (d["which"] == "+" and s_p[i] == 0) or (d["which"] == "-" and s_m[i] == 0)]
        if any(_resolved(d) for d in rel):
            continue
        # [v1.2 C11] isolated = both neighbours exist and are resolved, non-marginal,
        # non-trapped (NORMAL / NORMAL_REVERSED / CENTER); tau_pm <= TAU_MAX holds here because
        # otherwise the point itself would be UNRESOLVED
        isolated = (0 < i < N - 1 and cls[i - 1] in (NORMAL, NORMAL_REVERSED, CENTER)
                    and cls[i + 1] in (NORMAL, NORMAL_REVERSED, CENTER)
                    and tau_p[i] <= TAU_MAX and tau_m[i] <= TAU_MAX)
        if isolated and (cls[i] == DOUBLY_MARGINAL
                         or (rel and all(d["type"] == DEGENERATE for d in rel))):
            exempt[i] = True
            continue
        uncertain = True
        unexplained.append(int(i))
    degenerate_near_trapped = False
    for d in roots:
        if d["type"] != DEGENERATE:
            continue
        nb = set()
        for p in d["points"]:
            nb.update((p - 1, p, p + 1))
        nb = [p for p in nb if 0 <= p < N]
        if trapped[nb].any():
            degenerate_near_trapped = True
    uncertain = uncertain or degenerate_near_trapped

    if data_status != "OK" or (cls == INVALID).any():
        geometry = "UNDETERMINED"
    elif ah_roots:
        geometry = "AH_CANDIDATE"
    elif (cls == FUTURE_TRAPPED).any():
        geometry = "FUTURE_TRAPPED_NO_AH_CANDIDATE"
    elif uncertain or unres_pt.any():
        # [v1.2 C11] any UNRESOLVED point: geometry at best UNCERTAIN
        geometry = "UNCERTAIN"
    else:
        past = ((cls == PAST_TRAPPED).any()
                or any(d["type"] == PAST_MOTS and _resolved(d) for d in roots))
        future = any(d["type"] == FUTURE_MOTS for d in roots)
        benign = np.isin(cls, (NORMAL, NORMAL_REVERSED, CENTER)) | exempt
        if past and not future:
            geometry = "PAST_TRAPPED_ONLY"
        elif benign.all() and not future and not past:
            geometry = "NO_TRAPPED_SPHERES"
        else:
            # not covered by the §7 list (e.g. a non-candidate future MOTS without trapped
            # points, or past and future MOTS together): conservative choice
            geometry = "UNCERTAIN"
    marginal_present = bool(np.isin(cls, (FUTURE_MARGINAL, PAST_MARGINAL, DOUBLY_MARGINAL)).any())

    status = {
        "data": data_status, "constraints": c_status, "geometry": geometry,
        "marginal_present": marginal_present, "errors": errors, "warnings": warnings,
        "numerical_failures": failures, "n_invalid_points": int(invalid_pt.sum()),
        "uncertain_points": unexplained, "degenerate_root_near_trapped": degenerate_near_trapped,
        **extra,
        "n_unresolved_points": int(unres_pt.sum()),
        "n_unresolved": {"ham": int((ham_status == "UNRESOLVED").sum()),
                         "mom": int((mom_status == "UNRESOLVED").sum()),
                         "hamM": int((hamM_status == "UNRESOLVED").sum())},
        "n_violated": {"ham": int((ham_status == "VIOLATED").sum()),
                       "mom": int((mom_status == "VIOLATED").sum()),
                       "hamM": int((hamM_status == "VIOLATED").sum())},
    }

    return Diagnostics(
        r=r, R=R, center=center,
        Gamma=Gamma, U=U, dR=dR, sigma_dR=sig_dR, sigma_Gamma=sig_G,
        dGamma=dG, sigma_dGamma=sig_dG, dKB=dKB, sigma_dKB=sig_dKB, dM=dM, sigma_dM=sig_dM,
        M=M, sigma_M=sig_M, compactness=compact, M_over_R3=MR3,
        M_over_R3_extrapolated=bool(extrap_mask.any()),
        M_over_R3_extrapolated_mask=extrap_mask,
        Rtheta_plus=Rth_p, Rtheta_minus=Rth_m,
        sigma_Rtheta_plus=sig_Rth, sigma_Rtheta_minus=sig_Rth.copy(),
        theta_plus=theta_p, theta_minus=theta_m, tau_plus=tau_p, tau_minus=tau_m,
        sign_plus=s_p, sign_minus=s_m, sphere_class=cls,
        roots=roots, outer_ah_candidate=outer_ah, projections=proj,
        ham=ham, ham_rel=ham_rel, mom=mom, mom_rel=mom_rel, hamM=hamM, hamM_rel=hamM_rel,
        sigma_ham=sig_ham, sigma_mom=sig_mom, sigma_hamM=sig_hamM,
        ham_status=ham_status, mom_status=mom_status, hamM_status=hamM_status,
        norms=norms, M_rho=M_rho, M_flux=M_flux, M_int=M_int, M_prop=M_prop,
        center_checks=cc, status=status, kappa_theta=kappa_theta, kappa_c=kappa_c,
    )

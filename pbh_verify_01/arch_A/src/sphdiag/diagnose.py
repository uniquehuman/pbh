"""diagnose(): constraints, Misner-Sharp mass, null expansions, sphere classification
(contract sections 5-9).  The module measures; it never evolves data or solves constraints."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import brentq

from .fd import EPS, ROUND_C, FDGrid, even_extrapolate_to_zero, lagrange_eval
from .matter import Projections, ScalarField, VlasovMoments
from .slice import SphericalSlice

FOURPI = 4.0 * np.pi
TOL_FLOOR = 1.0e3 * EPS

_CLASS_TABLE = {
    (1, -1): "NORMAL",
    (-1, 1): "NORMAL_REVERSED",
    (-1, -1): "FUTURE_TRAPPED",
    (1, 1): "PAST_TRAPPED",
    (0, -1): "FUTURE_MARGINAL",
    (-1, 0): "FUTURE_MARGINAL",
    (0, 1): "PAST_MARGINAL",
    (1, 0): "PAST_MARGINAL",
    (0, 0): "DOUBLY_MARGINAL",
}
MARGINAL = ("FUTURE_MARGINAL", "PAST_MARGINAL", "DOUBLY_MARGINAL")
TRAPPED = ("FUTURE_TRAPPED", "PAST_TRAPPED")


class Sign(int):
    """Sign in {+1, 0, -1}; also compares equal to the strings '+', '0', '-'."""
    _STR = {1: "+", 0: "0", -1: "-"}

    def __new__(cls, v):
        v = int(v)
        if v not in (-1, 0, 1):
            raise ValueError(v)
        return super().__new__(cls, v)

    def __eq__(self, other):
        if isinstance(other, str):
            return self._STR[int(self)] == other
        return int(self) == other

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = int.__hash__

    def __str__(self):
        return self._STR[int(self)]

    def __repr__(self):
        return f"Sign({self._STR[int(self)]!r})"


@dataclass
class Diagnostics:
    r: np.ndarray
    R: np.ndarray
    Gamma: np.ndarray
    U: np.ndarray
    dR: np.ndarray
    M: np.ndarray
    sigma_M: np.ndarray
    compactness: np.ndarray
    M_over_R3: np.ndarray
    M_over_R3_extrapolated: bool
    Rtheta_plus: np.ndarray
    Rtheta_minus: np.ndarray
    sigma_Rtheta_plus: np.ndarray
    sigma_Rtheta_minus: np.ndarray
    theta_plus: np.ndarray
    theta_minus: np.ndarray
    sphere_class: np.ndarray
    roots: list
    outer_ah_candidate: dict | None
    projections: Projections | None
    ham: np.ndarray
    ham_rel: np.ndarray
    mom: np.ndarray
    mom_rel: np.ndarray
    hamM: np.ndarray
    hamM_rel: np.ndarray
    sigma_ham: np.ndarray
    sigma_mom: np.ndarray
    sigma_hamM: np.ndarray
    norms: dict
    M_rho: np.ndarray
    M_flux: np.ndarray
    M_int: np.ndarray
    M_prop: np.ndarray
    center_checks: dict
    status: dict
    # extras (not required by the contract, useful for audits)
    sigma_Gamma: np.ndarray = field(default=None, repr=False)
    sigma_dR: np.ndarray = field(default=None, repr=False)
    dGamma: np.ndarray = field(default=None, repr=False)
    dKB: np.ndarray = field(default=None, repr=False)
    dM: np.ndarray = field(default=None, repr=False)
    sigma_dGamma: np.ndarray = field(default=None, repr=False)
    sigma_dKB: np.ndarray = field(default=None, repr=False)
    constraint_point_status: dict = field(default_factory=dict, repr=False)
    rel_undefined: dict = field(default_factory=dict, repr=False)
    tau_plus: np.ndarray = field(default=None, repr=False)
    tau_minus: np.ndarray = field(default=None, repr=False)


# ====================================================================== helpers
def _nan(N):
    return np.full(N, np.nan)


def _empty_norms():
    return {nm: {reg: {k: np.nan for k in ("Linf", "L2vol", "Linf_rel")}
                 for reg in ("all", "center", "bulk", "outer")} for nm in ("ham", "mom", "hamM")}


def _sign_with_tol(f, tau):
    s = np.sign(f).astype(int)
    s[np.abs(f) <= tau] = 0
    return s


def _interp_local(r, y, x, valid):
    """Cubic (4-point Lagrange) interpolation of y at x; linear if the window has invalid points."""
    N = r.size
    k = int(np.searchsorted(r, x))
    k = min(max(k, 1), N - 1)
    j0 = min(max(k - 2, 0), N - 4)
    sl = slice(j0, j0 + 4)
    if np.all(valid[sl]) and np.all(np.isfinite(y[sl])):
        return lagrange_eval(r[sl], y[sl], x)
    i0, i1 = k - 1, k
    w = (x - r[i0]) / (r[i1] - r[i0])
    return float((1 - w) * y[i0] + w * y[i1])


def _find_roots(which, r, R, M, Gamma, f, tau, s, fo, tauo, valid, s_robust):
    """s_robust: per-point threshold kappa*(sigma + sigma_noise) used only for the 'robust' flag."""
    N = r.size
    roots = []
    found = {}

    def search_sign(start, step, key):
        k = start
        for _ in range(4):
            if k < 0 or k >= N or not valid[k]:
                break
            if s[k] != 0:
                found[key] = k
                return int(s[k])
            k += step
        found[key] = None
        return 0

    # candidate intervals, vectorised (the per-root work below is O(1))
    sg = np.sign(np.where(valid, f, np.nan))
    cand = (valid & (f == 0.0))
    cand[:-1] |= valid[:-1] & valid[1:] & (sg[:-1] * sg[1:] < 0)
    for i in np.flatnonzero(cand):
        i = int(i)
        exact = f[i] == 0.0
        change = (not exact and i + 1 < N and valid[i + 1] and f[i + 1] != 0.0
                  and np.sign(f[i]) != np.sign(f[i + 1]))
        if not (exact or change):
            continue
        if exact:
            r_lin = r_cub = float(r[i])
            il, ih = max(i - 1, 0), min(i + 1, N - 1)
            slope = (f[ih] - f[il]) / (r[ih] - r[il]) if ih > il else 0.0
            tau_star = float(tau[i])
            inner = search_sign(i - 1, -1, "in")
            outer = search_sign(i + 1, +1, "out")
            cell = (float(r[il]), float(r[ih]))
        else:
            r_lin = float(r[i] - f[i] * (r[i + 1] - r[i]) / (f[i + 1] - f[i]))
            j0 = min(max(i - 1, 0), N - 4)
            sl = slice(j0, j0 + 4)
            if np.all(valid[sl]):
                xs, ys = r[sl], f[sl]
                r_cub = float(brentq(lambda x: lagrange_eval(xs, ys, x), r[i], r[i + 1], xtol=1e-15 * max(1.0, abs(r[i])), rtol=4 * EPS))
            else:
                r_cub = r_lin
            slope = (f[i + 1] - f[i]) / (r[i + 1] - r[i])
            w = (r_cub - r[i]) / (r[i + 1] - r[i])
            tau_star = float((1 - w) * tau[i] + w * tau[i + 1])
            inner = search_sign(i, -1, "in")
            outer = search_sign(i + 1, +1, "out")
            cell = (float(r[i]), float(r[i + 1]))
        if slope != 0.0 and np.isfinite(slope):
            dr_tau = tau_star / abs(slope)
        else:
            dr_tau = cell[1] - cell[0]
        r_lo = max(min(r_lin, r_cub) - dr_tau, float(r[0]))
        r_hi = min(max(r_lin, r_cub) + dr_tau, float(r[-1]))
        R_star = _interp_local(r, R, r_cub, valid)
        M_star = _interp_local(r, M, r_cub, valid)
        G_star = _interp_local(r, Gamma, r_cub, valid)
        inside = (r > r_lo) & (r < r_hi)
        Rvals = [_interp_local(r, R, r_lo, valid), _interp_local(r, R, r_hi, valid), R_star] + list(R[inside])
        fo_star = _interp_local(r, fo, r_cub, valid)
        k = min(max(int(np.searchsorted(r, r_cub)), 1), N - 1)
        w = (r_cub - r[k - 1]) / (r[k] - r[k - 1])
        tauo_star = float((1 - w) * tauo[k - 1] + w * tauo[k])
        if not np.isfinite(fo_star):
            other = 0
        elif abs(fo_star) <= tauo_star:
            other = 0
        else:
            other = int(np.sign(fo_star))
        rtype = {-1: "FUTURE_MOTS", 1: "PAST_MOTS", 0: "DEGENERATE"}[other]
        ah = bool(which == "+" and rtype == "FUTURE_MOTS" and G_star > 0 and inner == -1 and outer == 1)
        # robustness against grid-scale noise (extension; does not change type / ah_candidate)
        rob_o = float((1 - w) * s_robust[k - 1] + w * s_robust[k])
        robust = bool(all(found[key] is not None and abs(f[found[key]]) > s_robust[found[key]] for key in ("in", "out"))
                      and abs(fo_star) > rob_o)
        roots.append({
            "which": which, "r": r_cub, "R": R_star, "R_lo": float(min(Rvals)), "R_hi": float(max(Rvals)),
            "M": M_star, "type": rtype, "ah_candidate": ah, "other_sign": Sign(other),
            "inner_sign": Sign(inner), "outer_sign": Sign(outer),
            # extras
            "r_lo": r_lo, "r_hi": r_hi, "r_linear": r_lin, "r_cubic": r_cub, "Gamma": G_star,
            "compactness": 2.0 * M_star / R_star if R_star > 0 else np.nan, "index": i, "exact_grid_zero": bool(exact),
            "robust": robust,
        })
    return roots


def _geometry_status(classes, roots, r, splus, sminus, data_ok):
    """Contract section 7 priority: UNDETERMINED > AH_CANDIDATE > FUTURE_TRAPPED_NO_AH_CANDIDATE >
    UNCERTAIN > PAST_TRAPPED_ONLY > NO_TRAPPED_SPHERES (fallback: UNCERTAIN)."""
    N = classes.size
    reasons = []
    marginal_present = bool(np.any(np.isin(classes, MARGINAL)))
    if not data_ok or np.any(classes == "INVALID"):
        return "UNDETERMINED", marginal_present, ["INVALID spheres or non-OK data status"]
    if any(rt["ah_candidate"] for rt in roots):
        return "AH_CANDIDATE", marginal_present, reasons
    if np.any(classes == "FUTURE_TRAPPED"):
        return "FUTURE_TRAPPED_NO_AH_CANDIDATE", marginal_present, reasons
    # ---- UNCERTAIN conditions
    uncertain = False
    is_marg = np.isin(classes, MARGINAL)
    for i in np.flatnonzero(np.isin(classes, ("FUTURE_MARGINAL", "DOUBLY_MARGINAL"))):
        zero_which = set()
        if splus[i] == 0:
            zero_which.add("+")
        if sminus[i] == 0:
            zero_which.add("-")
        dr = max(r[i] - r[max(i - 1, 0)], r[min(i + 1, N - 1)] - r[i])
        assigned = any(rt["type"] in ("FUTURE_MOTS", "PAST_MOTS") and rt["which"] in zero_which
                       and rt["r_lo"] - dr <= r[i] <= rt["r_hi"] + dr for rt in roots)
        # run of marginal points containing i, sandwiched between trapped spheres?
        lo = i
        while lo - 1 >= 0 and is_marg[lo - 1]:
            lo -= 1
        hi = i
        while hi + 1 < N and is_marg[hi + 1]:
            hi += 1
        sandwiched = (lo - 1 >= 0 and hi + 1 < N and classes[lo - 1] in TRAPPED and classes[hi + 1] in TRAPPED)
        if not assigned and not sandwiched:
            uncertain = True
            reasons.append(f"{classes[i]} at index {i} (r={r[i]:.6g}) not assigned to a FUTURE/PAST_MOTS root")
    for rt in roots:
        if rt["type"] == "DEGENERATE":
            i = rt["index"]
            nb = classes[max(i - 1, 0):min(i + 3, N)]
            if np.any(np.isin(nb, TRAPPED)):
                uncertain = True
                reasons.append(f"DEGENERATE root at r={rt['r']:.6g} next to a trapped sphere")
    if uncertain:
        return "UNCERTAIN", marginal_present, reasons
    if np.any(np.isin(classes, ("PAST_TRAPPED", "PAST_MARGINAL"))):
        return "PAST_TRAPPED_ONLY", marginal_present, reasons
    if np.all(np.isin(classes, ("NORMAL", "NORMAL_REVERSED", "CENTER"))):
        return "NO_TRAPPED_SPHERES", marginal_present, reasons
    # remaining marginal spheres assigned to MOTS roots without any trapped sphere: be conservative
    reasons.append("marginal spheres assigned to MOTS roots but no trapped region on the grid")
    return "UNCERTAIN", marginal_present, reasons


def _norm_block(X, Xrel, excluded, w, r, regions):
    out = {}
    for reg, idx in regions.items():
        d = {"Linf": np.nan, "L2vol": np.nan, "Linf_rel": np.nan}
        if idx.size == 0:
            out[reg] = d
            continue
        sel = idx[~excluded[idx]]
        if sel.size == 0 or not np.all(np.isfinite(X[sel])):
            out[reg] = d
            continue
        d["Linf"] = float(np.max(np.abs(X[sel])))
        if idx.size >= 2:
            integrand = np.where(excluded[idx], 0.0, np.nan_to_num(X[idx]) ** 2 * w[idx])
            wi = np.where(excluded[idx], 0.0, w[idx])
            den = np.trapezoid(wi, r[idx])
            d["L2vol"] = float(np.sqrt(np.trapezoid(integrand, r[idx]) / den)) if den > 0 else np.nan
        else:
            d["L2vol"] = float(abs(X[sel][0]))
        rel = Xrel[sel]
        rel = rel[np.isfinite(rel)]
        d["Linf_rel"] = float(np.max(rel)) if rel.size else np.nan
        out[reg] = d
    return out


def _invalid(slc, matter, problems):
    N = slc.N if isinstance(slc, SphericalSlice) else 0
    nan = _nan(N)
    status = {"data": "INVALID_INPUT",
              "constraints": "NOT_EVALUATED" if matter is None else "VIOLATED",
              "geometry": "UNDETERMINED", "marginal_present": False, "messages": list(problems)}
    return Diagnostics(
        r=np.asarray(getattr(slc, "r", nan)) if N else nan, R=nan.copy(), Gamma=nan.copy(), U=nan.copy(), dR=nan.copy(),
        M=nan.copy(), sigma_M=nan.copy(), compactness=nan.copy(), M_over_R3=nan.copy(), M_over_R3_extrapolated=False,
        Rtheta_plus=nan.copy(), Rtheta_minus=nan.copy(), sigma_Rtheta_plus=nan.copy(), sigma_Rtheta_minus=nan.copy(),
        theta_plus=nan.copy(), theta_minus=nan.copy(), sphere_class=np.array(["INVALID"] * N, dtype="<U32"),
        roots=[], outer_ah_candidate=None, projections=None,
        ham=nan.copy(), ham_rel=nan.copy(), mom=nan.copy(), mom_rel=nan.copy(), hamM=nan.copy(), hamM_rel=nan.copy(),
        sigma_ham=nan.copy(), sigma_mom=nan.copy(), sigma_hamM=nan.copy(), norms=_empty_norms(),
        M_rho=nan.copy(), M_flux=nan.copy(), M_int=nan.copy(), M_prop=nan.copy(),
        center_checks={"applicable": False, "Gamma0": np.nan, "KA_minus_KB0": np.nan, "sigma_Gamma0": np.nan,
                       "Gamma0_minus_1": np.nan, "extrapolated": False},
        status=status)


# ====================================================================== main entry
def diagnose(slc, matter=None, *, kappa_theta: float = 3.0, kappa_c: float = 10.0,
             noise_guard: bool = False) -> Diagnostics:
    """Physical diagnostics of one spherical 3+1 slice (contract v1.0, section 10).

    noise_guard (extension, default off = contract-literal): add a grid-scale-noise term to every
    derivative error estimate sigma(f'), see REPORT.md.  The per-root 'robust' flag and
    status['ah_candidate_robust'] are always computed with the noise term."""
    problems: list[str] = []
    if not isinstance(slc, SphericalSlice):
        problems.append(f"slc must be a SphericalSlice, got {type(slc).__name__}")
        return _invalid(slc, matter, problems)
    problems += slc.validate()
    if matter is not None:
        if not isinstance(matter, (ScalarField, VlasovMoments, Projections)):
            problems.append(f"matter must be ScalarField, VlasovMoments, Projections or None, got {type(matter).__name__}")
        elif slc.N > 0:
            problems += matter.validate(slc)
    for nm, k in (("kappa_theta", kappa_theta), ("kappa_c", kappa_c)):
        try:
            kf = float(k)
            if not (np.isfinite(kf) and kf >= 0):
                problems.append(f"{nm} must be finite and >= 0")
        except (TypeError, ValueError):
            problems.append(f"{nm} must be a number")
    if problems:
        return _invalid(slc, matter, problems)
    with np.errstate(all="ignore"):
        return _diagnose(slc, matter, float(kappa_theta), float(kappa_c), bool(noise_guard))


def _diagnose(slc: SphericalSlice, matter, kth: float, kc: float, ng: bool = False) -> Diagnostics:
    r, A, R, KA, KB = slc.r, slc.A, slc.R, slc.KA, slc.KB
    N = r.size
    g = FDGrid(r, slc.center)
    at_c = R == 0.0            # only the vertex centre (validated)
    pos = ~at_c
    msgs: list[str] = []

    # ---------------------------------------------------------------- derivatives (section 8)
    if slc.dR is not None:
        dR4 = dR6 = slc.dR
        sdR = np.zeros(N)
        rnd1 = ROUND_C * EPS * np.abs(slc.dR)
        d2R4, rnd2 = g.deriv(slc.dR, +1, 4, 1)
        d2R6, _ = g.deriv(slc.dR, +1, 6, 1)
    else:
        dR4, rnd1 = g.deriv(R, -1, 4, 1)
        dR6, _ = g.deriv(R, -1, 6, 1)
        sdR = np.abs(dR4 - dR6) + rnd1
        d2R4, rnd2 = g.deriv(R, -1, 4, 2)
        d2R6, _ = g.deriv(R, -1, 6, 2)
    if slc.dA is not None:
        dA4 = dA6 = slc.dA
        rndA = np.zeros(N)
    else:
        dA4, rndA = g.deriv(A, +1, 4, 1)
        dA6, _ = g.deriv(A, +1, 6, 1)
    if slc.dKB is not None:
        dKB4 = slc.dKB
        sdKB = np.zeros(N)
    else:
        dKB4, rndK = g.deriv(KB, +1, 4, 1)
        dKB6, _ = g.deriv(KB, +1, 6, 1)
        sdKB = np.abs(dKB4 - dKB6) + rndK
        if ng:
            sdKB = sdKB + g.noise_gain(4, 1) * g.noise_amplitude(KB, +1)

    Gamma = dR4 / A
    Gamma6 = dR6 / A
    # grid-scale noise of R' (always computed; enters sigma only with noise_guard)
    if slc.dR is not None:
        snR = np.zeros(N)          # analytic R': no finite-difference noise
    else:
        snR = g.noise_gain(4, 1) * g.noise_amplitude(R, -1)
    if ng:
        sdR = sdR + snR
    sGamma = sdR / A
    dG4 = (d2R4 - Gamma * dA4) / A
    dG6 = (d2R6 - Gamma6 * dA6) / A
    sdG = np.abs(dG4 - dG6) + (rnd2 + np.abs(Gamma) * rndA) / A
    if ng:
        nR2 = (g.noise_gain(4, 1) * g.noise_amplitude(slc.dR, +1)) if slc.dR is not None \
            else g.noise_gain(4, 2) * g.noise_amplitude(R, -1)
        nA = 0.0 if slc.dA is not None else g.noise_gain(4, 1) * g.noise_amplitude(A, +1)
        sdG = sdG + (nR2 + np.abs(Gamma) * nA) / A

    # ---------------------------------------------------------------- mass, expansions (sections 6, 7)
    U = -R * KB
    M = 0.5 * R * ((1.0 - Gamma) * (1.0 + Gamma) + U * U)
    M6 = 0.5 * R * ((1.0 - Gamma6) * (1.0 + Gamma6) + U * U)
    sigma_M = R * np.abs(Gamma) * sGamma
    compactness = np.where(pos, 2.0 * M / np.where(pos, R, 1.0), 0.0)
    M_over_R3 = np.where(pos, M / np.where(pos, R, 1.0) ** 3, np.nan)
    extrap = False
    if np.any(at_c):
        # M/R^3 is even in r: polynomial in r^2 through the first three points with R > 0
        M_over_R3[0] = even_extrapolate_to_zero(r[1:4], M_over_R3[1:4])
        extrap = True
    Rtp = 2.0 * (U + Gamma)
    Rtm = 2.0 * (U - Gamma)
    sRt = 2.0 * sGamma
    Rsafe = np.where(pos, R, 1.0)
    theta_p = np.where(pos, Rtp / Rsafe, np.nan)
    theta_m = np.where(pos, Rtm / Rsafe, np.nan)

    floor = TOL_FLOOR * (np.abs(U) + np.abs(Gamma))
    tau_p = np.maximum(kth * sRt, floor)
    tau_m = np.maximum(kth * sRt, floor)

    geo_arrays = [Gamma, U, dR4, M, sigma_M, compactness, Rtp, Rtm, sRt, tau_p]
    geo_ok = np.ones(N, dtype=bool)
    for a in geo_arrays:
        geo_ok &= np.isfinite(a)
    geo_ok &= np.where(pos, np.isfinite(theta_p) & np.isfinite(theta_m) & np.isfinite(M_over_R3), True)
    if extrap and not np.isfinite(M_over_R3[0]):
        geo_ok[0] = False

    sp_ = np.zeros(N, dtype=int)
    sm_ = np.zeros(N, dtype=int)
    sp_[geo_ok] = _sign_with_tol(Rtp[geo_ok], tau_p[geo_ok])
    sm_[geo_ok] = _sign_with_tol(Rtm[geo_ok], tau_m[geo_ok])
    classes = np.empty(N, dtype="<U32")
    for i in range(N):
        if not geo_ok[i]:
            classes[i] = "INVALID"
        elif at_c[i]:
            classes[i] = "CENTER"
        else:
            classes[i] = _CLASS_TABLE[(int(sp_[i]), int(sm_[i]))]

    sn_Rt = 2.0 * snR / A if not ng else np.zeros(N)       # already inside sRt when ng
    rob_thr = np.maximum(kth * (sRt + sn_Rt), floor)
    roots = (_find_roots("+", r, R, M, Gamma, Rtp, tau_p, sp_, Rtm, tau_m, geo_ok, rob_thr)
             + _find_roots("-", r, R, M, Gamma, Rtm, tau_m, sm_, Rtp, tau_p, geo_ok, rob_thr))
    roots.sort(key=lambda d: (d["r"], d["which"]))
    cands = [rt for rt in roots if rt["ah_candidate"]]
    outer = max(cands, key=lambda d: d["r"]) if cands else None

    # ---------------------------------------------------------------- centre checks (section 3)
    cc: dict[str, Any] = {"applicable": slc.center in ("vertex", "cell")}
    if slc.center == "vertex":
        cc.update(Gamma0=float(Gamma[0]), sigma_Gamma0=float(sGamma[0]), KA_minus_KB0=float(KA[0] - KB[0]),
                  extrapolated=False)
    elif slc.center == "cell":
        cc.update(Gamma0=even_extrapolate_to_zero(r[:3], Gamma[:3]), sigma_Gamma0=float(sGamma[0]),
                  KA_minus_KB0=even_extrapolate_to_zero(r[:3], (KA - KB)[:3]), extrapolated=True)
    else:
        cc.update(Gamma0=np.nan, sigma_Gamma0=np.nan, KA_minus_KB0=np.nan, extrapolated=False)
    cc["Gamma0_minus_1"] = cc["Gamma0"] - 1.0 if cc["applicable"] else np.nan

    # ---------------------------------------------------------------- matter and constraints (sections 5, 6, 9)
    nan = _nan(N)
    proj = None
    ham = ham_rel = mom = mom_rel = hamM = hamM_rel = nan
    sig_ham = sig_mom = sig_hamM = nan
    M_rho = M_flux = M_int = M_prop = nan
    dM4 = nan
    norms = _empty_norms()
    cps: dict = {}
    rel_undef: dict = {}
    matter_ok = True
    if matter is None:
        cstat = "NOT_EVALUATED"
    else:
        proj = matter.projections(slc, grid=g, noise_guard=ng) if isinstance(matter, ScalarField) \
            else matter.projections(slc, grid=g)
        rho, jr = proj.rho, proj.j_r
        s_rho = proj.sigma_rho if proj.sigma_rho is not None else np.zeros(N)
        s_jr = proj.sigma_j_r if proj.sigma_j_r is not None else np.zeros(N)
        # Hamiltonian [F7, F8]
        T_h = [np.where(pos, 2.0 * (1.0 - Gamma) * (1.0 + Gamma) / Rsafe ** 2, np.nan),
               np.where(pos, -4.0 * dG4 / (A * Rsafe), np.nan),
               4.0 * KA * KB, 2.0 * KB ** 2, -4.0 * FOURPI * rho]
        # momentum [F9]
        T_m = [dKB4, np.where(pos, -(dR4 / Rsafe) * (KA - KB), np.nan), FOURPI * jr]
        # mass form [F10]: M' by finite differences of M (4th) and M6 (6th)
        # rounding error of M itself: summation + propagated rounding of Gamma
        M_err = ROUND_C * EPS * 0.5 * R * (1.0 + Gamma ** 2 + U ** 2) + R * np.abs(Gamma) * rnd1 / A
        dM4, rndM = g.deriv(M, -1, 4, 1, abs_err=M_err)
        dM6, _ = g.deriv(M6, -1, 6, 1)
        T_M = [dM4, -FOURPI * R ** 2 * rho * dR4, -FOURPI * R ** 2 * U * jr]
        sig_ham = np.where(pos, 4.0 * np.abs(Gamma) * sGamma / Rsafe ** 2 + 4.0 * sdG / (A * Rsafe), np.nan) \
            + 4.0 * FOURPI * s_rho
        sig_mom = np.where(pos, sdKB + np.abs(KA - KB) * sdR / Rsafe, np.nan) + FOURPI * s_jr
        if ng:
            rndM = rndM + g.noise_gain(4, 1) * g.noise_amplitude(M, -1)
        sig_hamM = (np.abs(dM4 - dM6) + rndM) + FOURPI * R ** 2 * np.abs(rho) * sdR \
            + FOURPI * R ** 2 * (np.abs(dR4) * s_rho + np.abs(U) * s_jr)
        out = {}
        for name, T, sig, excl in (("ham", T_h, sig_ham, at_c), ("mom", T_m, sig_mom, at_c),
                                   ("hamM", T_M, sig_hamM, np.zeros(N, dtype=bool))):
            X = np.sum(T, axis=0)
            S = np.sum(np.abs(T), axis=0)
            undefined = ~(S > 0)
            Xrel = np.where(undefined, np.nan, np.abs(X) / np.where(undefined, 1.0, S))
            thr = kc * sig + TOL_FLOOR * S
            # NaN -> False; inf residual with inf threshold (inf <= inf) must not pass either
            ok = np.isfinite(X) & np.isfinite(thr) & (np.abs(X) <= thr)
            st = np.where(ok, "CONSISTENT", "VIOLATED").astype("<U16")
            st[excl] = "EXCLUDED_CENTER"
            X = np.where(excl, np.nan, X)
            Xrel = np.where(excl, np.nan, Xrel)
            out[name] = (X, Xrel, st)
            cps[name] = st
            rel_undef[name] = undefined & ~excl
        ham, ham_rel, _ = out["ham"]
        mom, mom_rel, _ = out["mom"]
        hamM, hamM_rel, _ = out["hamM"]
        # integral masses
        dRr = dR4
        M_rho = g.cumint(FOURPI * R ** 2 * rho * dRr, +1)
        M_flux = g.cumint(FOURPI * R ** 2 * U * jr, +1)
        M_int = M[0] + M_rho + M_flux
        M_prop = g.cumint(FOURPI * R ** 2 * A * rho, +1)
        # norms
        k_c = max(4, N // 20)
        posidx = np.flatnonzero(pos)
        c_idx = posidx[:k_c]
        o_idx = np.arange(N - 4, N)
        mask = np.ones(N, dtype=bool)
        mask[c_idx] = False
        mask[o_idx] = False
        mask[at_c] = False
        regions = {"all": np.arange(N), "center": c_idx, "bulk": np.flatnonzero(mask), "outer": o_idx}
        wvol = FOURPI * R ** 2 * A
        norms = {nm: _norm_block(out[nm][0], out[nm][1], (at_c if nm != "hamM" else np.zeros(N, dtype=bool)),
                                 wvol, r, regions) for nm in ("ham", "mom", "hamM")}
        # finiteness of matter-dependent outputs
        for a in (proj.rho, proj.j_r, proj.S_rr, proj.S_thth, hamM, sig_hamM, M_rho, M_flux, M_int, M_prop):
            if not np.all(np.isfinite(a)):
                matter_ok = False
        for a in (ham, mom, sig_ham, sig_mom):
            if not np.all(np.isfinite(a[pos])):
                matter_ok = False
        if not matter_ok:
            msgs.append("non-finite matter/constraint outputs")
        viol = any(np.any(cps[nm] == "VIOLATED") for nm in cps)
        cstat = "VIOLATED" if (viol or not matter_ok) else "CONSISTENT_WITH_TRUNCATION"

    # ---------------------------------------------------------------- statuses
    geo_fail = not np.all(geo_ok)
    if geo_fail:
        msgs.append(f"non-finite geometric outputs at indices {np.flatnonzero(~geo_ok)[:10].tolist()}")
    if not np.all(np.isfinite(dG4[pos])) or not np.all(np.isfinite(dKB4)):
        geo_fail = True
        msgs.append("non-finite derivative outputs")
    data = "NUMERICAL_FAILURE" if (geo_fail or not matter_ok) else "OK"
    gstat, marg, reasons = _geometry_status(classes, roots, r, sp_, sm_, data == "OK")
    status = {"data": data, "constraints": cstat, "geometry": gstat, "marginal_present": marg,
              "degenerate_root_present": any(rt["type"] == "DEGENERATE" for rt in roots),
              "ah_candidate_robust": (all(rt["robust"] for rt in cands) if cands else None),
              "noise_guard": ng,
              # signs resolved by tau but not by kappa*(sigma + grid-scale noise): classification noise-limited
              "noise_sensitive": bool(np.any(geo_ok & pos & (((np.abs(Rtp) > tau_p) & (np.abs(Rtp) <= rob_thr))
                                                         | ((np.abs(Rtm) > tau_m) & (np.abs(Rtm) <= rob_thr))))),
              "messages": msgs + reasons}

    return Diagnostics(
        r=r.copy(), R=R.copy(), Gamma=Gamma, U=U, dR=np.array(dR4, dtype=float), M=M, sigma_M=sigma_M,
        compactness=compactness, M_over_R3=M_over_R3, M_over_R3_extrapolated=extrap,
        Rtheta_plus=Rtp, Rtheta_minus=Rtm, sigma_Rtheta_plus=sRt, sigma_Rtheta_minus=sRt.copy(),
        theta_plus=theta_p, theta_minus=theta_m, sphere_class=classes, roots=roots, outer_ah_candidate=outer,
        projections=proj, ham=ham, ham_rel=ham_rel, mom=mom, mom_rel=mom_rel, hamM=hamM, hamM_rel=hamM_rel,
        sigma_ham=sig_ham, sigma_mom=sig_mom, sigma_hamM=sig_hamM, norms=norms,
        M_rho=M_rho, M_flux=M_flux, M_int=M_int, M_prop=M_prop, center_checks=cc, status=status,
        sigma_Gamma=sGamma, sigma_dR=sdR, dGamma=dG4, dKB=dKB4, dM=dM4, sigma_dGamma=sdG, sigma_dKB=sdKB,
        constraint_point_status=cps, rel_undefined=rel_undef, tau_plus=tau_p, tau_minus=tau_m)

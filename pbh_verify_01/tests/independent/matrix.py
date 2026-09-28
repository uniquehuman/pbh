"""Independent verification matrix, PBH-VERIFY-01 (verification_plan.md section 3).

Each case is a function (sd, res) -> None that fills a Result.  Expected values come only from
independent_reference/ (closed forms, own FKG integrator, own constraint-ODE solver, sympy
derivatives).  independent_reference.fd_ref is used ONLY for the scale of truncation tolerances.
"""
from collections import OrderedDict

import numpy as np
from scipy.optimize import brentq

from common import (EPS, Result, D, has, arr, classes, status, sg, roots_of, build, proj, zero_proj, floor,  # noqa: F401
                    hmin_of, hmax_of, compare, convergence, check_classes, exact_root_tol, check_root, run_case,
                    fd_ref)
from independent_reference import analytic as AN
from independent_reference.fkg import FKG
from independent_reference.flux_data import FluxData
from independent_reference.grids import make_grid, regions, sinh_map, exp_map
from independent_reference.norms import l2vol

CASES = OrderedDict()


def case(tid, title):
    def deco(fn):
        CASES[tid] = (fn, title)
        return fn
    return deco


NCONV = (100, 200, 400, 800)

DIAG_FIELDS = ["Gamma", "U", "dR", "M", "sigma_M", "compactness", "M_over_R3", "M_over_R3_extrapolated",
               "Rtheta_plus", "Rtheta_minus", "sigma_Rtheta_plus", "sigma_Rtheta_minus", "theta_plus",
               "theta_minus", "sphere_class", "roots", "outer_ah_candidate", "projections", "ham", "ham_rel",
               "mom", "mom_rel", "hamM", "hamM_rel", "sigma_ham", "sigma_mom", "sigma_hamM", "norms", "M_rho",
               "M_flux", "M_int", "M_prop", "center_checks", "status"]
ROOT_KEYS = ["which", "r", "R", "R_lo", "R_hi", "M", "type", "ah_candidate", "other_sign", "inner_sign", "outer_sign"]

# ---------------------------------------------------------------------------- helpers


def _win_max(a, w=3):
    a = np.abs(np.nan_to_num(np.asarray(a, float), nan=0.0))
    out = a.copy()
    for k in range(1, w + 1):
        out[k:] = np.maximum(out[k:], a[:-k])
        out[:-k] = np.maximum(out[:-k], a[k:])
    return out


def class_deltas(r, v, ref):
    """pointwise sign-band delta = 3*(10*local max |E_ref| + floor)   (plan 1.6)."""
    L, hm = r[-1] - r[0], hmin_of(r)
    S = 2 + np.max(np.abs(v["U"]) + np.abs(v["Gamma"]))
    fl = floor(S, L, hm, 1)
    dp = 3 * (10 * _win_max(ref["Rtp"] - v["Rtp"]) + fl)
    dm = 3 * (10 * _win_max(ref["Rtm"] - v["Rtm"]) + fl)
    return dp, dm


def expected_roots(sl, rmin, rmax, zero_tol=1e-9):
    out = []
    for which in "+-":
        oth = "-" if which == "+" else "+"
        for rs in sl.roots(which, rmin, rmax):
            rr = np.array([rs])
            f1 = float(sl.rtheta_fn(which, 1)(rr)[0])
            ov = float(sl.rtheta_fn(oth)(rr)[0])
            other = 0 if abs(ov) <= zero_tol else int(np.sign(ov))
            G = float(sl.exact(rr, with_matter=False)["Gamma"][0])
            inner, outer = -int(np.sign(f1)), int(np.sign(f1))
            typ = "FUTURE_MOTS" if other < 0 else ("PAST_MOTS" if other > 0 else "DEGENERATE")
            ah = which == "+" and G > 0 and other < 0 and inner < 0 and outer > 0
            out.append(dict(which=which, r=rs, type=typ, ah=ah, other=other if other != 0 else None,
                            inner=inner, outer=outer))
    return out


def match_roots(res, label, sl, r, d, v, ref, rmin, rmax, R_scale, allow_degenerate_extra=False):
    exp = expected_roots(sl, rmin, rmax)
    got = roots_of(d)
    used = set()
    for e in exp:
        cands = [(abs(q["R"] - float(sl.R_fn()(np.array([e["r"]]))[0])), k) for k, q in enumerate(got)
                 if q["which"] == e["which"] and k not in used]
        if not res.check(cands, "%s: expected %s-root at r=%.6g not reported" % (label, e["which"], e["r"])):
            continue
        k = min(cands)[1]
        used.add(k)
        key = "Rtp" if e["which"] == "+" else "Rtm"
        i = int(np.clip(np.searchsorted(r, e["r"]), 1, len(r) - 1))
        lo, hi = max(0, i - 3), min(len(r), i + 3)
        ref_err = float(np.max(np.abs(ref[key][lo:hi] - v[key][lo:hi])))
        R_star, tol, slack = exact_root_tol(sl, e["which"], e["r"], r, ref_err, R_scale)
        check_root(res, "%s/root%s@%.4g" % (label, e["which"], R_star), got[k], R_star, tol, slack, e)
    extra = [q for k, q in enumerate(got) if k not in used]
    if allow_degenerate_extra:
        extra = [q for q in extra if q["type"] != "DEGENERATE" or q["ah"]]
    res.check(not extra, "%s: %d spurious roots reported: %s" % (label, len(extra),
                                                                   [(q["which"], q["R"], q["type"]) for q in extra][:5]))
    return exp, got


def check_outer_ah(res, label, d, exp_R):
    o = D(d, "outer_ah_candidate")
    if exp_R is None:
        res.check(o is None, "%s: outer_ah_candidate should be None, got %r" % (label, o))
    else:
        if res.check(o is not None, "%s: outer_ah_candidate missing (expected R=%.6g)" % (label, exp_R)):
            Ro = float(o["R"] if isinstance(o, dict) else getattr(o, "R"))
            res.check(abs(Ro - exp_R) <= 1e-3 * max(1.0, abs(exp_R)),
                      "%s: outer_ah_candidate R=%.6g, expected %.6g" % (label, Ro, exp_R))


def sym_family(sd, res, fam, sl, grids, center, fields, pmin=3.5, pmin_edge=2.5, edge_fields=(),
               matter="exact", derivs=False, classes_check=True, roots_range=None, R_scale=1.0,
               extra_checks=None):
    """Run a SymSlice on successively refined grids; compare fields with exact, record orders.
    fields: list of (label, module_field, key, S_fn, d)."""
    errs = {f[0]: [] for f in fields}
    hs = []
    outs = []
    for r in grids:
        v = sl.exact(r)
        m = proj(sd, v) if matter == "exact" else (zero_proj(sd, len(r)) if matter == "zero" else None)
        d, _ = build(sd, r, v, center, m, derivs=derivs)
        ref = fd_ref.pipeline(r, v["A"], v["R"], v["KA"], v["KB"], center, v["rho"], v["jr"])
        g = "%s/N%d" % (fam, len(r))
        for lab, mf, key, Sfn, dd in fields:
            errs[lab].append(compare(res, "%s/%s" % (g, lab), arr(d, mf), v[key], ref[key], r, v["R"], v["A"],
                                     center, Sfn(v), dd))
        if classes_check:
            dp, dm = class_deltas(r, v, ref)
            check_classes(res, g, classes(d), v["Rtp"], v["Rtm"], v["R"], center, dp, dm)
        if roots_range is not None:
            match_roots(res, g, sl, r, d, v, ref, roots_range[0], roots_range[1], R_scale)
        if extra_checks is not None:
            extra_checks(res, g, r, v, d, ref)
        hs.append(hmax_of(r))
        outs.append((r, v, d, ref))
    if len(grids) >= 3:
        for lab, *_ in fields:
            regs_main = ("bulk",) if lab in edge_fields else ("bulk", "center", "outer")
            convergence(res, "%s/%s" % (fam, lab), hs, errs[lab], pmin, regs_main)
            if lab in edge_fields:
                _edge_conv(res, fam, lab, hs, errs[lab], pmin_edge)
    return outs


def _edge_conv(res, fam, lab, hs, errs, pmin_edge):
    """edge regions: nested one-sided stencils lose one order (pmin_edge = 2.5 for expected 3);
    ham at the centre in L_inf: the 1/R^2 and 1/R factors turn an O(h^4) error of Gamma, Gamma'
    into O(h^2) at fixed grid index (R = O(h)) -> pmin 1.8 (expected 2)."""
    convergence(res, "%s/%s[outer]" % (fam, lab), hs, errs, pmin_edge, ("outer",))
    if lab == "ham":
        convergence(res, "%s/%s[center]" % (fam, lab), hs, errs, 1.8, ("center",), kinds=("Linf",))
        convergence(res, "%s/%s[center]" % (fam, lab), hs, errs, pmin_edge, ("center",), kinds=("L2vol",))
    else:
        convergence(res, "%s/%s[center]" % (fam, lab), hs, errs, pmin_edge, ("center",))


S_M = lambda v: np.nanmax(np.abs(v["R"])) * (1 + np.nanmax(v["Gamma"]**2) + np.nanmax(v["U"]**2)) / 2  # noqa: E731
S_RT = lambda v: 2 + np.nanmax(np.abs(v["U"]) + np.abs(v["Gamma"]))  # noqa: E731
S_H = lambda v: np.nanmax(np.nan_to_num(v["ham_absterms"])) + 1.0 / (v["R"][-1] - v["R"][0])**2  # noqa: E731
S_MOM = lambda v: np.nanmax(np.nan_to_num(v["mom_absterms"])) + np.max(np.abs(v["KB"])) / (v["R"][-1] - v["R"][0]) + 1e-300  # noqa: E731
S_HM = lambda v: np.nanmax(np.nan_to_num(v["hamM_absterms"])) + S_M(v) / (v["R"][-1] - v["R"][0])  # noqa: E731

F_BASIC = [("M", "M", "M", S_M, 1), ("Rtp", "Rtheta_plus", "Rtp", S_RT, 1), ("Rtm", "Rtheta_minus", "Rtm", S_RT, 1)]
F_RES = [("ham", "ham", "ham", S_H, 2), ("mom", "mom", "mom", S_MOM, 1), ("hamM", "hamM", "hamM", S_HM, 2)]


def geom(res, label, d, expected):
    g = status(d, "geometry")
    if isinstance(expected, (set, tuple, list)):
        res.check(g in expected, "%s: geometry %s not in %s" % (label, g, sorted(expected)))
    else:
        res.check(g == expected, "%s: geometry %s, expected %s" % (label, g, expected))
    return g


def not_violated(res, label, d, hard=True):
    c = status(d, "constraints")
    if hard:
        res.check(c != "VIOLATED", "%s: constraints status VIOLATED on consistent data" % label)
    res.metrics.setdefault("constraint_status", {})[label] = c
    return c


def _cumtrap(f, r):
    return np.concatenate([[0.0], np.cumsum(0.5 * np.diff(r) * (f[1:] + f[:-1]))])


def quad_check(res, label, mod, exact_cum, integrand, r, S, extra=0.0, integrand_ref=None):
    """TOL_Q (plan 1.3, v1.1): 2*max(|E_trap|, |E_simpson|) of the exact integrand on this grid
    + 10 * integral |integrand_ref - integrand| (FD error of the integrand, own FD) + floor."""
    from scipy.integrate import cumulative_simpson
    mod = np.asarray(mod, float)
    trap = _cumtrap(integrand, r)
    simp = cumulative_simpson(integrand, x=r, initial=0.0)
    Etrap = trap + exact_cum[0] - exact_cum
    Esimp = simp + exact_cum[0] - exact_cum
    tol = 2 * np.maximum(np.abs(Etrap), np.abs(Esimp)) + floor(S, r[-1] - r[0], hmin_of(r), 1) + extra
    if integrand_ref is not None:
        tol = tol + 10 * _cumtrap(np.abs(np.asarray(integrand_ref) - integrand), r)
    err = np.abs(mod - exact_cum)
    ok = np.all(np.isfinite(mod)) and np.all(err <= tol)
    k = int(np.nanargmax(err - tol)) if np.all(np.isfinite(mod)) else 0
    res.check(ok, "%s: quadrature err %.3e > tol %.3e at r=%.4g (E_trap %.2e)" % (label, err[k], tol[k], r[k], Etrap[k]))
    res.metrics.setdefault("quadrature", {})[label] = dict(max_err=float(np.nanmax(err)), tol_at_max=float(tol[k]),
                                                            max_Etrap=float(np.max(np.abs(Etrap))))
    return err, tol


# ============================================================================ V: API and algebra

@case("V00", "API presence (contract section 10)")
def t_V00(sd, res):
    r = np.linspace(0, 6, 61)
    sl = AN.schwarzschild_pg()
    rr = np.linspace(0.5, 6, 61)
    v = sl.exact(rr)
    d, slc = build(sd, rr, v, "none", zero_proj(sd, len(rr)))
    for f in DIAG_FIELDS:
        res.check(has(d, f), "Diagnostics lacks field %s" % f)
    st = D(d, "status")
    for k in ("data", "constraints", "geometry"):
        res.check(k in st if isinstance(st, dict) else hasattr(st, k), "status lacks key %s" % k)
    nm = D(d, "norms")
    for name in ("ham", "mom", "hamM"):
        for reg in ("all", "center", "bulk", "outer"):
            for kind in ("Linf", "L2vol", "Linf_rel"):
                try:
                    nm[name][reg][kind]
                except Exception:  # noqa: BLE001
                    res.check(False, "norms[%s][%s][%s] missing" % (name, reg, kind))
    cc = D(d, "center_checks")
    for k in ("Gamma0", "KA_minus_KB0"):
        res.check(k in cc if isinstance(cc, dict) else hasattr(cc, k), "center_checks lacks %s" % k)
    rts = D(d, "roots")
    if res.check(len(rts) >= 1, "no roots on PG slice (cannot check root keys)"):
        q = rts[0]
        for k in ROOT_KEYS:
            res.check((k in q) if isinstance(q, dict) else hasattr(q, k), "root lacks key %s" % k)
    P = D(d, "projections")
    for k in ("rho", "j_r", "S_rr", "S_thth"):
        res.check(hasattr(P, k) or (isinstance(P, dict) and k in P), "Projections lacks %s" % k)
    for arrf in ("M", "Rtheta_plus", "theta_plus", "ham"):
        res.check(len(np.asarray(D(d, arrf))) == len(rr), "field %s has wrong length" % arrf)
    res.check(len(classes(d)) == len(rr), "sphere_class has wrong length")
    _ = r


@case("V01", "pi_from_dtphi closed form")
def t_V01(sd, res):
    rng = np.random.default_rng(1)
    n = 1000
    dt, dp = rng.normal(size=n), rng.normal(size=n)
    al, be = rng.uniform(0.3, 3, n), rng.uniform(-2, 2, n)
    Pi = np.asarray(sd.pi_from_dtphi(dt, dp, al, be), float)
    ex = (dt - be * dp) / al
    tol = 8 * EPS * (np.abs(dt) + np.abs(be * dp)) / al
    err = np.abs(Pi - ex)
    res.metric("max_err_over_tol", float(np.max(err / tol)))
    res.check(np.all(err <= tol), "pi_from_dtphi deviates: max err/tol = %.2f" % np.max(err / tol))


def _grids_small(N, L=4.0, r0=0.5):
    return [("vertex", "uni", make_grid("uni", N, L=L, center="vertex")),
            ("vertex", "str", make_grid("str", N, L=L, center="vertex")),
            ("cell", "uni", make_grid("uni", N, L=L, center="cell")),
            ("cell", "str", make_grid("str", N, L=L, center="cell")),
            ("none", "uni", make_grid("uni", N, r0=r0, r1=L, center="none")),
            ("none", "str", make_grid("str", N, r0=r0, r1=L, center="none"))]


def _P(P, k):
    return np.asarray(P[k] if isinstance(P, dict) else getattr(P, k), float)


@case("V02", "matter projections (F3-F5, adapter) on polynomial data")
def t_V02(sd, res):
    sl = AN.polynomial_scalar()
    for center, kind, r in _grids_small(41):
        v = sl.exact(r)
        slc = sd.SphericalSlice(r, v["A"], v["R"], v["KA"], v["KB"], center=center)
        P = sd.ScalarField(v["phi"], v["Pi"], sl.m).projections(slc)
        S = np.max(np.abs(v["rho"]) + np.abs(v["jr"]) + np.abs(v["Srr"]) + np.abs(v["Sthth"]))
        tol = floor(S, r[-1] - r[0], hmin_of(r), 1)
        for mk, ek in (("rho", "rho"), ("j_r", "jr"), ("S_rr", "Srr"), ("S_thth", "Sthth")):
            e = np.max(np.abs(_P(P, mk) - v[ek]))
            res.check(e <= tol, "%s/%s ScalarField %s err %.3e > %.3e" % (center, kind, mk, e, tol))
        d = sd.diagnose(slc, sd.ScalarField(v["phi"], v["Pi"], sl.m))
        Pd = D(d, "projections")
        res.check(np.max(np.abs(_P(Pd, "j_r") - v["jr"])) <= tol, "%s/%s diagnose().projections.j_r wrong" % (center, kind))
        J = 0.01 * (1 + r) * np.cos(r)
        Pv = sd.VlasovMoments(v["rho"], J, v["Srr"], v["Sthth"]).projections(slc)
        res.check(np.all(_P(Pv, "rho") == v["rho"]), "%s/%s Vlasov rho != E" % (center, kind))
        res.check(np.all(np.abs(_P(Pv, "j_r") - v["A"] * J) <= 4 * EPS * np.abs(v["A"] * J)),
                  "%s/%s Vlasov j_r != A*J (max diff %.2e)" % (center, kind, np.max(np.abs(_P(Pv, "j_r") - v["A"] * J))))
        res.check(np.all(_P(Pv, "S_rr") == v["Srr"]) and np.all(_P(Pv, "S_thth") == v["Sthth"]),
                  "%s/%s Vlasov S != (p,q)" % (center, kind))
        Pin = sd.Projections(v["rho"], v["jr"], v["Srr"], v["Sthth"])
        dp = sd.diagnose(slc, Pin)
        Po = D(dp, "projections")
        res.check(np.all(_P(Po, "rho") == v["rho"]) and np.all(_P(Po, "j_r") == v["jr"]),
                  "%s/%s Projections not passed through unchanged" % (center, kind))


@case("V03", "FD polynomial exactness: M, R theta, Gamma, U, dR, momentum residual at round-off")
def t_V03(sd, res):
    sl = AN.polynomial_slice()
    for N in (41, 81):
        for center, kind, r in _grids_small(N, L=3.0):
            v = sl.exact(r)
            d, _ = build(sd, r, v, center, proj(sd, v))
            L, hm = r[-1] - r[0], hmin_of(r)
            g = "%s/%s/N%d" % (center, kind, N)
            for mf, key, S in (("dR", "dR", np.max(np.abs(v["dR"]))), ("Gamma", "Gamma", np.max(np.abs(v["Gamma"]))),
                               ("U", "U", np.max(np.abs(v["U"])) + 1e-300), ("M", "M", S_M(v)),
                               ("Rtheta_plus", "Rtp", S_RT(v)), ("Rtheta_minus", "Rtm", S_RT(v)),
                               ("mom", "mom", S_MOM(v))):
                m = arr(d, mf)
                ok = np.isfinite(v[key])
                e = np.max(np.abs(m[ok] - v[key][ok]))
                tol = floor(S, L, hm, 1)
                res.metric("%s/%s" % (g, mf), dict(err=e, tol=tol))
                res.check(e <= tol, "%s %s err %.3e > round-off floor %.3e (polynomial must be exact)" % (g, mf, e, tol))


@case("V04", "analytic derivatives dA, dR, dKB: exact M, R theta, sigma = 0")
def t_V04(sd, res):
    for sl, center, r in ((AN.manufactured(), "vertex", np.linspace(0, 6, 64)),
                          (AN.minkowski_height(), "vertex", np.linspace(0, 6, 64)),
                          (AN.schwarzschild_iso(), "none", np.linspace(0.1, 10, 64))):
        v = sl.exact(r)
        d, _ = build(sd, r, v, center, proj(sd, v), derivs=True)
        g = sl.name
        for mf, key, S in (("M", "M", S_M(v)), ("Rtheta_plus", "Rtp", S_RT(v)), ("Rtheta_minus", "Rtm", S_RT(v))):
            e = np.max(np.abs(arr(d, mf) - v[key]))
            tol = 100 * EPS * S
            res.check(e <= tol, "%s %s err %.3e > %.3e with analytic derivatives" % (g, mf, e, tol))
        for sf in ("sigma_M", "sigma_Rtheta_plus", "sigma_Rtheta_minus"):
            s = arr(d, sf)
            res.check(np.all(s[np.isfinite(s)] == 0), "%s %s != 0 with analytic derivatives (max %.2e)" % (g, sf, np.nanmax(np.abs(s))))


# ============================================================================ M: Minkowski

@case("M10", "Minkowski trivial slice (vertex/cell/none)")
def t_M10(sd, res):
    sl = AN.minkowski_trivial()
    for N in (33, 129):
        for center, r in (("vertex", np.linspace(0, 10, N)), ("cell", (np.arange(N) + 0.5) * 10 / N),
                          ("none", np.linspace(1, 10, N))):
            g = "%s/N%d" % (center, N)
            v = sl.exact(r)
            d, slc = build(sd, r, v, center, zero_proj(sd, N))
            L, hm = r[-1] - r[0], hmin_of(r)
            R = v["R"]
            res.check(np.max(np.abs(arr(d, "M"))) <= floor(r[-1], L, hm, 1), "%s |M| too large: %.2e" % (g, np.max(np.abs(arr(d, "M")))))
            res.check(np.max(np.abs(arr(d, "compactness"))) <= floor(1, L, hm, 1), "%s compactness != 0" % g)
            res.check(np.max(np.abs(arr(d, "Rtheta_plus") - 2)) <= floor(2, L, hm, 1), "%s Rtheta+ != 2" % g)
            res.check(np.max(np.abs(arr(d, "Rtheta_minus") + 2)) <= floor(2, L, hm, 1), "%s Rtheta- != -2" % g)
            tp = arr(d, "theta_plus")
            pos = R > 0
            res.check(np.max(np.abs(tp[pos] * R[pos] - 2)) <= floor(2, L, hm, 1), "%s theta+ != 2/R" % g)
            if center == "vertex":
                res.check(np.isnan(tp[0]) and np.isnan(arr(d, "theta_minus")[0]), "%s theta at R=0 not NaN (masked?)" % g)
            cl = classes(d)
            exp_cl = np.where(R == 0, "CENTER", "NORMAL") if center == "vertex" else np.full(N, "NORMAL")
            res.check(np.all(cl == exp_cl), "%s classes %s" % (g, sorted(set(cl) - {"NORMAL"})))
            res.check(len(D(d, "roots")) == 0, "%s spurious roots %d" % (g, len(D(d, "roots"))))
            res.check(D(d, "outer_ah_candidate") is None, "%s outer_ah_candidate not None" % g)
            geom(res, g, d, "NO_TRAPPED_SPHERES")
            res.check(status(d, "data") == "OK", "%s data status %s" % (g, status(d, "data")))
            for f, dd in (("ham", 2), ("mom", 1), ("hamM", 2)):
                x = arr(d, f)
                x = x[np.isfinite(x)] if f != "hamM" else x
                tol = floor({"ham": 1.0 / L**2, "mom": 1.0 / L, "hamM": 1.0}[f], L, hm, dd)
                res.check(np.all(np.isfinite(x)) and np.max(np.abs(x)) <= tol, "%s %s not ~0: %.2e > %.2e" % (g, f, np.max(np.abs(x)), tol))
            not_violated(res, g, d)
            d0 = sd.diagnose(slc, None)
            res.check(np.all(np.isnan(arr(d0, "ham"))), "%s matter=None: ham not NaN" % g)
            res.check(status(d0, "constraints") == "NOT_EVALUATED", "%s matter=None: constraints %s" % (g, status(d0, "constraints")))


@case("M11", "Minkowski height-function slice t=T(r), K_A != K_B, A != 1: M=0, no roots, residual convergence")
def t_M11(sd, res):
    sl = AN.minkowski_height()
    for center, kind in (("vertex", "uni"), ("vertex", "str"), ("cell", "uni")):
        grids = [make_grid(kind, N, L=6.0, center=center) for N in NCONV]

        def extra(res, g, r, v, d, ref):
            res.check(len(D(d, "roots")) == 0, "%s: spurious roots" % g)
            geom(res, g, d, "NO_TRAPPED_SPHERES")
            not_violated(res, g, d, hard=False)
        sym_family(sd, res, "%s/%s" % (center, kind), sl, grids, center, F_BASIC + F_RES, matter="zero",
                   edge_fields=("ham", "hamM"), extra_checks=extra)


@case("M12", "Milne/hyperboloidal slice in chi and areal coordinates")
def t_M12(sd, res):
    for sl, L in ((AN.milne_chi(2.0), 3.0), (AN.milne_areal(2.0), 2.0 * np.sinh(3.0))):
        grids = [make_grid("uni", N, L=L, center="vertex") for N in NCONV]

        def extra(res, g, r, v, d, ref):
            res.check(len(D(d, "roots")) == 0, "%s: spurious roots" % g)
            geom(res, g, d, "NO_TRAPPED_SPHERES")
        sym_family(sd, res, sl.name, sl, grids, "vertex", F_BASIC + F_RES, matter="zero",
                   edge_fields=("ham", "hamM"), extra_checks=extra)


@case("M13", "Milne chi in [0,12] on coarse grids: sigma band honesty, no false horizon")
def t_M13(sd, res):
    sl = AN.milne_chi(2.0)
    for N in (48, 96):
        r = np.linspace(0, 12, N)
        v = sl.exact(r)
        d, _ = build(sd, r, v, "vertex", zero_proj(sd, N))
        ref = fd_ref.pipeline(r, v["A"], v["R"], v["KA"], v["KB"], "vertex")
        g = "N%d" % N
        cl = classes(d)
        # hard: never a resolved wrong sign; module's own band is what counts, so delta = 0 here would be
        # too strict -> use plan 1.6 delta
        dp, dm = class_deltas(r, v, ref)
        check_classes(res, g, cl, v["Rtp"], v["Rtm"], v["R"], "vertex", dp, dm)
        res.check(not any(q["ah"] for q in roots_of(d)), "%s: ah_candidate on Minkowski" % g)
        gs = status(d, "geometry")
        res.check(gs not in ("AH_CANDIDATE", "FUTURE_TRAPPED_NO_AH_CANDIDATE"), "%s: geometry %s on Minkowski" % (g, gs))
        res.metric(g, dict(geometry=gs, n_past_marginal=int(np.sum(cl == "PAST_MARGINAL")),
                           n_past_trapped=int(np.sum(cl == "PAST_TRAPPED")), n_normal=int(np.sum(cl == "NORMAL"))))
        if gs == "PAST_TRAPPED_ONLY":
            res.note("%s: PAST_TRAPPED_ONLY issued for a Minkowski slice from unresolved marginal signs (contract question K3c)" % g)


# ============================================================================ S: Schwarzschild

def _schw_extra(expect_R_ah, geometry):
    def extra(res, g, r, v, d, ref):
        for f in ("M", "Rtheta_plus", "Rtheta_minus", "theta_plus", "theta_minus"):
            res.check(np.all(np.isfinite(arr(d, f))), "%s: non-finite %s (coordinate singularity?)" % (g, f))
        check_outer_ah(res, g, d, expect_R_ah)
        geom(res, g, d, geometry)
        for q in roots_of(d):
            if q["ah"]:
                res.check(abs(q["M"] - 1.0) <= 1e-6, "%s: M at root %.9g != 1" % (g, q["M"]))
    return extra


@case("S20", "Schwarzschild Painleve-Gullstrand (A=1): horizon at R=2M, AH candidate")
def t_S20(sd, res):
    sl = AN.schwarzschild_pg()
    for kind in ("uni", "str"):
        grids = [make_grid(kind, N, r0=0.5, r1=12.0, center="none") for N in NCONV]
        sym_family(sd, res, "PG/%s" % kind, sl, grids, "none", F_BASIC + F_RES, matter="zero",
                   edge_fields=("ham", "hamM"), roots_range=(0.5, 12.0), R_scale=1.0,
                   extra_checks=_schw_extra(2.0, "AH_CANDIDATE"))


@case("S21", "Schwarzschild ingoing Eddington-Finkelstein / Kerr-Schild (A!=1, K_A!=K_B)")
def t_S21(sd, res):
    sl = AN.schwarzschild_ks()
    for kind in ("uni", "str"):
        grids = [make_grid(kind, N, r0=0.5, r1=12.0, center="none") for N in NCONV]
        sym_family(sd, res, "KS/%s" % kind, sl, grids, "none", F_BASIC + F_RES, matter="zero",
                   edge_fields=("ham", "hamM"), roots_range=(0.5, 12.0), R_scale=1.0,
                   extra_checks=_schw_extra(2.0, "AH_CANDIDATE"))


def _iso_checks(res, g, r, v, d, ref):
    got = roots_of(d)
    res.check(not any(q["ah"] for q in got), "%s: FALSE ah_candidate at the Einstein-Rosen throat" % g)
    for q in got:
        res.check(q["type"] == "DEGENERATE", "%s: root %s at R=%.6g has type %s (expected DEGENERATE)" % (g, q["which"], q["R"], q["type"]))
        res.check(abs(q["R"] - 2.0) < 0.05, "%s: root away from the throat at R=%.6g" % (g, q["R"]))
    check_outer_ah(res, g, d, None)
    cl = classes(d)
    marg = np.isin(cl, ["FUTURE_MARGINAL", "PAST_MARGINAL", "DOUBLY_MARGINAL"])
    gs = status(d, "geometry")
    res.check(gs not in ("AH_CANDIDATE", "FUTURE_TRAPPED_NO_AH_CANDIDATE", "PAST_TRAPPED_ONLY", "UNDETERMINED"),
              "%s: geometry %s on the time-symmetric isotropic slice" % (g, gs))
    if not marg.any():
        res.check(gs == "NO_TRAPPED_SPHERES", "%s: geometry %s, expected NO_TRAPPED_SPHERES" % (g, gs))
    else:
        st = D(d, "status")
        mp = st.get("marginal_present") if isinstance(st, dict) else getattr(st, "marginal_present", None)
        if gs == "NO_TRAPPED_SPHERES":
            res.check(bool(mp), "%s: marginal classes present but status.marginal_present = %r" % (g, mp))
        elif gs == "UNCERTAIN":
            res.note("%s: UNCERTAIN at the throat (literal reading of contract 1.0, question K3a')" % g)
        res.metric("%s/marginal" % g, dict(n=int(marg.sum()), geometry=gs, marginal_present=mp))
    # classes in the reversed region / exterior are covered by check_classes (sign rule of plan 1.6)
    res.metric("%s/n_reversed" % g, int(np.sum(cl == "NORMAL_REVERSED")))


@case("S22", "Schwarzschild isotropic K=0 slice: throat (doubly marginal), reversed region, no false AH")
def t_S22(sd, res):
    sl = AN.schwarzschild_iso()
    for kind in ("uni", "str"):
        grids = [make_grid(kind, N, r0=0.2, r1=10.0, center="none") for N in (201, 401, 801, 1601)]
        sym_family(sd, res, "iso/%s" % kind, sl, grids, "none", F_BASIC + F_RES, matter="zero",
                   edge_fields=("ham", "hamM"), extra_checks=_iso_checks)
    for k in (8, 16):
        h = 0.4 / k
        r = 0.1 + h * np.arange(int(round(9.9 / h)) + 1)
        r[k] = 0.5
        for derivs in (False, True):
            v = sl.exact(r)
            d, _ = build(sd, r, v, "none", zero_proj(sd, len(r)), derivs=derivs)
            ref = fd_ref.pipeline(r, v["A"], v["R"], v["KA"], v["KB"], "none")
            g = "iso/node/h%.4g/%s" % (h, "analytic" if derivs else "FD")
            dp, dm = class_deltas(r, v, ref)
            check_classes(res, g, classes(d), v["Rtp"], v["Rtm"], v["R"], "none", dp, dm)
            _iso_checks(res, g, r, v, d, ref)
            e = np.max(np.abs(arr(d, "M") - 1.0))
            res.check(e <= 10 * np.max(np.abs(ref["M"] - 1)) + floor(1, r[-1] - r[0], h, 1) if not derivs else e <= 1e-12,
                      "%s: M error %.2e" % (g, e))


@case("S23", "horizon exactly on a grid node: one root, FUTURE_MARGINAL node, AH candidate")
def t_S23(sd, res):
    sl = AN.schwarzschild_pg()
    r = np.linspace(1, 3, 41)
    r[20] = 2.0
    v = sl.exact(r)
    d, _ = build(sd, r, v, "none", zero_proj(sd, 41))
    got = [q for q in roots_of(d) if q["which"] == "+"]
    res.check(len(got) == 1, "expected exactly one + root, got %d: %s" % (len(got), [(q["R"], q["type"]) for q in got]))
    res.check(not [q for q in roots_of(d) if q["which"] == "-"], "spurious - roots")
    if got:
        res.check(abs(got[0]["R"] - 2.0) <= 1e-12, "root R=%.15g != 2" % got[0]["R"])
        res.check(got[0]["ah"] and got[0]["type"] == "FUTURE_MOTS", "root not an AH candidate: %s" % got[0])
    cl = classes(d)
    res.check(cl[20] == "FUTURE_MARGINAL", "class at the node r=2M is %s" % cl[20])
    res.check(np.all(cl[:20] == "FUTURE_TRAPPED") and np.all(cl[21:] == "NORMAL"), "classes around the node wrong")
    geom(res, "node", d, "AH_CANDIDATE")


@case("S24", "grid windows entirely inside / outside the horizon: no false edge roots")
def t_S24(sd, res):
    sl = AN.schwarzschild_pg()
    for (a, b), cls_exp, gexp in (((0.5, 1.9), "FUTURE_TRAPPED", "FUTURE_TRAPPED_NO_AH_CANDIDATE"),
                                  ((2.05, 12.0), "NORMAL", "NO_TRAPPED_SPHERES")):
        for N in (50, 200):
            r = np.linspace(a, b, N)
            v = sl.exact(r)
            d, _ = build(sd, r, v, "none", zero_proj(sd, N))
            g = "[%.3g,%.3g]/N%d" % (a, b, N)
            res.check(np.all(classes(d) == cls_exp), "%s classes %s" % (g, sorted(set(classes(d)))))
            res.check(len(D(d, "roots")) == 0, "%s spurious roots %s" % (g, [(q["which"], q["R"]) for q in roots_of(d)]))
            res.check(D(d, "outer_ah_candidate") is None, "%s outer_ah_candidate not None" % g)
            geom(res, g, d, gexp)


@case("S25", "synthetic PBH-in-FLRW snapshot: trapped shells, candidate selection, precedence")
def t_S25(sd, res):
    for shells, lab in ((((1.0, 3.0),), "one_shell"), (((1.0, 3.0), (0.5, 8.0)), "two_shells")):
        sl = AN.synthetic_pbh_flrw(shells=shells)
        exp = expected_roots(sl, 1e-6, 30.0)
        cands = [e for e in exp if e["ah"]]
        exp_outer = max(cands, key=lambda e: e["r"])["r"] if cands else None
        res.metric("%s/expected_roots" % lab, [(e["which"], e["r"], e["type"], e["ah"]) for e in exp])
        grids = [make_grid("uni", N, L=30.0, center="vertex") for N in (401, 801, 1601)]

        def extra(res, g, r, v, d, ref, exp_outer=exp_outer):
            check_outer_ah(res, g, d, exp_outer)
            geom(res, g, d, "AH_CANDIDATE")
        sym_family(sd, res, lab, sl, grids, "vertex", F_BASIC, matter=None, roots_range=(1e-6, 30.0),
                   R_scale=1.0, extra_checks=extra)


# ============================================================================ F: FLRW

def _flrw_common(sd, res, H, geometry, exp_outer):
    a = 1.3
    sl = AN.flrw(a=a, H=H)
    rho = 3 * H**2 / (8 * np.pi)
    Lr = (3 / abs(H)) / a
    for center in ("vertex", "cell"):
        for N in (64, 128, 256):
            r = make_grid("uni", N, L=Lr, center=center)
            v = sl.exact(r)
            ref = fd_ref.pipeline(r, v["A"], v["R"], v["KA"], v["KB"], center, v["rho"], v["jr"])
            for mlab, m in (("Projections", proj(sd, v)),
                            ("Vlasov", sd.VlasovMoments(v["rho"], np.zeros(N), np.zeros(N), np.zeros(N)))):
                d, _ = build(sd, r, v, center, m)
                g = "H%+.1f/%s/N%d/%s" % (H, center, N, mlab)
                L, hm = r[-1] - r[0], hmin_of(r)
                R = v["R"]
                Mex = H**2 * R**3 / 2
                res.check(np.max(np.abs(arr(d, "M") - Mex)) <= floor(np.max(Mex), L, hm, 1), "%s M != H^2R^3/2" % g)
                res.check(np.max(np.abs(arr(d, "M") - 4 * np.pi / 3 * rho * R**3)) <= floor(np.max(Mex), L, hm, 1) + 1e-14 * np.max(Mex),
                          "%s M != (4pi/3) rho R^3" % g)
                mr3 = arr(d, "M_over_R3")
                res.check(np.all(np.isfinite(mr3)) and np.max(np.abs(mr3 - H**2 / 2)) <= floor(H**2 / 2, L, hm, 3),
                          "%s M/R^3 != H^2/2 (max dev %.2e)" % (g, np.nanmax(np.abs(mr3 - H**2 / 2))))
                ext = np.atleast_1d(np.asarray(D(d, "M_over_R3_extrapolated")))
                if center == "vertex":
                    res.check(bool(ext[0]) and (ext.size == 1 or not np.any(ext[1:])), "%s M_over_R3_extrapolated flag wrong" % g)
                comp = arr(d, "compactness")
                res.check(np.max(np.abs(comp - H**2 * R**2)) <= floor(np.max(H**2 * R**2), L, hm, 1), "%s compactness" % g)
                dp, dm = class_deltas(r, v, ref)
                check_classes(res, g, classes(d), v["Rtp"], v["Rtm"], R, center, dp, dm)
                match_roots(res, g, sl, r, d, v, ref, 1e-9, r[-1], 1 / abs(H))
                check_outer_ah(res, g, d, exp_outer)
                geom(res, g, d, geometry)
                res.check(status(d, "geometry") != "AH_CANDIDATE", "%s: FLRW reported as AH candidate" % g)
                for f, S, dd in (("ham", 16 * np.pi * rho + 6 * H**2, 2), ("mom", abs(H) / L, 1), ("hamM", np.max(Mex) / L, 2)):
                    x = arr(d, f)
                    ok = R > 0 if f != "hamM" else np.ones(N, bool)
                    tol = floor(S, L, hm, dd)
                    res.check(np.all(np.abs(x[ok]) <= tol), "%s %s != 0: max %.2e > %.2e" % (g, f, np.nanmax(np.abs(x[ok])), tol))
                not_violated(res, g, d)
                integ = 4 * np.pi * R**2 * rho * a
                Mcum = 4 * np.pi / 3 * rho * a**3 * (r**3 - r[0]**3)
                quad_check(res, g + "/M_rho", arr(d, "M_rho"), Mcum, integ, r, np.max(Mex))
                quad_check(res, g + "/M_prop", arr(d, "M_prop"), Mcum, integ, r, np.max(Mex))
                quad_check(res, g + "/M_int", arr(d, "M_int"), Mex, integ, r, np.max(Mex))
                res.check(np.max(np.abs(arr(d, "M_flux"))) <= floor(np.max(Mex), L, hm, 1), "%s M_flux != 0" % g)


@case("F30", "expanding FLRW dust: M=(4pi/3)rho R^3, Hubble sphere PAST marginal, PAST_TRAPPED beyond")
def t_F30(sd, res):
    _flrw_common(sd, res, 0.4, "PAST_TRAPPED_ONLY", None)


@case("F31", "contracting FLRW dust: Hubble sphere FUTURE_MOTS bounding the trapped region from inside -> not AH")
def t_F31(sd, res):
    _flrw_common(sd, res, -0.4, "FUTURE_TRAPPED_NO_AH_CANDIDATE", None)


@case("F32", "Hubble sphere between the last two nodes: root at the outer edge")
def t_F32(sd, res):
    a, H = 1.3, 0.4
    sl = AN.flrw(a=a, H=H)
    for N in (32, 64, 128):
        rH = 1 / (a * H)
        r = np.linspace(0, rH * (1 + 0.5 / (N - 1)), N)
        v = sl.exact(r)
        d, _ = build(sd, r, v, "vertex", proj(sd, v))
        got = roots_of(d)
        g = "N%d" % N
        if res.check(len(got) == 1, "%s: expected one root, got %d" % (g, len(got))):
            q = got[0]
            res.check(q["which"] == "-" and q["type"] == "PAST_MOTS" and not q["ah"], "%s: root %s" % (g, q))
            res.check(abs(q["R"] - 1 / H) <= floor(1 / H, r[-1], hmin_of(r), 1), "%s: R*=%.15g vs %.15g" % (g, q["R"], 1 / H))
        geom(res, g, d, "PAST_TRAPPED_ONLY")


# ============================================================================ K: FKG

def _fkg_slice_checks(sd, res, g, fk, t, N, hard_root=True):
    s = fk.state(t)
    a, H = s["a"], s["H"]
    r = np.linspace(0, 2 / (a * H), N)
    sl = fk.slice(t, r)
    slc = sd.SphericalSlice(r, sl["A"], sl["R"], sl["KA"], sl["KB"], center="vertex")
    d = sd.diagnose(slc, sd.ScalarField(sl["phi"], sl["Pi"], fk.m))
    P = D(d, "projections")
    R = sl["R"]
    L, hm = r[-1], hmin_of(r)
    rho, p = s["rho"], s["p"]
    res.check(np.max(np.abs(_P(P, "rho") - rho)) <= 100 * EPS * rho, "%s rho" % g)
    res.check(np.max(np.abs(_P(P, "S_rr") - p)) <= 100 * EPS * rho, "%s S_rr != p" % g)
    res.check(np.max(np.abs(_P(P, "S_thth") - p)) <= 100 * EPS * rho, "%s S_thth != p" % g)
    res.check(np.max(np.abs(_P(P, "j_r"))) <= floor(abs(s["dphi"] * s["phi"]) / L, L, hm, 1) + 1e-300,
              "%s spurious spatial gradient: j_r = %.2e" % (g, np.max(np.abs(_P(P, "j_r")))))
    ham = arr(d, "ham")
    hl = np.maximum(np.r_[np.diff(r), r[-1] - r[-2]], np.r_[r[1] - r[0], np.diff(r)])
    pos = R > 0
    tol = 1e3 * EPS * (6 * H**2 + 16 * np.pi * rho + 20 / (a * hl[pos] * R[pos]))
    dev = np.abs(ham[pos] - 6 * s["F_res"])
    res.check(np.all(dev <= tol), "%s ham - 6 F_res: max dev %.2e (tol at max %.2e)" % (g, np.max(dev), tol[np.argmax(dev - tol)]))
    res.check(np.all(np.abs(arr(d, "mom")[pos]) <= floor(abs(H) / L + 4 * np.pi * abs(s["dphi"] * s["phi"]) / L, L, hm, 1)),
              "%s mom != 0" % g)
    res.check(np.max(np.abs(arr(d, "M") - H**2 * R**3 / 2)) <= floor(H**2 * R[-1]**3, L, hm, 1), "%s M != H^2R^3/2" % g)
    got = roots_of(d)
    if hard_root and res.check(len(got) == 1, "%s: expected one root (Hubble sphere), got %d" % (g, len(got))):
        q = got[0]
        res.check(q["which"] == "-" and q["type"] == "PAST_MOTS" and not q["ah"], "%s: root %s" % (g, q))
        res.check(abs(q["R"] - 1 / H) <= floor(1 / H, L, hm, 1), "%s: Hubble root R*=%.12g vs %.12g" % (g, q["R"], 1 / H))
    geom(res, g, d, "PAST_TRAPPED_ONLY")
    return d, s


@case("K40", "homogeneous massive scalar (own FKG ODE): rho, p (instantaneous != 0), ham = 6 F_res, mom = 0")
def t_K40(sd, res):
    fk = FKG(m=1.0, H0_over_m=0.2, t_end=60.0)
    ts = 30 + np.linspace(0, 2 * np.pi, 9)[:-1]
    ratios = []
    for N in (100, 200, 400):
        for t in ts:
            d, s = _fkg_slice_checks(sd, res, "N%d/t%.3f" % (N, t), fk, t, N)
            ratios.append(float(_P(D(d, "projections"), "S_rr")[0] / s["rho"]))
    res.metric("p_over_rho_range", [min(ratios), max(ratios)])
    res.check(min(ratios) <= -0.7 and max(ratios) >= 0.7, "instantaneous p/rho does not swing (range %s)" % [min(ratios), max(ratios)])
    avg = {}
    for q in (8.0, 20.0, 45.0):
        t0 = brentq(lambda t: fk.m / fk.state(t)["H"] - q, 0.01, 59.0)
        rr_, pp_, tt = fk.period_average(t0, 32, 4)
        _, w = fk.average_weights(t0, 32, 4)
        Srr = []
        for t in tt:
            s = fk.state(t)
            r = np.linspace(0, 2 / (s["a"] * s["H"]), 16)
            sl = fk.slice(t, r)
            slc = sd.SphericalSlice(r, sl["A"], sl["R"], sl["KA"], sl["KB"], center="vertex")
            Srr.append(_P(sd.ScalarField(sl["phi"], sl["Pi"], fk.m).projections(slc), "S_rr")[0])
        pm = float(np.dot(w, Srr))
        res.check(abs(pm - pp_) <= 1e-10 * rr_, "m/H=%g: module <p> %.6e vs reference %.6e" % (q, pm, pp_))
        avg[q] = pm / rr_
    res.metric("avg_p_over_rho", avg)
    res.check(all(abs(avg[q]) <= 0.15 / q for q in avg), "|<p>/<rho>| exceeds 0.15 H/m (window leakage bound): %s" % avg)


@case("K41", "FKG with loose ODE tolerance: module ham reproduces 6*F_res (sign and size)")
def t_K41(sd, res):
    fk = FKG(m=1.0, H0_over_m=0.2, t_end=40.0, rtol=1e-6, atol=1e-9)
    for t in (20.0, 30.0, 35.0):
        s = fk.state(t)
        d, _ = _fkg_slice_checks(sd, res, "loose/t%.1f" % t, fk, t, 200)
        H, rho = s["H"], s["rho"]
        res.metric("F_res/H^2@t%.0f" % t, s["F_res"] / H**2)
        tol_bulk = 1e3 * EPS * (6 * H**2 + 16 * np.pi * rho) * 10
        res.check(abs(6 * s["F_res"]) >= 10 * tol_bulk, "t=%g: Friedmann residual too small to be a meaningful test (%.2e)" % (t, s["F_res"]))
        ham = arr(d, "ham")
        mid = ham[len(ham) // 2]
        res.check(np.sign(mid) == np.sign(s["F_res"]), "t=%g: sign of ham (%.2e) differs from F_res (%.2e)" % (t, mid, s["F_res"]))


# ============================================================================ X: scalar flux data

def _flux_family(sd, res, fam, fd, kind, Ns, matter_kind="scalar"):
    errs = {}
    hs = []
    outs = []
    for N in Ns:
        r = make_grid(kind, N, L=fd.r_max, center="vertex")
        v = fd.exact(r)
        if matter_kind == "scalar":
            m = sd.ScalarField(v["phi"], v["Pi"], fd.m)
            rp = fd_ref.scalar_projections(r, v["A"], v["phi"], v["Pi"], fd.m, "vertex")
        else:
            m = sd.VlasovMoments(v["rho"], v["J"], v["Srr"], v["Sthth"])
            rp = dict(rho=v["rho"], jr=v["jr"])
        d, _ = build(sd, r, v, "vertex", m)
        ref = fd_ref.pipeline(r, v["A"], v["R"], v["KA"], v["KB"], "vertex", rp["rho"], rp["jr"])
        g = "%s/N%d" % (fam, N)
        for lab, mf, key, Sfn, dd in F_BASIC + F_RES:
            ex_unc = 10 * np.nanmax(np.abs(v[key])) if lab in ("ham", "mom", "hamM") else 1e-11 * Sfn(v)
            errs.setdefault(lab, []).append(compare(res, "%s/%s" % (g, lab), arr(d, mf), v[key], ref[key], r, v["R"],
                                                    v["A"], "vertex", Sfn(v), dd, extra=ex_unc))
        P = D(d, "projections")
        Sp = np.max(np.abs(v["rho"]) + np.abs(v["jr"]))
        for mk, ek in (("rho", "rho"), ("j_r", "jr")):
            errs.setdefault(mk, []).append(compare(res, "%s/proj_%s" % (g, mk), _P(P, mk), v[ek], rp[ek], r, v["R"], v["A"],
                                                   "vertex", Sp, 1 if matter_kind == "scalar" else 0))
        R, A = v["R"], v["A"]
        Smass = np.max(np.abs(v["M"])) + np.max(np.abs(v["M_flux"]))
        irho, iflux = 4 * np.pi * R**2 * v["rho"] * v["dR"], 4 * np.pi * R**2 * v["U"] * v["jr"]
        irho_r, iflux_r = 4 * np.pi * R**2 * rp["rho"] * ref["dR"], 4 * np.pi * R**2 * ref["U"] * rp["jr"]
        quad_check(res, g + "/M_rho", arr(d, "M_rho"), v["M_rho"], irho, r, Smass, 1e-11 * Smass, irho_r)
        quad_check(res, g + "/M_flux", arr(d, "M_flux"), v["M_flux"], iflux, r, Smass, 1e-11 * Smass, iflux_r)
        quad_check(res, g + "/M_prop", arr(d, "M_prop"), v["M_prop"], 4 * np.pi * R**2 * A * v["rho"], r, Smass,
                   1e-11 * Smass, 4 * np.pi * R**2 * A * rp["rho"])
        e1, t1 = quad_check(res, g + "/M_int-vs-M_MS(ode)", arr(d, "M_int"), v["M"], irho + iflux, r, Smass,
                            1e-11 * Smass + 10 * np.abs(ref["M"][0] - v["M"][0]), irho_r + iflux_r)
        mf = arr(d, "M_flux")[-1]
        res.check(np.sign(mf) == np.sign(v["M_flux"][-1]) and abs(v["M_flux"][-1]) >= 0.05 * abs(v["M"][-1]),
                  "%s: M_flux(r_max) = %.4e, reference %.4e (sign/size)" % (g, mf, v["M_flux"][-1]))
        dMMS = arr(d, "M") - arr(d, "M_rho")
        res.metric("%s/M_MS-M_rho@rmax" % g, dict(module=float(dMMS[-1]), ref_flux=float(v["M_flux"][-1])))
        cs = not_violated(res, g, d, hard=False)
        if N >= 200:
            res.check(cs != "VIOLATED", "%s: constraints VIOLATED on consistent data (if sigma omits the FD error of the "
                      "matter projections this is contract question K24)" % g)
        res.check(status(d, "geometry") == "NO_TRAPPED_SPHERES", "%s geometry %s" % (g, status(d, "geometry")))
        hs.append(hmax_of(r))
        outs.append((r, v, d))
    for lab in errs:
        if lab in ("ham", "hamM"):
            convergence(res, "%s/%s" % (fam, lab), hs, errs[lab], 3.5, ("bulk",))
            _edge_conv(res, fam, lab, hs, errs[lab], 2.5)
        else:
            convergence(res, "%s/%s" % (fam, lab), hs, errs[lab], 3.5, ("bulk", "center", "outer"))
    return outs


@case("X50", "EKG data with radial flux (own constraint-ODE solver): residual convergence, M_MS, M_rho, M_flux")
def t_X50(sd, res):
    fd = FluxData()
    for kind in ("uni", "str"):
        _flux_family(sd, res, "flux/%s" % kind, fd, kind, NCONV)


@case("X51", "flux sign flip (Pi -> -Pi, K_B -> -K_B)")
def t_X51(sd, res):
    N = 400
    r = make_grid("uni", N, L=8.0, center="vertex")
    base = FluxData()
    vb = base.exact(r)
    for lab, fdx in (("Pi", FluxData(sPi=-1.0)), ("KB", FluxData(sKB=-1.0))):
        vf = fdx.exact(r)
        res.check(np.max(np.abs(vf["M_flux"] + vb["M_flux"])) <= 1e-12 * np.max(np.abs(vb["M_flux"])),
                  "reference antisymmetry of M_flux broken for flip %s (reference failure -> VOID)" % lab)
        out = {}
        for tag, v, fdd in (("+", vb, base), ("-", vf, fdx)):
            d, _ = build(sd, r, v, "vertex", sd.ScalarField(v["phi"], v["Pi"], fdd.m))
            Smass = np.max(np.abs(v["M"])) + np.max(np.abs(v["M_flux"]))
            rp = fd_ref.scalar_projections(r, v["A"], v["phi"], v["Pi"], fdd.m, "vertex")
            _, tol = quad_check(res, "%s%s/M_flux" % (lab, tag), arr(d, "M_flux"), v["M_flux"],
                                4 * np.pi * v["R"]**2 * v["U"] * v["jr"], r, Smass, 1e-11 * Smass,
                                4 * np.pi * v["R"]**2 * v["U"] * rp["jr"])
            out[tag] = (arr(d, "M_flux"), tol, v)
            res.check(np.sign(arr(d, "M_flux")[-1]) == np.sign(v["M_flux"][-1]), "flip %s%s: wrong sign of M_flux" % (lab, tag))
        s = np.abs(out["+"][0] + out["-"][0])
        res.check(np.all(s <= out["+"][1] + out["-"][1]), "flip %s: M_flux(+)+M_flux(-) max %.2e not ~0" % (lab, np.max(s)))
        res.metric("flip_%s" % lab, dict(Mflux_plus=float(out["+"][0][-1]), Mflux_minus=float(out["-"][0][-1]),
                                         ref_plus=float(vb["M_flux"][-1]), ref_minus=float(vf["M_flux"][-1])))


@case("X52", "Vlasov adapter on the flux data: j_r = A J, residual convergence, M_flux")
def t_X52(sd, res):
    fd = FluxData()
    _flux_family(sd, res, "vlasov", fd, "uni", (200, 400, 800), matter_kind="vlasov")


# ============================================================================ C: violated constraints

def _bump(r, c=3.0, w=0.8, k=0):
    x = (r - c) / w
    inside = np.abs(x) < 1
    b = np.zeros_like(r)
    db = np.zeros_like(r)
    xi = x[inside]
    b[inside] = np.exp(1 - 1 / (1 - xi**2))
    db[inside] = b[inside] * (-2 * xi / (1 - xi**2)**2) / w
    return (b, db) if k else b


@case("C60", "deliberately violated constraints: exact linear response, localisation, VIOLATED, no normalisation")
def t_C60(sd, res):
    fd = FluxData()
    N = 400
    r = make_grid("uni", N, L=8.0, center="vertex")
    v = fd.exact(r)
    b, db = _bump(r, k=1)
    supp = b > 0
    near = np.convolve(supp.astype(float), np.ones(9), "same") > 0  # support +- 4 nodes
    R, KA, KB = v["R"], v["KA"], v["KB"]

    def run(KA_=None, KB_=None, rho_=None, jr_=None):
        vv = dict(v)
        if KA_ is not None:
            vv["KA"] = KA_
        if KB_ is not None:
            vv["KB"] = KB_
        m = sd.Projections(v["rho"] if rho_ is None else rho_, v["jr"] if jr_ is None else jr_, v["Srr"], v["Sthth"])
        return build(sd, r, vv, "vertex", m)[0]

    d0 = run()
    base = {f: arr(d0, f) for f in ("ham", "mom", "hamM")}
    base_err = {f: np.nanmax(np.abs(base[f] - v[f])) for f in base}
    res.metric("base_truncation", base_err)
    ref_db = fd_ref.deriv(b, r, "vertex", 1)
    edb = 10 * _win_max(ref_db - db) + floor(np.max(np.abs(db)), 8.0, hmin_of(r), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        invr = np.where(R > 0, 1 / R, 0.0)
    for eps in (1e-2, 1e-4, 1e-6):
        cases = {
            "KA": (dict(KA_=KA + eps * b), dict(ham=4 * KB * eps * b, mom=-invr * eps * b, hamM=0 * b), {}),
            "rho": (dict(rho_=v["rho"] + eps * b), dict(ham=-16 * np.pi * eps * b, hamM=-4 * np.pi * R**2 * v["dR"] * eps * b,
                                                        mom=0 * b), {}),
            "jr": (dict(jr_=v["jr"] + eps * b), dict(mom=4 * np.pi * eps * b, hamM=4 * np.pi * R**3 * KB * eps * b,
                                                     ham=0 * b), {}),
            "KB": (dict(KB_=KB + eps * b), dict(mom=eps * db + invr * eps * b,
                                                ham=4 * KA * eps * b + 4 * KB * eps * b + 2 * eps**2 * b**2), {"mom": eps * edb}),
        }
        for lab, (kw, dexp, extra_tol) in cases.items():
            d1 = run(**kw)
            g = "%s/eps%.0e" % (lab, eps)
            for f, De in dexp.items():
                x1 = arr(d1, f)
                Dm = x1 - base[f]
                T = np.nan_to_num(v["%s_absterms" % f]) + np.abs(De)
                tol = 100 * EPS * (np.abs(np.nan_to_num(x1)) + np.abs(np.nan_to_num(base[f])) + T) + extra_tol.get(f, 0.0)
                ok = R > 0 if f != "hamM" else np.ones(N, bool)
                err = np.abs(Dm - De)
                bad = ok & ~(err <= tol)
                res.check(not bad.any(), "%s %s: response != exact at %d pts (max err %.2e, tol there %.2e, r=%.3g)"
                          % (g, f, int(bad.sum()), np.max(np.where(ok, err, 0)), tol[np.argmax(np.where(ok, err - tol, -np.inf))],
                             r[np.argmax(np.where(ok, err - tol, -np.inf))]))
                outside = ok & ~(near if lab == "KB" else supp)
                res.check(np.all(np.abs(Dm[outside]) <= tol[outside]), "%s %s: response leaks outside the bump (max %.2e)"
                          % (g, f, np.max(np.abs(Dm[outside]))))
                if np.max(np.abs(De)) > 0:
                    k = int(np.argmax(np.abs(De)))
                    relm = arr(d1, f + "_rel")[k]
                    rel_min = 0.5 * abs(De[k]) / (np.nan_to_num(v["%s_absterms" % f])[k] + abs(De[k]))
                    res.check(relm >= rel_min, "%s %s_rel at the peak %.3e < %.3e: normalised away" % (g, f, relm, rel_min))
                    res.metric("%s/%s" % (g, f), dict(peak_exact=float(De[k]), peak_module=float(Dm[k]), rel=float(relm)))
            big = max(np.max(np.abs(De)) / max(base_err[f], 1e-300) for f, De in dexp.items() if np.max(np.abs(De)) > 0)
            cs = status(d1, "constraints")
            if big >= 1e3:
                res.check(cs == "VIOLATED", "%s: constraints status %s, expected VIOLATED (perturbation/truncation = %.1e)" % (g, cs, big))
            res.metrics.setdefault("status", {})[g] = dict(constraints=cs, perturbation_over_truncation=big)


# ============================================================================ T: coordinate transformation

@case("T70", "radial coordinate transformation: same physical spheres, two coordinates")
def t_T70(sd, res):
    # (a) flux data, vertex, r = 8 sinh(2x)/sinh(2)
    fd = FluxData()
    f, fp = sinh_map(8.0, 2.0)
    for N in (201, 401, 801):
        x = np.linspace(0, 1, N)
        rn = f(x)
        rn[0] = 0.0
        vr = fd.values(rn)
        vx = fd.transformed_values(x, f, fp)
        e = fd.exact(rn)
        fpx = fp(x)
        dr_, _ = build(sd, rn, vr, "vertex", sd.ScalarField(vr["phi"], vr["Pi"], fd.m))
        dx_, _ = build(sd, x, vx, "vertex", sd.ScalarField(vx["phi"], vx["Pi"], fd.m))
        rp_r = fd_ref.scalar_projections(rn, vr["A"], vr["phi"], vr["Pi"], fd.m)
        rp_x = fd_ref.scalar_projections(x, vx["A"], vx["phi"], vx["Pi"], fd.m)
        ref_r = fd_ref.pipeline(rn, vr["A"], vr["R"], vr["KA"], vr["KB"], "vertex", rp_r["rho"], rp_r["jr"])
        ref_x = fd_ref.pipeline(x, vx["A"], vx["R"], vx["KA"], vx["KB"], "vertex", rp_x["rho"], rp_x["jr"])
        g = "flux/N%d" % N
        for lab, mf, key, S, dd, scale in (("M", "M", "M", np.max(np.abs(e["M"])), 1, 1.0),
                                           ("Rtp", "Rtheta_plus", "Rtp", 4.0, 1, 1.0),
                                           ("Rtm", "Rtheta_minus", "Rtm", 4.0, 1, 1.0),
                                           ("ham", "ham", "ham", np.nanmax(e["ham_absterms"]), 2, 1.0),
                                           ("mom", "mom", "mom", np.nanmax(e["mom_absterms"]), 1, fpx),
                                           ("hamM", "hamM", "hamM", np.nanmax(e["hamM_absterms"]), 2, fpx)):
            mr, mx = arr(dr_, mf), arr(dx_, mf) / scale
            er_ = np.abs(ref_r[key] - e[key])
            ex_ = np.abs(ref_x[key] / scale - e[key])
            tol = 10 * (_win_max(er_) + _win_max(ex_)) + floor(S, 8.0, hmin_of(rn), dd) + floor(S, 1.0, hmin_of(x), dd) * np.maximum(1, 1 / scale) \
                + 20 * np.nanmax(np.abs(e[key]) if lab in ("ham", "mom", "hamM") else 0)
            ok = np.isfinite(e[key]) & (rn > 0 if lab in ("ham", "mom") else True)
            dev = np.abs(mr - mx)
            bad = ok & ~(dev <= tol)
            res.check(not bad.any(), "%s %s: coordinate dependence |Q_r - Q_x| max %.2e > tol (%d pts)" % (g, lab, np.nanmax(dev[ok]), int(bad.sum())))
            res.metrics.setdefault("coord_dev", {})["%s/%s" % (g, lab)] = float(np.nanmax(dev[ok]))
        for mf in ("M_rho", "M_flux", "M_prop", "M_int"):
            dev = np.abs(arr(dr_, mf) - arr(dx_, mf))
            Smass = np.max(np.abs(e["M"])) + np.max(np.abs(e["M_flux"]))
            integ = {"M_rho": 4 * np.pi * e["R"]**2 * e["rho"], "M_flux": 4 * np.pi * e["R"]**2 * e["U"] * e["jr"],
                     "M_prop": 4 * np.pi * e["R"]**2 * e["A"] * e["rho"],
                     "M_int": 4 * np.pi * e["R"]**2 * (e["rho"] + e["U"] * e["jr"])}[mf]
            exact_cum = {"M_rho": e["M_rho"], "M_flux": e["M_flux"], "M_prop": e["M_prop"], "M_int": e["M"]}[mf]
            tr = np.concatenate([[0], np.cumsum(0.5 * np.diff(rn) * (integ[1:] + integ[:-1]))])
            ix = integ * fpx
            tx = np.concatenate([[0], np.cumsum(0.5 * np.diff(x) * (ix[1:] + ix[:-1]))])
            tol = 2 * (np.abs(tr - exact_cum) + np.abs(tx - exact_cum)) + 2e-11 * Smass + floor(Smass, 8.0, hmin_of(rn), 1)
            res.check(np.all(dev <= tol), "%s %s coordinate dependence max %.2e" % (g, mf, np.max(dev)))
    # (b) KS none, exponential map; (c) height function, sinh map
    import sympy as sp
    for sl, center, fmap, r0, r1, lab in ((AN.schwarzschild_ks(), "none", None, 0.5, 12.0, "KS"),
                                          (AN.minkowski_height(), "vertex", None, 0.0, 6.0, "height")):
        if center == "none":
            fx = r0 + (r1 - r0) * (sp.exp(2 * AN.x) - 1) / (sp.exp(2) - 1)
        else:
            fx = r1 * sp.sinh(2 * AN.x) / sp.sinh(2)
        slx = sl.transformed(fx)
        fnum = sp.lambdify(AN.x, fx, "numpy")
        Rs = {}
        for N in (200, 400, 800):
            x = np.linspace(0, 1, N)
            rn = np.asarray(fnum(x), float)
            rn[0], rn[-1] = r0, r1
            for tag, sli, co in (("r", sl, rn), ("x", slx, x)):
                v = sli.exact(co)
                d, _ = build(sd, co, v, center, zero_proj(sd, N))
                ref = fd_ref.pipeline(co, v["A"], v["R"], v["KA"], v["KB"], center)
                g = "%s/%s/N%d" % (lab, tag, N)
                dp, dm = class_deltas(co, v, ref)
                check_classes(res, g, classes(d), v["Rtp"], v["Rtm"], v["R"], center, dp, dm)
                compare(res, g + "/M", arr(d, "M"), v["M"], ref["M"], co, v["R"], v["A"], center, S_M(v), 1)
                compare(res, g + "/Rtp", arr(d, "Rtheta_plus"), v["Rtp"], ref["Rtp"], co, v["R"], v["A"], center, S_RT(v), 1)
                if lab == "KS":
                    match_roots(res, g, sli, co, d, v, ref, co[0], co[-1], 1.0)
                    Rs[(tag, N)] = [q["R"] for q in roots_of(d) if q["ah"]]
        if lab == "KS":
            for N in (200, 400, 800):
                a_, b_ = Rs.get(("r", N), []), Rs.get(("x", N), [])
                if a_ and b_:
                    res.metric("KS/N%d/root_R_r_vs_x" % N, [a_[0], b_[0], abs(a_[0] - b_[0])])


# ============================================================================ Z: centre and edges

@case("Z80", "regular centre: vertex/cell, parity, R=0 limits, M/R^3 extrapolation, center_checks")
def t_Z80(sd, res):
    for irregular in (0.0, 0.1):
        sl = AN.polynomial_slice(irregular=irregular)
        for N in (41, 81):
            for center, kind in (("vertex", "uni"), ("vertex", "str"), ("cell", "uni"), ("cell", "str")):
                r = make_grid(kind, N, L=3.0, center=center)
                v = sl.exact(r)
                d, _ = build(sd, r, v, center, proj(sd, v))
                ref = fd_ref.pipeline(r, v["A"], v["R"], v["KA"], v["KB"], center, v["rho"], v["jr"])
                g = "irr%.1f/%s/%s/N%d" % (irregular, center, kind, N)
                L, hm = r[-1] - r[0], hmin_of(r)
                cc = D(d, "center_checks")
                G0 = float(cc["Gamma0"] if isinstance(cc, dict) else getattr(cc, "Gamma0"))
                KK0 = float(cc["KA_minus_KB0"] if isinstance(cc, dict) else getattr(cc, "KA_minus_KB0"))
                if center == "vertex":
                    res.check(abs(arr(d, "M")[0]) <= 1e-14 * S_M(v), "%s M(0) != 0" % g)
                    res.check(abs(arr(d, "compactness")[0]) <= 1e-14, "%s compactness(0) != 0" % g)
                    res.check(abs(arr(d, "Rtheta_plus")[0] - 2) <= floor(2, L, hm, 1), "%s Rtheta+(0) != 2" % g)
                    res.check(abs(arr(d, "Rtheta_minus")[0] + 2) <= floor(2, L, hm, 1), "%s Rtheta-(0) != -2" % g)
                    res.check(np.isnan(arr(d, "theta_plus")[0]) and np.isnan(arr(d, "theta_minus")[0]), "%s theta(0) not NaN" % g)
                    cl = classes(d)
                    res.check(cl[0] == "CENTER" and not np.any(cl[1:] == "CENTER"), "%s CENTER class misplaced" % g)
                    lim = sl.M_over_R3_at0()
                    e0 = abs(arr(d, "M_over_R3")[0] - lim)
                    e1 = abs(v["M"][1] / v["R"][1]**3 - lim)
                    res.check(e0 <= e1 + floor(abs(lim) + 1, L, hm, 1), "%s M/R^3(0) extrapolation err %.2e worse than first-point value %.2e" % (g, e0, e1))
                    ext = np.atleast_1d(np.asarray(D(d, "M_over_R3_extrapolated")))
                    res.check(bool(ext[0]) and (ext.size == 1 or not np.any(ext[1:])), "%s extrapolated flag" % g)
                    res.check(abs(G0 - 1) <= 10 * abs(ref["Gamma"][0] - 1) + floor(1, L, hm, 1), "%s Gamma0 = %.15g" % (g, G0))
                    res.check(abs(KK0 - irregular) <= 1e-12, "%s KA_minus_KB0 = %.6g, expected %.3g" % (g, KK0, irregular))
                else:
                    cl = classes(d)
                    res.check(not np.any(cl == "CENTER"), "%s CENTER class on a cell grid" % g)
                    for f in ("M", "Rtheta_plus", "Rtheta_minus", "theta_plus", "theta_minus"):
                        res.check(np.all(np.isfinite(arr(d, f))), "%s non-finite %s on a cell grid" % (g, f))
                    res.check(abs(G0 - 1) <= abs(v["Gamma"][0] - 1) + 10 * abs(ref["Gamma"][0] - v["Gamma"][0]) + floor(1, L, hm, 1),
                              "%s Gamma0 extrapolation %.15g worse than nearest point" % (g, G0))
                    dk = (v["KA"] - v["KB"])[0]
                    res.check(abs(KK0 - irregular) <= abs(dk - irregular) + 1e-12, "%s KA_minus_KB0 = %.6g, expected %.3g" % (g, KK0, irregular))
                res.check(len(D(d, "roots")) == 0, "%s spurious roots near centre: %s" % (g, [(q["which"], q["R"]) for q in roots_of(d)]))
    sl = AN.manufactured()
    lim = sl.M_over_R3_at0()
    errs = []
    for N in (64, 128, 256, 512):
        r = make_grid("uni", N, L=6.0, center="vertex")
        v = sl.exact(r)
        d, _ = build(sd, r, v, "vertex", proj(sd, v))
        errs.append(abs(arr(d, "M_over_R3")[0] - lim))
        near = [q for q in roots_of(d) if q["R"] < 0.5]
        res.check(not near, "manufactured N%d: roots near the centre %s" % (N, near))
    res.metric("manufactured_M_over_R3_0_err", errs)
    res.check(all(errs[i + 1] < errs[i] for i in range(3)) or max(errs) < 1e-10, "M/R^3(0) extrapolation not converging: %s" % errs)


# ============================================================================ D: derivatives, sigma

@case("D90", "manufactured solution: convergence order of derivatives and sigma bounds the true error")
def t_D90(sd, res):
    sl = AN.manufactured()
    fields = [("dR", "dR", "dR", lambda v: np.max(np.abs(v["dR"])), 1)] + F_BASIC + F_RES
    sig = {"M": "sigma_M", "Rtp": "sigma_Rtheta_plus", "Rtm": "sigma_Rtheta_minus", "mom": "sigma_mom",
           "ham": "sigma_ham", "hamM": "sigma_hamM"}
    for center in ("vertex", "cell", "none"):
        for kind in ("uni", "str"):
            if center == "none":
                grids = [make_grid(kind, N, r0=0.5, r1=6.0, center="none") for N in (64, 128, 256, 512)]
            else:
                grids = [make_grid(kind, N, L=6.0, center=center) for N in (64, 128, 256, 512)]

            def extra(res, g, r, v, d, ref, center=center):
                res.check(status(d, "constraints") == "VIOLATED", "%s: non-solution data not flagged VIOLATED" % g)
                if len(r) < 256:
                    return
                reg = regions(len(r), v["R"], center)["bulk"]
                L, hm = r[-1] - r[0], hmin_of(r)
                for lab, mf, key, Sfn, dd in fields:
                    if lab not in sig:
                        continue
                    err = np.abs(arr(d, mf) - v[key])
                    s = arr(d, sig[lab])
                    fl = floor(Sfn(v), L, hm, dd)
                    sel = reg & np.isfinite(err) & (err > 100 * fl)
                    if sel.sum() < 10:
                        res.metrics.setdefault("sigma", {})["%s/%s" % (g, lab)] = "at round-off (%d pts)" % sel.sum()
                        continue
                    frac = float(np.mean(err[sel] <= 3 * s[sel]))
                    ratio = float(np.max(err[reg & np.isfinite(err)]) / max(np.nanmax(s[reg]), 1e-300))
                    need = 0.8 if lab in ("ham", "hamM") else 0.9
                    res.check(frac >= need, "%s sigma(%s) bounds |err| at only %.2f of bulk points (< %.1f)" % (g, lab, frac, need))
                    if lab not in ("ham", "hamM"):
                        res.check(1 / 3 <= ratio <= 3, "%s ||err||/||sigma||(%s) = %.2f outside [1/3,3]" % (g, lab, ratio))
                    res.metrics.setdefault("sigma", {})["%s/%s" % (g, lab)] = dict(frac=frac, ratio=ratio, n=int(sel.sum()))
            sym_family(sd, res, "%s/%s" % (center, kind), sl, grids, center, fields, edge_fields=("ham", "hamM"),
                       classes_check=True, extra_checks=extra)


# ============================================================================ R: robustness

@case("R95", "robustness: NaN/inf/invalid input and overflow never give NO_TRAPPED_SPHERES")
def t_R95(sd, res):
    fd = FluxData()
    N = 100
    r0 = make_grid("uni", N, L=8.0, center="vertex")
    v0 = fd.values(r0)

    def attempt(label, r, v, matter_mod=None, expect_data=("INVALID_INPUT",), geometry_hard=True):
        try:
            phi, Pi = v["phi"].copy(), v["Pi"].copy()
            if matter_mod:
                phi, Pi = matter_mod(phi, Pi)
            slc = sd.SphericalSlice(r, v["A"], v["R"], v["KA"], v["KB"], center="vertex")
            d = sd.diagnose(slc, sd.ScalarField(phi, Pi, fd.m))
        except (ValueError, ArithmeticError) as e:
            res.note("%s: RAISED %s: %s (accepted, contract question K8)" % (label, type(e).__name__, e))
            res.metrics.setdefault("outcome", {})[label] = "RAISED " + type(e).__name__
            return
        ds, gs, cs = status(d, "data"), status(d, "geometry"), status(d, "constraints")
        res.metrics.setdefault("outcome", {})[label] = dict(data=ds, geometry=gs, constraints=cs)
        res.check(ds in expect_data, "%s: data status %s, expected %s" % (label, ds, expect_data))
        res.check(ds != "OK", "%s: data status OK for invalid input" % label)
        if geometry_hard:
            res.check(gs == "UNDETERMINED", "%s: geometry %s, expected UNDETERMINED" % (label, gs))
            res.check(gs != "NO_TRAPPED_SPHERES", "%s: failure reported as NO_TRAPPED_SPHERES" % label)
        else:
            res.check(cs != "CONSISTENT_WITH_TRUNCATION", "%s: constraints %s with NaN matter" % (label, cs))

    def mod(key, fn):
        v = {k: (np.array(x, float, copy=True) if isinstance(x, np.ndarray) else x) for k, x in v0.items()}
        v[key] = fn(v[key])
        return v

    def set_at(i, val):
        def f(a):
            a[i] = val
            return a
        return f
    try:
        slc = sd.SphericalSlice(r0, v0["A"], v0["R"], v0["KA"], v0["KB"], center="vertex")
        dc = sd.diagnose(slc, sd.ScalarField(v0["phi"], v0["Pi"], fd.m))
        res.check(status(dc, "data") == "OK", "control: clean data gives data status %s" % status(dc, "data"))
    except Exception as e:  # noqa: BLE001
        res.check(False, "control: clean data raised %s" % e)
    attempt("a:NaN_in_A", r0, mod("A", set_at(50, np.nan)))
    attempt("b:inf_in_KB", r0, mod("KB", set_at(50, np.inf)))
    attempt("c:NaN_in_Pi", r0, v0, matter_mod=lambda p, q: (p, set_at(50, np.nan)(q)), geometry_hard=False)
    attempt("d1:A_negative", r0, mod("A", set_at(50, -0.1)))
    attempt("d2:A_zero", r0, mod("A", set_at(50, 0.0)))
    attempt("e:R_negative", r0, mod("R", set_at(50, -1.0)))
    rr = r0.copy()
    rr[50], rr[51] = rr[51], rr[50]
    attempt("f:r_not_monotone", rr, v0)
    r6 = np.linspace(0, 1, 6)
    attempt("g:N_lt_8", r6, {k: (x[:6] if isinstance(x, np.ndarray) else x) for k, x in fd.values(r6).items()})
    vh = mod("A", lambda a: (a.__setitem__(slice(40, 43), 1e-200), a)[1])
    attempt("h:overflow", r0, vh, expect_data=("NUMERICAL_FAILURE",))


# ============================================================================ N: norms

@case("N99", "reported norms: L_inf per region equals recomputation from the module's own arrays")
def t_N99(sd, res):
    fd = FluxData()
    for N in (200, 400):
        r = make_grid("uni", N, L=8.0, center="vertex")
        v = fd.values(r)
        d, _ = build(sd, r, v, "vertex", sd.ScalarField(v["phi"], v["Pi"], fd.m))
        nm = D(d, "norms")
        reg = regions(N, v["R"], "vertex")
        for name in ("ham", "mom", "hamM"):
            x = arr(d, name)
            ok = np.ones(N, bool) if name == "hamM" else v["R"] > 0
            for rg, mask in reg.items():
                m = mask & ok
                mx = float(np.max(np.abs(x[m])))
                rep = float(nm[name][rg]["Linf"])
                res.check(abs(rep - mx) <= 10 * EPS * mx + 1e-300, "N%d norms[%s][%s][Linf] = %.6e, recomputed %.6e" % (N, name, rg, rep, mx))
            A = v["A"]
            l2n = l2vol(np.where(ok, x, np.nan), r, v["R"], A, reg["all"] & ok, normalized=False)
            l2u = l2vol(np.where(ok, x, np.nan), r, v["R"], A, reg["all"] & ok, normalized=True)
            rep = float(nm[name]["all"]["L2vol"])
            which = "unnormalised" if abs(rep - l2n) <= 1e-6 * l2n else ("volume-normalised" if abs(rep - l2u) <= 1e-6 * l2u else "neither")
            res.metrics.setdefault("L2vol_convention", {})["N%d/%s" % (N, name)] = dict(reported=rep, unnorm=l2n, norm=l2u, match=which)
            if which == "neither":
                res.note("N%d %s: L2vol matches neither convention (contract question K7)" % (N, name))


def run_all(ids=None, verbose=True):
    results = []
    for tid, (fn, title) in CASES.items():
        if ids and tid not in ids:
            continue
        r = run_case(fn, tid, title)
        results.append(r)
        if verbose:
            print("%-4s %-5s %s%s" % (tid, r.status, title, ("  [%d fails]" % len(r.fails)) if r.fails else ""), flush=True)
    return results


__all__ = ["CASES", "run_all", "Result"]

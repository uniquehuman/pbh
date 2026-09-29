"""Shared helpers for the independent verification matrix (PBH-VERIFY-01).

The implementation under test is imported from SPHDIAG_PATH (default <pilot>/src).  Expected
values come ONLY from independent_reference/ or closed forms; nothing here calls sphdiag to
produce an expected value.
"""
import importlib
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from independent_reference import fd_ref  # noqa: E402
from independent_reference.norms import norms as _norms, observed_order  # noqa: E402
from independent_reference.grids import regions  # noqa: E402

EPS = float(np.finfo(float).eps)
SPHDIAG_PATH = os.path.abspath(os.environ.get("SPHDIAG_PATH", os.path.join(ROOT, "src")))

_SD = None
_SD_ERR = None


def load_sphdiag():
    """Import sphdiag from SPHDIAG_PATH; returns (module or None, error message or None)."""
    global _SD, _SD_ERR
    if _SD is not None or _SD_ERR is not None:
        return _SD, _SD_ERR
    if SPHDIAG_PATH not in sys.path:
        sys.path.insert(0, SPHDIAG_PATH)
    try:
        _SD = importlib.import_module("sphdiag")
        missing = [n for n in ("SphericalSlice", "ScalarField", "VlasovMoments", "Projections",
                               "pi_from_dtphi", "diagnose") if not hasattr(_SD, n)]
        if missing:
            _SD_ERR = "sphdiag at %s lacks API names %s" % (SPHDIAG_PATH, missing)
            _SD = None
    except Exception as e:  # noqa: BLE001
        _SD_ERR = "sphdiag not importable from SPHDIAG_PATH=%s: %s: %s" % (SPHDIAG_PATH, type(e).__name__, e)
        _SD = None
    return _SD, _SD_ERR


# ------------------------------------------------------------------ result object

class Result:
    def __init__(self, tid, title):
        self.id = tid
        self.title = title
        self.status = "PASS"
        self.metrics = {}
        self.fails = []
        self.notes = []

    def check(self, cond, msg):
        if not bool(cond):
            self.fails.append(msg)
            if self.status == "PASS":
                self.status = "FAIL"
        return bool(cond)

    def note(self, msg):
        self.notes.append(msg)

    def metric(self, key, val):
        self.metrics[key] = val

    def to_dict(self):
        return dict(id=self.id, title=self.title, status=self.status, fails=self.fails[:50],
                    n_fails=len(self.fails), notes=self.notes[:50], metrics=_jsonable(self.metrics))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else str(v)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def run_case(fn, tid, title):
    sd, err = load_sphdiag()
    res = Result(tid, title)
    if sd is None:
        res.status = "SKIP"
        res.notes.append(err)
        return res
    try:
        fn(sd, res)
    except Exception as e:  # noqa: BLE001
        res.status = "ERROR"
        res.fails.append("%s: %s" % (type(e).__name__, e))
        res.notes.append(traceback.format_exc()[-3000:])
    return res


# ------------------------------------------------------------------ access to module outputs

def D(d, name):
    if isinstance(d, dict):
        return d[name]
    return getattr(d, name)


def has(d, name):
    if isinstance(d, dict):
        return name in d
    return hasattr(d, name)


def arr(d, name):
    return np.asarray(D(d, name), dtype=float)


def classes(d):
    return np.asarray([str(c).split(".")[-1] for c in D(d, "sphere_class")])


def status(d, key):
    s = D(d, "status")
    v = s[key] if isinstance(s, dict) else getattr(s, key)
    return str(v).split(".")[-1]


def sg(v):
    """normalise a sign field to -1/0/+1 (None if missing)."""
    if v is None:
        return None
    if isinstance(v, str):
        t = v.strip().lower().replace("−", "-")
        if t in ("+", "+1", "1", "plus", "pos", "positive", ">0"):
            return 1
        if t in ("-", "-1", "minus", "neg", "negative", "<0"):
            return -1
        if t in ("0", "zero", "+0", "-0", "=0"):
            return 0
        raise ValueError("unrecognised sign %r" % v)
    return int(np.sign(float(v)))


def rget(root, key):
    return root[key] if isinstance(root, dict) else getattr(root, key)


def roots_of(d):
    out = []
    for q in (D(d, "roots") or []):
        w = str(rget(q, "which")).replace("−", "-")
        out.append(dict(which="+" if w.startswith("+") or w in ("plus", "p") else "-",
                        r=float(rget(q, "r")), R=float(rget(q, "R")), R_lo=float(rget(q, "R_lo")),
                        R_hi=float(rget(q, "R_hi")), M=float(rget(q, "M")), type=str(rget(q, "type")).split(".")[-1],
                        ah=bool(rget(q, "ah_candidate")), other=sg(rget(q, "other_sign")),
                        inner=sg(rget(q, "inner_sign")), outer=sg(rget(q, "outer_sign"))))
    return out


def build(sd, r, v, center, matter=None, derivs=False, **kw):
    extra = dict(dA=v["dA"], dR=v["dR"], dKB=v["dKB"]) if derivs else {}
    extra.update(kw)
    slc = sd.SphericalSlice(np.asarray(r, float), np.asarray(v["A"], float), np.asarray(v["R"], float),
                            np.asarray(v["KA"], float), np.asarray(v["KB"], float), center=center, **extra)
    return sd.diagnose(slc, matter), slc


def proj(sd, v, fp=None):
    jr = v["jr"] if fp is None else v["jr"] * fp
    return sd.Projections(np.asarray(v["rho"], float), np.asarray(jr, float), np.asarray(v["Srr"], float),
                          np.asarray(v["Sthth"], float))


def zero_proj(sd, n):
    z = np.zeros(n)
    return sd.Projections(z, z.copy(), z.copy(), z.copy())


# ------------------------------------------------------------------ tolerances

def floor(S, L, hmin, d):
    """round-off floor (verification_plan.md 1.3, v2): 100 eps S (L/hmin)^d * c_d, c_0 = c_1 = 1, c_2 = 12.
    c_2: each nested first derivative multiplies round-off by up to sum|w| h = 10.67 (one-sided
    5-point stencil); v1 implicitly used ~1 for the second factor (v1: c_2 = 1)."""
    c = 12.0 if d >= 2 else 1.0
    return 100.0 * EPS * float(S) * (float(L) / float(hmin))**d * c


def hmin_of(r):
    return float(np.min(np.diff(r)))


def hmax_of(r):
    return float(np.max(np.diff(r)))


def ref_pipeline(r, v, center, with_matter=True):
    return fd_ref.pipeline(r, v["A"], v["R"], v["KA"], v["KB"], center,
                           v["rho"] if with_matter else None, v["jr"] if with_matter else None)


def compare(res, label, mod, exact, ref, r, R, A, center, S, d, regions_req=("center", "bulk", "outer"),
            extra=0.0, factor=10.0, hard=True):
    """Compare module array with exact; tolerance per region/kind = factor*||ref-exact|| + floor (+extra).
    Returns module error norms."""
    mod = np.asarray(mod, float)
    exact = np.asarray(exact, float)
    e_mod = mod - exact
    if isinstance(ref, (list, tuple)):
        # v2: envelope over admissible formulations (pointwise max |ref_k - exact|)
        e_ref = np.nanmax(np.abs(np.array([np.asarray(q, float) - exact for q in ref])), axis=0)
    else:
        e_ref = np.asarray(ref, float) - exact
    fl = floor(S, r[-1] - r[0], hmin_of(r), d) + extra
    nm = _norms(e_mod, r, R, A, center)
    nr = _norms(e_ref, r, R, A, center)
    nf = _norms(np.full_like(mod, fl), r, R, A, center)
    # NaN where exact finite -> failure (masking or spurious NaN)
    bad = np.isfinite(exact) & ~np.isfinite(mod)
    if hard:
        res.check(not np.any(bad), "%s: module NaN/inf where exact is finite at %d points (first r=%s)"
                  % (label, int(bad.sum()), r[np.argmax(bad)] if bad.any() else None))
    out = {}
    for reg in regions_req + ("all",):
        out[reg] = {}
        for kind in ("Linf", "L2vol"):
            em, er, ff = nm[reg][kind], nr[reg][kind], nf[reg][kind]
            tol = factor * (er if np.isfinite(er) else 0.0) + (ff if np.isfinite(ff) else fl)
            out[reg][kind] = dict(err=em, tol=tol, ref=er, floor=ff)
            if hard and reg in regions_req and np.isfinite(em):
                res.check(em <= tol, "%s[%s,%s]: err %.3e > tol %.3e (ref %.3e, floor %.3e)"
                          % (label, reg, kind, em, tol, er, ff))
    res.metrics.setdefault("fields", {})[label] = out
    return out


def convergence(res, label, hs, errs, pmin, regions_req=("bulk",), kinds=("Linf", "L2vol")):
    """errs: list of compare() outputs for successively refined grids with max spacings hs."""
    out = {}
    for reg in regions_req:
        for kind in kinds:
            ec = errs[-2][reg][kind]["err"]
            ef = errs[-1][reg][kind]["err"]
            fl = errs[-1][reg][kind]["floor"]
            p = observed_order(ec, ef, hs[-2], hs[-1])
            allp = [observed_order(errs[i][reg][kind]["err"], errs[i + 1][reg][kind]["err"], hs[i], hs[i + 1])
                    for i in range(len(hs) - 1)]
            out["%s/%s" % (reg, kind)] = dict(p=p, p_all=allp, e=[e[reg][kind]["err"] for e in errs])
            if ef > 100 * fl:
                res.check(np.isfinite(p) and p >= pmin, "%s order[%s,%s] = %.2f < %.1f (errors %s)"
                          % (label, reg, kind, p, pmin, ["%.2e" % e[reg][kind]["err"] for e in errs]))
            else:
                out["%s/%s" % (reg, kind)]["at_roundoff_floor"] = True
    res.metrics.setdefault("orders", {})[label] = out
    return out


# ------------------------------------------------------------------ classification rules

CLASS_PAIRS = {
    "NORMAL": [(1, -1)], "NORMAL_REVERSED": [(-1, 1)], "FUTURE_TRAPPED": [(-1, -1)], "PAST_TRAPPED": [(1, 1)],
    "FUTURE_MARGINAL": [(0, -1), (-1, 0)], "PAST_MARGINAL": [(0, 1), (1, 0)], "DOUBLY_MARGINAL": [(0, 0)],
}


def _compatible(s, ex, delta):
    se = int(np.sign(ex))
    if s == se:
        return True
    return s == 0 and abs(ex) <= delta


def check_classes(res, label, cls, eRtp, eRtm, R, center, delta_p, delta_m):
    """verification_plan.md 1.6: never a resolved wrong sign; '0' only inside delta."""
    nbad = 0
    first = []
    for i, c in enumerate(cls):
        if center == "vertex" and R[i] == 0:
            ok = c == "CENTER"
        elif c == "UNRESOLVED":
            # v2 (contract 1.2): admissible only where my own reference error envelope is within a factor 10 of
            # the UNRESOLVED threshold tau_max = 0.1 (delta = 3*(10 E_ref + floor) >= 0.01); UNRESOLVED on
            # well-resolved data means sigma is overestimated by > 10x (information hidden)
            dpi = delta_p[i] if np.ndim(delta_p) else delta_p
            dmi = delta_m[i] if np.ndim(delta_m) else delta_m
            ok = max(dpi, dmi) >= 0.01
        elif c not in CLASS_PAIRS:
            ok = False
        else:
            dpi = delta_p[i] if np.ndim(delta_p) else delta_p
            dmi = delta_m[i] if np.ndim(delta_m) else delta_m
            ok = any(_compatible(a, eRtp[i], dpi) and _compatible(b, eRtm[i], dmi)
                     for a, b in CLASS_PAIRS[c])
        if not ok:
            nbad += 1
            if len(first) < 5:
                first.append("i=%d R=%.6g class=%s exact(Rt+,Rt-)=(%.3e,%.3e)" % (i, R[i], c, eRtp[i], eRtm[i]))
    res.check(nbad == 0, "%s: %d points with inadmissible class: %s" % (label, nbad, "; ".join(first)))
    return nbad


def exact_root_tol(sl, which, r_star, r, ref_err, center_scale, R_fn=None):
    """Absolute tolerance for a root position in R (verification_plan.md S20):
       linear-interpolation bound (h^2/8)|f''|/|f'| + 10 E_ref/|f'|, mapped to R, + R-interpolation
       + round-off; and the slack for the [R_lo, R_hi] bracket: cubic bound (h^4/24)|f''''|/|f'| etc."""
    i = int(np.searchsorted(r, r_star)) - 1
    i = min(max(i, 0), len(r) - 2)
    h = r[i + 1] - r[i]
    loc = np.linspace(max(r[0], r[i] - h), min(r[-1], r[i + 1] + h), 41)
    f1 = abs(float(sl.rtheta_fn(which, 1)(np.array([r_star]))[0]))
    f2 = float(np.max(np.abs(sl.rtheta_fn(which, 2)(loc))))
    f4 = float(np.max(np.abs(sl.rtheta_fn(which, 4)(loc))))
    Rf = sl.R_fn()
    Rp = abs(float(sl.R_fn(1)(np.array([r_star]))[0]))
    Rpp = float(np.max(np.abs(sl.R_fn(2)(loc))))
    R_star = float(Rf(np.array([r_star]))[0])
    fd_part = 10 * ref_err / f1
    tol = (h**2 / 8 * f2 / f1 + fd_part) * Rp + h**2 / 8 * Rpp + 100 * EPS * center_scale * (r[-1] - r[0]) / hmin_of(r)
    slack = (h**4 / 24 * f4 / f1 + fd_part) * Rp + h**2 / 8 * Rpp + 100 * EPS * center_scale * (r[-1] - r[0]) / hmin_of(r)
    return R_star, tol, slack


def check_root(res, label, q, R_star, tol, slack, exp):
    """exp: dict with which, type, ah, other, inner, outer (None = don't care)."""
    res.check(abs(q["R"] - R_star) <= tol, "%s: |R*-R_exact| = %.3e > tol %.3e (R*=%.12g, exact %.12g)"
              % (label, abs(q["R"] - R_star), tol, q["R"], R_star))
    res.check(q["R_lo"] - slack <= R_star <= q["R_hi"] + slack,
              "%s: exact root %.12g outside reported [R_lo,R_hi]=[%.12g,%.12g] (slack %.2e)"
              % (label, R_star, q["R_lo"], q["R_hi"], slack))
    res.check(q["R_lo"] <= q["R"] <= q["R_hi"], "%s: R* not inside its own interval" % label)
    for k in ("which", "type", "ah", "other", "inner", "outer"):
        if exp.get(k) is not None:
            res.check(q[k] == exp[k], "%s: root field %s = %r, expected %r" % (label, k, q[k], exp[k]))
    res.metrics.setdefault("roots", {})[label] = dict(R=q["R"], R_exact=R_star, err=abs(q["R"] - R_star), tol=tol,
                                                        R_lo=q["R_lo"], R_hi=q["R_hi"], type=q["type"], ah=q["ah"])


def warnings_of(d):
    """status['warnings'] normalised to a list of lower-case strings (list, dict or str accepted)."""
    st = D(d, "status")
    w = st.get("warnings") if isinstance(st, dict) else getattr(st, "warnings", None)
    if w is None:
        return None
    if isinstance(w, dict):
        return [str(k).lower() + ":" + str(v).lower() for k, v in w.items() if v not in (False, None, 0, "", [])]
    if isinstance(w, str):
        return [w.lower()]
    return [str(x).lower() for x in w]


def marginal_present(d):
    st = D(d, "status")
    return st.get("marginal_present") if isinstance(st, dict) else getattr(st, "marginal_present", None)


def ham_ref(ref):
    """reference for 'ham' tolerances: the envelope over formulations (v2)."""
    return ref.get("ham_variants", ref["ham"])

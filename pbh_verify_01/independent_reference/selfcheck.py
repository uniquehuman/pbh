#!/usr/bin/env python3
"""Self-validation of the independent references (verification_plan.md section 5, RS1-RS6).
Does not import sphdiag.  Writes results/reference_selfcheck.json.

Run from the pilot root:  python3 -m independent_reference.selfcheck
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from independent_reference import analytic as AN  # noqa: E402
from independent_reference import derive_sympy, fd_ref  # noqa: E402
from independent_reference.fkg import FKG  # noqa: E402
from independent_reference.flux_data import FluxData  # noqa: E402
from independent_reference.grids import make_grid, regions  # noqa: E402
from independent_reference.norms import observed_order  # noqa: E402


def rs3():
    out = {}
    a = FKG(formulation="Hdot", t_end=60.0)
    b = FKG(formulation="constraint", t_end=60.0)
    ts = np.linspace(0, 60, 601)
    sa, sb = a.state(ts), b.state(ts)
    amp = np.max(np.abs(sa["phi"]))
    out["two_formulations_max_dphi_over_amp"] = float(np.max(np.abs(sa["phi"] - sb["phi"])) / amp)
    out["two_formulations_max_dH_over_H"] = float(np.max(np.abs(sa["H"] - sb["H"]) / sa["H"]))
    out["max_Fres_over_H2"] = float(np.max(np.abs(sa["F_res"]) / sa["H"]**2))
    out["ok_formulations"] = out["two_formulations_max_dphi_over_amp"] < 1e-9 and out["two_formulations_max_dH_over_H"] < 1e-9
    out["ok_Fres"] = out["max_Fres_over_H2"] < 1e-11
    avg = {}
    for q in (8.0, 20.0, 45.0, 100.0):
        if q > a.m / a.state(59.0)["H"]:
            fk = FKG(formulation="Hdot", t_end=200.0)
        else:
            fk = a
        t0 = brentq(lambda t: fk.m / fk.state(t)["H"] - q, 0.01, fk.sol.t[-1] - 30)
        r_, p_, _ = fk.period_average(t0, 256)
        avg[q] = p_ / r_
    out["avg_p_over_rho"] = avg
    qs = sorted(avg)
    out["avg_p_scaling_exponent_last_pair"] = float(np.log(abs(avg[qs[-2]] / avg[qs[-1]])) / np.log(qs[-1] / qs[-2]))
    # bound: residual leakage of the 4-period Hann window from the amplitude decay (d ln Phi^2/dt = -3H)
    # is ~ 3 H T_w / (64 pi) = 0.12 H/m; the physical <p>/<rho> is O((H/m)^2).  Criterion 0.15 H/m.
    out["avg_bound_ratio"] = {q: abs(avg[q]) / (0.15 / q) for q in qs}
    out["ok_avg_decreasing"] = all(abs(avg[q]) <= 0.15 / q for q in qs)
    loose = FKG(rtol=1e-6, atol=1e-9, t_end=40.0)
    out["loose_Fres_over_H2_at_t20_30_35"] = [float(loose.state(t)["F_res"] / loose.state(t)["H"]**2) for t in (20., 30., 35.)]
    out["ok"] = out["ok_formulations"] and out["ok_Fres"] and out["ok_avg_decreasing"]
    return out


def rs4():
    out = {}
    r = np.linspace(0, 8, 1601)
    for lab, kw in (("base", {}), ("flipPi", dict(sPi=-1.0)), ("flipKB", dict(sKB=-1.0))):
        fd = FluxData(**kw)
        fd2 = FluxData(rtol=1e-11, atol=1e-16, **kw)
        e = fd.exact(r)
        dM = float(np.max(np.abs(fd2.values(r)["M"] - e["M"])))
        def rel(x, t):  # pointwise |X|/sum|T_k| where the terms are not negligible (> 1e-4 of their max; below that the ratio is round-off of ~1e-20 numbers)
            t = np.nan_to_num(t)
            m = t > 1e-4 * np.max(t)
            return float(np.nanmax(np.abs(x[m]) / t[m]))
        hr, mr, Hr = rel(e["ham"], e["ham_absterms"]), rel(e["mom"], e["mom_absterms"]), rel(e["hamM"], e["hamM_absterms"])
        out[lab] = dict(M_rtol_diff=dM, ham_rel_max=hr, mom_rel_max=mr, hamM_rel_max=Hr,
                        min_Ainv2=float(np.min(1 / e["A"]**2)), max_2M_over_R=float(np.max(2 * e["M"][1:] / r[1:])),
                        M_rmax=float(e["M"][-1]), Mflux_rmax=float(e["M_flux"][-1]),
                        flux_fraction=float(e["M_flux"][-1] / e["M"][-1]),
                        ok=dM < 1e-11 and hr < 1e-9 and mr < 1e-12 and Hr < 1e-9 and abs(e["M_flux"][-1] / e["M"][-1]) >= 0.05)
    # v2 variants: FLRW background (U62), central bump (U62), kink (U61; residuals checked away from the kink)
    rr = np.linspace(0, 6, 2401)
    for lab, kw, rmax, mask_fn in (("v2_background", dict(background=(0.03, 0.02), r_max=6.0), 6.0, None),
                                   ("v2_bump", dict(background=(0.03, 0.02), r_max=6.0, bump=(0.1, 0.0, 0.04)), 6.0, None),
                                   ("v2_kink", dict(kink=(1.0, 0.25)), 8.0, lambda q: np.abs(q - 0.25) > 2e-3)):
        q = rr if rmax == 6.0 else r
        fd = FluxData(**kw)
        fd2 = FluxData(rtol=1e-11, atol=1e-16, **kw)
        e = fd.exact(q)
        m = np.ones(q.size, bool) if mask_fn is None else mask_fn(q)

        def rel2(x, t):
            t = np.nan_to_num(t)
            mm = m & (t > 1e-4 * np.max(t))
            return float(np.nanmax(np.abs(x[mm]) / t[mm]))
        dM = float(np.max(np.abs(fd2.values(q)["M"] - e["M"])))
        hr, mr = rel2(e["ham"], e["ham_absterms"]), rel2(e["mom"], e["mom_absterms"])
        out[lab] = dict(M_rtol_diff=dM, ham_rel_max=hr, mom_rel_max=mr, min_Ainv2=float(np.min(1 / e["A"]**2)),
                        ok=dM < 1e-10 and hr < 1e-8 and mr < 1e-11)
        if lab == "v2_background":
            # exact FLRW limit: with the perturbation removed the data must be A = 1, M = (4 pi/3) rho_b r^3
            fb = FluxData(phi0=0.0, Pi0=0.0, kb0=0.0, **kw)
            eb = fb.exact(q)
            rho_b = 0.5 * 0.02**2 + 0.5 * 0.03**2
            out[lab]["pure_background_A_minus_1"] = float(np.max(np.abs(eb["A"] - 1)))
            out[lab]["pure_background_M_rel_err"] = float(np.max(np.abs(eb["M"][1:] - 4 * np.pi / 3 * rho_b * q[1:]**3)
                                                             / (4 * np.pi / 3 * rho_b * q[1:]**3)))
            out[lab]["ok"] &= out[lab]["pure_background_A_minus_1"] < 1e-12 and out[lab]["pure_background_M_rel_err"] < 1e-9
    b = FluxData().exact(r)
    f = FluxData(sPi=-1.0).exact(r)
    out["antisymmetry_Mflux"] = float(np.max(np.abs(b["M_flux"] + f["M_flux"])))
    out["ok"] = all(out[k]["ok"] for k in ("base", "flipPi", "flipKB", "v2_background", "v2_bump", "v2_kink")) \
        and out["antisymmetry_Mflux"] < 1e-13
    return out


def rs5():
    out = {}
    sl = AN.manufactured()
    for center, kw in (("vertex", dict(L=6.0)), ("cell", dict(L=6.0)), ("none", dict(r0=0.5, r1=6.0))):
        for kind in ("uni", "str"):
            errs, hs = {}, []
            for N in (64, 128, 256, 512):
                rr = make_grid(kind, N, center=center, **kw)
                v = sl.exact(rr)
                p = fd_ref.pipeline(rr, v["A"], v["R"], v["KA"], v["KB"], center, v["rho"], v["jr"])
                reg = regions(N, v["R"], center)["bulk"]
                for k in ("dR", "M", "Rtp", "ham", "mom", "hamM"):
                    errs.setdefault(k, []).append(float(np.nanmax(np.abs(p[k] - v[k])[reg])))
                hs.append(float(np.max(np.diff(rr))))
            orders = {k: observed_order(e[-2], e[-1], hs[-2], hs[-1]) for k, e in errs.items()}
            out["%s/%s" % (center, kind)] = dict(orders=orders, errs=errs,
                                                 ok=all(3.5 <= o <= 4.6 for o in orders.values()))
    ps = AN.polynomial_slice()
    worst = 0.0
    for center in ("vertex", "cell", "none"):
        for kind in ("uni", "str"):
            rr = make_grid(kind, 41, L=3.0, r0=0.5, r1=3.0, center=center)
            v = ps.exact(rr)
            p = fd_ref.pipeline(rr, v["A"], v["R"], v["KA"], v["KB"], center, v["rho"], v["jr"])
            for k in ("dR", "M", "Rtp", "mom"):
                worst = max(worst, float(np.nanmax(np.abs(p[k] - v[k]))))
    out["polynomial_exactness_worst"] = worst
    out["ok"] = all(v["ok"] for k, v in out.items() if isinstance(v, dict)) and worst < 1e-12
    return out


def rs6():
    out = {}
    cases = [(AN.schwarzschild_pg(), "+", 0.5, 12, 2.0), (AN.schwarzschild_ks(), "+", 0.5, 12, 2.0),
             (AN.flrw(1.3, 0.4), "-", 1e-6, 6, 1 / (1.3 * 0.4)), (AN.flrw(1.3, -0.4), "+", 1e-6, 6, 1 / (1.3 * 0.4))]
    ok = True
    for sl, w, a, b, rexp in cases:
        rts = sl.roots(w, a, b)
        e = abs(rts[0] - rexp) if len(rts) == 1 else np.inf
        out["%s/%s" % (sl.name, w)] = dict(roots=rts, err=e)
        ok &= e < 1e-13
    iso = AN.schwarzschild_iso()
    rp, rm = iso.roots("+", 0.1, 10), iso.roots("-", 0.1, 10)
    out["iso_throat"] = dict(plus=rp, minus=rm)
    ok &= len(rp) == 1 and len(rm) == 1 and abs(rp[0] - 0.5) < 1e-12
    for shells, lab in ((((1.0, 3.0),), "one_shell"), (((1.0, 3.0), (0.5, 8.0)), "two_shells")):
        s = AN.synthetic_pbh_flrw(shells=shells)
        out["synthetic/" + lab] = dict(plus=s.roots("+", 1e-6, 30), minus=s.roots("-", 1e-6, 30))
    out["synthetic_ok"] = (len(out["synthetic/one_shell"]["plus"]) == 2 and len(out["synthetic/one_shell"]["minus"]) == 1
                           and len(out["synthetic/two_shells"]["plus"]) == 4 and len(out["synthetic/two_shells"]["minus"]) == 1)
    out["ok"] = bool(ok and out["synthetic_ok"])
    return out


def main():
    t0 = time.time()
    res = {}
    sym = derive_sympy.run_all(verbose=False)
    res["RS1_RS2_sympy"] = dict(checks=sym, ok=all(sym.values()))
    res["RS3_FKG"] = rs3()
    res["RS4_flux_data"] = rs4()
    res["RS5_fd_ref"] = rs5()
    res["RS6_roots"] = rs6()
    res["runtime_s"] = time.time() - t0
    res["all_ok"] = all(v["ok"] for k, v in res.items() if isinstance(v, dict))
    for k, v in res.items():
        if isinstance(v, dict):
            print("%-16s %s" % (k, "OK" if v["ok"] else "FAIL"))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, float) and not np.isfinite(o):
            return str(o)
        return o
    with open(os.path.join(ROOT, "results", "reference_selfcheck.json"), "w") as fh:
        json.dump(conv(res), fh, indent=1)
    print("all_ok =", res["all_ok"], " (%.0f s)" % res["runtime_s"])
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

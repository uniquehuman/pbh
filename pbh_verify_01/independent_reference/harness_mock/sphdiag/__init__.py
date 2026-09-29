"""HARNESS SELF-TEST MOCK of the sphdiag API, contract v1.1 (verification role's own code).

NOT the implementation under test and NOT a reference for expected values.  Its only purpose is
to check that the independent test matrix runs end to end and that it FAILS on deliberately
injected bugs (mutation testing, verification_plan.md section 5, HS).  Written from the contract
text and the verification role's own derivations; shares nothing with src/ or arch_A/.

Mutations (env SPHDIAG_MOCK_MUTATION, comma separated):
  v1 set:  F10sign jr_sign jr_noA swap_theta ah_nosides nan_ok naive_mass sigma_zero order2
  v2 set:  ah_plus_only   (v1.0 rule: candidate only for a theta_+ zero -> misses Gamma < 0 horizons)
           past_marg      (v1.0 rule: unresolved (+,0) -> PAST_TRAPPED_ONLY)
           no_noise       (sigma_noise = 0 in tau)
           no_unresolved  (UNRESOLVED never reported)
           R0_ok          (R = 0 away from a vertex centre accepted)
           trap_quad      (trapezoid quadrature instead of 4th order)
           no_warnings    (status['warnings'] always empty)
           l2_normalised  (L2vol divided by the volume)
  v1.2:    no_unres_class (tau > 0.1 not turned into class UNRESOLVED / no status cap)
"""
import os
import types

import numpy as np

from independent_reference import fd_ref

_MUT = set(filter(None, os.environ.get("SPHDIAG_MOCK_MUTATION", "").split(",")))
EPS = np.finfo(float).eps
SQ924 = np.sqrt(924.0)


def pi_from_dtphi(dtphi, dphi_dr, alpha, beta):
    return (np.asarray(dtphi, float) - np.asarray(beta, float) * np.asarray(dphi_dr, float)) / np.asarray(alpha, float)


class Projections:
    def __init__(self, rho, j_r, S_rr, S_thth):
        self.rho, self.j_r, self.S_rr, self.S_thth = (np.asarray(x, float) for x in (rho, j_r, S_rr, S_thth))

    def projections(self, slc):
        return self


class SphericalSlice:
    def __init__(self, r, A, R, KA, KB, center="vertex", alpha=None, beta=None, dA=None, dR=None, dKB=None):
        self.r, self.A, self.R, self.KA, self.KB = (np.array(x, float) for x in (r, A, R, KA, KB))
        self.center = center
        self.dA, self.dR, self.dKB = dA, dR, dKB


def _order():
    return 2 if "order2" in _MUT else 4


def _D(f, slc, parity, order=None):
    return fd_ref.deriv(f, slc.r, slc.center, parity, order or _order())


def _sig(f, slc, parity):
    """sigma(f') = window-max |D4 - D6| + 10 eps sum|w f| (contract 1.1 section 9; window: proposal K11)."""
    if "sigma_zero" in _MUT:
        return np.zeros(len(f))
    d4 = _D(f, slc, parity)
    d6 = fd_ref.deriv(f, slc.r, slc.center, parity, 6)
    return _win(np.abs(d4 - d6)) + 10 * EPS * fd_ref.deriv_absw(f, slc.r, slc.center, parity)


class ScalarField:
    def __init__(self, phi, Pi, m_phi):
        self.phi, self.Pi, self.m = np.asarray(phi, float), np.asarray(Pi, float), float(m_phi)

    def projections(self, slc):
        dphi = _D(self.phi, slc, 1)
        Phi = dphi / slc.A
        V = 0.5 * self.m**2 * self.phi**2
        jr = (1 if "jr_sign" in _MUT else -1) * self.Pi * dphi
        P = Projections(0.5 * self.Pi**2 + 0.5 * Phi**2 + V, jr, 0.5 * self.Pi**2 + 0.5 * Phi**2 - V,
                        0.5 * self.Pi**2 - 0.5 * Phi**2 - V)
        sd = _sig(self.phi, slc, 1)
        P.sig_rho = np.abs(Phi) * sd / slc.A
        P.sig_jr = np.abs(self.Pi) * sd
        return P


class VlasovMoments:
    def __init__(self, E, J, p, q):
        self.E, self.J, self.p, self.q = (np.asarray(x, float) for x in (E, J, p, q))

    def projections(self, slc):
        return Projections(self.E, self.J if "jr_noA" in _MUT else slc.A * self.J, self.p, self.q)


def _win(a, w=2):
    a = np.abs(np.nan_to_num(a))
    o = a.copy()
    for k in range(1, w + 1):
        o[k:] = np.maximum(o[k:], a[:-k])
        o[:-k] = np.maximum(o[:-k], a[k:])
    return o


def _noise(f):
    """sigma_noise: max over a 7-point window of |Delta^6 f| / sqrt(924)."""
    n = len(f)
    if "no_noise" in _MUT or n < 7:
        return np.zeros(n)
    d6 = np.abs(np.diff(np.nan_to_num(f), 6)) / SQ924      # centred at k+3
    c = np.zeros(n)
    c[3:n - 3] = d6
    c[:3] = d6[0]
    c[n - 3:] = d6[-1]
    return _win(c, 3)


def _invalid(n, why, data="INVALID_INPUT"):
    nan = np.full(n, np.nan)
    z = {k: nan.copy() for k in ("Gamma", "U", "dR", "M", "sigma_M", "compactness", "M_over_R3", "Rtheta_plus",
                                  "Rtheta_minus", "sigma_Rtheta_plus", "sigma_Rtheta_minus", "theta_plus", "theta_minus",
                                  "ham", "ham_rel", "mom", "mom_rel", "hamM", "hamM_rel", "sigma_ham", "sigma_mom",
                                  "sigma_hamM", "M_rho", "M_flux", "M_int", "M_prop")}
    z.update(M_over_R3_extrapolated=np.zeros(n, bool), sphere_class=np.array(["INVALID"] * n), roots=[],
             outer_ah_candidate=None, projections=None, norms={}, center_checks={"Gamma0": np.nan, "KA_minus_KB0": np.nan},
             status={"data": data, "constraints": "NOT_EVALUATED", "geometry": "UNDETERMINED", "marginal_present": False,
                     "warnings": [why]})
    return types.SimpleNamespace(**z)


def _regions(n, R, center):
    pos = np.where(R > 0)[0] if center == "vertex" else np.arange(n)
    nc = max(4, n // 20)
    c = np.zeros(n, bool)
    c[pos[:nc]] = True
    o = np.zeros(n, bool)
    o[-4:] = True
    b = ~(c | o)
    b[R <= 0] = False
    return {"all": np.ones(n, bool), ("inner" if center == "none" else "center"): c, "bulk": b, "outer": o}


def _cum(f, r):
    """cumulative integral: 4th order (cubic through 4 neighbouring nodes, 2-point Gauss per interval)."""
    if "trap_quad" in _MUT:
        return np.concatenate([[0.0], np.cumsum(0.5 * np.diff(r) * (f[1:] + f[:-1]))])
    n = len(r)
    out = np.zeros(n)
    g = np.array([-1, 1]) / np.sqrt(3.0)
    for i in range(n - 1):
        s = min(max(i - 1, 0), n - 4)
        xs, fs = r[s:s + 4], f[s:s + 4]
        a, b = r[i], r[i + 1]
        tot = 0.0
        for gx in g:
            x = 0.5 * (a + b) + 0.5 * (b - a) * gx
            L = [np.prod([(x - xs[m]) / (xs[k] - xs[m]) for m in range(4) if m != k]) for k in range(4)]
            tot += 0.5 * (b - a) * np.dot(L, fs)
        out[i + 1] = out[i] + tot
    return out


def diagnose(slc, matter=None, *, kappa_theta=3.0, kappa_c=10.0):
    r, A, R, KA, KB = slc.r, slc.A, slc.R, slc.KA, slc.KB
    n = len(r)
    center = slc.center
    if "nan_ok" in _MUT:
        A, R, KA, KB = (np.nan_to_num(x, nan=1.0, posinf=1.0) for x in (A, R, KA, KB))
        slc.A, slc.R, slc.KA, slc.KB = A, R, KA, KB
    else:
        if n < 8 or any(len(x) != n for x in (A, R, KA, KB)):
            return _invalid(n, "shape")
        if not all(np.all(np.isfinite(x)) for x in (r, A, R, KA, KB)):
            return _invalid(n, "non-finite input")
        if np.any(np.diff(r) <= 0) or np.any(A <= 0) or np.any(R < 0):
            return _invalid(n, "monotonicity/positivity")
        if center == "vertex" and r[0] != 0:
            return _invalid(n, "vertex needs r[0]=0")
        zero = np.where(R == 0)[0]
        if "R0_ok" not in _MUT and zero.size and not (center == "vertex" and list(zero) == [0]):
            return _invalid(n, "R=0 away from a vertex centre")
    warns = []
    analytic = slc.dR is not None
    if analytic:
        dR, sdR = np.asarray(slc.dR, float), np.zeros(n)
        dKB, sdKB = np.asarray(slc.dKB, float), np.zeros(n)
    else:
        dR, sdR = _D(R, slc, -1), _sig(R, slc, -1)
        dKB, sdKB = _D(KB, slc, 1), _sig(KB, slc, 1)
    G = dR / A
    sG = sdR / A
    U = -R * KB
    with np.errstate(all="ignore"):
        M = 0.5 * R * (1 - G**2 + U**2)
    sM = R * np.abs(G) * sG
    Rtp, Rtm = 2 * (U + G), 2 * (U - G)
    if "swap_theta" in _MUT:
        Rtp, Rtm = Rtm, Rtp
    sRt = 2 * sG
    z = R == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        thp, thm = Rtp / R, Rtm / R
        comp = np.where(z, 0.0, 2 * M / R)
        mr3 = M / R**3
    thp[z] = np.nan
    thm[z] = np.nan
    ext = np.zeros(n, bool)
    if np.any(z):
        i = np.where(z)[0][0]
        R1, R2 = R[i + 1]**2, R[i + 2]**2
        mr3[i] = mr3[i + 1] - R1 * (mr3[i + 2] - mr3[i + 1]) / (R2 - R1)
        ext[i] = True
    # classes: tau = max(kappa (sigma + 2 sn(U) + 2 sn(Gamma)), floor), floor = 1e3 eps max(1, |U|+|Gamma|) (K2)
    snU, snG = _noise(U), _noise(G)
    tau = np.maximum(kappa_theta * (sRt + 2 * snU + 2 * snG), 1e3 * EPS * np.maximum(1.0, np.abs(U) + np.abs(G)))
    tp = tm = tau
    sp_ = np.where(np.abs(Rtp) <= tp, 0, np.sign(Rtp)).astype(int)
    sm_ = np.where(np.abs(Rtm) <= tm, 0, np.sign(Rtm)).astype(int)
    table = {(1, -1): "NORMAL", (-1, 1): "NORMAL_REVERSED", (-1, -1): "FUTURE_TRAPPED", (1, 1): "PAST_TRAPPED",
             (0, -1): "FUTURE_MARGINAL", (-1, 0): "FUTURE_MARGINAL", (0, 1): "PAST_MARGINAL", (1, 0): "PAST_MARGINAL",
             (0, 0): "DOUBLY_MARGINAL"}
    cls = np.array([table[(a, b)] for a, b in zip(sp_, sm_)], dtype=object)
    # v1.2: tau_+ or tau_- > tau_max = 0.1 -> class UNRESOLVED (sign treated as unresolved)
    unres_pt = (tp > 0.1) | (tm > 0.1)
    if "no_unres_class" in _MUT:
        unres_pt[:] = False
    cls[unres_pt] = "UNRESOLVED"
    sp_[unres_pt] = 0
    sm_[unres_pt] = 0
    cls[z] = "CENTER"
    bad = ~(np.isfinite(Rtp) & np.isfinite(Rtm) & np.isfinite(M))
    cls[bad & ~z] = "INVALID"
    # roots
    roots = []
    root_idx = []
    for which, f, tf, g, tg in (("+", Rtp, tp, Rtm, tm), ("-", Rtm, tm, Rtp, tp)):
        s_self = sp_ if which == "+" else sm_
        for i in range(n - 1):
            if z[i] or z[i + 1] or bad[i] or bad[i + 1]:
                continue
            at_node = f[i] == 0
            if not (f[i] * f[i + 1] < 0 or at_node):
                continue
            w = 0.0 if at_node else f[i] / (f[i] - f[i + 1])
            rs = r[i] + w * (r[i + 1] - r[i])
            Rs = R[i] + w * (R[i + 1] - R[i])
            lo, hi = max(0, i - 1), min(n, i + 3)
            Rc = Rs
            if hi - lo >= 4 and not at_node:
                c = np.polyfit(r[lo:hi] - rs, f[lo:hi], 3)
                rts = np.roots(c)
                rts = rts[np.isreal(rts)].real + rs
                rts = rts[(rts >= r[i]) & (rts <= r[i + 1])]
                if rts.size:
                    Rc = np.interp(rts[0], r, R)
            slope = abs(f[i + 1] - f[i]) / abs(R[i + 1] - R[i]) if R[i + 1] != R[i] else np.inf
            band = (tf[i] + tf[i + 1]) / 2 / slope if np.isfinite(slope) and slope > 0 else 0.0
            R_lo, R_hi = min(Rs, Rc) - band, max(Rs, Rc) + band
            gv = g[i] + w * (g[i + 1] - g[i])
            gt = tg[i] + w * (tg[i + 1] - tg[i])
            other = 0 if abs(gv) <= gt else int(np.sign(gv))
            Gv = G[i] + w * (G[i + 1] - G[i])
            Gt = sG[i] + w * (sG[i + 1] - sG[i]) + 1e3 * EPS * max(1.0, abs(Gv))
            inner = next((int(s_self[k]) for k in range(i if not at_node else i - 1, max(-1, i - 4), -1)
                          if k >= 0 and s_self[k] != 0 and not z[k]), 0)
            outer = next((int(s_self[k]) for k in range(i + 1, min(n, i + 5)) if s_self[k] != 0), 0)
            typ = "FUTURE_MOTS" if other < 0 else ("PAST_MOTS" if other > 0 else "DEGENERATE")
            if "ah_nosides" in _MUT:
                ah = which == "+" and other < 0
            elif "ah_plus_only" in _MUT:
                ah = which == "+" and Gv > Gt and other < 0 and inner < 0 and outer > 0
            else:
                # v1.1: theta_out = theta_+ if Gamma > 0 else theta_-; negative on the smaller-R side
                if Gv > Gt:
                    ah = which == "+" and other < 0 and inner < 0 and outer > 0
                elif Gv < -Gt:
                    ah = which == "-" and other < 0 and outer < 0 and inner > 0
                else:
                    ah = False
            roots.append(dict(which=which, r=float(rs), R=float(Rs), R_lo=float(R_lo), R_hi=float(R_hi),
                              M=float(M[i] + w * (M[i + 1] - M[i])), type=typ, ah_candidate=bool(ah),
                              other_sign={1: "+", -1: "-", 0: "0"}[other], inner_sign={1: "+", -1: "-", 0: "0"}[inner],
                              outer_sign={1: "+", -1: "-", 0: "0"}[outer]))
            root_idx.append(i)
    cands = [q for q in roots if q["ah_candidate"]]
    outer_ah = max(cands, key=lambda q: q["R"]) if cands else None
    # geometry status (contract 1.1 section 7)
    margset = ["FUTURE_MARGINAL", "PAST_MARGINAL", "DOUBLY_MARGINAL"]
    trapped = np.isin(cls, ["FUTURE_TRAPPED", "PAST_TRAPPED"])
    marg_idx = np.where(np.isin(cls, margset))[0]
    known = [(i, q) for i, q in zip(root_idx, roots) if q["type"] != "DEGENERATE" and q["inner_sign"] != "0"
             and q["outer_sign"] != "0" and q["inner_sign"] != q["outer_sign"]]
    unexplained = False
    ok_nb = lambda j: 0 <= j < n and cls[j] in ("NORMAL", "NORMAL_REVERSED")  # noqa: E731  (v1.2: resolved non-marginal)
    for k in marg_idx:
        if any(i - 1 <= k <= i + 2 for i, _ in known):
            continue
        iso = (cls[k] == "DOUBLY_MARGINAL" and tp[k] <= 0.1 and tm[k] <= 0.1 and ok_nb(k - 1) and ok_nb(k + 1))
        if iso:
            continue
        if "past_marg" in _MUT and cls[k] == "PAST_MARGINAL":
            continue
        unexplained = True
    degen_near_trap = any(q["type"] == "DEGENERATE" and (trapped[max(i - 1, 0):i + 3].any())
                          for i, q in zip(root_idx, roots))
    past_known = any(q["type"] == "PAST_MOTS" and q["inner_sign"] != "0" and q["outer_sign"] != "0" for q in roots)
    if np.any(cls == "INVALID"):
        geo = "UNDETERMINED"
    elif cands:
        geo = "AH_CANDIDATE"
    elif np.any(cls == "FUTURE_TRAPPED"):
        geo = "FUTURE_TRAPPED_NO_AH_CANDIDATE"
    elif unexplained or degen_near_trap or np.any(cls == "UNRESOLVED"):   # v1.2 cap
        geo = "UNCERTAIN"
    elif np.any(cls == "PAST_TRAPPED") or past_known or ("past_marg" in _MUT and np.any(cls == "PAST_MARGINAL")):
        geo = "PAST_TRAPPED_ONLY"
    else:
        geo = "NO_TRAPPED_SPHERES"
    marginal_present = bool(marg_idx.size)
    # matter and residuals
    P = matter.projections(slc) if matter is not None else None
    out = dict(Gamma=G, U=U, dR=dR, M=M, sigma_M=sM, compactness=comp, M_over_R3=mr3, M_over_R3_extrapolated=ext,
               Rtheta_plus=Rtp, Rtheta_minus=Rtm, sigma_Rtheta_plus=sRt, sigma_Rtheta_minus=sRt.copy(),
               theta_plus=thp, theta_minus=thm, sphere_class=cls, roots=roots, outer_ah_candidate=outer_ah, projections=P)
    data = "OK"
    cstat = "NOT_EVALUATED"
    nanv = np.full(n, np.nan)
    if P is None:
        out.update(ham=nanv, ham_rel=nanv, mom=nanv, mom_rel=nanv, hamM=nanv, hamM_rel=nanv, sigma_ham=nanv,
                   sigma_mom=nanv, sigma_hamM=nanv, norms={}, M_rho=nanv, M_flux=nanv, M_int=nanv, M_prop=nanv)
    else:
        rho, jr = P.rho, P.j_r
        s_rho = getattr(P, "sig_rho", np.zeros(n))
        s_jr = getattr(P, "sig_jr", np.zeros(n))
        if not (np.all(np.isfinite(rho)) and np.all(np.isfinite(jr))):
            data = "INVALID_INPUT"
        dG, sdG = _D(G, slc, 1), _sig(G, slc, 1)
        dM, sdM = _D(M, slc, -1), _sig(M, slc, -1)
        with np.errstate(all="ignore"):
            hT = [2 * (1 - G**2) / R**2, -4 * dG / (A * R), 4 * KA * KB, 2 * KB**2, -16 * np.pi * rho]
            mT = [dKB, -(dR / R) * (KA - KB), 4 * np.pi * jr]
            HT = [dM, -4 * np.pi * R**2 * rho * dR, -4 * np.pi * R**2 * U * jr]
            ham, mom = sum(hT), sum(mT)
            if "F10sign" in _MUT:
                hamM = -(R**2 * dR / 4) * ham + R**3 * KB * mom
                hamM[z] = sum(HT)[z]
            else:
                hamM = sum(HT)
            sh = 4 * np.abs(G) * sG / R**2 + 4 * sdG / (A * R) + 16 * np.pi * s_rho
            sm = sdKB + np.abs(KA - KB) * sdR / R + 4 * np.pi * s_jr
            sH = sdM + 4 * np.pi * R**2 * np.abs(rho) * sdR + 4 * np.pi * R**2 * (np.abs(dR) * s_rho + np.abs(U) * s_jr)
            Th, Tm, TH = (sum(np.abs(t) for t in T) for T in (hT, mT, HT))
            hr, mr, Hr = np.abs(ham) / Th, np.abs(mom) / Tm, np.abs(hamM) / TH
        for x in (ham, mom, hr, mr, sh, sm):
            x[z] = np.nan
        hl = np.maximum(np.r_[np.diff(r), r[-1] - r[-2]], np.r_[r[1] - r[0], np.diff(r)])
        Lr = r[-1] - r[0]
        SX = {"h": (11 * Lr / hl)**2 / Lr**2, "m": (11 * Lr / hl) * (np.max(np.abs(KB)) + 1 / Lr) / Lr,
              "H": (11 * Lr / hl)**2 * (np.max(np.abs(M)) / Lr + 1)}
        viol = np.zeros(n, bool)
        unres = np.zeros(n, bool)
        evaluated = np.zeros(n, bool)
        for X, s, T, key in ((ham, sh, Th, "h"), (mom, sm, Tm, "m"), (hamM, sH, TH, "H")):
            with np.errstate(invalid="ignore"):
                ok = np.isfinite(X)
                evaluated |= ok
                viol |= ok & (np.abs(np.nan_to_num(X)) > kappa_c * np.nan_to_num(s) + 1e3 * EPS * np.maximum(np.nan_to_num(T), SX[key]))
                unres |= ok & (np.nan_to_num(s) > 0.1 * np.nan_to_num(T))
        reg = _regions(n, R, center)
        creg = reg["inner" if center == "none" else "center"]
        if data != "OK":
            cstat = "NOT_EVALUATED"
        elif viol.any():
            cstat = "VIOLATED"
        elif "no_unresolved" not in _MUT and (unres.sum() > 0.05 * evaluated.sum() or (unres & creg).any()):
            cstat = "UNRESOLVED"
        else:
            cstat = "CONSISTENT_WITH_TRUNCATION"
        norms = {}
        nrel_undef = 0
        for name, X, Xr in (("ham", ham, hr), ("mom", mom, mr), ("hamM", hamM, Hr)):
            nrel_undef += int(np.sum(np.isfinite(X) & ~np.isfinite(Xr)))
            norms[name] = {}
            for rg, mask in reg.items():
                m = mask & np.isfinite(X)
                w = 4 * np.pi * R**2 * A
                l2 = vol = 0.0
                idx = np.where(m)[0]
                for k in range(idx.size - 1):
                    a_, b_ = idx[k], idx[k + 1]
                    if b_ == a_ + 1:
                        l2 += 0.5 * (r[b_] - r[a_]) * (X[a_]**2 * w[a_] + X[b_]**2 * w[b_])
                        vol += 0.5 * (r[b_] - r[a_]) * (w[a_] + w[b_])
                if "l2_normalised" in _MUT and vol > 0:
                    l2 /= vol
                norms[name][rg] = {"Linf": float(np.max(np.abs(X[m]))) if m.any() else np.nan, "L2vol": float(np.sqrt(l2)),
                                   "Linf_rel": float(np.nanmax(Xr[m])) if m.any() and np.isfinite(Xr[m]).any() else np.nan}
        if nrel_undef:
            warns.append("n_rel_undefined=%d" % nrel_undef)
        Mrho = _cum(4 * np.pi * R**2 * rho * dR, r)
        Mflux = _cum(4 * np.pi * R**2 * U * jr, r) * (0 if "naive_mass" in _MUT else 1)
        Mint = M[0] + Mrho + Mflux
        mis = np.max(np.abs(Mint - M)) / (np.max(np.abs(M)) + np.max(sM))
        if mis > 1e-6:
            warns.append("M_int_mismatch=%.3g" % mis)
        if center == "cell":
            warns.append("cell: ball [0, r0] not included in integrals")
        out.update(ham=ham, ham_rel=hr, mom=mom, mom_rel=mr, hamM=hamM, hamM_rel=Hr, sigma_ham=sh, sigma_mom=sm,
                   sigma_hamM=sH, norms=norms, M_rho=Mrho, M_flux=Mflux, M_int=Mint,
                   M_prop=_cum(4 * np.pi * R**2 * A * rho, r))
    if center == "vertex":
        G0, K0, sG0 = G[0], KA[0] - KB[0], sG[0]
    else:
        r1, r2 = r[0]**2, r[1]**2
        G0 = G[0] - r1 * (G[1] - G[0]) / (r2 - r1)
        d_ = KA - KB
        K0 = d_[0] - r1 * (d_[1] - d_[0]) / (r2 - r1)
        sG0 = sG[0]
    if center in ("vertex", "cell"):
        if abs(G0 - 1) > max(kappa_theta * sG0, 1e-8) or abs(K0) > 1e-8 * max(np.max(np.abs(KA)), np.max(np.abs(KB)), 1e-300):
            warns.append("center_deviation: Gamma0-1=%.3g, KA-KB(0)=%.3g" % (G0 - 1, K0))
    if np.any(M < -kappa_theta * sM - 1e3 * EPS * np.abs(R)):
        warns.append("negative_mass: min M_MS=%.3g" % np.nanmin(M))
    fin = np.isfinite(M) & np.isfinite(Rtp) & np.isfinite(Rtm)
    if not np.all(fin[~z]):
        data = "NUMERICAL_FAILURE"
        geo = "UNDETERMINED"
        cstat = "NOT_EVALUATED"
    if data == "INVALID_INPUT" and "nan_ok" in _MUT:
        data = "OK"
    if "no_warnings" in _MUT:
        warns = []
    out.update(center_checks={"Gamma0": float(G0), "KA_minus_KB0": float(K0)},
               status={"data": data, "constraints": cstat, "geometry": geo, "marginal_present": marginal_present,
                       "warnings": warns, "n_unresolved_points": int(np.sum(cls == "UNRESOLVED"))})
    return types.SimpleNamespace(**out)

"""PBH-VERIFY-01 physics derivation, part 2: exact reference backgrounds.

For each background the 4-metric is obtained by an explicit coordinate transformation
(pull-back) from a textbook form, then (alpha, beta, A, R) are read off and compared with the
closed forms quoted in derivation.md.  K_A, K_B follow from the ADM formula
    K_A = (A beta' + beta A' - dA/dt)/(alpha A),  K_B = (beta R' - dR/dt)/(alpha R)
which core.py verified against -gamma gamma nabla n with 4D Christoffels.
Then: vacuum/FLRW constraints (F8, F9), M_MS (F11), R theta_pm (F12), F13, and the marginal
spheres with the contract's sign-pattern classification.
Run: cd /root/PBH/pbh_verify_01 && PYTHONPATH=.pydeps python3 derivation/backgrounds.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import sympy as sp
from common import check, info, pullback, summary

t = sp.Symbol('t', real=True)
r = sp.Symbol('r', positive=True)   # radial label > 0 in all backgrounds below
M, ell = sp.symbols('M ell', positive=True)
T, x, tau, chi = sp.symbols('T x tau chi', real=True)


def slice_vars(g, tt, rr):
    """g: 2x2 (t,r)-block of the metric in slice coordinates."""
    A2 = sp.simplify(g[1, 1])
    be = sp.simplify(g[0, 1] / g[1, 1])
    al2 = sp.simplify(-g[0, 0] + A2 * be**2)
    return al2, be, A2


def K_of(al, be, A, R, tt, rr):
    KA = (A * sp.diff(be, rr) + be * sp.diff(A, rr) - sp.diff(A, tt)) / (al * A)
    KB = (be * sp.diff(R, rr) - sp.diff(R, tt)) / (al * R)
    return sp.simplify(KA), sp.simplify(KB)


def diag(name, al, be, A, R, tt, rr, claimed, rho=0, jr=0, assum=None):
    print(f"\n== {name} ==")
    KA, KB = K_of(al, be, A, R, tt, rr)
    Rp = sp.diff(R, rr)
    Gm = sp.simplify(Rp / A)
    U = sp.simplify(-R * KB)
    Mms = sp.simplify(R / 2 * (1 - Gm**2 + U**2))
    thp = sp.simplify(2 * (U + Gm))   # R theta+
    thm = sp.simplify(2 * (U - Gm))   # R theta-
    R3 = 2 * (1 - Gm**2) / R**2 - 4 * sp.diff(Gm, rr) / (A * R)
    Ham = sp.simplify(R3 + 4 * KA * KB + 2 * KB**2 - 16 * sp.pi * rho)
    Mom = sp.simplify(sp.diff(KB, rr) - Rp / R * (KA - KB) + 4 * sp.pi * jr)
    for k, v in [('alpha', al), ('beta', be), ('A', A), ('R', R)]:
        info(k, sp.simplify(v))
    info("K_A", KA); info("K_B", KB); info("Gamma", Gm); info("U", U)
    info("M_MS", Mms); info("R theta+", thp); info("R theta-", thm)
    check(f"{name}: Hamiltonian constraint F8 = 0", Ham)
    check(f"{name}: momentum constraint F9 = 0", Mom)
    check(f"{name}: F13 (R theta+)(R theta-)/4 = 2M/R - 1", thp * thm / 4 - (2 * Mms / R - 1))
    for key, val in claimed.items():
        ref = {'K_A': KA, 'K_B': KB, 'M_MS': Mms, 'Rth+': thp, 'Rth-': thm, 'Gamma': Gm, 'U': U}[key]
        check(f"{name}: claimed {key} = {val}", ref - val)
    return dict(KA=KA, KB=KB, G=Gm, U=U, M=Mms, thp=thp, thm=thm)


def classify_root(name, d, which, rstar, rr, subs_extra=None, eps=sp.Rational(1, 1000)):
    """Contract sec. 7 classification of a root of R theta_which at r = rstar (numeric check)."""
    subs_extra = subs_extra or {}
    f = d['thp'] if which == '+' else d['thm']
    other = d['thm'] if which == '+' else d['thp']
    val = sp.simplify(f.subs(rr, rstar).subs(subs_extra))
    if val.is_Float or (val.is_number and not val.free_symbols):
        val = sp.Integer(0) if abs(sp.N(val)) < 1e-12 else val   # numeric root from nsolve
    o = sp.N(other.subs(rr, rstar).subs(subs_extra))
    fin = sp.N(f.subs(rr, rstar - eps * abs(rstar)).subs(subs_extra))
    fout = sp.N(f.subs(rr, rstar + eps * abs(rstar)).subs(subs_extra))
    G0 = sp.N(d['G'].subs(rr, rstar).subs(subs_extra))
    sgn = lambda v: '+' if v > 0 else ('-' if v < 0 else '0')
    typ = 'FUTURE_MOTS' if o < 0 else ('PAST_MOTS' if o > 0 else 'DEGENERATE')
    ah = (which == '+' and o < 0 and fin < 0 and fout > 0 and G0 > 0)
    ok = check(f"{name}: R theta{which} = 0 at r* = {rstar}", val)
    print(f"      other expansion sign {sgn(o)}, inner {sgn(fin)}, outer {sgn(fout)}, Gamma={G0:.4g}"
          f"  -> {typ}, ah_candidate={ah}")
    return typ, ah


RES = {}
# ------------------------------------------------------------------ Minkowski
print("\n######## Minkowski ########")
# (a) tau-hyperboloids with areal radial coordinate: t_M = sqrt(tau^2 + x^2), x_M = x  (tau > 0)
gM = sp.diag(-1, 1)
tM = sp.sqrt(t**2 + r**2)
g = pullback(gM, [tM, r], [t, r])
al2, be, A2 = slice_vars(g, t, r)
check("Mink hyperboloid(tau, areal): alpha = 1, beta = -r/tau, A = tau/sqrt(tau^2+r^2)",
      [al2 - 1, be + r / t, A2 - t**2 / (t**2 + r**2)])
RES['mink_hyp'] = diag("Minkowski tau-hyperboloid (tau=t>0, areal r)", 1, -r / t, t / sp.sqrt(t**2 + r**2), r, t, r,
                       {'K_A': -1 / t, 'K_B': -1 / t, 'M_MS': 0})
# (b) Milne: t_M = tau cosh chi, x_M = tau sinh chi (same slices, comoving chi)
g = pullback(gM, [t * sp.cosh(r), t * sp.sinh(r)], [t, r])
al2, be, A2 = slice_vars(g, t, r)
check("Milne: alpha=1, beta=0, A=tau", [al2 - 1, be, A2 - t**2])
RES['milne'] = diag("Minkowski Milne (tau=t, chi=r)", 1, 0, t, t * sp.sinh(r), t, r,
                    {'K_A': -1 / t, 'K_B': -1 / t, 'M_MS': 0, 'Rth+': 2 * sp.exp(r), 'Rth-': -2 * sp.exp(-r)})
# (c) stationary hyperboloidal slicing: t_M = T + sqrt(ell^2 + x^2)
g = pullback(gM, [t + sp.sqrt(ell**2 + r**2), r], [t, r])
al2, be, A2 = slice_vars(g, t, r)
s = sp.sqrt(ell**2 + r**2)
check("Mink stationary hyperboloidal: alpha = s/ell, beta = -r s/ell^2, A = ell/s",
      [al2 - s**2 / ell**2, be + r * s / ell**2, A2 - ell**2 / s**2])
RES['mink_hyp_stat'] = diag("Minkowski stationary hyperboloidal (t_M = T + sqrt(ell^2+r^2))", s / ell, -r * s / ell**2,
                            ell / s, r, t, r, {'K_A': -1 / ell, 'K_B': -1 / ell, 'M_MS': 0})
for key in ['mink_hyp', 'milne', 'mink_hyp_stat']:
    d = RES[key]
    bad = 0
    for rv in [sp.Rational(1, 10), 1, 10]:
        vp = sp.N(d['thp'].subs({r: rv, t: 1, ell: 1})); vm = sp.N(d['thm'].subs({r: rv, t: 1, ell: 1}))
        bad += 0 if (vp > 0 and vm < 0) else 1
    check(f"{key}: R theta+ > 0, R theta- < 0 at r = 0.1, 1, 10 (NORMAL, no marginal spheres)", sp.Integer(bad))
print("   (numeric spot checks r = 0.1, 1, 10: all NORMAL)")

# ------------------------------------------------------------------ Schwarzschild
print("\n######## Schwarzschild ########")
f = 1 - 2 * M / r
# ingoing EF: ds^2 = -f dv^2 + 2 dv dr + ...
gEF = sp.Matrix([[-f, 1], [1, 0]])
# (a) Kerr-Schild: v = t + r
g = pullback(gEF, [t + r, r], [t, r])
al2, be, A2 = slice_vars(g, t, r)
check("Kerr-Schild: A^2 = 1+2M/r, beta = (2M/r)/(1+2M/r), alpha^2 = 1/(1+2M/r)",
      [A2 - (1 + 2 * M / r), be - (2 * M / r) / (1 + 2 * M / r), al2 - 1 / (1 + 2 * M / r)])
RES['ks'] = diag("Schwarzschild ingoing Eddington-Finkelstein / Kerr-Schild", 1 / sp.sqrt(1 + 2 * M / r),
                 (2 * M / r) / (1 + 2 * M / r), sp.sqrt(1 + 2 * M / r), r, t, r,
                 {'M_MS': M, 'K_B': 2 * M / (r**2 * sp.sqrt(1 + 2 * M / r)),
                  'K_A': -2 * M * (1 + M / r) / (r**2 * (1 + 2 * M / r)**sp.Rational(3, 2)),
                  'Rth+': 2 * (1 - 2 * M / r) / sp.sqrt(1 + 2 * M / r)})
classify_root("KS", RES['ks'], '+', 2 * M, r, {M: 1})
# (b) Painleve-Gullstrand: v = T + h(r), h' = 1/(1+sqrt(2M/r))
h = sp.Function('h')
g = pullback(gEF, [t + h(r), r], [t, r]).subs(sp.Derivative(h(r), r), 1 / (1 + sp.sqrt(2 * M / r)))
g = g.applyfunc(sp.simplify)
al2, be, A2 = slice_vars(g, t, r)
check("PG: alpha = 1, beta = +sqrt(2M/r), A = 1", [al2 - 1, be - sp.sqrt(2 * M / r), A2 - 1])
RES['pg'] = diag("Schwarzschild Painleve-Gullstrand (ingoing)", 1, sp.sqrt(2 * M / r), 1, r, t, r,
                 {'M_MS': M, 'K_B': sp.sqrt(2 * M / r**3), 'K_A': -sp.sqrt(2 * M / r**3) / 2,
                  'Rth+': 2 * (1 - sp.sqrt(2 * M / r)), 'Rth-': -2 * (1 + sp.sqrt(2 * M / r))})
classify_root("PG", RES['pg'], '+', 2 * M, r, {M: 1})
# (c) isotropic, t = const (time-symmetric, K_ij = 0): r_s = r (1 + M/2r)^2
gS = sp.diag(-(1 - 2 * M / r), 1 / (1 - 2 * M / r))   # in (t_s, r_s) with r_s as symbol 'r'
rs = r * (1 + M / (2 * r))**2
g = pullback(gS.subs(r, rs), [t, rs], [t, r])
al2, be, A2 = slice_vars(g, t, r)
psi = 1 + M / (2 * r)
check("isotropic: A = psi^2, beta = 0, alpha^2 = ((1-M/2r)/(1+M/2r))^2", [A2 - psi**4, be, al2 - ((1 - M / (2 * r)) / psi)**2])
RES['iso'] = diag("Schwarzschild isotropic, t = const (K = 0)", (1 - M / (2 * r)) / psi, 0, psi**2, psi**2 * r, t, r,
                  {'M_MS': M, 'K_A': 0, 'K_B': 0, 'Gamma': (1 - M / (2 * r)) / (1 + M / (2 * r))})
d = RES['iso']
check("isotropic: both R theta+ and R theta- vanish at r = M/2 (throat = bifurcation sphere, R = 2M)",
      [d['thp'].subs(r, M / 2), d['thm'].subs(r, M / 2), (psi**2 * r).subs(r, M / 2) - 2 * M])
print("      r > M/2: (+,-) NORMAL;  r < M/2: (-,+) NORMAL_REVERSED (Gamma < 0);  r = M/2: (0,0) DOUBLY_MARGINAL -> DEGENERATE root")
check("isotropic: inversion symmetry r -> M^2/(4r) maps R to R", sp.simplify((psi**2 * r).subs(r, M**2 / (4 * r)) - psi**2 * r))

# ------------------------------------------------------------------ FLRW
print("\n######## FLRW (dust-free generic a(t); rho from the Friedmann equation) ########")
a = sp.Function('a', positive=True)(t)
H = sp.diff(a, t) / a
for k, Sk, nm in [(0, r, 'flat'), (1, sp.sin(r), 'closed'), (-1, sp.sinh(r), 'open')]:
    rho = 3 * (H**2 + k / a**2) / (8 * sp.pi)
    d = diag(f"FLRW {nm} (k={k}), comoving r", 1, 0, a, a * Sk, t, r,
             {'K_A': -H, 'K_B': -H, 'U': a * Sk * H, 'Gamma': sp.diff(Sk, r),
              'M_MS': sp.Rational(4, 3) * sp.pi * rho * (a * Sk)**3}, rho=rho)
    RES[nm] = d
# explicit marginal spheres, expanding (H>0) and contracting (H<0), numeric a = 1
Hs = sp.Symbol('H', real=True)
print("\n-- FLRW marginal spheres (a = 1 at the instant; H numeric) --")
for nm, Sk in [('flat', r), ('closed', sp.sin(r)), ('open', sp.sinh(r))]:
    d = RES[nm]
    sub = lambda e: e.subs(sp.Derivative(a, t), Hs * a).subs(a, 1)
    dd = {kk: sp.simplify(sub(v)) for kk, v in d.items()}
    for Hv in [sp.Rational(3, 2), -sp.Rational(3, 2)]:
        sols = {}
        for which in ['+', '-']:
            fexpr = dd['thp'] if which == '+' else dd['thm']
            fe = sp.simplify(fexpr.subs(Hs, Hv))
            rng = (sp.Rational(1, 1000), sp.pi - sp.Rational(1, 1000)) if nm == 'closed' else (sp.Rational(1, 1000), 5)
            # scan for sign changes
            fl = sp.lambdify(r, fe, 'math')
            xs = [float(rng[0]) + (float(rng[1]) - float(rng[0])) * i / 2000 for i in range(2001)]
            for i in range(2000):
                if fl(xs[i]) * fl(xs[i + 1]) < 0:
                    root = sp.nsolve(fe, r, (xs[i] + xs[i + 1]) / 2)
                    ddH = {kk: v.subs(Hs, Hv) for kk, v in dd.items()}
                    typ, ah = classify_root(f"FLRW {nm} H={Hv}", ddH, which, sp.Float(root, 15), r)
                    Rst = sp.N((Sk).subs(r, root))
                    print(f"      R* = {Rst:.6f}, H R* = {sp.N(Hv*Rst):.6f}, 2M/R at root = "
                          f"{sp.N((2*ddH['M']/Sk).subs(r, root)):.6f}")
# analytic statements
check("flat expanding: R theta- = 0 exactly at R = 1/H (Hubble sphere), where 2M/R = 1",
      [sp.simplify(2 * (Hs * r - 1)).subs(r, 1 / Hs), sp.simplify((Hs**2 * r**2) .subs(r, 1 / Hs) - 1)])
check("open expanding: marginal sphere tanh(chi) = 1/(aH) (exists since aH > 1 when rho > 0)",
      sp.simplify((2 * (Hs * sp.sinh(r) - sp.cosh(r))).subs(r, sp.atanh(1 / Hs))))
check("closed expanding: theta- = 0 at tan(chi) = 1/(aH); theta+ = 0 at tan(chi) = -1/(aH) (chi > pi/2)",
      [sp.simplify((2 * (Hs * sp.sin(r) - sp.cos(r))).subs(r, sp.atan(1 / Hs))),
       sp.simplify((2 * (Hs * sp.sin(r) + sp.cos(r))).subs(r, sp.pi - sp.atan(1 / Hs)))])
# closed FLRW at maximal expansion (H = 0): equator is doubly marginal but there is no horizon
dcl = RES['closed']
at_turn = lambda e: sp.simplify(e.subs(sp.Derivative(a, t), 0))
check("closed FLRW at turnaround (adot=0): R theta+ = 2 cos chi, R theta- = -2 cos chi -> (0,0) at chi = pi/2 with 2M/R = 1",
      [at_turn(dcl['thp']).subs(r, sp.pi / 2), at_turn(dcl['thm']).subs(r, sp.pi / 2),
       sp.simplify(at_turn(2 * dcl['M'] / (a * sp.sin(r))).subs(r, sp.pi / 2) - 1)])
# asymptotics that matter for the sign tolerance: R theta- -> 0^- on hyperboloidal/Milne slices
check("tau-hyperboloid: R theta- ~ -tau/r -> 0^- as r -> oo (sign tolerance near the outer edge!)",
      sp.limit(RES['mink_hyp']['thm'] * r, r, sp.oo) + t)
check("Milne: R theta- = -2 e^{-chi} -> 0^-", sp.limit(RES['milne']['thm'], r, sp.oo))
sys.exit(1 if summary() else 0)

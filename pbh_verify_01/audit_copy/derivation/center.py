"""PBH-VERIFY-01 physics derivation, part 3: regular centre, and slope of theta at a marginal sphere.

(1) Taylor expansions at a regular centre for smooth data with the contract's parities
    (A, K_A, K_B, phi, Pi even; R odd).
(2) Radial derivative of R theta_pm at a marginal sphere, reduced with the constraints
    (independent derivation; used to discuss when a theta+ root can be an outer boundary).
Run: cd /root/PBH/pbh_verify_01 && PYTHONPATH=.pydeps python3 derivation/center.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import sympy as sp
from common import check, info, summary

r = sp.Symbol('r', positive=True)
a0, a2, a4, R1, R3, R5, ka0, ka2, kb0, kb2, p0, p2, P0, P2, m = sp.symbols(
    'a0 a2 a4 R1 R3 R5 kA0 kA2 kB0 kB2 phi0 phi2 Pi0 Pi2 m', real=True)
A = a0 + a2 * r**2 + a4 * r**4
R = R1 * r + R3 * r**3 + R5 * r**5
KA = ka0 + ka2 * r**2
KB = kb0 + kb2 * r**2
phi = p0 + p2 * r**2
Pi = P0 + P2 * r**2
V = m**2 * phi**2 / 2
Rp = sp.diff(R, r)
G = Rp / A
U = -R * KB
Phi = sp.diff(phi, r) / A
rho = Pi**2 / 2 + Phi**2 / 2 + V
jr = -Pi * sp.diff(phi, r)
Ham = 2 * (1 - G**2) / R**2 - 4 * sp.diff(G, r) / (A * R) + 4 * KA * KB + 2 * KB**2 - 16 * sp.pi * rho
Mom = sp.diff(KB, r) - Rp / R * (KA - KB) + 4 * sp.pi * jr
Mms = R / 2 * (1 - G**2 + U**2)
ser = lambda e, n=3: sp.series(e, r, 0, n).removeO()

print("== (1) regular centre, generic even/odd data (no conditions imposed) ==")
info("Gamma", sp.simplify(ser(G, 3)))
Hs = sp.expand(ser(Ham * r**2, 3))
info("r^2 * Ham, leading", sp.factor(Hs.coeff(r, 0)))
check("Ham has a 1/r^2 pole with coefficient 2(1-Gamma0^2)/R1^2  -> finite only if Gamma(0)=1",
      Hs.coeff(r, 0) - 2 * (1 - (R1 / a0)**2) / R1**2)
Ms = sp.expand(ser(Mom * r, 3))
check("Mom has a 1/r pole with coefficient -(K_A(0)-K_B(0))  -> finite only if K_A(0)=K_B(0)",
      Ms.coeff(r, 0) + (ka0 - kb0))
check("j_r = -Pi phi' = -2 Pi0 phi2 r + O(r^3)  (odd, O(r))", ser(jr, 3) + 2 * P0 * p2 * r)
MmsS = sp.expand(ser(Mms, 4))
check("M_MS without Gamma(0)=1: M ~ (R1/2)(1 - R1^2/a0^2) r  (O(r): conical defect mass)",
      MmsS.coeff(r, 1) - R1 / 2 * (1 - R1**2 / a0**2))

print("\n== regular centre with Gamma(0)=1 (R1 = a0) and K_A(0)=K_B(0) = k0 ==")
k0 = sp.Symbol('k0', real=True)
reg = {R1: a0, ka0: k0, kb0: k0}
Gr, Hr, Mr, Mmr, Ur = [sp.expand(ser(e.subs(reg), n)) for e, n in [(G, 3), (Ham, 1), (Mom, 2), (Mms, 5), (U, 3)]]
info("Gamma", sp.factor(Gr))
check("Gamma = 1 + O(r^2)", Gr.coeff(r, 0) - 1)
info("Gamma_2 (coefficient of r^2)", sp.factor(Gr.coeff(r, 2)))
H0 = sp.simplify(Hr.coeff(r, 0))
info("Ham(0)", H0)
R3c = sp.simplify(ser((2 * (1 - G**2) / R**2 - 4 * sp.diff(G, r) / (A * R)).subs(reg), 1))
info("3R(0)", R3c)
check("3R(0) = -12 Gamma_2 / a0^2  (Gamma_2 = coefficient of r^2 in Gamma)", R3c + 12 * Gr.coeff(r, 2) / a0**2)
rho0 = (rho.subs(r, 0))
check("Ham(0) = 3R(0) + 6 k0^2 - 16 pi rho(0)", H0 - (R3c + 6 * k0**2 - 16 * sp.pi * rho0))
check("Mom = O(r): no r^0 term", Mr.coeff(r, 0))
info("Mom, O(r) coefficient", sp.factor(Mr.coeff(r, 1)))
check("M_MS = O(r^3): no r^1 term", [Mmr.coeff(r, 1), Mmr.coeff(r, 2)])
MR3 = sp.simplify(Mmr.coeff(r, 3) / a0**3)
info("lim M_MS/R^3", MR3)
check("lim_{r->0} M_MS/R^3 = (4 pi/3) rho(0) + Ham(0)/12   [= (4pi/3)rho_c on shell]",
      MR3 - (sp.Rational(4, 3) * sp.pi * rho0 + H0 / 12))
check("lim M_MS/R^3 = 3R(0)/12 + k0^2/2  (pure geometry)", MR3 - (R3c / 12 + k0**2 / 2))
Rtp = sp.expand(ser((2 * (U + G)).subs(reg), 3)); Rtm = sp.expand(ser((2 * (U - G)).subs(reg), 3))
check("R theta+ = +2 - 2 a0 k0 r + O(r^2);  R theta- = -2 - 2 a0 k0 r + O(r^2)",
      [Rtp.coeff(r, 0) - 2, Rtp.coeff(r, 1) + 2 * a0 * k0, Rtm.coeff(r, 0) + 2, Rtm.coeff(r, 1) + 2 * a0 * k0])
check("U = -R K_B = -a0 k0 r + O(r^3)  (odd)", [Ur.coeff(r, 1) + a0 * k0, Ur.coeff(r, 2)])
check("compactness 2M/R = O(r^2)", sp.expand(ser((2 * Mms / R).subs(reg), 2)).coeff(r, 1))
KAmKB = sp.expand(ser((KA - KB).subs(reg), 3))
check("K_A - K_B = O(r^2) automatically once K_A(0)=K_B(0) (even data)", [KAmKB.coeff(r, 0), KAmKB.coeff(r, 1)])
HM = sp.diff(Mms, r) - 4 * sp.pi * R**2 * (rho * Rp + U * jr)
HMs = sp.expand(ser(HM.subs(reg), 3))
check("H_M = M' - 4piR^2(rho R' + U j_r) = O(r^2) at a regular centre", [HMs.coeff(r, 0), HMs.coeff(r, 1)])
check("H_M leading coefficient = (a0^3/4) Ham(0)  (from corrected F10: +R^2R'/4 Ham)",
      sp.simplify(HMs.coeff(r, 2) - a0**3 / 4 * H0))

print("\n== (2) slope of R theta at a marginal sphere, reduced with the constraints ==")
# point values (the root condition holds at a point, so derivatives stay independent symbols)
Av, Rv, Rpv, Rppv, Apv, KAv, KBv, KBpv, rhov, jv = sp.symbols("A R Rp Rpp Ap KA KB KBp rho j", real=True)
Gv = Rpv / Av
Gpv = Rppv / Av - Rpv * Apv / Av**2
Upv = -Rpv * KBv - Rv * KBpv
Hv = 2 * (1 - Gv**2) / Rv**2 - 4 * Gpv / (Av * Rv) + 4 * KAv * KBv + 2 * KBv**2 - 16 * sp.pi * rhov
Mv = KBpv - Rpv / Rv * (KAv - KBv) + 4 * sp.pi * jv
c1, c2 = sp.symbols('c1 c2')
for sgn, name, rootKB in [(1, '+', Gv / Rv), (-1, '-', -Gv / Rv)]:
    # theta_(sgn) = 0  <=>  U = -sgn*Gamma  <=>  K_B = sgn*Gamma/R
    lhs = (Upv + sgn * Gpv) / Av          # d/d(proper length) of (U +/- Gamma) = s^a d_a
    target = sgn * (1 - 8 * sp.pi * Rv**2 * (rhov - sgn * jv / Av)) / (2 * Rv)
    expr = (lhs - target - c1 * Hv - c2 * Mv).subs(KBv, rootKB)
    sol = sp.solve([sp.expand(expr).coeff(rhov), sp.expand(expr).coeff(jv)], [c1, c2], dict=True)[0]
    rem = sp.simplify(expr.subs(sol))
    check(f"at theta{name}=0: s^a d_a (U {name} Gamma) = {'+' if sgn>0 else '-'}[1 - 8 pi R^2 (rho {'-' if sgn>0 else '+'} J)]/(2R)"
          f" + c1 Ham + c2 Mom, J = j_r/A", rem)
    info("   c1, c2", (sp.simplify(sol[c1]), sp.simplify(sol[c2])))
print("   => on shell, a theta+ root is crossed from - (inside) to + (outside) iff 8 pi R^2 (rho - J) < 1;")
print("      flat FLRW Hubble sphere (theta- root): 8 pi R^2 rho = 3 > 1 -> theta- goes - to + outward (past MOTS).")
sys.exit(1 if summary() else 0)

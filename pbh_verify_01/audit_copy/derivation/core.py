"""PBH-VERIFY-01 physics derivation, part 1: geometry of the general spherical 3+1 metric.

ds^2 = -al^2 dt^2 + A^2 (dr + be dt)^2 + R^2 dOmega^2,   al, be, A, R functions of (t, r).

Independent routes:
  * K_ij from the ADM formula (3D covariant derivative of the shift) AND from -gamma gamma nabla n
    with 4D Christoffels;
  * constraints from the 4D Einstein tensor (G_nn, G_n r) AND from the 3D formulas of the contract;
  * theta_pm from q^{ab} nabla_a k_b (4D Christoffels) AND from the area-element Lie derivative;
  * T_ab from the variation of the matter action with respect to g^{ab}.
Run: cd /root/PBH/pbh_verify_01 && PYTHONPATH=.pydeps python3 derivation/core.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import sympy as sp
from common import check, info, christoffel, ricci, summary

T0 = time.time()
t, r, th, ph = sp.symbols('t r theta phi', real=True)
X = [t, r, th, ph]
al = sp.Function('alpha', positive=True)(t, r)
be = sp.Function('beta', real=True)(t, r)
A = sp.Function('A', positive=True)(t, r)
R = sp.Function('R', positive=True)(t, r)

g = sp.Matrix([[-al**2 + A**2 * be**2, A**2 * be, 0, 0],
               [A**2 * be, A**2, 0, 0],
               [0, 0, R**2, 0],
               [0, 0, 0, R**2 * sp.sin(th)**2]])
ginv = g.inv().applyfunc(sp.simplify)
sqrtmg = al * A * R**2 * sp.sin(th)
print("== D0: 4-metric ==")
check("det g = -(al A R^2 sin th)^2", g.det() + sqrtmg**2)
check("g^{-1}: g^tt=-1/al^2, g^tr=be/al^2, g^rr=1/A^2-be^2/al^2",
      [ginv[0, 0] + 1 / al**2, ginv[0, 1] - be / al**2, ginv[1, 1] - (1 / A**2 - be**2 / al**2)])

Gam = christoffel(g, ginv, X)
print(f"   (4D Christoffels done, {time.time()-T0:.1f}s)")

# ---------------- D1: normals ----------------
print("\n== D1: normals, Gamma, U ==")
n_lo = sp.Matrix([-al, 0, 0, 0])
n_up = (ginv * n_lo).applyfunc(sp.simplify)
check("n^a = (1, -be, 0, 0)/al", n_up - sp.Matrix([1, -be, 0, 0]) / al)
check("n.n = -1", (n_lo.T * n_up)[0] + 1)
s_up = sp.Matrix([0, 1 / A, 0, 0])
s_lo = (g * s_up).applyfunc(sp.simplify)
check("s.s = +1", (s_lo.T * s_up)[0] - 1)
check("s.n = 0", (n_lo.T * s_up)[0])
# s is future-independent: s^a is tangent to the slice (s^t = 0) and points to increasing r.
Rp = sp.diff(R, r)
U = sp.simplify(sum(n_up[a] * sp.diff(R, X[a]) for a in range(4)))
Gm = sp.simplify(sum(s_up[a] * sp.diff(R, X[a]) for a in range(4)))
info("U = n^a d_a R", U)
check("Gamma = s^a d_a R = R'/A", Gm - Rp / A)

# ---------------- K_ij two ways ----------------
print("\n== D1b: extrinsic curvature, two independent routes ==")
Y = [r, th, ph]
gam = sp.diag(A**2, R**2, R**2 * sp.sin(th)**2)
gaminv = gam.inv()
Gam3 = christoffel(gam, gaminv, Y)
beta_up = [be, 0, 0]
beta_lo = [sum(gam[i, j] * beta_up[j] for j in range(3)) for i in range(3)]
Dbeta = sp.zeros(3, 3)
for i in range(3):
    for j in range(3):
        Dbeta[i, j] = sp.diff(beta_lo[j], Y[i]) - sum(Gam3[k][i][j] * beta_lo[k] for k in range(3))
K_adm = sp.zeros(3, 3)
for i in range(3):
    for j in range(3):
        K_adm[i, j] = sp.simplify(-(sp.diff(gam[i, j], t) - Dbeta[i, j] - Dbeta[j, i]) / (2 * al))

# 4D route: K_ab = - gamma_a^c gamma_b^d nabla_c n_d
nabla_n = sp.zeros(4, 4)
for a in range(4):
    for b in range(4):
        nabla_n[a, b] = sp.diff(n_lo[b], X[a]) - sum(Gam[c][a][b] * n_lo[c] for c in range(4))
proj = sp.eye(4) + n_up * n_lo.T   # proj[a, c] = gamma^a_c = delta^a_c + n^a n_c  (upper a, lower c)
# gamma_a^c (lower a, upper c) = delta + n_a n^c -> matrix P[a, c]
P = sp.eye(4) + n_lo * n_up.T
K4 = (-(P * nabla_n * P.T)).applyfunc(sp.simplify)
check("K_ij(ADM formula) == -gamma gamma nabla n (4D Christoffels), spatial block",
      K_adm - K4[1:, 1:])
check("K_ab n^b = 0 (K is spatial)", (K4 * n_up).applyfunc(sp.simplify))
check("unprojected -nabla_i n_j on spatial coords equals K_ij (since n_i = 0)",
      (-nabla_n[1:, 1:]) - K_adm)
KA = sp.simplify(K_adm[0, 0] / A**2)
KB = sp.simplify(K_adm[1, 1] / R**2)
check("K^phi_phi = K^theta_theta", K_adm[2, 2] / (R**2 * sp.sin(th)**2) - KB)
check("K_ij off-diagonal = 0", [K_adm[0, 1], K_adm[0, 2], K_adm[1, 2]])
info("K_A = K^r_r", KA)
info("K_B = K^th_th", KB)
check("F1: U = n^a d_a R = -R K_B", U + R * KB)
divn = sp.simplify(sum(sp.diff(sqrtmg * n_up[a], X[a]) for a in range(4)) / sqrtmg)
check("K = gamma^ij K_ij = -nabla_a n^a = K_A + 2 K_B", divn + KA + 2 * KB)
# FLRW sign check of convention: al=1, be=0, A=a(t), R=a(t) r  => K_A=K_B=-H
a_ = sp.Function('a', positive=True)(t)
flrw = {al: 1, be: 0, A: a_, R: a_ * r}
check("sign convention: expanding FLRW gives K_A=K_B=-adot/a, K=-3H",
      [KA.subs(flrw).doit() + sp.diff(a_, t) / a_, KB.subs(flrw).doit() + sp.diff(a_, t) / a_])

# ---------------- D4: 3-Ricci scalar ----------------
print("\n== D4: 3-Ricci scalar ==")
Ric3 = ricci(Gam3, Y)
R3 = sp.simplify(sum(gaminv[i, j] * Ric3[i, j] for i in range(3) for j in range(3)))
GG = Rp / A
F7 = 2 * (1 - GG**2) / R**2 - 4 * sp.diff(GG, r) / (A * R)
check("F7: 3R = 2(1-Gamma^2)/R^2 - 4 Gamma'/(A R)", R3 - F7)
info("3R (expanded form)", sp.simplify(R3))

# ---------------- D5: constraints from 4D Einstein tensor ----------------
print("\n== D5: constraints via 4D Einstein tensor (independent route) ==")
t1 = time.time()
Ric4 = ricci(Gam, X)
R4 = sp.simplify(sum(ginv[a, b] * Ric4[a, b] for a in range(4) for b in range(4)))
Gein = Ric4 - g * R4 / 2
print(f"   (4D Ricci/Einstein done, {time.time()-t1:.1f}s)")
Gnn = sum(n_up[a] * n_up[b] * Gein[a, b] for a in range(4) for b in range(4))
Gnr = sum(n_up[a] * Gein[a, 1] for a in range(4))    # G_ab n^a gamma^b_r, gamma^b_r = delta^b_r
Ham_geom = R3 + 4 * KA * KB + 2 * KB**2
check("F8 (Gauss): 3R + K^2 - K_ijK^ij = 3R + 4K_AK_B + 2K_B^2 == 2 G_ab n^a n^b", Ham_geom - 2 * Gnn)
KAp, KBp = sp.diff(KA, r), sp.diff(KB, r)
# Codazzi: D_j(K^j_r - delta^j_r K) computed with 3D Christoffels
Kmix = sp.diag(KA, KB, KB)
Ktot = KA + 2 * KB
div_r = 0
for j in range(3):
    div_r += sp.diff(Kmix[j, 0], Y[j])
    for k in range(3):
        div_r += Gam3[j][j][k] * Kmix[k, 0] - Gam3[k][j][0] * Kmix[j, k]
div_r -= sp.diff(Ktot, r)
check("Codazzi (3D route): D_j(K^j_r - delta^j_r K) = -2K_B' + 2(R'/R)(K_A-K_B)",
      div_r - (-2 * KBp + 2 * Rp / R * (KA - KB)))
check("Codazzi (4D route): D_j(K^j_r - delta K) == -G_ab n^a gamma^b_r  (= 8 pi j_r on shell)",
      div_r + Gnr)
Mom_geom = KBp - Rp / R * (KA - KB)
check("F9: M := K_B' - (R'/R)(K_A-K_B) + 4 pi j_r with j_r = -G(n,e_r)/(8pi) vanishes identically",
      Mom_geom - Gnr / 2)
# sign-flip control (should FAIL, demonstrating the test is sensitive)
ctrl = sp.simplify(KBp + Rp / R * (KA - KB) - Gnr / 2)
print(f"   control (wrong sign of (R'/R) term) is nonzero: {ctrl != 0}")

# ---------------- D2: stress tensor from the action ----------------
print("\n== D2: stress tensor from L = -1/2 (grad phi)^2 - V, and projections ==")
f = sp.Function('varphi', real=True)(t, r)
Vf = sp.Function('V')
# generic variation: symmetric inverse-metric symbols h^{ab}
hs = sp.symbols('h0:10')
H = sp.Matrix(4, 4, lambda i, j: 0)
idx = 0
for i in range(4):
    for j in range(i, 4):
        H[i, j] = hs[idx]; H[j, i] = hs[idx]; idx += 1
dphi = sp.symbols('dphi0:4')
Lsym = -sp.Rational(1, 2) * sum(H[a, b] * dphi[a] * dphi[b] for a in range(4) for b in range(4)) - sp.Symbol('Vv')
# T_ab = -2 dL/dg^{ab} + g_ab L ; dL/dg^{ab} for a != b gets factor 1/2 (symmetric symbol counted twice)
Tgen = sp.zeros(4, 4)
Gsym = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f'g{min(i,j)}{max(i,j)}'))
idx = 0
for i in range(4):
    for j in range(i, 4):
        dL = sp.diff(Lsym, hs[idx]) * (1 if i == j else sp.Rational(1, 2))
        Tgen[i, j] = -2 * dL + Gsym[i, j] * Lsym
        Tgen[j, i] = Tgen[i, j]
        idx += 1
F2 = sp.Matrix(4, 4, lambda a, b: dphi[a] * dphi[b] - Gsym[a, b] *
               (sp.Rational(1, 2) * sum(H[c, d] * dphi[c] * dphi[d] for c in range(4) for d in range(4)) + sp.Symbol('Vv')))
check("F2: T_ab = -2 dL/dg^ab + g_ab L = d_a phi d_b phi - g_ab[1/2 (grad phi)^2 + V]", Tgen - F2)

dph = [sp.diff(f, x) for x in X]
kin = sum(ginv[a, b] * dph[a] * dph[b] for a in range(4) for b in range(4))
T = sp.Matrix(4, 4, lambda a, b: dph[a] * dph[b] - g[a, b] * (kin / 2 + Vf(f)))
Pi = sp.simplify(sum(n_up[a] * dph[a] for a in range(4)))
check("Pi = n^a d_a phi = (d_t phi - be d_r phi)/al", Pi - (sp.diff(f, t) - be * sp.diff(f, r)) / al)
Phi = sp.diff(f, r) / A
rho = sp.simplify(sum(n_up[a] * n_up[b] * T[a, b] for a in range(4) for b in range(4)))
jr = sp.simplify(-sum(n_up[a] * T[a, 1] for a in range(4)))
# spatial projection: gamma_i^a = delta_i^a for coordinate spatial i (n_i = 0) -> S_ij = T_ij
Srr = sp.simplify(T[1, 1] / A**2)
Sthth = sp.simplify(T[2, 2] / R**2)
check("F3: rho = 1/2 Pi^2 + 1/2 Phi^2 + V", rho - (Pi**2 / 2 + Phi**2 / 2 + Vf(f)))
check("F4: j_r = -gamma_r^a n^b T_ab = -Pi phi'", jr + Pi * sp.diff(f, r))
check("F5: S^r_r = 1/2 Pi^2 + 1/2 Phi^2 - V", Srr - (Pi**2 / 2 + Phi**2 / 2 - Vf(f)))
check("F5: S^th_th = 1/2 Pi^2 - 1/2 Phi^2 - V", Sthth - (Pi**2 / 2 - Phi**2 / 2 - Vf(f)))
check("F5: S^ph_ph = S^th_th", T[3, 3] / (R**2 * sp.sin(th)**2) - Sthth)
# kinematic sign of j_r: outgoing wave in flat space phi = F(t - r) gives j_r > 0
Fw = sp.Function('F')
flat = {al: 1, be: 0, A: 1, R: r}
jr_out = (jr.subs(flat).subs(f, Fw(t - r))).doit()
check("orientation: outgoing wave phi=F(t-r) in flat space has j_r = +F'^2 > 0",
      sp.simplify(jr_out - sp.Subs(sp.diff(Fw(sp.Symbol('u')), sp.Symbol('u')), sp.Symbol('u'), t - r).doit()**2))
# field equation and conservation: nabla^a T_ab = (box phi - V') d_b phi
box = sp.simplify(sum(sp.diff(sqrtmg * sum(ginv[a, b] * dph[b] for b in range(4)), X[a]) for a in range(4)) / sqrtmg)
Tmix = (ginv * T)  # T^a_b
for bidx, name in [(0, 't'), (1, 'r')]:
    divT = 0
    for a in range(4):
        divT += sp.diff(Tmix[a, bidx], X[a])
        for c in range(4):
            divT += Gam[a][a][c] * Tmix[c, bidx] - Gam[c][a][bidx] * Tmix[a, c]
    check(f"Bianchi/matter: nabla_a T^a_{name} = (box phi - V'(phi)) d_{name} phi",
          divT - (box - sp.diff(Vf(f), f)) * dph[bidx])

# ---------------- D3: homogeneous limit (FKG) ----------------
print("\n== D3: homogeneous Friedmann-Klein-Gordon (F6), k = 0, +1, -1 ==")
ph0 = sp.Function('phi0', real=True)(t)
m = sp.Symbol('m_phi', positive=True)
Vq = lambda x: m**2 * x**2 / 2
Hh = sp.diff(a_, t) / a_
for k, Rk in [(0, a_ * r), (1, a_ * sp.sin(r)), (-1, a_ * sp.sinh(r))]:
    sub = {al: 1, be: 0, A: a_, R: Rk}
    Gtt = sp.simplify(Gein[0, 0].subs(sub).doit())
    Grr = sp.simplify((Gein[1, 1] / A**2).subs(sub).doit())
    Gthth = sp.simplify((Gein[2, 2] / R**2).subs(sub).doit())
    rho0 = sp.diff(ph0, t)**2 / 2 + Vq(ph0)
    p0 = sp.diff(ph0, t)**2 / 2 - Vq(ph0)
    check(f"k={k:+d}: G_tt = 8 pi rho  <=>  H^2 + k/a^2 = (8pi/3) rho", Gtt - 3 * (Hh**2 + k / a_**2))
    check(f"k={k:+d}: G^r_r = G^th_th = -(2 a''/a + H^2 + k/a^2)  (= 8 pi p)",
          [Grr + (2 * sp.diff(a_, t, 2) / a_ + Hh**2 + k / a_**2), Gthth - Grr])
    # Hdot from the two Friedmann equations
    Hdot = sp.diff(Hh, t)
    # a''/a = Hdot + H^2 ; combine: Hdot = -4 pi (rho + p) + k/a^2
    boxk = box.subs(f, ph0).subs(sub).doit()
    check(f"k={k:+d}: box phi = -(phi'' + 3 H phi')  =>  KG: phi''+3H phi'+m^2 phi = 0",
          sp.simplify(boxk + sp.diff(ph0, t, 2) + 3 * Hh * sp.diff(ph0, t)))
    check(f"k={k:+d}: slice values K_A = K_B = -H, Pi = phi', U = R H, rho,S^r_r,S^th_th = rho0,p0,p0",
          [KA.subs(sub).doit() + Hh, KB.subs(sub).doit() + Hh, Pi.subs(f, ph0).subs(sub).doit() - sp.diff(ph0, t),
           U.subs(sub).doit() - Rk * Hh,
           rho.subs(f, ph0).subs(sub).doit().subs(Vf(ph0), Vq(ph0)) - rho0,
           Srr.subs(f, ph0).subs(sub).doit().subs(Vf(ph0), Vq(ph0)) - p0,
           Sthth.subs(f, ph0).subs(sub).doit().subs(Vf(ph0), Vq(ph0)) - p0])
# Hdot identity (algebraic consequence): with G_tt and G_rr equations
Hs, Hd, rs, ps, ks, as_ = sp.symbols('H Hdot rho p k a')
# 3(H^2 + k/a^2) = 8 pi rho ; -(2(Hdot + H^2) + H^2 + k/a^2) = 8 pi p  -> Hdot = -4pi(rho+p) + k/a^2
sol = sp.solve([3 * (Hs**2 + ks / as_**2) - 8 * sp.pi * rs, -(2 * (Hd + Hs**2) + Hs**2 + ks / as_**2) - 8 * sp.pi * ps], [Hd, rs])
check("Hdot = -4 pi (rho + p) + k/a^2 ; for scalar rho+p = phidot^2 -> k=0: Hdot = -4 pi phidot^2",
      sol[Hd] - (-4 * sp.pi * (rs + ps) + ks / as_**2) .subs(rs, sol[rs]))

# ---------------- D7: Misner-Sharp ----------------
print("\n== D7: Misner-Sharp mass ==")
gradR2 = sp.simplify(sum(ginv[a, b] * sp.diff(R, X[a]) * sp.diff(R, X[b]) for a in range(2) for b in range(2)))
check("g^{ab} d_aR d_bR (t,r block) = Gamma^2 - U^2", gradR2 - (Gm**2 - U**2))
M_MS = R / 2 * (1 - gradR2)
check("F11: M_MS = (R/2)(1 - (R'/A)^2 + R^2 K_B^2)", M_MS - R / 2 * (1 - (Rp / A)**2 + R**2 * KB**2))

# ---------------- D8: null expansions ----------------
print("\n== D8: null expansions ==")
qinv = (ginv + n_up * n_up.T - s_up * s_up.T).applyfunc(sp.simplify)
check("q^{ab} = g^ab + n^a n^b - s^a s^b is the sphere projector (only angular comps: 1/R^2, 1/(R^2 sin^2))",
      qinv - sp.diag(0, 0, 1 / R**2, 1 / (R**2 * sp.sin(th)**2)))
thetas = {}
for sgn, name in [(1, '+'), (-1, '-')]:
    k_up = n_up + sgn * s_up
    k_lo = (g * k_up).applyfunc(sp.simplify)
    check(f"k_{name} = n {name} s is null, n.k = -1", [(k_lo.T * k_up)[0], (n_lo.T * k_up)[0] + 1])
    nab_k = sp.Matrix(4, 4, lambda a, b: sp.diff(k_lo[b], X[a]) - sum(Gam[c][a][b] * k_lo[c] for c in range(4)))
    theta = sp.simplify(sum(qinv[a, b] * nab_k[a, b] for a in range(4) for b in range(4)))
    theta_area = sp.simplify(sum(k_up[a] * sp.diff(sp.log(R**2), X[a]) for a in range(4)))
    check(f"theta_{name}: q^ab nabla_a k_b == k^a d_a ln(R^2) (area route)", theta - theta_area)
    check(f"F12: theta_{name} = (2/R)(U {name} Gamma) = 2({name}R'/(AR) - K_B)",
          [theta - 2 / R * (U + sgn * Gm), theta - 2 * (sgn * Rp / (A * R) - KB)])
    thetas[name] = theta
check("F13: R^2 theta+ theta- /4 = U^2 - Gamma^2 = 2 M_MS/R - 1",
      [R**2 * thetas['+'] * thetas['-'] / 4 - (U**2 - Gm**2), R**2 * thetas['+'] * thetas['-'] / 4 - (2 * M_MS / R - 1)])
# boost invariance of the product: k+ -> e^eta k+, k- -> e^-eta k-  (another slicing through same sphere)
print("   note: a local boost of (n,s) rescales k+ -> e^eta k+, k- -> e^-eta k-, so theta+theta- (and M_MS) are slicing-independent;")
print("         theta+ and theta- separately are not; their signs are.")

# ---------------- D7b: Kodama vector / current (integral form of F10) ----------------
print("\n== D7b: Kodama vector and the flux term ==")
# K^a = eps^{ab} d_b R on the (t,r) 2-space, eps^{tr} = +1/sqrt(-det g_2) = 1/(al A)
eps2 = sp.Matrix([[0, 1 / (al * A)], [-1 / (al * A), 0]])
Kod = [sum(eps2[a, b] * sp.diff(R, X[b]) for b in range(2)) for a in range(2)] + [0, 0]
Kod = sp.Matrix(Kod)
check("Kodama K^a = Gamma n^a - U s^a  (K.grad R = 0; K = static Killing field in Schwarzschild)",
      Kod - (Gm * n_up - U * s_up))
check("K^a d_a R = 0", sum(Kod[a] * sp.diff(R, X[a]) for a in range(4)))
TnK = sp.simplify(sum(n_up[a] * T[a, b] * Kod[b] for a in range(4) for b in range(4)))
check("4 pi R^2 A * T(n,K) = 4 pi R^2 (rho R' + U j_r): the on-shell F10 integrand is the Kodama energy flux through the slice",
      A * TnK - (rho * Rp + U * jr))
# Milligan et al. 2504.02600v2 eq. (6c) (comoving gauge beta=0, alpha=e^varphi, A=e^{lambda/2})
vp_, lam_ = sp.Function('varphi_l')(t, r), sp.Function('lambda_l')(t, r)
subsM = {be: 0, al: sp.exp(vp_), A: sp.exp(lam_ / 2)}
MomF = (Mom_geom + 4 * sp.pi * jr).subs(subsM).doit()
rhs6c = (sp.diff(lam_, t) * Rp + 2 * sp.diff(R, t) * sp.diff(vp_, r) - 2 * sp.diff(R, t, r)) / R - 8 * sp.pi * sp.diff(f, t) * sp.diff(f, r)
check("Milligan+ (6c): 8 pi phidot phi' = (lamdot R' + 2 Rdot varphi' - 2 Rdot')/R  <=>  2 alpha * F9 = 0", sp.simplify(2 * sp.exp(vp_) * MomF - rhs6c))
# Bloomfield+ 1504.02071v1 eq. (8a),(8b),(13): U = e^-phi Rdot, Gamma = e^-lambda/2 R', Gamma^2 = 1 + U^2 - 2m/R
check("Bloomfield+ (8a),(8b): U = e^{-phi} Rdot, Gamma = e^{-lambda/2} R' in comoving gauge",
      [U.subs(subsM).doit() - sp.exp(-vp_) * sp.diff(R, t), (Gm.subs(subsM).doit()) - sp.exp(-lam_ / 2) * Rp])

# ---------------- D6: mass-form identity (F10) on a slice ----------------
print("\n== D6: mass-form identity F10 (slice functions of r) ==")
As, Rs, KAs, KBs, rhos, js = [sp.Function(nm)(r) for nm in ('A', 'R', 'K_A', 'K_B', 'rho', 'j_r')]
Gs = sp.diff(Rs, r) / As
Us = -Rs * KBs
Ms = Rs / 2 * (1 - Gs**2 + Us**2)
Hc = 2 * (1 - Gs**2) / Rs**2 - 4 * sp.diff(Gs, r) / (As * Rs) + 4 * KAs * KBs + 2 * KBs**2 - 16 * sp.pi * rhos
Mc = sp.diff(KBs, r) - sp.diff(Rs, r) / Rs * (KAs - KBs) + 4 * sp.pi * js
HM = sp.diff(Ms, r) - 4 * sp.pi * Rs**2 * (rhos * sp.diff(Rs, r) + Us * js)
ok_contract = check("F10 as written in contract v1.0-draft: H_M = -(R^2R'/4) H + R^3 K_B M",
                    HM - (-(Rs**2 * sp.diff(Rs, r) / 4) * Hc + Rs**3 * KBs * Mc))
check("F10 corrected: H_M = +(R^2R'/4) H + R^3 K_B M", HM - ((Rs**2 * sp.diff(Rs, r) / 4) * Hc + Rs**3 * KBs * Mc))
# uniqueness of coefficients: H_M = c1 H + c2 M with c1, c2 free functions -> solve from rho and j_r coefficients
c1 = sp.simplify(sp.diff(HM, rhos) / sp.diff(Hc, rhos))
c2 = sp.simplify(sp.diff(HM, js) / sp.diff(Mc, js))
info("coefficients forced by the rho and j_r terms: c1", c1)
info("                                              c2", c2)
# M' in areal gauge: R = r
areal = sp.simplify((sp.diff(Ms, r) - 4 * sp.pi * Rs**2 * (rhos * sp.diff(Rs, r) + Us * js)).subs(Rs, r).doit())
check("areal gauge (R=r): on-shell M' = 4 pi r^2 (rho - r K_B j_r) = 4 pi r^2 (E - r K_B A J)  (V1 form, context_audit C4)",
      (((Rs**2 * sp.diff(Rs, r) / 4) * Hc + Rs**3 * KBs * Mc).subs(Rs, r).doit()) - areal)

print(f"\n(total time {time.time()-T0:.1f}s)")
sys.exit(1 if summary() else 0)

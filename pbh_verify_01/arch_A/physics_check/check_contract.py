"""Symbolic check of the physics in physics_contract.md v1.0 (F1-F13, F6, limitation of section 7).

Run:  PYTHONPATH=/root/PBH/pbh_verify_01/.pydeps python3 physics_check/check_contract.py
Every check is an independent derivation from the 4-metric / action; prints OK or raises.
"""
import sympy as sp

t, r, th, ph = sp.symbols('t r theta phi', real=True)
X = [t, r, th, ph]
al = sp.Function('alpha', positive=True)(t, r)
be = sp.Function('beta', real=True)(t, r)
A = sp.Function('A', positive=True)(t, r)
R = sp.Function('R', positive=True)(t, r)
f = sp.Function('f', real=True)(t, r)          # scalar field phi
m = sp.Symbol('m', positive=True)


def simp(e):
    return sp.simplify(sp.expand(e))


def check(name, expr):
    e = simp(expr)
    if e != 0:
        raise AssertionError(f"{name}: residual {e}")
    print(f"OK  {name}")


# ---------------------------------------------------------------- 4-metric
# ds^2 = -al^2 dt^2 + A^2 (dr + be dt)^2 + R^2 dOmega^2
g = sp.zeros(4, 4)
g[0, 0] = -al**2 + A**2 * be**2
g[0, 1] = g[1, 0] = A**2 * be
g[1, 1] = A**2
g[2, 2] = R**2
g[3, 3] = R**2 * sp.sin(th)**2
gi = sp.simplify(g.inv())


def christoffel(g, gi, X):
    n = len(X)
    G = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                G[a][b][c] = sp.simplify(sum(gi[a, d] * (sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b])
                                                         - sp.diff(g[b, c], X[d])) for d in range(n)) / 2)
    return G


G4 = christoffel(g, gi, X)
n_lo = sp.Matrix([-al, 0, 0, 0])
n_up = sp.simplify(gi * n_lo)
check("n^a = (1, -beta, 0, 0)/alpha", (n_up - sp.Matrix([1, -be, 0, 0]) / al).norm()**2)
s_up = sp.Matrix([0, 1 / A, 0, 0])
s_lo = g * s_up


def cov_lo(v_lo):
    """nabla_a v_b"""
    return sp.Matrix(4, 4, lambda a, b: sp.diff(v_lo[b], X[a]) - sum(G4[c][a][b] * v_lo[c] for c in range(4)))


# ---------------------------------------------------------------- extrinsic curvature, F1
Dn = cov_lo(n_lo)
proj = sp.eye(4) + n_up * n_lo.T          # gamma^a_b = delta^a_b + n^a n_b
Kab = -sp.simplify(proj.T * Dn * proj)     # K_ab = -gamma_a^c gamma_b^d nabla_c n_d
# spatial part and mixed components
gam = sp.Matrix([[A**2, 0, 0], [0, R**2, 0], [0, 0, R**2 * sp.sin(th)**2]])
Ksp = Kab[1:, 1:]
KA = simp((gam.inv() * Ksp)[0, 0])
KB = simp((gam.inv() * Ksp)[1, 1])
# ADM form: K_ij = -(1/2al)(d_t gam_ij - D_i be_j - D_j be_i)
# theta-theta: D_th be_th = -Gamma^r_thth be_r = (R R'/A^2) A^2 be
KB_adm = -(2 * R * sp.diff(R, t) - 2 * R * sp.diff(R, r) * be) / (2 * al) / R**2
check("K^th_th = -(d_t R - be R')/(al R)", KB - KB_adm)
U = sp.simplify((n_up.T * sp.Matrix([sp.diff(R, x) for x in X]))[0])
check("F1: U = n.grad R = -R K_B", U + R * KB)
Gam = sp.simplify((s_up.T * sp.Matrix([sp.diff(R, x) for x in X]))[0])
check("F1: Gamma = s.grad R = R'/A", Gam - sp.diff(R, r) / A)
Ktr = simp(-sp.simplify((gi * Dn).trace()))
check("K = gamma^ij K_ij = -nabla.n", Ktr - (KA + 2 * KB))

# ---------------------------------------------------------------- stress tensor, F2-F5
V = m**2 * f**2 / 2
df = sp.Matrix([sp.diff(f, x) for x in X])
kin = (df.T * gi * df)[0]
T = sp.Matrix(4, 4, lambda a, b: df[a] * df[b] - g[a, b] * (kin / 2 + V))
Pi = (sp.diff(f, t) - be * sp.diff(f, r)) / al
check("Pi = n^a d_a phi", (n_up.T * df)[0] - Pi)
rho = simp((n_up.T * T * n_up)[0])
Phi = sp.diff(f, r) / A
check("F3 rho", rho - (Pi**2 / 2 + Phi**2 / 2 + V))
j_lo = -(proj.T * T * n_up)            # j_a = -gamma_a^b n^c T_bc
check("F4 j_r = -Pi phi'", simp(j_lo[1]) + Pi * sp.diff(f, r))
S_lo = proj.T * T * proj
Srr = simp(S_lo[1, 1] / A**2)
Sthth = simp(S_lo[2, 2] / R**2)
check("F5 S^r_r", Srr - (Pi**2 / 2 + Phi**2 / 2 - V))
check("F5 S^th_th", Sthth - (Pi**2 / 2 - Phi**2 / 2 - V))

# ---------------------------------------------------------------- 3-geometry, F7
Y = [r, th, ph]
gam_i = gam.inv()
G3 = christoffel(gam, gam_i, Y)


def ricci_scalar(G, gi_, Y):
    n = len(Y)
    Ric = sp.zeros(n, n)
    for b in range(n):
        for d in range(n):
            Ric[b, d] = sum(sp.diff(G[a][b][d], Y[a]) - sp.diff(G[a][b][a], Y[d])
                            + sum(G[a][a][e] * G[e][b][d] - G[a][d][e] * G[e][b][a] for e in range(n))
                            for a in range(n))
    return sp.simplify(sum(gi_[b, d] * Ric[b, d] for b in range(n) for d in range(n)))


R3 = ricci_scalar(G3, gam_i, Y)
Gm = sp.diff(R, r) / A
R3_contract = 2 * (1 - Gm**2) / R**2 - 4 * sp.diff(Gm, r) / (A * R)
check("F7 3-Ricci scalar", R3 - R3_contract)

# ---------------------------------------------------------------- constraints F8, F9 (derived from D_j(K^j_i - delta K) = 8 pi j_i)
KA_, KB_ = sp.Function('KA')(r), sp.Function('KB')(r)
Kmix = sp.diag(KA_, KB_, KB_)             # K^i_j
Kt = KA_ + 2 * KB_
Ar, Rr = sp.Function('Ar', positive=True)(r), sp.Function('Rr', positive=True)(r)
gam_r = sp.diag(Ar**2, Rr**2, Rr**2 * sp.sin(th)**2)
G3r = christoffel(gam_r, gam_r.inv(), Y)
# D_j M^j_i for mixed tensor M = K - delta K
Mmix = Kmix - sp.eye(3) * Kt
divM_r = sum(sp.diff(Mmix[j, 0], Y[j]) + sum(G3r[j][j][k] * Mmix[k, 0] for k in range(3))
             - sum(G3r[k][j][0] * Mmix[j, k] for k in range(3)) for j in range(3))
jr = sp.Symbol('j_r')
Rp = sp.diff(Rr, r)
mom_contract = sp.diff(KB_, r) - (Rp / Rr) * (KA_ - KB_) + 4 * sp.pi * jr
check("F9 momentum constraint (divided by -2)", (divM_r - 8 * sp.pi * jr) / (-2) - mom_contract)
KK = (Kmix * Kmix).trace()
check("F8 K^2 - K_ij K^ij", Kt**2 - KK - (4 * KA_ * KB_ + 2 * KB_**2))

# ---------------------------------------------------------------- F10, F11
rh = sp.Function('rho')(r)
Gr = Rp / Ar
Ur = -Rr * KB_
R3r = 2 * (1 - Gr**2) / Rr**2 - 4 * sp.diff(Gr, r) / (Ar * Rr)
Ham = R3r + 4 * KA_ * KB_ + 2 * KB_**2 - 16 * sp.pi * rh
Mms = Rr / 2 * (1 - Gr**2 + Ur**2)
HamM = sp.diff(Mms, r) - 4 * sp.pi * Rr**2 * (rh * Rp + Ur * jr)
check("F10 mass-form identity (+ sign)", HamM - (Rr**2 * Rp / 4 * Ham + Rr**3 * KB_ * mom_contract))
check("F10 with the draft (-) sign FAILS", 0 if simp(HamM - (-Rr**2 * Rp / 4 * Ham + Rr**3 * KB_ * mom_contract)) != 0 else 1)
# F11 from 4-metric: M = R/2 (1 - g^ab d_a R d_b R)
dR4 = sp.Matrix([sp.diff(R, x) for x in X])
M4 = R / 2 * (1 - (dR4.T * gi * dR4)[0])
check("F11 M_MS = R/2 (1 - Gamma^2 + U^2)", M4 - R / 2 * (1 - Gam**2 + U**2))

# ---------------------------------------------------------------- F12, F13: expansions from q^ab nabla_a k_b
q_up = sp.zeros(4, 4)
q_up[2, 2] = 1 / R**2
q_up[3, 3] = 1 / (R**2 * sp.sin(th)**2)
for sgn, name in ((1, '+'), (-1, '-')):
    k_lo = n_lo + sgn * s_lo
    th_ = simp(sum(q_up[a, b] * cov_lo(k_lo)[a, b] for a in range(4) for b in range(4)))
    check(f"F12 theta{name} = q^ab nabla_a k_b = 2(+-R'/(AR) - K_B)", th_ - 2 * (sgn * sp.diff(R, r) / (A * R) - KB))
    check(f"F12 theta{name} = (2/R) k.grad R", th_ - 2 / R * (U + sgn * Gam))
    kk = simp(((n_up + sgn * s_up).T * g * (n_up + sgn * s_up))[0])
    check(f"k{name} null", kk)
    check(f"n.k{name} = -1", simp((n_lo.T * (n_up + sgn * s_up))[0]) + 1)
thp = 2 / R * (U + Gam)
thm = 2 / R * (U - Gam)
check("F13 R^2 th+ th- /4 = 2M/R - 1", R**2 * thp * thm / 4 - (2 * M4 / R - 1))

# ---------------------------------------------------------------- centre expansion of R theta_pm
a0, a2, k0, k2 = sp.symbols('a0 a2 k0 k2')
rr = sp.Symbol('rr')
Aser = a0 + a2 * rr**2
# regular: Gamma(0)=1 => R'(0) = A(0); R odd
b3 = sp.Symbol('b3')
Rser = a0 * rr + b3 * rr**3
KBs = k0 + k2 * rr**2
Rtp = 2 * (-Rser * KBs + sp.diff(Rser, rr) / Aser)
ser = sp.series(Rtp, rr, 0, 2).removeO()
check("centre: R theta+ = 2 - 2 A0 KB0 r + O(r^2)", ser - (2 - 2 * a0 * k0 * rr))

# ---------------------------------------------------------------- section-7 limitation: d(R theta+)/dl on a theta+ root, on the constraints
Rp_, Ap_ = sp.symbols('Rp Ap')
# solve constraints for Gamma' and U' at a point where U = -Gamma (theta+ = 0)
Gs, Us, Rs, As, KAs, rhos, Js = sp.symbols('G U Rs As KAs rho J')
KBs_ = Us / (-Rs)
Gp = sp.Symbol('Gp')
ham_eq = 2 * (1 - Gs**2) / Rs**2 - 4 * Gp / (As * Rs) + 4 * KAs * KBs_ + 2 * KBs_**2 - 16 * sp.pi * rhos
Gp_sol = sp.solve(ham_eq, Gp)[0]
# U' = -R' K_B - R K_B'; K_B' from momentum: K_B' = (R'/R)(KA-KB) - 4 pi j_r, j_r = A J, R' = A Gamma
KBp = (As * Gs / Rs) * (KAs - KBs_) - 4 * sp.pi * As * Js
Up = -As * Gs * KBs_ - Rs * KBp
dRtp_dl = 2 * (Up + Gp_sol) / As
on_root = dRtp_dl.subs(Us, -Gs)
check("sec.7: d(R theta+)/dl = (1/R)(1 - 8 pi R^2 (rho - J)) on theta+ = 0",
      on_root - (1 - 8 * sp.pi * Rs**2 * (rhos - Js)) / Rs)

# ---------------------------------------------------------------- F6 FLRW / FKG
aa, H, kk_ = sp.symbols('a H k')
for k, Sk in ((0, rr), (1, sp.sin(rr)), (-1, sp.sinh(rr))):
    Rf = aa * Sk
    Af = aa
    Gf = sp.diff(Rf, rr) / Af
    R3f = 2 * (1 - Gf**2) / Rf**2 - 4 * sp.diff(Gf, rr) / (Af * Rf)
    check(f"F6 3R = 6k/a^2 for k={k}", R3f - 6 * k / aa**2)
    check(f"F6 Gamma = S_k' for k={k}", Gf - sp.diff(Sk, rr))
    # Hamiltonian with K_A=K_B=-H: 6H^2 + 6k/a^2 = 16 pi rho -> H^2 + k/a^2 = 8pi rho/3
    rhof = (R3f + 6 * H**2) / (16 * sp.pi)
    check(f"F6 Friedmann k={k}", H**2 + k / aa**2 - sp.Rational(8, 3) * sp.pi * rhof)
    Mf = Rf / 2 * (1 - Gf**2 + (Rf * H)**2)
    check(f"FLRW M_MS = 4pi/3 rho R^3 (k={k})", Mf - sp.Rational(4, 3) * sp.pi * rhof * Rf**3)

# FKG: d/dt of Friedmann with KG gives Hdot = -4 pi phidot^2 + k/a^2
tt = sp.Symbol('tt')
av = sp.Function('a')(tt)
fv = sp.Function('f')(tt)
Hv = sp.diff(av, tt) / av
kg = sp.diff(fv, tt, 2) + 3 * Hv * sp.diff(fv, tt) + m**2 * fv
for k in (0, 1, -1):
    fr = Hv**2 + k / av**2 - sp.Rational(8, 3) * sp.pi * (sp.diff(fv, tt)**2 / 2 + m**2 * fv**2 / 2)
    dfr = sp.diff(fr, tt)
    # substitute phi'' from KG and use Friedmann -> dfr = 2H (Hdot + 4 pi phidot^2 - k/a^2)
    dfr = dfr.subs(sp.diff(fv, tt, 2), sp.solve(kg, sp.diff(fv, tt, 2))[0])
    Hdot = sp.diff(Hv, tt)
    check(f"F6 Hdot = -4 pi phidot^2 + k/a^2 (k={k})",
          sp.simplify(dfr - 2 * Hv * (Hdot + 4 * sp.pi * sp.diff(fv, tt)**2 - k / av**2)))

# ---------------------------------------------------------------- Painleve-Gullstrand (used in tests)
Mb = sp.Symbol('M', positive=True)
bPG = sp.sqrt(2 * Mb / r)
KA_pg = sp.diff(bPG, r)                      # K_rr = D_r beta_r, alpha = 1, A = 1
KB_pg = bPG / r
check("PG Hamiltonian vacuum", 4 * KA_pg * KB_pg + 2 * KB_pg**2)
check("PG momentum vacuum", sp.diff(KB_pg, r) - (KA_pg - KB_pg) / r)
check("PG M_MS = M", r / 2 * (1 - 1 + r**2 * KB_pg**2) - Mb)
print("all symbolic checks passed")

# PG K_A from the general 4-metric K_ab (independent of the ADM shortcut above)
subs_pg = {al: 1, A: 1, R: r, be: sp.sqrt(2 * Mb / r)}
KA_gen = sp.simplify(KA.subs(subs_pg).doit())
KB_gen = sp.simplify(KB.subs(subs_pg).doit())
check("PG K_A from 4-metric", KA_gen - KA_pg)
check("PG K_B from 4-metric", KB_gen - KB_pg)
print("all symbolic checks passed (incl. PG)")

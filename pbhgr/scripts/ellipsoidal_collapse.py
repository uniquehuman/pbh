"""Bond-Myers (1996) ellipsoidal collapse of a homogeneous ellipsoid in a matter-dominated background, external tide
in linear theory, Zel'dovich initial conditions (lambda_1 >= lambda_2 >= lambda_3, delta = sum lambda,
e = (l1 - l3)/(2 delta), p = (l1 + l3 - 2 l2)/(2 delta)).  Each axis is frozen when it has collapsed to f_freeze of
its maximum; the collapse times t_1 <= t_2 <= t_3 are reported relative to the spherical collapse time of the same
delta (e = p = 0).  Then the distribution of (e, p) for peaks of height nu (Sheth-Mo-Tormen 2001, eq. A3) gives the
distribution of t_3/t_sph, i.e. the expected shift of the deadline map with oblateness/prolateness.
Usage: python3 ellipsoidal_collapse.py [nu=4] [f_freeze=0.01]"""
import sys, numpy as np
from scipy.integrate import solve_ivp, quad
nu = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
f_freeze = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

from scipy.special import elliprd
def b_coeffs(R):
    """Interior shape coefficients b_i = R1 R2 R3 int_0^inf dtau / ((R_i^2+tau) prod_j sqrt(R_j^2+tau)) - 2/3
    = (2/3) R1 R2 R3 R_D(R_j^2, R_k^2, R_i^2) - 2/3 (Carlson's R_D)."""
    R = np.asarray(R, dtype=float); pr = np.prod(R); x = R**2
    return np.array([2.0 / 3.0 * pr * elliprd(x[1], x[2], x[0]) - 2.0 / 3.0,
                     2.0 / 3.0 * pr * elliprd(x[0], x[2], x[1]) - 2.0 / 3.0,
                     2.0 / 3.0 * pr * elliprd(x[0], x[1], x[2]) - 2.0 / 3.0])

def collapse_times(e, p, delta_i=1e-2, t_i=1.0):
    """EdS: a = (t/t_i)^{2/3}, rho_b = 1/(6 pi G t^2) with G = 1.  Returns (t1, t2, t3) in units of t_i."""
    lam = delta_i * np.array([(1 + 3 * e + p) / 3, (1 - 2 * p) / 3, (1 - 3 * e + p) / 3])   # l1 >= l2 >= l3
    R0 = 1.0 - lam                                        # Zel'dovich at t_i (a_i = 1), comoving unit size
    V0 = (2.0 / (3.0 * t_i)) * (1.0 - 2.0 * lam)          # dR/dt = H (1 - 2 lambda) for D ~ a
    frozen = [False] * 3; Rfreeze = [None] * 3; tcol = [np.nan] * 3; Rmax = R0.copy()
    def rhs(t, y):
        R = np.maximum(y[:3].copy(), 1e-9); V = y[3:].copy()
        for i in range(3):
            if frozen[i]: R[i] = Rfreeze[i]
        a = (t / t_i) ** (2.0 / 3.0); rho_b = 1.0 / (6 * np.pi * t**2)
        delta = a**3 / np.prod(R) - 1.0                   # Zel'dovich volume already carries delta_i
        b = b_coeffs(R)
        lam_ext = (lam - delta_i / 3) * a                 # linear external tide
        acc = -4 * np.pi * rho_b * R * ((1 + delta) / 3 + 0.5 * b * delta + lam_ext)
        for i in range(3):
            if frozen[i]: acc[i] = 0.0; V[i] = 0.0
        return np.concatenate([V, acc])
    y = np.concatenate([R0, V0]); t = t_i
    t_end = 1e5 * t_i
    while t < t_end and not all(frozen):
        def ev(tt, yy):
            return min(yy[i] - f_freeze * Rmax[i] for i in range(3) if not frozen[i])
        ev.terminal = True; ev.direction = -1
        sol = solve_ivp(rhs, (t, t_end), y, events=ev, rtol=1e-8, atol=1e-12, method='DOP853', max_step=0.02 * t + 1e-9)
        t = sol.t[-1]; y = sol.y[:, -1]
        Rmax = np.maximum(Rmax, y[:3])
        if sol.status == 1:
            i = int(np.argmin([y[j] - f_freeze * Rmax[j] if not frozen[j] else np.inf for j in range(3)]))
            frozen[i] = True; Rfreeze[i] = y[i]; tcol[i] = t
        else:
            break
    return np.array(tcol) / t_i

t_sph = collapse_times(0.0, 0.0)[2]
print(f"nu={nu}, f_freeze={f_freeze}: spherical collapse (top-hat) at t_sph = {t_sph:.1f} t_i for delta_i = 1e-2 (top-hat expectation: t_i (1.686/delta_i)^1.5 = {(1.686/1e-2)**1.5:.0f} t_i)")
print("e     p     | t1/t_sph  t2/t_sph  t3/t_sph")
grid = []
for e in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
    for p in (-0.5 * e, 0.0, 0.5 * e) if e > 0 else (0.0,):
        tc = collapse_times(e, p) / t_sph
        grid.append((e, p, tc)); print(f"{e:4.2f} {p:5.2f} | {tc[0]:8.3f} {tc[1]:8.3f} {tc[2]:8.3f}")
# Zel'dovich (e, p) distribution for a peak of height nu (Sheth, Mo & Tormen 2001, eq. A3)
def p_ep(e, p):
    return 1125.0 / np.sqrt(10 * np.pi) * e * (e**2 - p**2) * nu**5 * np.exp(-2.5 * nu**2 * (3 * e**2 + p**2))
es = np.linspace(0.002, 0.6, 300); pe = np.array([quad(lambda p: p_ep(e, p), -e, e)[0] for e in es])
pe /= np.trapezoid(pe, es); cdf = np.cumsum(pe) * (es[1] - es[0])
e_med, e_90 = es[np.searchsorted(cdf, 0.5)], es[np.searchsorted(cdf, 0.9)]
print(f"\nZel'dovich ellipticity for nu={nu}: median e = {e_med:.3f}, 90th percentile e = {e_90:.3f}, most probable e ~ {es[np.argmax(pe)]:.3f}")
for lab, ee in (("median", e_med), ("90 %", e_90)):
    tc = collapse_times(ee, 0.0) / t_sph
    print(f"  e = {ee:.3f} (p = 0): t1/t_sph = {tc[0]:.3f}, t3/t_sph = {tc[2]:.3f}")

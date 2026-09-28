"""Own Friedmann-Klein-Gordon integrator (homogeneous massive scalar, V = m^2 phi^2 / 2, G = c = 1).

Formulation 'Hdot' (default, used for the slices):
    phi' = dphi,  dphi' = -3 H dphi - m^2 phi,  (ln a)' = H,  H' = -4 pi dphi^2
The Friedmann constraint H^2 = (8 pi/3) rho is NOT imposed; F_res = H^2 - 8 pi rho / 3 is a
diagnostic (controlled by the ODE tolerance).
Formulation 'constraint' (cross-check only): H = +sqrt(8 pi rho / 3).

Equations: standard (e.g. Mukhanov, Physical Foundations of Cosmology, eqs. 5.21-5.22 with
8 pi G = 8 pi); the 3+1 embedding (R = a r, A = a, K^r_r = K^th_th = -H, Pi = dphi/dt) follows
from verification_plan.md section 2 and derive_sympy.py (RS2: FLRW).
"""
import numpy as np
from scipy.integrate import solve_ivp


class FKG:
    def __init__(self, m=1.0, H0_over_m=0.2, t_end=40.0, rtol=1e-13, atol=1e-16, formulation="Hdot"):
        self.m = m
        self.formulation = formulation
        H0 = H0_over_m * m
        phi0 = H0 / (m * np.sqrt(4 * np.pi / 3))
        if formulation == "Hdot":
            y0 = [phi0, 0.0, 0.0, H0]
            f = self._rhs_hdot
        else:
            y0 = [phi0, 0.0, 0.0]
            f = self._rhs_con
        self.sol = solve_ivp(f, (0.0, t_end), y0, method="DOP853", rtol=rtol, atol=atol, dense_output=True)
        assert self.sol.success

    def _rhs_hdot(self, t, y):
        phi, dphi, lna, H = y
        return [dphi, -3 * H * dphi - self.m**2 * phi, H, -4 * np.pi * dphi**2]

    def _H_con(self, phi, dphi):
        return np.sqrt(8 * np.pi / 3 * (0.5 * dphi**2 + 0.5 * self.m**2 * phi**2))

    def _rhs_con(self, t, y):
        phi, dphi, lna = y
        H = self._H_con(phi, dphi)
        return [dphi, -3 * H * dphi - self.m**2 * phi, H]

    def state(self, t):
        y = self.sol.sol(t)
        if self.formulation == "Hdot":
            phi, dphi, lna, H = y
        else:
            phi, dphi, lna = y
            H = self._H_con(phi, dphi)
        rho = 0.5 * dphi**2 + 0.5 * self.m**2 * phi**2
        p = 0.5 * dphi**2 - 0.5 * self.m**2 * phi**2
        return dict(t=t, phi=phi, dphi=dphi, a=np.exp(lna), H=H, rho=rho, p=p,
                    F_res=H**2 - 8 * np.pi * rho / 3)

    def slice(self, t, r):
        s = self.state(t)
        n = len(r)
        one = np.ones(n)
        return dict(r=r, A=s["a"] * one, R=s["a"] * r, KA=-s["H"] * one, KB=-s["H"] * one,
                    phi=s["phi"] * one, Pi=s["dphi"] * one, m=self.m, state=s)

    def period_average(self, t0, nph=64):
        """Trapezoid average over one period 2 pi/m starting at t0; returns (<rho>, <p>, times)."""
        ts = t0 + np.linspace(0.0, 2 * np.pi / self.m, nph + 1)
        s = self.state(ts)
        w = np.ones(nph + 1)
        w[0] = w[-1] = 0.5
        w /= w.sum()
        return float(np.dot(w, s["rho"])), float(np.dot(w, s["p"])), ts

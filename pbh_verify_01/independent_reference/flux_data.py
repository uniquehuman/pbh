"""Own constraint-ODE solver: spherically symmetric Einstein-Klein-Gordon initial data with
nonzero radial momentum flux, areal coordinate R = r.

Derivation (own; verification_plan.md section 2):
  prescribe phi(r), Pi(r), K_B(r) (even functions);
  momentum constraint  K_B' - (1/r)(K_A - K_B) + 4 pi j_r = 0,  j_r = -Pi phi'
      => K_A = K_B + r K_B' - 4 pi r Pi phi'                          (algebraic)
  mass function M = (r/2)(1 - A^-2 + r^2 K_B^2)  =>  A^-2 = 1 - 2M/r + r^2 K_B^2
  Hamiltonian constraint in mass form (equivalent to H = 0 when the momentum constraint holds,
  identity checked symbolically in derive_sympy.py RS1):  M' = 4 pi r^2 (rho + U j_r), U = -r K_B,
  with rho = Pi^2/2 + phi'^2 A^-2 / 2 + m^2 phi^2 / 2:
      M' = 4 pi r^2 [Pi^2/2 + phi'^2 (1 + r^2 K_B^2)/2 + m^2 phi^2/2 + r K_B Pi phi'] - 4 pi r phi'^2 M
  (linear, no division by r), M(0) = 0.
Integrated with DOP853 together with
  M_rho,ref'  = 4 pi r^2 rho            (naive mass, R' = 1)
  M_flux,ref' = 4 pi r^2 U j_r = 4 pi r^3 K_B Pi phi'
  M_prop,ref' = 4 pi r^2 A rho
"""
import numpy as np
from scipy.integrate import solve_ivp

from .constraints_ref import exact_diag


class FluxData:
    def __init__(self, phi0=0.25, wphi=1.0, Pi0=0.15, wPi=np.sqrt(2.0), kb0=0.4, wkb=2.0, m=1.0,
                 r_max=8.0, sPi=1.0, sKB=1.0, rtol=1e-13, atol=1e-18):
        self.p = dict(phi0=phi0, wphi=wphi, Pi0=sPi * Pi0, wPi=wPi, kb0=sKB * kb0, wkb=wkb, m=m, r_max=r_max)
        self.m = m
        self.r_max = r_max
        self.sol = solve_ivp(self._rhs, (0.0, r_max * 1.0000001), [0.0, 0.0, 0.0, 0.0], method="DOP853",
                             rtol=rtol, atol=atol, dense_output=True)
        assert self.sol.success

    # prescribed even functions and analytic derivatives
    def phi(self, r, k=0):
        a, w = self.p["phi0"], self.p["wphi"]
        g = a * np.exp(-r**2 / w**2)
        if k == 0:
            return g
        if k == 1:
            return -2 * r / w**2 * g
        return (4 * r**2 / w**4 - 2 / w**2) * g

    def Pi(self, r):
        return self.p["Pi0"] * np.exp(-r**2 / self.p["wPi"]**2)

    def KB(self, r, k=0):
        a, w = self.p["kb0"], self.p["wkb"]
        g = a * np.exp(-r**2 / w**2)
        if k == 0:
            return g
        if k == 1:
            return -2 * r / w**2 * g
        return (4 * r**2 / w**4 - 2 / w**2) * g

    def _rhs(self, r, y):
        M = y[0]
        m = self.m
        dph, Pi, KB, ph = self.phi(r, 1), self.Pi(r), self.KB(r), self.phi(r)
        dM = 4 * np.pi * r**2 * (0.5 * Pi**2 + 0.5 * dph**2 * (1 + r**2 * KB**2) + 0.5 * m**2 * ph**2
                                 + r * KB * Pi * dph) - 4 * np.pi * r * dph**2 * M
        Ainv2 = self._Ainv2(r, M, KB)
        rho = 0.5 * Pi**2 + 0.5 * dph**2 * Ainv2 + 0.5 * m**2 * ph**2
        dMrho = 4 * np.pi * r**2 * rho
        dMflux = 4 * np.pi * r**3 * KB * Pi * dph
        dMprop = 4 * np.pi * r**2 * rho / np.sqrt(Ainv2)
        return [dM, dMrho, dMflux, dMprop]

    @staticmethod
    def _Ainv2(r, M, KB):
        r = np.asarray(r, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            Mr = np.where(r > 0, M / np.where(r > 0, r, 1.0), 0.0)
        return 1 - 2 * Mr + r**2 * KB**2

    def values(self, r):
        r = np.asarray(r, float)
        Y = self.sol.sol(r)
        M, Mrho, Mflux, Mprop = Y
        M = np.where(r == 0, 0.0, M)
        dph, d2ph, Pi, KB, dKB, ph = self.phi(r, 1), self.phi(r, 2), self.Pi(r), self.KB(r), self.KB(r, 1), self.phi(r)
        B = self._Ainv2(r, M, KB)
        A = 1 / np.sqrt(B)
        dM = np.array([self._rhs(ri, [Mi, 0, 0, 0])[0] for ri, Mi in zip(r, M)])
        with np.errstate(divide="ignore", invalid="ignore"):
            dB = np.where(r > 0, -2 * dM / np.where(r > 0, r, 1) + 2 * M / np.where(r > 0, r, 1)**2, 0.0) \
                + 2 * r * KB**2 + 2 * r**2 * KB * dKB
        dA = -0.5 * B**-1.5 * dB
        KA = KB + r * dKB - 4 * np.pi * r * Pi * dph
        rho = 0.5 * Pi**2 + 0.5 * dph**2 * B + 0.5 * self.m**2 * ph**2
        jr = -Pi * dph
        Srr = 0.5 * Pi**2 + 0.5 * dph**2 * B - 0.5 * self.m**2 * ph**2
        Sthth = 0.5 * Pi**2 - 0.5 * dph**2 * B - 0.5 * self.m**2 * ph**2
        return dict(r=r, A=A, dA=dA, R=r.copy(), dR=np.ones_like(r), d2R=np.zeros_like(r), KA=KA, KB=KB, dKB=dKB,
                    phi=ph, dphi=dph, Pi=Pi, rho=rho, jr=jr, Srr=Srr, Sthth=Sthth,
                    M=M, dM=dM, M_rho=Mrho, M_flux=Mflux, M_prop=Mprop, J=jr / A)

    def exact(self, r):
        v = self.values(r)
        d = exact_diag(v["A"], v["dA"], v["R"], v["dR"], v["d2R"], v["KA"], v["KB"], v["dKB"], v["rho"], v["jr"])
        d.update(v)
        d["M_ode"] = v["M"]
        return d

    def transformed_values(self, x, fmap, fprime):
        """Same data in coordinate x, r = fmap(x): A_x = A f', R_x = f, j_x = j_r f', scalars unchanged."""
        rr = fmap(x)
        v = self.values(rr)
        fp = fprime(x)
        out = dict(v)
        out.update(r=x, A=v["A"] * fp, R=rr, jr=v["jr"] * fp)
        return out

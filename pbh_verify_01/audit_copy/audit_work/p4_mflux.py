"""P4: sign of the flux mass M_flux = int 4 pi R^2 U j_r dr (contract §6) and M_int = M_MS on the constraints."""
import sys; sys.path.insert(0, "tests/dev")
import numpy as np
import devbg as bg
from sphdiag import SphericalSlice, Projections, diagnose
for N in (100, 200, 400):
    r = np.linspace(0.3, 4.0, N)
    kw, ex = bg.generic_slice(r)          # rho, j_r defined by H = M = 0 (derived independently in devbg)
    d = diagnose(SphericalSlice(center="none", **kw), Projections(ex["rho"], ex["j_r"], 0*r, 0*r))
    ref_flux = np.concatenate([[0], np.cumsum(0.5*np.diff(r)*((4*np.pi*d.R**2*d.U*ex["j_r"])[1:] + (4*np.pi*d.R**2*d.U*ex["j_r"])[:-1]))])
    print(N, "status", d.status["constraints"], "| max|M_int - M| =", f"{np.max(np.abs(d.M_int - ex['M'])):.3e}",
          "| M_flux[-1] =", f"{d.M_flux[-1]:.5f}", " trapz(+4piR^2 U j_r) =", f"{ref_flux[-1]:.5f}",
          "| max|M[0]+M_rho-M_flux - M| =", f"{np.max(np.abs(d.M[0]+d.M_rho-d.M_flux-ex['M'])):.3e}")

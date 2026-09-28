"""P8: cell centre + wrong R parity -> O(1), non-convergent errors near the centre with a benign status."""
import sys; sys.path.insert(0, "tests/dev")
import numpy as np
import devbg as bg
from sphdiag import SphericalSlice, Projections, diagnose
for N in (50, 100, 200, 400):
    r = 4.0*(np.arange(N)+0.5)/N
    kw, ex = bg.generic_slice(r)
    d = diagnose(SphericalSlice(center="cell", **kw), Projections(ex["rho"], ex["j_r"], 0*r, 0*r))
    print(N, {k: d.status[k] for k in ("data","constraints","geometry")}, "n_viol", d.status["n_violated"],
          "| Gamma err[0:3]", np.round(np.abs(d.Gamma[:3]-ex["Gamma"][:3]), 3),
          "| compactness[0:2]", np.round(d.compactness[:2], 3), "exact", np.round(2*ex["M"][:2]/d.R[:2], 5),
          "| Gamma0_extrap", round(d.center_checks["Gamma0"], 3))
# collapsing centre: U = -R*KB with KB > 0 large; exact Gamma = 1 (R = r, A = 1): are false trapped spheres produced?
N = 60; r = 2.0*(np.arange(N)+0.5)/N; o = np.ones_like(r)
for kb in (1.0, 2.0, 30.0):
    d = diagnose(SphericalSlice(r, o, r.copy(), kb*o, kb*o, center="cell"))
    print("KB=", kb, "exact Rtheta_+[0] =", 2*(-r[0]*kb+1), " code:", round(d.Rtheta_plus[0],3), d.sphere_class[:3], d.status["geometry"])

"""P6: Minkowski hyperboloidal slice far out (Rtheta_- -> 0^-): false marginal spheres / UNCERTAIN?"""
import sys; sys.path.insert(0, "tests/dev")
import numpy as np
import devbg as bg
from sphdiag import SphericalSlice, Projections, diagnose
for rmax, N in ((1e3, 400), (1e5, 400), (1e5, 100), (1e7, 200)):
    r = np.geomspace(0.5, rmax, N)
    kw, ex = bg.minkowski_hyperboloid(r)
    kw.pop("alpha"); kw.pop("beta")
    d = diagnose(SphericalSlice(center="none", **kw), Projections(0*r,0*r,0*r,0*r))
    cls, cnt = np.unique(d.sphere_class, return_counts=True)
    print(f"rmax={rmax:.0e} N={N}:", d.status["geometry"], d.status["constraints"], dict(zip(cls, cnt)),
          "min|Rth-|=%.2e max tau=%.2e" % (np.min(np.abs(d.Rtheta_minus)), np.max(d.tau_minus)))
    # uniform grid, as in test
for rmax, N in ((1e3, 2000), (1e5, 2000)):
    r = np.linspace(0.5, rmax, N)
    kw, ex = bg.minkowski_hyperboloid(r); kw.pop("alpha"); kw.pop("beta")
    d = diagnose(SphericalSlice(center="none", **kw), Projections(0*r,0*r,0*r,0*r))
    cls, cnt = np.unique(d.sphere_class, return_counts=True)
    print(f"uniform rmax={rmax:.0e} N={N}:", d.status["geometry"], d.status["constraints"], dict(zip(cls, cnt)), "err Rth- max %.2e" % np.max(np.abs(d.Rtheta_minus-ex['Rtm'])))

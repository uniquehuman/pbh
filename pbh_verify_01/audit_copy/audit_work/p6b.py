import sys; sys.path.insert(0, "tests/dev")
import numpy as np
import devbg as bg
from sphdiag import SphericalSlice, Projections, diagnose
r = np.linspace(0.5, 1e5, 2000)
kw, ex = bg.minkowski_hyperboloid(r); kw.pop("alpha"); kw.pop("beta")
d = diagnose(SphericalSlice(center="none", **kw), Projections(0*r,0*r,0*r,0*r))
print(d.status["n_violated"])
for nm in ("ham","mom","hamM"):
    st = d[nm+"_status"]; i = np.flatnonzero(st=="VIOLATED")
    for j in i[:5]:
        print(nm, j, r[j], d[nm][j], d["sigma_"+nm][j], d[nm+"_rel"][j])

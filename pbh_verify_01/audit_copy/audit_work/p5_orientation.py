"""P5: AH detection vs orientation of the radial label (contract §2: r is 'any monotone label').
Schwarzschild PG slice, M=1, relabelled x = 20 - R (R decreasing in x)."""
import sys; sys.path.insert(0, "tests/dev")
import numpy as np
import devbg as bg
from sphdiag import SphericalSlice, Projections, diagnose
R = bg.stretched(400, 0.4, 12.0)
kw, ex = bg.schwarzschild_pg(R, 1.0)
d = diagnose(SphericalSlice(center="none", **kw), Projections(0*R,0*R,0*R,0*R))
print("increasing label :", d.status["geometry"], [ (x["which"], round(x["R"],6), x["type"], x["ah_candidate"]) for x in d.roots])
Rr = R[::-1].copy(); x = 20.0 - Rr
kw2 = dict(r=x, A=np.ones_like(x), R=Rr, KA=kw["KA"][::-1].copy(), KB=kw["KB"][::-1].copy())
d2 = diagnose(SphericalSlice(center="none", **kw2), Projections(0*x,0*x,0*x,0*x))
print("decreasing label :", d2.status["geometry"], [ (x_["which"], round(x_["R"],6), x_["type"], x_["ah_candidate"], x_["inner_sign"], x_["outer_sign"]) for x_ in d2.roots])
print("M identical:", np.allclose(d2.M[::-1], d.M, atol=1e-6), " classes (first, last):", d2.sphere_class[0], d2.sphere_class[-1])

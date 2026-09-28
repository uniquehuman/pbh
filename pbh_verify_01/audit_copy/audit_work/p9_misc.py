"""P9: white hole, contracting FLRW, interior R = 0, M_int check, NaN only in unused fields."""
import sys; sys.path.insert(0, "tests/dev")
import numpy as np
import devbg as bg
from sphdiag import SphericalSlice, Projections, ScalarField, diagnose
def show(tag, d):
    print(tag, "|", d.status["data"], d.status["constraints"], d.status["geometry"], "|",
          [(x["which"], round(x["R"],4), x["type"], x["ah_candidate"], str(x["inner_sign"]), str(x["outer_sign"])) for x in d.roots])
r = bg.stretched(300, 0.4, 12.0)
kw, ex = bg.schwarzschild_pg(r, 1.0)
kwW = dict(kw, KA=-kw["KA"], KB=-kw["KB"])          # time reverse: white hole (outgoing PG)
show("white hole PG      ", diagnose(SphericalSlice(center="none", **kwW), Projections(0*r,0*r,0*r,0*r)))
H = -1.5; x = np.linspace(0.1, 2.0, 200); o = np.ones_like(x)
show("contracting FLRW   ", diagnose(SphericalSlice(x, o, x.copy(), -H*o, -H*o, center="none"), Projections(3*H**2/(8*np.pi)*o, 0*x,0*x,0*x)))
# closed FLRW expanding beyond equator (Gamma<0): second PAST_MOTS on theta_+ with Gamma<0
chi = np.linspace(0.05, 2.9, 300); o = np.ones_like(chi); H=1.5
show("closed FLRW H>0    ", diagnose(SphericalSlice(chi, o, np.sin(chi), -H*o, -H*o, center="none"), Projections(3*(H**2+1)/(8*np.pi)*o, 0*chi,0*chi,0*chi)))
# interior R = 0 point (a crushed sphere in the middle of the grid), center='none'
x = np.linspace(0.0, 2.0, 81); R = np.abs(x - 1.0); o = np.ones_like(x)
d = diagnose(SphericalSlice(x, o, R, 0*x, 0*x, center="none"), Projections(0*x,0*x,0*x,0*x))
print("interior R=0       |", d.status["data"], d.status["constraints"], d.status["geometry"], "classes around:", d.sphere_class[38:43], "warnings:", d.status["warnings"], "errors:", d.status["errors"])
# M_int != M_MS (fluxless data, rho inconsistent at 1e-3 level) -- is M_int drift used in any status?

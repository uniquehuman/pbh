"""P1: parity of R used for R' at a regular centre (flat space, FLRW)."""
import numpy as np
from sphdiag import SphericalSlice, Projections, diagnose, RadialOperator
r = np.linspace(0, 2, 41)
one = np.ones_like(r)
for c, rr in (("vertex", r), ("cell", (np.arange(40)+0.5)*0.05)):
    o = np.ones_like(rr)
    d = diagnose(SphericalSlice(rr, o, rr.copy(), 0*rr, 0*rr, center=c), Projections(0*rr,0*rr,0*rr,0*rr))
    print(c, "Gamma[:4] =", np.round(d.Gamma[:4], 6), " exact 1")
    print(c, "M[:4] =", np.round(d.M[:4], 6), " exact 0; Rtheta_minus[:4]", np.round(d.Rtheta_minus[:4],4))
    print(c, "status", {k: d.status[k] for k in ("data","constraints","geometry")}, "cc Gamma0", d.center_checks["Gamma0"])
    print(c, "classes[:4]", d.sphere_class[:4], "n_violated", d.status["n_violated"], "failures", d.status["numerical_failures"])
    print(c, "M_over_R3[0]", d.M_over_R3[0])
op = RadialOperator(r, "vertex")
print("op.d(r,+1)[0] =", op.d(r, +1)[0], " op.d(r,-1)[0] =", op.d(r, -1)[0])

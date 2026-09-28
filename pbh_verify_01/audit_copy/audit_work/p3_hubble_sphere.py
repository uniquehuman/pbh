"""P3: expanding flat FLRW -- the Hubble sphere (theta_- = 0, theta_+ > 0) must be PAST_MOTS, never an AH candidate."""
import numpy as np
from sphdiag import SphericalSlice, Projections, diagnose
H = 1.5
r = np.linspace(0.1, 2.0, 200)
o = np.ones_like(r)
rho = 3*H**2/(8*np.pi)*o
d = diagnose(SphericalSlice(r, o, r.copy(), -H*o, -H*o, center="none"), Projections(rho, 0*r, 0*r, 0*r))
for x in d.roots:
    print({k: x[k] for k in ("which","R","type","ah_candidate","other_sign","inner_sign","outer_sign")})
print("status:", d.status["geometry"], " outer_ah_candidate R =", None if d.outer_ah_candidate is None else d.outer_ah_candidate["R"])
# PBH-like: Schwarzschild-PG interior BH (R<2M) embedded ... simplest: synthetic slice with a real AH at r=1
# and a Hubble-like theta_- root further out (U = u(r), Gamma = 1): u = -2 + r for r<~1.6, then grows past 1
r = np.linspace(0.2, 4.0, 400)
u = -2 + r + 0.0*r
u = np.where(r < 1.5, -2 + r, -0.5 + 1.0*(r - 1.5)**1*1.0 + 0.0)  # crosses -1 at r=1 (theta_+), +1 at r=3 (theta_-)
d = diagnose(SphericalSlice(r, np.ones_like(r), r.copy(), 0*r, -u/r, center="none"))
for x in d.roots:
    print({k: (round(x[k],4) if isinstance(x[k], float) else x[k]) for k in ("which","r","type","ah_candidate","inner_sign","outer_sign")})
print("status:", d.status["geometry"], " outer_ah_candidate r =", d.outer_ah_candidate["r"], d.outer_ah_candidate["which"], d.outer_ah_candidate["type"])

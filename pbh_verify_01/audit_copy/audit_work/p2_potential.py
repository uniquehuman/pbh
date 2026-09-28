"""P2: scalar potential normalisation V = m^2 phi^2 / 2 (contract §1, F3, F5)."""
import numpy as np
from sphdiag import SphericalSlice, ScalarField, diagnose
m, phi0 = 2.0, 0.2
r = np.linspace(0.5, 3, 60)                     # center='none': isolates from the centre parity issue
sf = ScalarField(phi0 + 0*r, 0*r, m)
o = np.ones_like(r)
p = sf.projections(SphericalSlice(r, o, r.copy(), 0*r, 0*r, center="none"))
print("rho =", p.rho[0], " expected V = 0.5 m^2 phi^2 =", 0.5*m**2*phi0**2)
# homogeneous slice (pure potential, Pi = 0: de Sitter-like moment): H^2 = 8 pi rho/3 with the CORRECT rho
rho_true = 0.5*m**2*phi0**2
H = np.sqrt(8*np.pi/3*rho_true)
d = diagnose(SphericalSlice(r, o, r.copy(), -H*o, -H*o, center="none"), sf)
print("consistent FKG slice: constraints =", d.status["constraints"], " ham_rel max =", np.nanmax(d.ham_rel))
H2 = np.sqrt(8*np.pi/3*2*rho_true)
d2 = diagnose(SphericalSlice(r, o, r.copy(), -H2*o, -H2*o, center="none"), sf)
print("slice built with V=m^2phi^2: constraints =", d2.status["constraints"], " ham_rel max =", np.nanmax(d2.ham_rel))

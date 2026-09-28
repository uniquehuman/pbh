import numpy as np
from sphdiag import SphericalSlice, Projections, diagnose
Ms = 0.1; N = 81; h = 2.8/(N-1)
for f in np.linspace(0, 0.95, 20):
    r = np.linspace(0.2, 3.0, N); r = r + (1 - r[29]) + f*h - 1e-9  # shell at fraction f of a cell after node 29
    A = np.where(r < 1, 1.0, 1/np.sqrt(1-2*Ms/r))
    d = diagnose(SphericalSlice(r, A, r.copy(), 0*r, 0*r, center="none"), Projections(0*r,0*r,0*r,0*r))
    print("f=%.2f" % f, d.status["constraints"], d.status["n_violated"], "max|ham|/(10 sig)=%.2f" % np.nanmax(np.abs(d.ham)/(10*d.sigma_ham)),
          "max|hamM|/(10 sig)=%.2f" % np.nanmax(np.abs(d.hamM)/(10*d.sigma_hamM)))

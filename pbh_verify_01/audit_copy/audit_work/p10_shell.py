"""P10: a missing thin mass shell (delta-function in rho) -- vacuum Projections, but the metric has a mass jump.
R = r, K = 0, A = 1 (r<1), A = (1-2M/r)^(-1/2) (r>1). Exact: H = -16 pi * (shell) at r = 1 only; M_MS jumps 0 -> M."""
import numpy as np
from sphdiag import SphericalSlice, Projections, diagnose
Ms = 0.1
for N in (41, 81, 161, 321):
    r = np.linspace(0.2, 3.0, N) + 0.0137   # shell between nodes
    A = np.where(r < 1, 1.0, 1/np.sqrt(1-2*Ms/r))
    d = diagnose(SphericalSlice(r, A, r.copy(), 0*r, 0*r, center="none"), Projections(0*r,0*r,0*r,0*r))
    i = np.argmin(np.abs(r-1))
    q = np.abs(d.ham)/(10*d.sigma_ham)
    print(N, d.status["constraints"], d.status["n_violated"], "max |ham|/(kappa_c sigma) = %.2f" % np.nanmax(q),
          " M jump seen: M[-1]-M[0] = %.4f" % (d.M[-1]-d.M[0]), " M_int[-1]-M_int[0] = %.4f" % (d.M_int[-1]-d.M_int[0]))

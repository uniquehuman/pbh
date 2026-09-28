"""P7: how large a genuine constraint violation is absorbed by kappa_c*sigma (status CONSISTENT_WITH_TRUNCATION)."""
import sys; sys.path.insert(0, "tests/dev")
import numpy as np
import devbg as bg
from sphdiag import SphericalSlice, Projections, diagnose
for N in (40, 80, 160):
    r = np.linspace(0.3, 4.0, N)
    kw, ex = bg.generic_slice(r)
    for eps in (0.3, 0.1, 0.03, 0.01, 0.003, 1e-3, 1e-4):
        d = diagnose(SphericalSlice(center="none", **kw), Projections(ex["rho"]*(1+eps), ex["j_r"], 0*r, 0*r))
        if d.status["constraints"] != "VIOLATED":
            print(f"N={N}: rho*(1+{eps}) -> {d.status['constraints']}  (max ham_rel={np.nanmax(d.ham_rel):.3f})")
            break
    for eps in (1.0, 0.3, 0.1, 0.03, 0.01, 1e-3):
        d = diagnose(SphericalSlice(center="none", **kw), Projections(ex["rho"], ex["j_r"]*(1+eps), 0*r, 0*r))
        if d.status["constraints"] != "VIOLATED":
            print(f"N={N}: j_r*(1+{eps}) -> {d.status['constraints']}  (max mom_rel={np.nanmax(d.mom_rel):.3f})")
            break
# drop the flux entirely (j_r = 0): mimics a code that forgot the radial flux
for N in (40, 80, 160):
    r = np.linspace(0.3, 4.0, N); kw, ex = bg.generic_slice(r)
    d = diagnose(SphericalSlice(center="none", **kw), Projections(ex["rho"], 0*r, 0*r, 0*r))
    print(f"N={N}: j_r := 0 ->", d.status["constraints"], d.status["n_violated"])

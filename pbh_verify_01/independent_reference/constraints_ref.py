"""Exact (no finite differences) evaluation of the slice diagnostics from values and ANALYTIC
radial derivatives. Formulas: own derivation, verification_plan.md section 2, confirmed
symbolically from the 4-metric in derive_sympy.py (RS1).

Inputs are numpy arrays of equal length:
  A, dA, R, dR, d2R, KA, KB, dKB, and optionally rho, jr (matter projections).
At points with R == 0 the residuals ham, mom are set to NaN (0/0; excluded as in contract §9);
M = 0 and hamM = M' - 0 there.
"""
import numpy as np


def exact_diag(A, dA, R, dR, d2R, KA, KB, dKB, rho=None, jr=None):
    A, dA, R, dR, d2R, KA, KB, dKB = (np.asarray(v, float) for v in (A, dA, R, dR, d2R, KA, KB, dKB))
    G = dR / A
    dG = (d2R * A - dR * dA) / A**2
    U = -R * KB
    dU = -dR * KB - R * dKB
    M = 0.5 * R * (1 - G**2 + U**2)
    dM = 0.5 * dR * (1 - G**2 + U**2) + R * (-G * dG + U * dU)
    out = dict(Gamma=G, dGamma=dG, U=U, M=M, dM=dM, dR=dR,
               Rtp=2 * (U + G), Rtm=2 * (U - G))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["theta_p"] = 2 * (U + G) / R
        out["theta_m"] = 2 * (U - G) / R
        out["M_over_R3"] = M / R**3
    out["theta_p"][R == 0] = np.nan
    out["theta_m"][R == 0] = np.nan
    if rho is not None:
        rho = np.asarray(rho, float)
        jr = np.asarray(jr, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            hT = [2 * (1 - G**2) / R**2, -4 * dG / (A * R), 4 * KA * KB, 2 * KB**2, -16 * np.pi * rho]
            mT = [dKB, -(dR / R) * (KA - KB), 4 * np.pi * jr]
        HT = [dM, -4 * np.pi * R**2 * rho * dR, -4 * np.pi * R**2 * U * jr]
        ham = sum(hT)
        mom = sum(mT)
        hamM = sum(HT)
        z = R == 0
        ham[z] = np.nan
        mom[z] = np.nan
        out.update(ham=ham, mom=mom, hamM=hamM,
                   ham_absterms=sum(np.abs(x) for x in hT),
                   mom_absterms=sum(np.abs(x) for x in mT),
                   hamM_absterms=sum(np.abs(x) for x in HT))
    return out

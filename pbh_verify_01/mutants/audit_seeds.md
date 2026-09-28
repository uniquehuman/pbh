# Secret seeds in audit_copy/ (not visible to the audit agent). Created 2026-09-28 before the audit.
| seed | class | file | change |
|---|---|---|---|
| S1 | M4a (flux sign, integral mass only) | diagnose.py | M_flux = cumint(-4πR²U j_r) |
| S2 | M5 (cosmological marginal → AH candidate) | diagnose.py | ah = rtype != DEGENERATE and Γ>0 and inner=-1 and outer=+1 (drops which=='+' and FUTURE_MOTS) |
| S3 | M3c (potential coefficient) | matter.py | V = m²φ² |
| S4 | M8 (parity of R at centre) | diagnose.py | R' with even ghost parity (+1) |
| S5 | M6 (circular test) | tests/dev/test_dev_api.py::test_fkg_homogeneous_scalar_slice | expected ρ and H computed by the module under test |

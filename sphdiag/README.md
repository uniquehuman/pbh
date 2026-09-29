# sphdiag

Physical diagnostics of spherically symmetric 3+1 slices, implementing
`physics_contract.md` v1.2 (PBH-VERIFY-01; implementation B v0.3). The package only **measures**: it does not
evolve data, solve constraints or impose boundary/regularity conditions.

Needs Python 3.12 and numpy/scipy. Import it with `PYTHONPATH=src`.

## Usage

```python
import numpy as np
from sphdiag import SphericalSlice, ScalarField, VlasovMoments, Projections, pi_from_dtphi, diagnose

r = np.linspace(0.4, 12.0, 400)                      # any strictly increasing label, N >= 8
M = 1.0                                              # Schwarzschild, Painleve-Gullstrand slice
slc = SphericalSlice(r, A=np.ones_like(r), R=r, KA=-0.5*np.sqrt(2*M/r**3),
                     KB=np.sqrt(2*M/r**3), center="none")   # "vertex" | "cell" | "none"
d = diagnose(slc, Projections(0*r, 0*r, 0*r, 0*r))   # matter: ScalarField / VlasovMoments /
                                                     # Projections / None
d.status          # {'data': 'OK', 'constraints': 'CONSISTENT_WITH_TRUNCATION',
                  #  'geometry': 'AH_CANDIDATE', 'marginal_present': ..., ...}
d.outer_ah_candidate["R"], d.outer_ah_candidate["R_lo"], d.outer_ah_candidate["R_hi"]
d.M, d.sigma_M, d.Rtheta_plus, d.sphere_class, d.norms["ham"]["bulk"]["L2vol"]
```

Inputs: `KA = K^r_r` and `KB = K^θ_θ` (mixed components, sign convention K = −3H for an
expanding FLRW slice). `alpha` and `beta` are stored but never used by the diagnostics. You can
pass analytic `dA, dR, dKB`. If you do, σ(R′) = σ(K_B′) = 0, and A′ enters Γ′ = (R″ − ΓA′)/A.
Scalar field: `ScalarField(phi, Pi, m_phi)` with Π = n^a∂_aφ (use `pi_from_dtphi` to convert
from ∂_tφ). Vlasov: `VlasovMoments(E, J, p, q)` gives ρ = E, j_r = A·J, S^r_r = p,
S^θ_θ = q.

`diagnose` never raises on bad data:

- Wrong shapes, a non-monotone `r`, N < 8, NaN/inf, A ≤ 0 or R < 0 give `data = INVALID_INPUT`
  and `geometry = UNDETERMINED`.
- Non-finite outputs at points other than the documented R = 0 points give
  `data = NUMERICAL_FAILURE` and `geometry = UNDETERMINED`.

## Result fields

These are the contract §10 fields plus extras:

| extra field | meaning |
|---|---|
| `sigma_dR, sigma_Gamma, dGamma, sigma_dGamma, dKB, sigma_dKB, dM, sigma_dM` | derivatives and their error estimates |
| `tau_plus, tau_minus, sign_plus, sign_minus` | zero band τ± and resolved signs s± of Rθ± |
| `ham_status, mom_status, hamM_status` | per-point `CONSISTENT` / `VIOLATED` / `INVALID` / `CENTER_EXCLUDED` / `NOT_EVALUATED` |
| `M_over_R3_extrapolated_mask` | points where M/R³ was replaced by its R → 0 limit |
| `status['errors'|'warnings'|'numerical_failures'|'n_violated'|'uncertain_points']` | details |
| root keys `r_lo, r_hi, Gamma, n_crossings, tangential, points` | details of each root |

Root signs (`other_sign`, `inner_sign`, `outer_sign`) are `Sign` objects. A `Sign` is an int
(−1/0/+1) that also compares equal to `'-'`, `'0'` and `'+'`.

## Statuses (contract v1.1)

- `status['constraints']`: `NOT_EVALUATED` (no matter or data != OK) > `VIOLATED` > `UNRESOLVED`
  (σ(X) > 0.1 Σ|T_k| at > 5 % of the points of any residual, or at any point of the
  `center`/`inner` region) > `CONSISTENT_WITH_TRUNCATION`. Per point: `ham_status` etc.
- AH candidate: zero of the outgoing expansion θ_out (θ₊ if Γ > 0, θ₋ if Γ < 0), other
  expansion < 0, θ_out < 0 on the smaller-R side and > 0 on the larger-R side; independent of
  the direction of the r label. `outer_ah_candidate` = largest R.
- [v1.2] Sphere class `UNRESOLVED` when τ₊ > 0.1 or τ₋ > 0.1 (not for `CENTER`). Any such point
  makes the geometry at best `UNCERTAIN` (`AH_CANDIDATE` / `FUTURE_TRAPPED_NO_AH_CANDIDATE`
  keep priority); `status['n_unresolved_points']`; warning if they lie inside (in R) an AH
  candidate. The isolated-`DOUBLY_MARGINAL` exemption needs both neighbours resolved
  `NORMAL`/`NORMAL_REVERSED`/`CENTER`.
- R = 0 is allowed only at r[0] with `center="vertex"`.
- Warnings (`status['warnings']`, plus `center_deviation`, `M_int_mismatch`, `n_M_negative`,
  `n_rel_undefined` keys) never change the data status.
- Norm regions: `all`, `center` (`inner` when `center="none"`), `bulk`, `outer`.

## Numerics

- The derivative stencils use Fornberg weights: 5 points (order 4) for the values and 7 points
  (order 6) for the error estimate σ(f′) = |D⁴f − D⁶f| + ε·Σ|w_k f_k|.
- At a regular centre the parity ghosts are A, K_A, K_B, φ, Π, Γ even and R, U, M odd.
  Stencils are one-sided at the outer edge and for `center="none"`.
- Zero band of Rθ±: τ = max(κ(σ(Rθ) + 2σ_noise(U) + 2σ_noise(Γ)), 1e3 ε(|U| + |Γ|)), where
  σ_noise is the windowed max of the undivided 6th difference divided by √924.
- Integral masses integrate from r[0] with the exact integral of the local cubic interpolant
  (order 4).

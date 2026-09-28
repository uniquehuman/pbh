# implementation_B.md: NUMERICAL IMPLEMENTATION role, architecture B

Pilot PBH-VERIFY-01, package `src/sphdiag`, contract `physics_contract.md` v1.0 (frozen).
Session 2026-09-28, 22:53–23:20 UTC (about 27 minutes of wall-clock time).

**Files I read:** `physics_contract.md`, `derivation.md`, `derivation/output.txt`, the head of
`derivation/backgrounds.py`, `context_audit.md` and `agent_logs/timeline.txt`.

**Files I did not read:** `tests/independent/`, `independent_reference/`, `verification_plan.md`,
`arch_A/`, `mutants/` and `mutation_report.md`. `mutation_report.md` was not on the forbidden list,
but I skipped it so that it cannot compromise independence.

## 1. What was built

| File | Content |
|---|---|
| `fd.py` | Vectorised Fornberg weights. `RadialOperator` builds 5-point (order-4) and 7-point (order-6) first-derivative stencils on the extended grid: 3 parity ghosts for `vertex`/`cell`, none for `none`. Windows are centred where possible and clipped to fully one-sided at the edges. The error estimate is `d_err`. `cumint` integrates exactly the local cubic interpolant through 4 nodes, including the ghosts (order 4). |
| `slice.py` | `SphericalSlice`. It validates the input and never raises. Structural problems (shape, N < 8, a non-monotone or non-finite r, a bad `center`, vertex r[0] ≠ 0, cell r[0] ≤ 0) set `structure_ok = False`. Point-wise problems (NaN/inf, A ≤ 0, R < 0, vertex R[0] ≠ 0) are stored in `bad_points` and `errors`. |
| `matter.py` | `Projections` (with optional `sigma_rho`, `sigma_j_r`), `ScalarField` (F3–F5; φ′ by the order-4 FD with φ even; it propagates σ(φ′) into σ(ρ) and σ(j_r)), `VlasovMoments` (ρ = E, j_r = A·J, S^r_r = p, S^θ_θ = q) and `pi_from_dtphi`. |
| `diagnose.py` | Everything in §5–§10: Γ, U, M_MS, the expansions, the classes, the roots, the residuals, σ, the per-point and global statuses, the norms, the integral masses, the centre checks and the status precedence. |

Dependencies are numpy and scipy (`brentq` only). Performance at N = 10⁵: about 1.4 s and about
125 MB peak (the (N, 7) stencil work arrays).

### Which quantity comes from what (no constraint is used to compute its own check)

- R′ = D⁴R (R odd), or the analytic `dR`.
- Γ = R′/A. U = −R K_B, taken from the input.
- M = (R/2)(1 − Γ² + U²) [F11].
- Γ′ = D⁴Γ (Γ even). If `dA` is given, Γ′ = (D⁴R′ − Γ A′)/A instead.
- K_B′ = D⁴K_B, or the analytic `dKB`.
- M′ = D⁴M (M odd). It is the FD derivative of the M_MS values themselves, not the chain rule. The
  chain rule would make 𝓗_M algebraically equal to (R²R′/4)𝓗 + R³K_B𝓜 and remove it as an
  independent check.
- M_int = M(r₀) + ∫4πR²(ρR′ + U j_r) dr uses only matter and geometry, never M′.
- Each order-6 quantity (Γ⁶, M⁶) is built from D⁶R. The error estimates are σ(Γ′) = |D⁴Γ⁴ − D⁶Γ⁶|
  and σ(M′) = |D⁴M⁴ − D⁶M⁶|, so they include the error inherited from Γ.

## 2. Interpretations of and deviations from the contract (flagged)

**I-1 (contract internally contradictory: §7 UNCERTAIN vs NO_TRAPPED_SPHERES).**
With the stated precedence UNCERTAIN > … > NO_TRAPPED_SPHERES, the literal UNCERTAIN rule covers
"FUTURE/DOUBLY_MARGINAL points not assigned to a root of known type (FUTURE_MOTS/PAST_MOTS)". That
rule already captures every "isolated marginal point with a DEGENERATE root and non-trapped
neighbours". So the explicit allowance of such points in NO_TRAPPED_SPHERES (with
`marginal_present`) could never be reached. Two expected outcomes in `derivation.md` §3 depend on
it: the isotropic Schwarzschild throat on a node, and closed FLRW at turnaround.

I implemented the physically intended reading. A FUTURE_MARGINAL or DOUBLY_MARGINAL point is
*explained*, and so does not make the slice UNCERTAIN, in any of three cases:

1. It belongs to a root of type FUTURE_MOTS or PAST_MOTS of the expansion that vanishes there,
   and that root has resolved, different signs on its two sides.
2. It belongs only to DEGENERATE roots and has no trapped neighbour. This case sets
   `marginal_present`.
3. It has trapped neighbours on **both** sides. I read "без trapped-соседей с обеих сторон"
   as "not flanked by trapped points on both sides", which is the more conservative reading.

In addition, any DEGENERATE root adjacent to a trapped point gives UNCERTAIN.

**I-2 (my own tightening of I-1).** A root does not explain a marginal point in two cases:

- It is tangential (the same sign inside and outside, i.e. MOTS pair birth, open question 3).
- It touches the grid edge (inner or outer sign 0).

A grid-resolved tangential zero therefore gives UNCERTAIN. The type field is still set from the
sign of the other expansion, as the contract says, and `tangential = True` is added.

**I-3 (§9 σ, deviation).** I added a round-off floor to σ(f′) = |D⁴f − D⁶f|: ε·Σ_k|w_k|(|f_k| +
e_k), where e_k is the propagated round-off of a derived field (Γ, M). For R′ this floor is the
honest bound ε·R/h, which the contract's 1e3·ε·Σ|T_k| floor does not cover when R/h ≳ 10³.

Without the floor, exact backgrounds with polynomial R (Minkowski, KS/PG in areal gauge) give
random false VIOLATED points, because |D⁴ − D⁶| is itself pure round-off noise there.

The floor is negligible whenever the truncation error exceeds about 10⁻¹³ relative. It is zero
for analytic derivatives, where σ(R′) = σ(K_B′) = 0 exactly as the contract says.

**I-4 (§9 σ, extension).** For `ScalarField`, ρ and j_r contain the FD derivative φ′, so
"input fields are exact" does not hold for them. I added 16π σ(ρ) to σ(𝓗), 4π σ(j_r) to σ(𝓜),
and 4πR²(|R′|σ(ρ) + |U|σ(j_r)) to σ(𝓗_M), with σ(ρ) = |Φ|σ(φ′)/A and σ(j_r) = |Π|σ(φ′).

For `Projections` and `VlasovMoments` these terms are zero, so the v1.0 formulas are reproduced
exactly.

**I-5 (§8 analytic derivatives).** "Error estimates = 0" is applied to σ(R′) and σ(K_B′). Γ′ and
M′ are second-derivative-type quantities and remain FD, so their σ is non-zero. `dA` is used only
in Γ′ = (R″ − ΓA′)/A, with R″ = D⁴(R′).

**I-6 (§9 norms).** L2vol is the unnormalised sqrt(∫X² 4πR²A dr), by the trapezoid rule over
consecutive finite points of the region. The regions are:

- `center`: the first max(4, N//20) points with R > 0.
- `outer`: the last 4 points.
- `bulk`: every other point with R > 0.
- `all`: everything.

R = 0 points (where only 𝓗_M is defined) appear in `all` only. For N < 28 the regions may
overlap.

**I-7 (§10 `M_over_R3_extrapolated`).** This is a single Python `bool`: True if any R = 0 point
was replaced by its limit. `M_over_R3_extrapolated_mask` gives the point-wise mask.

The limit used is lim M/R³ = −(Γ′² + ΓΓ″)/(2R′²) + K_B²/2, by L'Hôpital. It equals
³R(0)/12 + K_B(0)²/2 at a regular centre (`derivation.md` §4). Γ″ = D⁴(Γ′) with Γ′ odd, so the
limit converges at order 4 on odd-symmetric grids (§4). For a `cell` centre, `center_checks`
holds an even extrapolation of M/R³, which is only about O(h²) because of the 1 − Γ² cancellation.

**I-8 (§6).** The integral masses start at r[0] for every centre type, as the contract literally
says. For `cell` the start value is the numerical M_MS(r[0]), not 0 at r = 0.

**I-9 (§9 constraints with bad data).** The contract offers no fourth global value. When data ≠
OK, the global constraint status is VIOLATED if any finite point is violated, otherwise
NOT_EVALUATED. It is never CONSISTENT_WITH_TRUNCATION.

Per-point statuses have three extra values:

- `CENTER_EXCLUDED`: 𝓗 and 𝓜 at R = 0.
- `INVALID`: non-finite.
- `NOT_EVALUATED`: no matter.

**I-10 (input validation).**

- `center="vertex"` requires r[0] = 0 and R[0] = 0. Deviations up to 64ε relative are snapped to 0
  with a warning. A larger R[0] ≠ 0 gives INVALID_INPUT, because odd ghosts of R would be
  inconsistent.
- α and β are not used by the diagnostics, so bad values or shapes only produce warnings. The
  isotropic Schwarzschild lapse changes sign, and that data must not become INVALID_INPUT.
- Matter of an unsupported type gives INVALID_INPUT. `diagnose` never raises.
- Points whose matter input is non-finite are also given class INVALID (conservative).

**I-11 (§7 roots, details the contract leaves open).**

- Raw sign changes of Rθ between neighbours, plus exact zeros, are events. Events that share a
  run of band-zero points (|Rθ| ≤ τ) are merged into one root. This avoids root "clouds" from
  round-off.
- r* is the root of the cubic Lagrange interpolant through 4 nodes (brentq). For a merged
  cluster, r* is the median event.
- [r_lo, r_hi] is the hull of the linear and cubic roots of all events, r* ± τ*/|slope|, and the
  band-zero points, clipped to the neighbouring resolved nodes.
- [R_lo, R_hi] is the hull of cubic **and** linear interpolants of R at r_lo, r*, r_hi.
- inner_sign and outer_sign are the resolved signs just outside the cluster, or 0 at a grid edge.
- other_sign is the sign of the cubic-interpolated other expansion at r*, with tolerance
  τ_other + |cubic − linear|. It is set to 0 if the other expansion changes sign within
  [r_lo, r_hi].
- The signs are `Sign` objects: ints that also compare equal to '+', '0' and '-'. The contract does
  not fix int vs string.

**I-12 (§3 cell centre checks).** For `cell`, Gamma0 and KA_minus_KB0 are the even extrapolation
(quadratic in r²) through the first 3 points.

I did not modify the contract. I found no other internal contradiction. The corrected F10 sign (+)
is confirmed numerically in `test_dev_constraints.py`: a matter bump gives Δ𝓗_M = +(R²R′/4)Δ𝓗,
and the "−" form is rejected.

## 3. Developer tests (`tests/dev`, 80 tests, all passing, about 5 s)

Run with `PYTHONPATH=src python3 -m pytest tests/dev -q`.

**Exact backgrounds from `derivation.md` §3.** Values of Γ, U, M and Rθ±, plus classes, roots,
statuses and masses:

- Minkowski: t = const, τ-hyperboloid out to r/τ = 10³, Milne.
- Schwarzschild: KS and PG. The AH candidate is at R = 2M, and [R_lo, R_hi] contains it. An
  analytic-derivative variant gives exact M. A grid inside the horizon only gives
  FUTURE_TRAPPED_NO_AH_CANDIDATE.
- Schwarzschild isotropic: the throat both on and off a node gives DEGENERATE roots and
  NO_TRAPPED_SPHERES, with `marginal_present` on the node.
- FLRW: flat, closed and open, with H = ±3/2. R* = (H² + k)^{−1/2}, and the expected
  PAST_MOTS / FUTURE_MOTS roots and statuses appear. M_prop matches its analytic value. Closed
  FLRW at turnaround gives DOUBLY_MARGINAL.
- FKG: a homogeneous scalar-field slice.

**Synthetic geometry.** An AH with a trapped interior, the outermost of several candidates,
trapped-to-edge, a tangential zero, reversed orientation, UNDETERMINED over AH_CANDIDATE, and a
throat.

**NaN/inf handling.** NaN, +inf and −inf in each slice field; NaN in the matter; A ≤ 0; R < 0;
structural errors; overflow giving NUMERICAL_FAILURE; the NaN θ at the centre is not a failure.

**API completeness.** Every §10 field, root keys, norms keys and shapes; `pi_from_dtphi`; the
Vlasov adapter; the ScalarField formulas.

**Measured convergence orders** (grid doubling):

| Quantity | Measured order |
|---|---|
| D⁴ (vertex, cell, none; odd and even fields) | 3.7–4.1 |
| D⁶ | 5.4–6.3 |
| `cumint` | 3.94, 3.97 |
| σ(f′)/true error | 1.00–1.002 |

The generic non-areal slice has flux, and its matter is defined from 𝓗 = 𝓜 = 0 (test data only).
Its orders are:

| Quantity | vertex | cell |
|---|---|---|
| 𝓜 | 4.05, 4.03, 4.01 | 4.0 |
| M | 4.0 | 4.0 |
| M_int − M | 4.0 | 3.9–4.2 |
| 𝓗 bulk | 4.04, 3.86, 3.65 | 4.0 |
| 𝓗 all | 1.95–2.01 (see L-1) | 1.95–2.01 (see L-1) |
| 𝓗_M bulk | 4.0 | 4.0 |
| 𝓗_M all | 3.1–3.3 at the outer edge (see L-2) | 3.1–3.3 at the outer edge (see L-2) |

Other measured orders:

- M/R³ centre limit (uniform grid): 4.06, 4.03.
- Time-symmetric scalar "star" (ODE-solved M), 𝓗 away from the centre: 4.01, 4.00.

## 4. Known limitations

- **L-1.** Near a regular centre, 𝓗 converges only as h² (2(1 − Γ²)/R² with an O(h⁴) error in Γ
  divided by r²). σ(𝓗) tracks this, so there are no false VIOLATED points. It is inherent to F7
  with FD Γ (`derivation.md` open question 8).
- **L-2.** Γ′ and M′ are nested FD derivatives. Where the stencil window shifts (the outer edge and
  the one-sided `none` edge), the error of Γ is not smooth and these drop to about third order.
  The same happens at a `vertex`/`cell` centre when the grid map r(x) is not odd-smooth, because
  the mirrored grid then has a kink: M/R³(0) converges at about 3rd order on
  `devbg.stretched` grids, 4th on uniform ones. σ captures it.
- **L-3.** On a Minkowski τ-hyperboloid with r/τ = 10⁵ (N = 4000), the last grid point becomes
  PAST_MARGINAL and the status becomes PAST_TRAPPED_ONLY (open question 5). There Rθ₋ = −10⁻⁵
  while the honest round-off bound of the one-sided D(R) at R = 10⁵ (amplified by 1/A = 10⁵)
  is about 3·10⁻⁶. It is correct up to r/τ = 10⁴.
- **L-4.** Tangential zeros that do not fall on a node are missed, as the contract acknowledges.
- **L-5.** The parity of even fields (A, K_A, K_B, φ, Π) is assumed, not verified. Data of the
  wrong parity at a vertex or cell centre give wrong centre derivatives. Only
  `center_checks` (Γ(0), K_A(0) − K_B(0)) hints at it.
- **L-6.** Not implemented, as it is not in the contract:
  - the 8πR²(ρ − J) < 1 self-consistency check at θ₊ roots (open question 4);
  - a symmetric θ₋ AH-candidate for Γ < 0 (open question 6).
- **L-7.** σ is the |D⁴ − D⁶| heuristic plus round-off. It can underestimate where the D⁶ error is
  not small (under-resolved data). κ_c = 10 is the only safety margin.
- **L-8.** Peak memory is about 125 MB at N = 10⁵.

## 5. My own errors during development (honest log)

1. **Wrong test function.** The σ-tracking test used f = sin r·e^{−0.1r} with odd parity, but f is
   not odd. The ghosts then made the centre derivative wrong: error 2.6e-3 against σ = 2.6e-4.
   Detected by the test failing. Fixed the test with f = sin r·e^{−0.1r²}. This also shows L-5:
   the module trusts the declared parity.
2. **Test too strict.** I expected D(even)(0) = 0 exactly, but it is about 4e-16 (round-off in
   the Fornberg weights) and j_r(0) is about 2e-16. Detected by the test. Relaxed to 1e-14.
3. **R uncertainty interval too narrow.** [R_lo, R_hi] did not contain the exact R = 2M at the
   isotropic Schwarzschild throat. R′ = 0 there, so the interpolation error of R itself (about
   3e-7) dominates, and I had propagated only the r-uncertainty. Detected by a manual check
   against `derivation.md` §3. Fixed by adding the linear-vs-cubic spread of R.
4. **Round-off floor factor.** My first round-off floor used factor 2 plus double counting. It made
   the τ-hyperboloid marginal at the edge, about 10× the realised error. I reduced it to 1. This
   did not remove the r/τ = 10⁵ edge effect (L-3), which is a genuine precision limit.
5. **Tangential zero misclassified.** A grid-resolved tangential zero was first counted as
   "explained by FUTURE_MOTS" and gave NO_TRAPPED_SPHERES. Found while probing open question 3.
   Changed as in I-2.
6. **Overflow in the sign-change test.** `v[:-1]*v[1:]` overflowed for huge values and raised a
   RuntimeWarning. Replaced with a product of signs.
7. **Too slow.** Python loops (class lookup, blocks) made N = 10⁵ take about 10 s under
   tracemalloc. Vectorised; it now takes 1.4 s.
8. **α/β shape mismatch counted as input error.** A shape mismatch of α or β was first recorded as
   an error, which would make INVALID_INPUT from an unused input. Made it a warning.
9. **Over-tight test tolerances.**
   - The analytic-derivative KS root (1e-8): cubic root interpolation is O(h⁴) even with exact
     nodal values.
   - The M/R³ centre-limit convergence on a stretched grid, which led to L-2.
   - 𝓗_M "all" order > 3.0, which is fragile because of the outer edge. It is now asserted on the
     bulk region (> 3.5) with "all" > 2.8.

No physics formula errors were found in the implementation. F7–F13 matched the exact backgrounds
on the first run.

## 6. Time spent (UTC)

| Part | Time |
|---|---|
| Reading contract, derivation, audit | 22:53–23:00 (7 min) |
| `fd.py`, `slice.py`, `matter.py`, `diagnose.py` first version | 23:00–23:06 (6 min) |
| Smoke tests on exact backgrounds, fixes 3–4 | 23:06–23:08 (2 min) |
| Developer tests (7 files) and fixes 1, 2, 5–9 | 23:08–23:16 (8 min) |
| README, this log | 23:16–23:20 (4 min) |

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

---

# v0.2: fix round against contract v1.1 and audit findings (2026-09-28/29, 23:54–00:06 UTC)

Base: tag `impl-B-v0.1`. I followed the same role rules: I did not read `tests/independent/`,
`independent_reference/`, `verification_plan.md`, `arch_A/`, `mutants/` or `audit_copy/`. I read
the v1.0→v1.1 diff of `physics_contract.md` (§7, §9, §13).

## Changes, each mapped to a finding

| Change | Code | Finding / contract item |
|---|---|---|
| **AH candidate via the outgoing expansion.** θ_out = θ₊ if Γ > 0 at the root, θ₋ if Γ < 0. The other expansion must be < 0. θ_out must be < 0 on the smaller-R side and > 0 on the larger-R side. New root keys: `outgoing`, `smallR_sign`, `largeR_sign`. `outer_ah_candidate` is now the root with the largest R (it was the largest r). | `diagnose._find_roots` | A6 → C1 |
| **UNCERTAIN rule.** Marginal points of every type (now including PAST_MARGINAL) must be explained by a root of known type with resolved sides that differ. The "flanked by trapped points on both sides" exception is dropped. An isolated DOUBLY_MARGINAL/DEGENERATE point without trapped neighbours is exempt (NO_TRAPPED_SPHERES + `marginal_present`). | geometry status | C2 (was my I-1), A10 → C3 |
| **PAST_TRAPPED_ONLY rule.** Requires PAST_TRAPPED points or a PAST_MOTS root with resolved sides, and no FUTURE_MOTS. An unresolved (+,0) now gives UNCERTAIN. Cases outside the §7 list (e.g. a non-candidate future MOTS without trapped points) also give UNCERTAIN, the conservative fallback. | geometry status | A10 → C3 |
| **VIOLATED floor.** The threshold is now 1e3·ε·max(Σ\|T\|, S_X). S_X is the magnitude of the terms before cancellation, with stencil sums Σ\|w\|\|f\| for the derivative terms, so the floor never vanishes. σ_round was already in σ(f′) (C = 1). | `_pstatus` | C4 (was my I-3) |
| **φ′ error for ScalarField.** Already in v0.1 (I-4); unchanged. | `matter.py` | C5 |
| **σ_noise.** Max over a 7-point window of \|Δ⁶f\|/√924, with parity ghosts at a centre. The zero band is now τ± = max(κ(σ(Rθ) + 2σ_noise(U) + 2σ_noise(Γ)), floor). | `RadialOperator.sigma_noise` | C6 |
| **UNRESOLVED.** A point is UNRESOLVED if σ(X) > 0.1·Σ\|T\|. The global status is UNRESOLVED if > 5 % of the evaluated points *of one residual* are unresolved, or any point in the `center`/`inner` region. Priority: NOT_EVALUATED (no matter **or data ≠ OK**) > VIOLATED > UNRESOLVED > CONSISTENT. New key: `status['n_unresolved']`. | `_pstatus`, global status | A7, A11 → C7 |
| **R = 0 validation.** R = 0 anywhere except r[0] with `center="vertex"` gives INVALID_INPUT. | `slice.py` | A9 → C8 |
| **Warnings (C9).** Keys: `center_deviation`, `M_int_mismatch`, `n_M_negative`, `n_rel_undefined`. There is also a text warning that for `center="cell"` the ball [0, r₀] is excluded from the integrals. None of these changes the data status. | end of `diagnose` | A8, A12, A15 → C9 |
| **Norm region name.** The first region is called `inner` when `center="none"`. | `_region_names` | A16 → C10 |
| **K40.** No change. The Γ′ round-off ∝ ε/h² is inherent: a direct R″ stencil has the same ε/h² scaling, so there is no cheap reduction. | – | matrix K40 |

## Interpretations in v0.2 (flagged)

**I-13.** A point is UNRESOLVED only if σ(X) also exceeds the round-off floor 1e3·ε·max(Σ\|T\|,
S_X). R = 0 points are excluded from the resolution test (all 𝓗_M terms vanish identically there).
Without these two rules, exact round-off-level cases would be labelled UNRESOLVED:

- flat Minkowski (every 𝓗 term is at round-off level);
- every vertex grid (at R = 0, σ(M′) ~ h⁴ while Σ\|T\| = \|M′(0)\| ≈ 0).

**I-14 (contract §9 read literally is inconsistent for 𝓗_M in vacuum).** In vacuum Σ_k\|T_k\| of
𝓗_M is just \|M′\|, which is itself pure truncation error. So σ(M′) > 0.1\|M′\| everywhere, and
every vacuum slice (Schwarzschild, Minkowski hyperboloid) would be UNRESOLVED. For the resolution
test only, I replace the M′ term by the magnitude of its un-cancelled pieces:

(\|R′\|/2)(1 + Γ² + U²) + R(\|ΓΓ′\| + \|UU′\|).

𝓗_M itself, its relative value and its floor are unchanged.

**I-15.** The 5 % rule is applied per residual rather than pooled over 𝓗, 𝓜 and 𝓗_M. Pooling
dilutes a localized defect: on the glued-ball kink (N = 101), 10 of 101 𝓗 points are unresolved
(10 %), but pooled that is only 3 %.

**I-16.** The centre-deviation warning uses κ = `kappa_theta` (3) for \|Γ(0) − 1\| >
max(κσ_Γ0, 1e-8). The M < 0 warning uses M < −(κσ_M + 1e3·ε·(R/2)(1 + Γ² + U²)); the round-off
term avoids warnings on exactly flat data, where σ_M = 0.

## Audit cases re-run on v0.2

| Case | Result |
|---|---|
| A6: PG with decreasing R | AH_CANDIDATE (θ₋ root, Γ < 0), R* = 2M in [R_lo, R_hi]. It matches the increasing-label run. |
| A7: glued balls R = \|x − 1\| | N = 100: VIOLATED. N = 101: UNRESOLVED. **N = 400: CONSISTENT_WITH_TRUNCATION.** |
| A10: hyperboloid to r/τ = 10⁷ | UNCERTAIN, no longer PAST_TRAPPED_ONLY. |
| A11: hyperboloid r ∈ [0.5, 1e5], N = 2000 | **Still VIOLATED at r₀.** |

**Open point on A7.** The kink affects about 12 points whatever N is, so their fraction falls below
5 % as the grid is refined. The v1.1 rule cannot catch a refined non-convergent kink. Only the
two-grid convergence check (which the contract itself requires) can.

**Open point on A11.** At the one-sided r₀ stencil Γ′ has an O(1) error while |D⁴ − D⁶| underestimates it
by about 100×. So the point satisfies \|X\| > 10σ, which also means σ < 0.1Σ\|T\|, i.e. "resolved"
by definition. Under the v1.1 rules a VIOLATED point is always resolved, and VIOLATED has priority.
This needs a contract-level decision, for example an under-resolution test on the inputs
(σ_noise(R)/h against R′) that is not part of the contract. I left it as is.

## Tests (v0.2)

There are 97 dev tests, all passing (about 8 s). The new file is `tests/dev/test_dev_v11.py`.
Tests that kill the A13 survivors:

- M_prop and L2vol with A ≠ 1 (generic slice, quadrature reference);
- σ_M with Γ ≠ 1 (KS);
- an unresolved PAST_MARGINAL gives UNCERTAIN;
- a direct `_find_roots` test of the other-sign check at r_lo/r_hi;
- cell Γ(0) is extrapolated (Γ[0] − 1 ≈ 1e-4, extrapolated value within 5 %);
- non-finite integral masses give NUMERICAL_FAILURE;
- FKG with analytic ρ, S, M = (4π/3)ρR³ and the M/R³ limit.

New physics tests:

- **Horizon with Γ < 0.**
- **Scalar field with flux and K ≠ 0.** Areal gauge. M comes from an ODE; K_A comes from 𝓜 = 0.
  Result: CONSISTENT, 𝓗 and 𝓜 converge at order about 4, M_int = M_ref to 1e-6. Flipping the sign
  of Π gives VIOLATED.
- **R = 0 away from the centre** gives INVALID_INPUT.
- **Glued-ball kink** is not CONSISTENT (N = 101).
- **A coarse centre** gives UNRESOLVED; the same data on a fine grid gives CONSISTENT.
- **No VIOLATED on flat space in a curvilinear label** (N = 20–1000).
- **Hyperboloid to 10⁷** is not PAST_TRAPPED_ONLY.
- **Noise** widens τ.
- **Every C9 warning** fires, and there are no spurious warnings on clean data.

**Quick mutation check.** I ran my own mutations in scratch copies, not the auditor's mutants. The
tests kill:

- M_prop without A;
- L2 without A;
- σ_M without \|Γ\|;
- dropping the other-sign end check;
- cell Γ0 = Γ[0];
- the orientation rule;
- noise in τ;
- the UNRESOLVED rule;
- σ(ρ) = 0 for ScalarField;
- the R = 0 check;
- removing all four mass-finiteness checks (removing only M_int and M_prop survived; the test now
  checks all three failure names).

"PAST_MARGINAL in the PAST_TRAPPED_ONLY condition" survives, but in v0.2 it is an *equivalent*
mutant. An unexplained PAST_MARGINAL is caught earlier by UNCERTAIN, and an explained one implies
a resolved PAST_MOTS root.

Performance at N = 10⁵: 1.6 s.

## My own errors in v0.2

1. **Literal UNRESOLVED gave false results.** The first implementation of UNRESOLVED followed §9
   literally and made the isotropic Schwarzschild slice and the Minkowski hyperboloid UNRESOLVED
   (𝓗_M in vacuum; see I-14). Detected by the existing background tests. Fixed with I-13 and I-14.
2. **5 % rule first pooled.** I first pooled the 5 % over all residuals, so glued balls
   stayed CONSISTENT at N = 101. Changed to per residual (I-15).
3. **Two new tests wrong.** The cell-Γ0 test used too coarse a grid (FD error ~1e-4 hid the
   extrapolation), and a test expected φ′ of a constant to be exactly 0. Both were test errors.

## Time

| Part | Time |
|---|---|
| Reading the v1.1 diff | 23:54–23:56 |
| Code | 23:56–00:00 |
| Tests and mutation check | 00:00–00:05 |
| Log | 00:05–00:07 |

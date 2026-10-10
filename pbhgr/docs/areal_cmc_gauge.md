# Spherical Einstein–Vlasov in the areal–CMC gauge (`pbhgr/cosmo_ev.py`)

The equations solved by `pbhgr.cosmo_ev`, as implemented and validated (FLRW exact to 1e-9, LTB dust collapse
with second-order convergence, static Einstein–Vlasov polytrope, LTB trapping times of every shell; see the
module docstring for the discrete form). Units G = c = 1.

## Metric and slicing

ds² = −α² dt² + A² (dR + β dt)² + R² dΩ², with the areal radius R as the radial coordinate and constant-mean-
curvature slices K = −3H(t) matching the FLRW exterior (α = 1, A = 1, β = −HR there). Normal n = (∂_t − β∂_R)/α,
radial unit vector ŝ = ∂_R/A. Matter in the normal frame: E, J = −T(n, ŝ), p = T(ŝ, ŝ), q = T(θ̂, θ̂).
The gauge is horizon-penetrating: 1/A² stays positive inside 2m/R > 1.

## Constraints solved on every slice (no evolved metric variable)

1. Extrinsic curvature: K^θ_θ = β/(αR), K^R_R = K − 2K^θ_θ.
2. Momentum constraint with the slicing condition, integrated outward from K^θ_θ(0) = K/3:
   ∂_R K^θ_θ + (3/R) K^θ_θ = K/R − 4πAJ, i.e. K^θ_θ(R) = K/3 − (4π/R³) ∫₀^R A J R′³ dR′; then β = αR K^θ_θ.
3. Misner–Sharp mass (independent of α): ∂_R m = 4πR² (E − R K^θ_θ A J), m(0) = 0.
4. Radial metric: 1/A² = 1 − 2m/R + R² (K^θ_θ)²  (FLRW: 1 − H²R² + H²R² = 1).
5. Lapse from ∂_t K = 3Ḣ:
   (1/(AR²)) ∂_R(R² ∂_Rα / A) = α [(K^R_R)² + 2(K^θ_θ)² + 4π(E + p + 2q)] + 3Ḣ, ∂_Rα(0) = 0, α(R_out) = 1.

## Particles

Unit rest mass, conserved angular momentum L, weight N. With the covariant radial momentum p_R and
Γ = √(1 + p_R²/A² + L²/R²) the geodesic equations are

- dR/dt = α p_R/(A²Γ) − β,
- dp_R/dt = −Γ ∂_Rα + p_R ∂_Rβ + (α/Γ) [p_R² ∂_RA/A³ + L²/R³],
- dτ/dt = α/Γ.

The code evolves the normal-frame momentum P = p_R/A, for which

- dR/dt = αP/(AΓ) − β,
- dP/dt = −(Γ/A) ∂_Rα + αP K^R_R + αL²/(ΓAR³).

Sources per cell of proper volume 4πR²A dR: E = ΣNΓ, J = ΣNP, p = ΣNP²/Γ, q = ΣN(L²/R²)/(2Γ). The grid is
comoving-areal, x = R/a(t), so the FLRW exterior is static in x and dx/dt = (dR/dt − HR)/a.

## Horizons and monitoring

θ₊ ∝ 1/A − β/α: the apparent horizon is 2m/R = 1 with K^θ_θ > 0; the cosmological anti-trapped surface has
K^θ_θ < 0. The unused evolution equation of A (equivalently K^R_R) is the Hamiltonian-consistency diagnostic,
together with mass conservation at R_out.

## Practical notes

- The areal gauge amplifies super-horizon errors by (HR)²: start quietly and deposit with cumulative,
  volume-interpolated weights.
- The comoving grid loses physical resolution as a(t) grows; re-stretch it (`--dR`) so that the core spacing
  stays below the horizon radius, and use ≥ 32 shells per cell in a lingering cold core (discreteness
  relaxation saturates there).
- Report horizon times both in the shells' proper time and in CMC coordinate time: the core clock lags the
  far-away clock by several t_H before trapping.

# Stage 1b (e ≠ 0): early-start 3D particle runs — design (5 Oct 2026)

Goal (plan §6.12): triaxial peaks at μ = 0.04–0.06 (first e = 0.13, 0.19; Yoo et al.'s e = 0.2 boundary 0.045–0.050),
after reproducing the validated e = 0 results with the same pipeline.

## Why a new start
The validated runs start at t_0 = 0.5 t_C(0) from the exact LTB slice. No exact solution exists for e ≠ 0, and the
gradient expansion is useless at t_0 (ε = k/aH ≈ 4). The only clean data are Yoo et al.'s long-wavelength CMC data at
ε_i = 0.2 (t_i = 2/(3H_i), H_i = 5k, a_i = 1), valid for any ζ(x), so the run must start at t_i and cover a factor
10⁴–10⁵ in time.

## Initial data (mode `yoo`)
- Profile (Yoo eq. 5.1): ζ = μ e^{−k²r²/6} [1 + (k²/6)(p(2X² − Y² − Z²) + 3e(Y² − Z²))], Ψ = e^{ζ/2}.
- q = −(4/3) ΔΨ/Ψ⁵, p_ij = Ψ⁻⁴[−(2/Ψ)(∂_i∂_jΨ − δ_ij ΔΨ/3) + (6/Ψ²)(∂_iΨ∂_jΨ − δ_ij (∂Ψ)²/3)];
  ψ = Ψ[1 − q/(6H_i²)], γ̃_ij = δ_ij − (4/5) p_ij/H_i², Ã_ij = (2/5) p_ij/H_i, K = −3H_i (the spherical restriction is
  `pbhgr/cosmo_id.py`). All derivatives of Ψ are analytic.
- Code variables in the coordinates of the validated runs, x' = a_ref x with a_ref = a(0.5 t_C(0)): χ = a_ref² ψ⁻⁴
  det(γ̃)^{−1/3}, h̃ = γ̃ det^{−1/3}, Ã_ij → det^{−1/3} Ã_ij, lapse 1, shift 0, Γ̃ from h̃ (GammaCalculator), fref = 1.
  (χ_far starts at a_ref² ≫ 1 and is O(1) at collapse, so the χ floor only acts in the puncture.)
- Matter from the constraints with the code's own operators (vacuum `Constraints`): E = Ham/16π, J_i = Mom_i/8π per
  level on a 3-ghost FillPatch; cold particles on the lattice: v_i = J_i/E, Γ = (1 − γ^{ij}v_iv_j)^{−1/2},
  u_i = Γ v_i, m = E √γ dV/Γ (CIC interpolation of E, J, χ, h̃ to the particles); τ_0 = t_i.

## Time stepping (mode `dt_mode = 1`)
- Coordinate light speed is √χ_far, so dt_0 = min(dt_multiplier dx_0/√χ_max, dt_frac (t + t_i)); the second limit
  (accuracy of the expansion) governs until t ≈ 0.4 t_C(0) for μ = 0.3.
- Adaptive subcycling: level l takes the parent's step (n_cycle = 1) while that satisfies its own CFL limit, otherwise
  half of it. In the accuracy-limited phase all levels share one small step (7 level-updates per coarse step instead of
  127), so the ≈ 300 early steps cost like ≈ 20 late ones.
- Quantities tied to the coordinate light speed are scaled with χ_far(t): Kreiss–Oliger σ → σ √χ_max, Γ-driver
  coefficient 0.75 → 0.75 χ_max (its gauge speed is √F in coordinate units, independent of χ).

## Horizon diagnostic for non-spherical runs
Trapped coordinate spheres: Θ = D_i s^i + K_ij s^i s^j − K of the spheres r = const (exact normal s_i ∝ x_i, needs first
derivatives of χ, h̃) on ≈ 100 directions × radii. r_in = largest r with max_Ω Θ ≤ 0 (a trapped sphere: sufficient for a
horizon), r_out = smallest r with min_Ω Θ ≥ 0; area A = ∫ √γ |∇r| r² dΩ, M = √(A/16π). Reduces to the ray diagnostic at
e = 0. Proper time of the matter in the shell [r_in, r_out] as before.

## Validation ladder
1. μ = 0: FLRW over a factor 10³ in a with adaptive steps and subcycling switches.
2. e = 0, μ = 0.3, 3 levels: shell check against LTB (R, M_MS at the shell's τ and label) up to the caustic.
3. e = 0, μ = 0.3, 7 levels: (M_AH, τ_AH) against the late-start runs and LTB.
4. e = 0.2, p = 0 at μ = 0.3 and 0.1 (Yoo's Fig. 7 drop times 20 and 70 t_H as a qualitative check), then the grid
   μ = 0.04–0.06 × e = 0.13, 0.19, 0.2.

## Cost (CCX33, 7 levels, 4.75 min per fully subcycled coarse step)
μ = 0.3 to 2.2 t_C(0): ≈ 300 cheap + 80 full steps ≈ 8–10 h. μ = 0.05 to 1.5 t_C(0): ≈ 300 cheap + 190 full ≈ 16 h.

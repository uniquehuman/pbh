# PBH — codes for primordial black hole formation in an early matter era

Scientific libraries from a numerical-relativity programme that asks whether asteroid-mass primordial black holes
(10¹⁹–10²⁰ g) can be all of the dark matter if they formed while the Universe was dominated by an oscillating
heavy scalar field — a "dust" era before reheating. The field oscillates ≥ 10⁹ times per Hubble time at these
masses, so it is modelled as collisionless matter: the **Einstein–Vlasov system**, in spherical symmetry and in
full 3D general relativity.

🌐 Results, figures, documents and data: **https://pbh.ivliev.org** (English and Russian).

## Libraries

| Path | What it is |
|---|---|
| [`pbhgr/pbhgr/`](pbhgr/pbhgr) | Python package. `cosmo_ev` — spherical Einstein–Vlasov solver in a horizon-penetrating areal–CMC gauge with an expanding FLRW exterior (fully constrained; shells with radial and angular momentum); equations in [`pbhgr/docs/areal_cmc_gauge.md`](pbhgr/docs/areal_cmc_gauge.md). `cosmo_id` — long-wavelength CMC initial data of Yoo, Escrivà, Harada & Kohri (2026) mapped onto shells. `ltb` — exact growing-mode Lemaître–Tolman–Bondi solution (collapse and trapping times, M_AH(τ)). `profiles` — BBKS mean peak profiles for lognormal spectra and flat-core profiles. `pm3d` — Newtonian 3D particle-mesh code for one triaxial peak in Einstein–de Sitter. `spherical_ev` and helpers — polar-areal Einstein–Vlasov with decay of the field into radiation, isolated initial data, static polytropes. C kernels in `pbhgr/csrc/`, pytest suite in `pbhgr/tests/`. |
| [`pbhgr/grteclyn/PBHVlasov/`](pbhgr/grteclyn/PBHVlasov) | Vlasov (particle) matter module for the 3D numerical-relativity code **GRTeclyn** (AMReX): deposition consistent across AMR levels, geodesic push with per-particle proper time, a profile-referenced 1+log gauge for cosmological collapse, early start from Yoo et al. long-wavelength data, ray and sphere apparent-horizon diagnostics, example parameter files. Design: [`pbhgr/docs/stage2_vlasov_design.md`](pbhgr/docs/stage2_vlasov_design.md) (Russian), [`pbhgr/docs/stage1b_design.md`](pbhgr/docs/stage1b_design.md). |
| [`pbhgr/grchombo/PBHCosmo/`](pbhgr/grchombo/PBHCosmo) | **GRChombo** example: scalar-field collapse of an LTB-type peak in an expanding box, with LTB initial data, a profile-referenced gauge and sphere tagging. |
| [`pbhgr/scripts/`](pbhgr/scripts) | Drivers: `v1_ladder.py` (1D validation ladder and production runs of `cosmo_ev`), `v1_static_test.py`, `pm3d_peak.py`, `ellipsoidal_collapse.py` (Bond–Myers). `scripts/grchombo/`: build script for the GRChombo/GRTeclyn stack, initial-data and parameter generators for the 3D modules (`ltb_to_grchombo.py`, `stage1b_params.py`), 3D-vs-LTB/1D comparisons and checkpoint diagnostics. |
| [`sphdiag/`](sphdiag) | Diagnostics of spherically symmetric 3+1 slices for any matter (scalar field, Vlasov moments, given stress tensor): constraint residuals, Misner–Sharp mass, light-ray expansions θ±, classification of spheres and apparent-horizon candidates, all with numerical-error estimates and validity flags. Measures only — it does not evolve or solve constraints. Formula specification [`physics_contract.md`](sphdiag/physics_contract.md), symbolic derivation [`derivation.md`](sphdiag/derivation.md). |

The history of each library is kept (restricted to these paths).

## Quick start

```bash
cd pbhgr
make                                            # builds the C kernel pbhgr/_kernels.so
make test                                       # pytest (add --runslow for the long tests)
python3 scripts/v1_ladder.py flrw --K 1000      # homogeneous FLRW check
python3 scripts/v1_ladder.py ltb --mu 0.3 --tend 6.5 --K 2000 --n 4 --stretch 4 --out runs/v1   # LTB peak

cd ../sphdiag
PYTHONPATH=src python3 -m pytest tests/dev      # 106 tests, a few seconds
```

Python ≥ 3.12 with numpy and scipy (`pbhgr` also uses psutil, pyyaml, h5py) and a C compiler. Units G = c = 1;
for the Yoo profile k = 1, H_i = 5, t_H = 100√6, M_H = 0.75 t_H.

The 3D modules are compiled inside GRTeclyn and GRChombo source trees: see
[`pbhgr/scripts/grchombo/build_grchombo_ax102.sh`](pbhgr/scripts/grchombo/build_grchombo_ax102.sh) and
[`pbhgr/docs/grchombo_ax102.md`](pbhgr/docs/grchombo_ax102.md) (Russian). Parameter files for early-start
3D runs come from `pbhgr/scripts/grchombo/stage1b_params.py`.

## What these codes have shown (October 2026)

- Spherical GR–Vlasov, cold, converged: μ_th(100 t_H) ≈ 0.039, μ_th(300 t_H) ≈ 0.019, μ_th(1000 t_H) ≈ 0.0086.
- 3D GR with PBHVlasov, spherical peak: horizon growth matches the exact LTB solution and the 1D code to
  ±0.01 t_C(0) over 0.1–0.6 M_H.
- 3D GR with PBHVlasov, ellipticity e = 0.2 (preliminary, see the caveat): μ = 0.3 forms a ≈ 1 M_H horizon at
  matter proper time τ = 2.09 t_C(0); μ = 0.1 forms a 0.35–0.56 M_H horizon at τ = 2.44 t_C(0).

**Caveat / known issue (10 October 2026):** early-start 3D runs (`pbh_vlasov.init = yoo`) develop an instability
at the faces of the finest adaptive-mesh level during the super-horizon phase. The e = 0.2 results carry a not yet
quantified error of order 10 % in the traceless extrinsic curvature Ã and are preliminary; the e = 0 checks
against LTB are unaffected. Details: the 10 October entry of the log, https://pbh.ivliev.org/en/log/

Results: https://pbh.ivliev.org/en/results/

## Licence

Copyright 2026 Sergey Ivliev.

- Code: **MIT-0** ([`LICENSE`](LICENSE)) — any use, no conditions.
- `pbhgr/grteclyn/PBHVlasov/` and `pbhgr/grchombo/PBHCosmo/`: **BSD-3-Clause** — derived from GRTeclyn and
  GRChombo, whose licence requires keeping their copyright notices.
- Documents: **CC0 1.0**.

See [`LICENSING.md`](LICENSING.md). Attribution is not required; if this helps your research, a citation is
appreciated ([`CITATION.cff`](CITATION.cff)).

Русская версия: [`README.ru.md`](README.ru.md).

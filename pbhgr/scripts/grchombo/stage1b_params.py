#!/usr/bin/env python3
"""Parameter file for a stage-1b early-start run (Yoo long-wavelength data, seven levels, geodesic phase to
0.5 t_C(0), then the profile-referenced 1+log lapse and the Gamma-driver) for given (mu, e, p).
Coordinates x' = a_ref x with a_ref = a(0.5 t_C(0)); box 15.36 comoving (6.27 r_m) as in the validated mu = 0.3 runs,
so that the resolution in units of t_C(0) and M_H is the same for every mu.
Usage: stage1b_params.py --mu 0.1 --e 0.2 --p 0 --stop 3.5 --out DIR"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from pbhgr.ltb import LTBGaussian  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, required=True)
ap.add_argument("--e", type=float, default=0.0)
ap.add_argument("--p", type=float, default=0.0)
ap.add_argument("--stop", type=float, default=3.5, help="stop at this multiple of t_C(0)")
ap.add_argument("--gauge_on", type=float, default=0.5, help="gauge switch at this multiple of t_C(0)")
ap.add_argument("--lapse_coeff", type=float, default=2.0)
ap.add_argument("--max_level", type=int, default=6, help="finest AMR level; each level is a 64^3 box of half the width "
                "(level l covers +-L/2^(l+1)); use 7 for mu <= 0.1 so the horizon has > 5 cells per radius")
ap.add_argument("--check_int", type=int, default=20, help="checkpoint interval in coarse steps")
ap.add_argument("--out", required=True)
a = ap.parse_args()

H_i, k = 5.0, 1.0
t_i = 2.0 / (3.0 * H_i)
t_H = (np.sqrt(6.0) / (k / H_i)) ** 3 * t_i  # 100 sqrt(6) for H_i = 5
M_H = 0.75 * t_H
tC0 = float(LTBGaussian(a.mu, k=k, H_i=H_i).tC(1e-6))  # exact central caustic time of the LTB solution
a_ref = (0.5 * tC0 / t_i) ** (2.0 / 3.0)
L = 15.360000 * a_ref  # 6194.411298 / 403.29929194892884 comoving box of the mu = 0.3 runs
dx0 = L / 64
template = f"""# Stage 1b early-start run: mu = {a.mu}, e = {a.e}, p = {a.p}; t_C(0) = {tC0:.4f}, a_ref = {a_ref:.6f}
# geodesic slicing to {a.gauge_on} t_C(0), then lapse_coeff {a.lapse_coeff} + Gamma-driver; stop at {a.stop} t_C(0); {a.max_level + 1} levels
amr.verbose = 1
amr.n_cell = 64 64 64
geometry.prob_extent = {L:.6f} {L:.6f} {L:.6f}
geometry.is_periodic = 1 1 1
geometry.center = {L/2:.6f} {L/2:.6f} {L/2:.6f}
amr.max_level = {a.max_level}
amr.regrid_int = -1
amr.max_grid_size = 16
amr.blocking_factor = 8
amr.n_error_buf = 0
evolution.dt_multiplier = 0.25
evolution.max_steps = 10000000
evolution.sigma = 0.3
evolution.stop_time = {a.stop * tC0 - t_i:.4f}
gauge.lapse_advec_coeff = 1.0
gauge.lapse_coeff = {a.lapse_coeff}
gauge.lapse_power = 1.0
gauge.shift_advec_coeff = 0.0
gauge.shift_Gamma_coeff = 0.75
gauge.eta = {1.0 / M_H:.6e}
ccz4.formulation = 1
pbh_vlasov.init = yoo
pbh_vlasov.H0 = {H_i}
pbh_vlasov.mu = {a.mu}
pbh_vlasov.ell_e = {a.e}
pbh_vlasov.ell_p = {a.p}
pbh_vlasov.kp = {k}
pbh_vlasov.a_ref = {a_ref:.10f}
pbh_vlasov.tau0 = {t_i:.12f}
pbh_vlasov.dt_mode = 1
pbh_vlasov.dt_frac = 0.05
pbh_vlasov.yoo_match_iter = 6
pbh_vlasov.gauge_on_time = {a.gauge_on * tC0 - t_i:.6f}
pbh_vlasov.n_per_dir = 2
pbh_vlasov.lattice_buffer = 4
pbh_vlasov.max_disp_cells = 0.8
pbh_vlasov.ah_interval = 0
pbh_vlasov.theta_rays = 1
pbh_vlasov.theta_ray_n = 256
pbh_vlasov.theta_ray_dr = 0.5
pbh_vlasov.theta_profile_interval = 10
pbh_vlasov.sphere_diag = 1
amr.plot_int = 20
amr.check_int = {a.check_int}
amr.plot_vars = chi lapse K rho_p h11 h22 A11 A22 shift1
rho_line_extraction.enabled = true
rho_line_extraction.end = {L - dx0/2:.6f} {L/2:.6f} {L/2:.6f}
chi_line_extraction.enabled = true
chi_line_extraction.end = {L - dx0/2:.6f} {L/2:.6f} {L/2:.6f}
K_line_extraction.enabled = true
K_line_extraction.end = {L - dx0/2:.6f} {L/2:.6f} {L/2:.6f}
lapse_line_extraction.enabled = true
lapse_line_extraction.end = {L - dx0/2:.6f} {L/2:.6f} {L/2:.6f}
"""
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, "params.txt"), "w").write(template)
setup = dict(mu=a.mu, e=a.e, p=a.p, t0=t_i, f0=t_i / tC0, tC0=tC0, tH=t_H, ti=t_i, a0=a_ref, rm=np.sqrt(6.0) / k,
             L=L, N=64, dx=dx0, MH=M_H, gauge_on_time=a.gauge_on * tC0 - t_i, stop=a.stop * tC0 - t_i,
             note="early start: run time 0 = t_i; x' = a_ref x")
json.dump(setup, open(os.path.join(a.out, "setup.json"), "w"), indent=1)
print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in setup.items()}))

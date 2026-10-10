"""Outcome-map scan driver for the spherical solver (Stage A, section 4 'Скан').

Axes (isolated-cloud family, see initial_data.shell_family):
    nu          = 2 M / R   initial compactness (amplitude proxy)
    flatness    profile exponent (1 = Gaussian, larger = flatter top)
    L_rms       tangential velocity rms in units of sqrt(M/R)
    Gamma_tdyn  Gamma * t_dyn, t_dyn = sqrt(R^3 / M)
Each point returns one of {BH_approach, bounce, dispersion, unresolved} plus diagnostics.
Independent points are embarrassingly parallel (multiprocessing), exactly the pattern the
specification prescribes for the production campaign (one run per node, small outcome files).
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .initial_data import shell_family
from .provenance import Provenance
from .spherical_ev import Grid, Run, RunConfig, solve_metric


def run_point(args):
    nu, flatness, L_rms, Gamma_tdyn, n_part, K, r_out_over_R, t_end_tdyn, seed, out_dir = args
    M = 1.0
    R = 2 * M / nu
    t_dyn = np.sqrt(R**3 / M)
    tag = f"nu{nu:.3f}_fl{flatness:.1f}_L{L_rms:.2f}_G{Gamma_tdyn:.3f}_n{n_part}_K{K}"
    rd = Path(out_dir) / tag
    params = dict(nu=nu, flatness=flatness, L_rms=L_rms, Gamma_tdyn=Gamma_tdyn, n_part=n_part, K=K,
                  r_out_over_R=r_out_over_R, t_end_tdyn=t_end_tdyn, M=M, R=R, t_dyn=t_dyn)
    prov = Provenance(rd, params, seed)
    rng = np.random.default_rng(seed)
    g = Grid(r_out=r_out_over_R * R, K=K)
    P = shell_family(M, R, n_part, rng, flatness=flatness, L_rms=L_rms)
    for _ in range(4):
        P.N *= M / solve_metric(g, P).M
    cfg = RunConfig(Gamma=Gamma_tdyn / t_dyn, t_end=t_end_tdyn * t_dyn, cfl=0.4, diag_every=50, seed=seed,
                    emit_fraction=0.25)
    run = Run(g, P, cfg)
    outcome = run.run()
    diag = run.diag
    comp = [d["max_2m_over_r"] for d in diag]
    mc = [d["momentum_constraint_rel_residual"] for d in diag if d["momentum_constraint_rel_residual"] == d["momentum_constraint_rel_residual"]]
    summary = {
        "outcome": outcome, "max_compactness": float(np.max(comp)), "t_at_max_tdyn": float(diag[int(np.argmax(comp))]["t"] / t_dyn),
        "final_M_in": diag[-1]["M_in"], "M_total_drift": float(np.max(np.abs([d["M_total"] for d in diag])) - diag[0]["M_total"]),
        "min_lapse": float(np.min([d["min_lapse"] for d in diag])), "decayed_rest_mass": diag[-1]["decayed_rest_mass"],
        "escaped_energy": diag[-1]["escaped_energy"], "steps": run.step, "t_end_reached_tdyn": run.t / t_dyn,
        "momentum_constraint_max": float(np.max(mc)) if mc else None, "n_particles_final": diag[-1]["n_particles"],
    }
    prov.finish({"constraint_norms": {"momentum_constraint_rel_residual_max": summary["momentum_constraint_max"],
                                      "ADM_mass_drift": summary["M_total_drift"]},
                 "horizon_diagnostics": {"outcome": outcome, "max_2m_over_r": summary["max_compactness"],
                                         "min_lapse": summary["min_lapse"]}})
    with open(rd / "diag.json", "w") as f:
        json.dump(diag, f)
    with open(rd / "summary.json", "w") as f:
        json.dump({**params, **summary}, f, indent=1)
    return {**params, **summary, "wall_s": prov.record["wall_time_s"], "peak_rss_MB": prov.record["peak_rss_MB"]}


def scan(points, out_dir, n_workers=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_workers = n_workers or max(1, os.cpu_count())
    t0 = time.time()
    with Pool(n_workers) as pool:
        rows = pool.map(run_point, [tuple(p) + (str(out_dir),) for p in points], chunksize=1)
    with open(out_dir / "scan_summary.json", "w") as f:
        json.dump(rows, f, indent=1)
    return rows, time.time() - t0

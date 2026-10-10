"""Run provenance record, as required by the specification (section 12, Reproducibility).

Every run writes: git commit, compiler version, compile flags, host CPU, MPI version
(none in the Python prototype), input parameters, RNG seed, start/end time, wall time,
peak RSS, constraint norms and horizon diagnostics (the latter two are supplied by the
solver through `finish(extra=...)`).
"""
from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e.__class__.__name__})"


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _git_commit():
    root = Path(__file__).resolve().parent.parent
    out = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(root), "status", "--porcelain"])
    return {"commit": out, "dirty": bool(dirty) and not dirty.startswith("unavailable")}


class Provenance:
    def __init__(self, run_dir: str | os.PathLike, params: dict, seed: int | None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.record = {
            "git": _git_commit(),
            "python": sys.version,
            "numpy": np.__version__,
            "compiler": _run(["gcc", "--version"]).splitlines()[0] if True else None,
            "compile_flags": "-O2 -fPIC -shared -std=c11 (csrc/kernels.c)",
            "host": platform.node(),
            "cpu": _cpu_model(),
            "n_cpu": os.cpu_count(),
            "mpi": "none (serial prototype)",
            "params": params,
            "seed": seed,
            "start_utc": datetime.now(timezone.utc).isoformat(),
        }

    def finish(self, extra: dict | None = None):
        self.record["end_utc"] = datetime.now(timezone.utc).isoformat()
        self.record["wall_time_s"] = time.time() - self.t0
        self.record["peak_rss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        if extra:
            self.record.update(extra)
        with open(self.run_dir / "provenance.json", "w") as f:
            json.dump(self.record, f, indent=2, default=_json_default)
        return self.record


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

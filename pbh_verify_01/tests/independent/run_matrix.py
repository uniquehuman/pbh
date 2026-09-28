#!/usr/bin/env python3
"""Run the independent verification matrix and write results/independent_matrix.json.

Usage (from the pilot root):
    SPHDIAG_PATH=src python3 tests/independent/run_matrix.py [--only V00,S20,...] [--out path]
SPHDIAG_PATH selects the implementation directory (default <pilot>/src).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from common import SPHDIAG_PATH, load_sphdiag  # noqa: E402
from matrix import CASES, run_all  # noqa: E402


def _git(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "independent_matrix.json"))
    a = ap.parse_args()
    ids = [x.strip() for x in a.only.split(",") if x.strip()] or None
    sd, err = load_sphdiag()
    print("SPHDIAG_PATH = %s  (%s)" % (SPHDIAG_PATH, "imported" if sd else err))
    t0 = time.time()
    results = run_all(ids)
    dt = time.time() - t0
    summary = {}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1
    print("-" * 78)
    for r in results:
        print("%-4s %-5s %s" % (r.id, r.status, r.title))
        for f in r.fails[:8]:
            print("        FAIL: %s" % f)
        if len(r.fails) > 8:
            print("        ... %d more" % (len(r.fails) - 8))
        for n in r.notes[:3]:
            print("        note: %s" % n.splitlines()[-1][:300])
        orders = r.metrics.get("orders", {})
        shown = 0
        for lab, o in orders.items():
            for k, val in o.items():
                if shown < 4 and "p" in val and val["p"] == val["p"]:
                    print("        order %-40s %-14s p=%.2f  e=%s" % (lab[:40], k, val["p"], ["%.1e" % e for e in val["e"]]))
                    shown += 1
    print("-" * 78)
    print("summary:", summary, "(%.1f s)" % dt)
    out = dict(generated=datetime.datetime.utcnow().isoformat() + "Z", sphdiag_path=SPHDIAG_PATH,
               sphdiag_importable=sd is not None, import_error=err, sphdiag_git=_git(SPHDIAG_PATH),
               verification_git=_git(ROOT), n_cases=len(CASES), summary=summary, runtime_s=dt,
               results=[r.to_dict() for r in results])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("written", a.out)
    return 0 if all(r.status in ("PASS", "SKIP") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

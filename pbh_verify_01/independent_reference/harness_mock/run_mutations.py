#!/usr/bin/env python3
"""HS: mutation test of the independent matrix using the harness mock (not the implementation).
Each mutation must be detected (>= 1 FAIL/ERROR).  Writes results/harness_mutation_test.json."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MUTS = ["", "F10sign", "jr_sign", "jr_noA", "swap_theta", "ah_nosides", "nan_ok", "naive_mass", "sigma_zero", "order2",
        # v2 (contract 1.1)
        "ah_plus_only", "past_marg", "no_noise", "no_unresolved", "R0_ok", "trap_quad", "no_warnings", "l2_normalised",
        # v1.2
        "no_unres_class"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "harness_mutation_test_v2.json"
out = {}
for m in MUTS:
    env = dict(os.environ, SPHDIAG_PATH=os.path.join(ROOT, "independent_reference", "harness_mock"),
               SPHDIAG_MOCK_MUTATION=m, PYTHONWARNINGS="ignore")
    js = os.path.join(ROOT, "results", "_mut_%s.json" % (m or "none"))
    subprocess.run([sys.executable, os.path.join(ROOT, "tests", "independent", "run_matrix.py"), "--out", js],
                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = json.load(open(js))
    failed = [r["id"] for r in d["results"] if r["status"] not in ("PASS", "SKIP")]
    out[m or "none"] = dict(failed=failed, detected=bool(failed) if m else None, clean=(not failed) if not m else None)
    os.remove(js)
    print(m or "none", "->", failed, flush=True)
ok = out["none"]["clean"] and all(v["detected"] for k, v in out.items() if k != "none")
out["all_ok"] = ok
json.dump(out, open(os.path.join(ROOT, "results", OUT), "w"), indent=1)
print("all_ok", ok)

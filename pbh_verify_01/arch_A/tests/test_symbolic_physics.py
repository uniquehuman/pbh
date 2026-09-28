"""Runs the independent symbolic derivation of the contract formulas (physics_check/*.py) when sympy
is available (it is installed in /root/PBH/pbh_verify_01/.pydeps; not a runtime dependency)."""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PC = os.path.join(HERE, "..", "physics_check")
PYDEPS = "/root/PBH/pbh_verify_01/.pydeps"


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = PYDEPS + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _have_sympy():
    r = subprocess.run([sys.executable, "-c", "import sympy"], env=_env(), capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _have_sympy(), reason="sympy not available")
@pytest.mark.parametrize("script", ["check_contract.py", "check_gauss_codazzi.py"])
def test_symbolic_derivations(script):
    r = subprocess.run([sys.executable, os.path.join(PC, script)], env=_env(), capture_output=True,
                       text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "passed" in r.stdout

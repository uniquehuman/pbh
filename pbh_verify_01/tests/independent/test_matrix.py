"""pytest wrapper for the independent verification matrix (PBH-VERIFY-01).

Run from the pilot root:
    SPHDIAG_PATH=src PYTHONPATH=.:tests/independent python3 -m pytest tests/independent -q
Skips cleanly (with the import error) when sphdiag is not importable from SPHDIAG_PATH.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_sphdiag, run_case  # noqa: E402
from matrix import CASES  # noqa: E402

_sd, _err = load_sphdiag()
pytestmark = pytest.mark.skipif(_sd is None, reason=str(_err))


@pytest.mark.parametrize("tid", list(CASES.keys()))
def test_case(tid):
    fn, title = CASES[tid]
    r = run_case(fn, tid, title)
    if r.status == "SKIP":
        pytest.skip("; ".join(r.notes))
    assert r.status == "PASS", "%s %s: %s\n%s" % (tid, r.status, title, "\n".join(r.fails[:20]))

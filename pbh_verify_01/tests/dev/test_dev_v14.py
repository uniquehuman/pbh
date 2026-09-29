"""Contract v1.4 (C14): accidental zeros of |D4 f - D6 f| -> max over {i-1, i, i+1}."""
import numpy as np

from sphdiag import Projections, RadialOperator, SphericalSlice, diagnose
import devbg as bg


def test_accidental_zero_of_difference_estimate():
    """Non-smooth nodal error (as for module-computed fields) on sin(r); one node outside the
    D4 stencil but inside the D6 stencil of row i is adjusted so that D4 f - D6 f = 0 exactly at
    row i while the true D4 error there stays ~3e-6 (the U62 pattern)."""
    N, i = 60, 20
    r = np.linspace(0, 3, N)
    op = RadialOperator(r, "none")
    idx6, W6 = op._stencils[6]
    k = np.arange(N)
    eps = 1e-6 * (-1.0) ** k * (1 + 0.3 * np.sin(k))
    diff = op.d(eps, 1, 4)[i] - op.d(eps, 1, 6)[i]
    eps[i + 3] += diff / W6[i][list(idx6[i]).index(i + 3)]
    f = np.sin(r) + eps
    d4, sig = op.d_err(f, f, 1)
    raw = np.abs(op.d(f, 1, 4) - op.d(f, 1, 6))
    true = np.abs(d4 - np.cos(r))
    assert raw[i] < 0.05 * true[i]                      # accidental (near) zero of |D4 - D6|
    assert true[i] / sig[i] < 1.5                       # v1.4: covered by the neighbours
    assert sig[i] >= max(raw[i - 1], raw[i + 1])
    assert true[i] / raw[i] > 20                        # v1.3 would have given |X|/sigma > 20


def test_margins_after_c14():
    rc = abs(bg.generic_slice(np.array([0.0]))[1]["rho"][0])
    for N in (100, 200, 400):
        r = np.linspace(0, 4.0, N)
        kw, ex = bg.generic_slice(r)
        d = diagnose(SphericalSlice(center="vertex", **kw),
                     Projections(ex["rho"], ex["j_r"], 0 * r, 0 * r))
        assert d.status["constraints"] == "CONSISTENT_WITH_TRUNCATION"
        for nm in ("ham", "mom", "hamM"):
            X, s = getattr(d, nm), getattr(d, "sigma_" + nm)
            m = np.isfinite(X)
            assert np.max(np.abs(X[m]) / s[m]) < 2.0
        if N >= 200:
            bump = 1e-4 * rc * np.exp(-((r - 0.3) / 0.1) ** 2)
            d = diagnose(SphericalSlice(center="vertex", **kw),
                         Projections(ex["rho"] + bump, ex["j_r"], 0 * r, 0 * r))
            assert d.status["constraints"] == "VIOLATED"

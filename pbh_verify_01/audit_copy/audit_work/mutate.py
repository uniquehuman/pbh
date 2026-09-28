"""Mini mutation run on the audit-patched copy (4 fixes) to measure developer-test sensitivity.
Each mutant: one textual replacement; survivor = all tests in tests/dev pass."""
import shutil, subprocess, sys, os, tempfile
BASE = "audit_work/patched/sphdiag"
MUTANTS = [
    ("M_prop without A", "diagnose.py", "op.cumint(FOUR_PI * R ** 2 * A * rho, +1)", "op.cumint(FOUR_PI * R ** 2 * rho, +1)"),
    ("L2vol dV without A", "diagnose.py", "dV = FOUR_PI * R ** 2 * A", "dV = FOUR_PI * R ** 2"),
    ("sigma_Rtheta = sigma_Gamma (no factor 2)", "diagnose.py", "sig_Rth = 2.0 * sig_G", "sig_Rth = sig_G"),
    ("sigma_M without |Gamma|", "diagnose.py", "sig_M = R * np.abs(Gamma) * sig_G", "sig_M = R * sig_G"),
    ("ah without G_star>0", "diagnose.py", "rtype == FUTURE_MOTS and G_star > 0", "rtype == FUTURE_MOTS"),
    ("ah without type check (which+ only)", "diagnose.py", 'which == "+" and rtype == FUTURE_MOTS and', 'which == "+" and'),
    ("kappa_theta ignored (tau=floor)", "diagnose.py", "tau_p = np.maximum(kappa_theta * sig_Rth, floor)", "tau_p = floor + 0 * sig_Rth"),
    ("kappa_c ignored (sigma*1)", "diagnose.py", "np.abs(X) > kappa_c * sig + 1e3 * EPS * Tabs", "np.abs(X) > 1.0 * sig + 1e3 * EPS * Tabs"),
    ("sig_ham drops sigma(Gamma') term", "diagnose.py", "+ 4.0 * sig_dG / (A * Rs)\n", "+ 0.0 * sig_dG / (A * Rs)\n"),
    ("hamM flux term uses j_r*|U|", "diagnose.py", "-FOUR_PI * R ** 2 * U * j_r)", "-FOUR_PI * R ** 2 * np.abs(U) * j_r)"),
    ("ScalarField sigma_rho = 0", "matter.py", "sig_rho = np.abs(Phi) * sig_dphi / A", "sig_rho = 0 * sig_dphi"),
    ("ScalarField Phi = phi' (no 1/A)", "matter.py", "Phi = dphi / A", "Phi = dphi"),
    ("centre region n_c = 4", "diagnose.py", "n_c = max(4, N // 20)", "n_c = 4"),
    ("PAST_MARGINAL not in PAST_TRAPPED_ONLY", "diagnose.py", "np.isin(cls, (PAST_TRAPPED, PAST_MARGINAL)).any()", "np.isin(cls, (PAST_TRAPPED,)).any()"),
    ("tangential root counted as explained", "diagnose.py", "and d[\"outer_sign\"] != 0 and not d[\"tangential\"] for d in rel)", "and d[\"outer_sign\"] != 0 for d in rel)"),
    ("other-sign endpoint check removed", "diagnose.py", "if not np.isfinite(ve) or ve * other < 0:", "if False:"),
    ("cell Gamma0 = Gamma[0] (no extrapolation)", "diagnose.py", "cc.update(Gamma0=_even_extrapolate0(r, Gamma),", "cc.update(Gamma0=float(Gamma[0]),"),
    ("M_over_R3 limit drops K_B^2/2", "diagnose.py", "lim = -(dG ** 2 + Gamma * ddG) / (2.0 * dR ** 2) + 0.5 * KB ** 2", "lim = -(dG ** 2 + Gamma * ddG) / (2.0 * dR ** 2)"),
    ("M_rho uses R' -> A (proper)", "diagnose.py", "op.cumint(FOUR_PI * R ** 2 * rho * dR, +1)", "op.cumint(FOUR_PI * R ** 2 * rho * A, +1)"),
    ("Vlasov j_r = J", "matter.py", "Projections(self.E, slc.A * self.J, self.p, self.q)", "Projections(self.E, self.J, self.p, self.q)"),
    ("failure check of M_prop etc removed", "diagnose.py", '("M_int", M_int), ("M_prop", M_prop)):', '):'),
]
res = []
for name, f, old, new in MUTANTS:
    tmp = tempfile.mkdtemp(dir="audit_work")
    shutil.copytree(BASE, os.path.join(tmp, "sphdiag"))
    p = os.path.join(tmp, "sphdiag", f)
    s = open(p).read()
    if s.count(old) != 1:
        res.append((name, "PATTERN-NOT-FOUND")); shutil.rmtree(tmp); continue
    open(p, "w").write(s.replace(old, new))
    env = dict(os.environ, PYTHONPATH=tmp)
    out = subprocess.run([sys.executable, "-m", "pytest", "tests/dev", "-q", "-x", "-p", "no:cacheprovider"],
                         env=env, capture_output=True, text=True)
    last = out.stdout.strip().splitlines()[-1]
    res.append((name, "KILLED" if out.returncode else "SURVIVED", last))
    shutil.rmtree(tmp)
for r in res: print(" | ".join(r))

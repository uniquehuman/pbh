#!/usr/bin/env python3
"""Generate mutant copies of src/sphdiag (pre-registered classes, mutation_report.md §0).

Usage: python3 mutants/make_mutants.py [--src src] [--out mutants]
Each mutant is mutants/<id>/src/sphdiag with exactly one textual change (asserted unique).
"""
import argparse
import shutil
from pathlib import Path

D, MT, FD = "diagnose.py", "matter.py", "fd.py"

MUTANTS = {
    "M1": ("areal vs coordinate radius in M_MS", [
        (D, "M = 0.5 * R * (1.0 - Gamma ** 2 + U ** 2)", "M = 0.5 * r * (1.0 - Gamma ** 2 + U ** 2)"),
        (D, "M6 = 0.5 * R * (1.0 - Gamma6 ** 2 + U ** 2)", "M6 = 0.5 * r * (1.0 - Gamma6 ** 2 + U ** 2)")]),
    "M2a": ("sign of K_ij (U = +R K_B)", [(D, "U = -R * KB", "U = R * KB")]),
    "M2b": ("orientation of the spatial normal s^a", [
        (D, "Gamma = dR / A\n", "Gamma = -dR / A\n"), (D, "Gamma6 = dR6 / A", "Gamma6 = -dR6 / A")]),
    "M3a": ("lost 4pi in the flux mass", [
        (D, "M_flux = op.cumint(FOUR_PI * R ** 2 * U * j_r, +1)", "M_flux = op.cumint(R ** 2 * U * j_r, +1)")]),
    "M3b": ("8 pi rho instead of 16 pi rho in the Hamiltonian constraint", [
        (D, "2.0 * KB ** 2, -4.0 * FOUR_PI * rho)", "2.0 * KB ** 2, -2.0 * FOUR_PI * rho)")]),
    "M3c": ("potential coefficient m^2 phi^2 instead of m^2 phi^2 / 2", [
        (MT, "return 0.5 * self.m_phi ** 2 * self.phi ** 2", "return self.m_phi ** 2 * self.phi ** 2")]),
    "M4a": ("sign of the radial flux j_r", [(MT, "j_r = -Pi * dphi", "j_r = Pi * dphi")]),
    "M4b": ("flux term omitted (mass form and integral mass)", [
        (D, "-FOUR_PI * R ** 2 * U * j_r)", "0.0 * j_r)"),
        (D, "M_flux = op.cumint(FOUR_PI * R ** 2 * U * j_r, +1)", "M_flux = 0.0 * op.cumint(FOUR_PI * R ** 2 * U * j_r, +1)")]),
    "M5": ("cosmological (past) marginal sphere accepted as AH candidate", [
        (D, 'ah = bool(which == "+" and rtype == FUTURE_MOTS and G_star > 0\n                  and inner == -1 and outer == 1)',
            'ah = bool(rtype != DEGENERATE and G_star > 0\n                  and inner == -1 and outer == 1)')]),
    "M7": ("NaN / invalid points treated as no collapse", [
        (D, "cls[invalid_pt] = INVALID", "cls[invalid_pt] = NORMAL"),
        (D, 'if data_status != "OK" or (cls == INVALID).any():\n        geometry = "UNDETERMINED"',
            'if False:\n        geometry = "UNDETERMINED"')]),
    "M8": ("even parity for R at the centre", [
        (D, "dR, sig_dR = op.d_err(R, R, -1)\n            dR6 = op.d(R, -1, 6)", "dR, sig_dR = op.d_err(R, R, +1)\n            dR6 = op.d(R, +1, 6)")]),
    "M9": ("2nd-order primary derivative instead of 4th", [
        (FD, "self._stencils[order] = self._build(order + 1, deriv=1)", "self._stencils[order] = self._build(3 if order == 4 else order + 1, deriv=1)")]),
    "M10": ("Gamma = R'/A^2 (gamma_rr instead of sqrt)", [
        (D, "Gamma = dR / A\n", "Gamma = dR / A ** 2\n"), (D, "Gamma6 = dR6 / A", "Gamma6 = dR6 / A ** 2")]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="src")
    ap.add_argument("--out", default="mutants")
    a = ap.parse_args()
    for mid, (desc, edits) in MUTANTS.items():
        dst = Path(a.out) / mid / "src"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(Path(a.src), dst, ignore=shutil.ignore_patterns("__pycache__"))
        for fn, old, new in edits:
            p = dst / "sphdiag" / fn
            s = p.read_text()
            assert s.count(old) == 1, (mid, fn, old)
            p.write_text(s.replace(old, new))
        (Path(a.out) / mid / "MUTANT.txt").write_text(f"{mid}: {desc}\n" + "\n".join(f"{fn}: {o!r} -> {n!r}" for fn, o, n in edits) + "\n")
        print(mid, "ok:", desc)


if __name__ == "__main__":
    main()

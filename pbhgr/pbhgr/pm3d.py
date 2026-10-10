"""Newtonian 3D particle-mesh code for one triaxial peak in an Einstein-de Sitter background (item 3b of the plan).

Units as in the 1D runs: G = c = 1, comoving wavenumber k_p = 1 (the profile's k), a_i = 1, H_i = 5, t_i = 2/(3 H_i),
rho_b a^3 = 3 H_i^2/(8 pi).  Comoving coordinates x in a periodic box of side L; canonical momentum p = a^2 dx/dt:
    dx/da = p / (a^3 H),   dp/da = -grad Phi / (a H),   lap Phi = (3/2) H_i^2 delta / a,   H = H_i a^{-3/2}.
Initial data (Zel'dovich, growing mode, at a = 1): zeta(x) = mu Psi(r_ell) + tail, delta_k = (2/5) (k/(a_i H_i))^2 zeta_k,
psi = -grad lap^{-1} delta, x = q + psi, p = H_i psi.  The peak is made triaxial through the profile itself:
r_ell^2 = sum_i (x_i/A_i)^2 with A_i^-2 proportional to the Zel'dovich eigenvalues (1+3e+p, 1-2p, 1-3e+p)/3 and
prod A_i = 1.  Small-scale tail: Gaussian random field, P_zeta flat in ln k between k1 and k2 with rms zeta_rms.
"""
from __future__ import annotations

import numpy as np

H_I = 5.0
T_I = 2.0 / (3.0 * H_I)
RHO_A3 = 3.0 * H_I**2 / (8.0 * np.pi)          # rho_b a^3


def zeldovich_axes(e: float, p: float):
    lam = np.array([(1 + 3 * e + p), (1 - 2 * p), (1 - 3 * e + p)]) / 3.0
    A = lam ** -0.5
    return A / np.prod(A) ** (1.0 / 3.0)


class PM3D:
    def __init__(self, N: int, L: float, Np: int, seed: int = 0, smooth: float = 0.0):
        self.N, self.L, self.Np = N, L, Np
        self.smooth = smooth                          # Gaussian force smoothing length in cells (0 = plain CIC/FFT)
        self.dx = L / N
        k1 = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        kz = 2 * np.pi * np.fft.rfftfreq(N, d=self.dx)
        self.kx, self.ky, self.kz = np.meshgrid(k1, k1, kz, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2[0, 0, 0] = 1.0
        self.rng = np.random.default_rng(seed)
        self.m_part = RHO_A3 * L**3 / Np**3           # mass per particle (a^3 rho_b V / N_p)

    # --- initial data
    def make_ic(self, mu, profile_spline, e=0.0, p=0.0, tail_rms=0.0, tail_k=(2.0, 10.0)):
        N, L = self.N, self.L
        xg = (np.arange(N) + 0.5) * self.dx - L / 2
        X, Y, Z = np.meshgrid(xg, xg, xg, indexing="ij")
        A = zeldovich_axes(e, p)
        r_ell = np.sqrt((X / A[0])**2 + (Y / A[1])**2 + (Z / A[2])**2)
        zeta = mu * profile_spline(np.minimum(r_ell, profile_spline.x[-1]))
        zk = np.fft.rfftn(zeta)
        if tail_rms > 0:
            kk = np.sqrt(self.k2); kk[0, 0, 0] = 0.0
            band = (kk >= tail_k[0]) & (kk <= tail_k[1])
            # P_zeta(k) d^3k/(2pi)^3 with dimensionless power flat in ln k: P(k) = 2 pi^2 zeta_rms^2 / (k^3 ln(k2/k1))
            Pk = np.where(band, 2 * np.pi**2 * tail_rms**2 / (np.maximum(kk, 1e-30)**3 * np.log(tail_k[1] / tail_k[0])), 0.0)
            noise = np.fft.rfftn(self.rng.standard_normal((N, N, N)))       # <|n_k|^2> = N^3
            tail_k = noise * np.sqrt(Pk * N**3 / L**3)                        # <|zeta_k|^2> = P(k) N^6/L^3 (discrete FT)
            self.tail_rms_grid = float(np.std(np.fft.irfftn(tail_k, s=(N, N, N))))
            zk = zk + tail_k
        else:
            self.tail_rms_grid = 0.0
        dk = 0.4 * self.k2 / H_I**2 * zk                 # delta_k = (2/5) k^2/(a_i H_i)^2 zeta_k  (a_i = 1)
        dk[0, 0, 0] = 0.0
        phik = -dk / self.k2                            # displacement potential: psi = -grad(phi), lap phi = delta -> phi_k = -delta_k/k^2
        psi = [np.fft.irfftn(1j * kc * phik, s=(N, N, N)) * -1.0 for kc in (self.kx, self.ky, self.kz)]   # psi = -grad phi
        # particles on a lattice (quiet start), displaced by CIC-interpolated psi
        q1 = (np.arange(self.Np) + 0.5) * (L / self.Np) - L / 2
        Q = np.stack(np.meshgrid(q1, q1, q1, indexing="ij"), axis=-1).reshape(-1, 3)
        disp = np.stack([self.interp(f, (Q + L / 2) % L) for f in psi], axis=-1)     # mesh arrays are indexed on [0, L): cell i <-> centred x = (i+0.5) dx - L/2
        self.x = (Q + disp + L / 2) % L
        self.p = H_I * disp                              # p = a^2 xdot = a^3 H psi at a = 1
        self.q = Q
        self.delta_lin = np.fft.irfftn(dk, s=(N, N, N))
        self.r_ell_q = np.sqrt(np.sum((Q / A) ** 2, axis=1))
        self.A = A
        return self

    # --- mesh operations
    def _cic_index(self, X):
        g = X / self.dx - 0.5
        i0 = np.floor(g).astype(np.int64); f = g - i0
        return i0 % self.N, (i0 + 1) % self.N, f

    def deposit(self, X):
        N = self.N
        ix0, ix1, fx = self._cic_index(X[:, 0]); iy0, iy1, fy = self._cic_index(X[:, 1]); iz0, iz1, fz = self._cic_index(X[:, 2])
        rho = np.zeros(N**3)
        for ix, wx in ((ix0, 1 - fx), (ix1, fx)):
            for iy, wy in ((iy0, 1 - fy), (iy1, fy)):
                for iz, wz in ((iz0, 1 - fz), (iz1, fz)):
                    rho += np.bincount((ix * N + iy) * N + iz, weights=wx * wy * wz, minlength=N**3)
        return rho.reshape(N, N, N)

    def interp(self, F, X):
        N = self.N
        ix0, ix1, fx = self._cic_index(X[:, 0]); iy0, iy1, fy = self._cic_index(X[:, 1]); iz0, iz1, fz = self._cic_index(X[:, 2])
        out = np.zeros(len(X))
        for ix, wx in ((ix0, 1 - fx), (ix1, fx)):
            for iy, wy in ((iy0, 1 - fy), (iy1, fy)):
                for iz, wz in ((iz0, 1 - fz), (iz1, fz)):
                    out += F[ix, iy, iz] * wx * wy * wz
        return out

    def forces(self, a):
        """-grad Phi at the particles (comoving gradient), Phi_k = -(3/2) H_i^2 delta_k /(a k^2)."""
        n_mean = self.Np**3 / self.N**3
        delta = self.deposit(self.x) / n_mean - 1.0
        dk = np.fft.rfftn(delta)
        phik = -1.5 * H_I**2 / a * dk / self.k2
        if self.smooth > 0:
            phik = phik * np.exp(-0.5 * self.k2 * (self.smooth * self.dx) ** 2)
        phik[0, 0, 0] = 0.0
        acc = []
        for kc in (self.kx, self.ky, self.kz):
            g = np.fft.irfftn(-1j * kc * phik, s=(self.N, self.N, self.N))     # -d Phi/dx_i
            acc.append(self.interp(g, self.x))
        return np.stack(acc, axis=-1), delta

    # --- leapfrog in a (kick-drift-kick)
    def step(self, a, da):
        H = H_I * a**-1.5
        if getattr(self, "_acc", None) is None or self._acc_a != a:
            self._acc, _ = self.forces(a)
        acc = self._acc
        self.p += 0.5 * da * acc / (a * H)
        a_mid = a + 0.5 * da; H_mid = H_I * a_mid**-1.5
        self.x = (self.x + da * self.p / (a_mid**3 * H_mid)) % self.L
        a_new = a + da; H_new = H_I * a_new**-1.5
        acc, delta = self.forces(a_new)
        self.p += 0.5 * da * acc / (a_new * H_new)
        self._acc, self._acc_a = acc, a_new                # reused as the opening kick of the next step
        return a_new, delta


def analyse(pm: PM3D, a: float, fracs=(0.1, 0.2, 0.3), radii_cells=(1.0, 2.0, 4.0, 8.0)):
    """Axes of the inner Lagrangian sets and enclosed-mass compaction around the densest point (physical units)."""
    L = pm.L; out = {"a": a}
    # densest point: use the Lagrangian centre set (r_ell_q < 0.05 r_m ~ innermost particles) centre of mass
    rm = np.sqrt(6.0)
    core = np.zeros(len(pm.x), bool); core[np.argsort(pm.r_ell_q)[:27]] = True      # the 27 innermost Lagrangian particles
    def com(sel):
        rel = (pm.x[sel] - pm.x[sel][0] + L / 2) % L - L / 2      # unwrap around one member
        return (pm.x[sel][0] + rel.mean(axis=0)) % L
    c = com(core)
    rel_all = (pm.x - c + L / 2) % L - L / 2
    for f in fracs:
        sel = pm.r_ell_q < f * rm
        if sel.sum() < 30: sel = np.zeros(len(pm.x), bool); sel[np.argsort(pm.r_ell_q)[:30]] = True
        rel = rel_all[sel]; rel = rel - rel.mean(axis=0)
        I = rel.T @ rel / len(rel)
        ev = np.sqrt(np.maximum(np.linalg.eigvalsh(I), 0.0)) * np.sqrt(5.0)     # semi-axes of the equivalent uniform ellipsoid (comoving)
        M = pm.m_part * sel.sum()
        axes_phys = a * ev[::-1]                                                 # a1 >= a2 >= a3, physical
        R_hoop = np.sqrt(0.5 * (axes_phys[0]**2 + axes_phys[1]**2))
        out[f"axes_{f}"] = axes_phys.tolist(); out[f"axes_com_{f}"] = ev[::-1].tolist(); out[f"M_{f}"] = M; out[f"hoop_{f}"] = 2 * M / max(R_hoop, 1e-30)
    r_com = np.sqrt(np.sum(rel_all**2, axis=1))
    for nc in radii_cells:                       # enclosed mass within nc cells of the centre; compaction in physical units
        R = a * nc * pm.dx
        M = pm.m_part * np.count_nonzero(r_com < nc * pm.dx)
        out[f"m_in_{nc:g}c"] = M; out[f"C_{nc:g}c"] = 2 * M / R
    out["rho_max_over_mean"] = float(np.max(pm.deposit(pm.x)) / (pm.Np**3 / pm.N**3))
    return out

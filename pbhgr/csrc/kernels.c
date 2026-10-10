/* Compiled kernels for the spherical Einstein-Vlasov solver (polar-areal gauge).
 *
 * mass_integrate: integrate the Hamiltonian constraint
 *     dm/dr = s_k * sqrt(1 - 2 m / r)
 * across K cells [r_e[k], r_e[k+1]] with piecewise-constant coordinate source
 * density s_k = (sum_i N_i E_i)_k / dr  (see spherical_ev.py for the derivation),
 * using nsub RK4 substeps per cell.  m_edge[0] must hold the inner boundary
 * mass (0 for a regular centre).  Returns the number of cells where the
 * argument of the square root became non-positive (horizon reached).
 */
#include <math.h>

static inline double rhs(double r, double m, double s) {
    if (r <= 0.0) return s;               /* regular centre: sqrt(1 - 2m/r) -> 1 */
    double a = 1.0 - 2.0 * m / r;
    if (a <= 0.0) return 0.0;             /* horizon: freeze (flagged by caller) */
    return s * sqrt(a);
}

int mass_integrate(int K, const double *r_edge, const double *s, double *m_edge, int nsub) {
    int bad = 0;
    for (int k = 0; k < K; ++k) {
        double r0 = r_edge[k];
        double h = (r_edge[k + 1] - r0) / nsub;
        double m = m_edge[k];
        double sk = s[k];
        for (int j = 0; j < nsub; ++j) {
            double r = r0 + j * h;
            double k1 = rhs(r, m, sk);
            double k2 = rhs(r + 0.5 * h, m + 0.5 * h * k1, sk);
            double k3 = rhs(r + 0.5 * h, m + 0.5 * h * k2, sk);
            double k4 = rhs(r + h, m + h * k3, sk);
            m += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        }
        double re = r_edge[k + 1];
        if (re > 0.0 && 1.0 - 2.0 * m / re <= 0.0) { bad++; m = 0.5 * re * (1.0 - 1e-12); }
        m_edge[k + 1] = m;
    }
    return bad;
}

/* Areal-CMC gauge (Stage A step 2): integrate the Misner-Sharp mass across cells
 *     dm/dR = (sE_k/dR) * sqrt(u) - R * Kth(R) * (sJ_k/dR),   u = 1 - 2m/R + R^2 Kth^2,
 * with sE_k = sum N W and sJ_k = sum N P in cell k (normal-frame energy and radial momentum),
 * Kth = K^theta_theta linear between the edge values kth_edge[k], kth_edge[k+1].
 * u = 1/A^2 must stay positive (areal gauge regular); returns the number of cells where it did not.
 */
int mass_integrate_cmc(int K, const double *r_edge, const double *sE, const double *sJ,
                       const double *kth_edge, double *m_edge, int nsub) {
    int bad = 0;
    for (int k = 0; k < K; ++k) {
        double r0 = r_edge[k], r1 = r_edge[k + 1];
        double dR = r1 - r0, h = dR / nsub;
        double m = m_edge[k];
        double v3 = (r1 * r1 * r1 - r0 * r0 * r0) / 3.0;          /* sources ~ R^2 within the cell */
        double se = sE[k] / v3, sj = sJ[k] / v3;
        double k0 = kth_edge[k], k1 = kth_edge[k + 1];
        /* the ODE is stiff at H R >> 1: |df/dm| ~ se r; keep h |df/dm| <= 0.3 (RK4 stability limit 2.8) */
        int ns = nsub;
        double lam = fabs(se) * r1;
        int need = (int)(dR * lam / 0.3) + 1;
        if (need > ns) ns = need;
        h = dR / ns;
        for (int j = 0; j < ns; ++j) {
            double rr[4] = {r0 + j * h, r0 + j * h + 0.5 * h, r0 + j * h + 0.5 * h, r0 + (j + 1) * h};
            double mm[4], kk[4], f[4];
            mm[0] = m;
            for (int s = 0; s < 4; ++s) {
                double r = rr[s];
                double kth = k0 + (k1 - k0) * (r - r0) / dR;
                double mloc = (s == 0) ? m : (s == 1) ? m + 0.5 * h * f[0] : (s == 2) ? m + 0.5 * h * f[1] : m + h * f[2];
                double u = (r > 0.0) ? 1.0 - 2.0 * mloc / r + r * r * kth * kth : 1.0;
                if (u <= 0.0) { u = 0.0; bad++; }
                f[s] = r * r * (se * sqrt(u) - r * kth * sj);
                (void)mm; (void)kk;
            }
            m += h * (f[0] + 2.0 * f[1] + 2.0 * f[2] + f[3]) / 6.0;
        }
        m_edge[k + 1] = m;
    }
    return bad;
}

/* Shell-code constraint solve along sorted shells (areal-CMC gauge).
 *   R[i] ascending, NW[i] = N_i W_i, NP[i] = N_i P_i, Kt = trace K = -3H.
 *   Outputs per shell: m_in[i] (Misner-Sharp mass just inside shell i), dm[i] (its own jump),
 *   kth[i] = K^th_th at the shell (half of its own momentum counted), u[i] = 1/A^2 at the shell
 *   (evaluated with m_in + dm/2, iterated to self-consistency).  Returns the number of shells with u <= 0.
 */
int shell_constraints(int n, const double *R, const double *NW, const double *NP, double Kt,
                      double *m_in, double *dm, double *kth, double *u) {
    int bad = 0;
    double m = 0.0, cumNPR = 0.0;
    for (int i = 0; i < n; ++i) {
        double r = R[i];
        double r3 = r * r * r;
        double k = (r > 0.0) ? Kt / 3.0 - (cumNPR + 0.5 * NP[i] * r) / r3 : Kt / 3.0;
        double d = NW[i];                                   /* first guess: sqrt(u) = 1 */
        double uu = 1.0;
        for (int it = 0; it < 3; ++it) {
            uu = (r > 0.0) ? 1.0 - 2.0 * (m + 0.5 * d) / r + r * r * k * k : 1.0;
            if (uu <= 0.0) { uu = 1e-14; }
            d = NW[i] * sqrt(uu) - r * k * NP[i];
        }
        if (uu <= 1e-14) bad++;
        m_in[i] = m; dm[i] = d; kth[i] = k; u[i] = uu;
        m += d;
        cumNPR += NP[i] * r;
    }
    return bad;
}

/* ---------------------------------------------------------------------------------------------------------------
   Warm shells near pericentre: RK4 substeps under the metric frozen on the edge grid (areal radii R_edge[0..K]),
   one shell at a time with its own (even) substep count.  cosmo = 1: a = (t/t_i)^{2/3}, H = 2/(3t), K = -3H;
   cosmo = 0: a = 1, H = K = 0.  Same right-hand side as CosmoRun.rhs (mirror convention: x may be negative).
   Outputs the state at t0 + dt/2 and at t0 + dt.  Returns the number of shells whose radius fell below 1e-12. */
static void metric_interp(int K, const double *Re, const double *al, const double *dal, const double *Ae,
                          const double *Kt, double R, double *alpha, double *dalpha, double *A, double *Kth) {
    int lo = 0, hi = K;
    if (R >= Re[K]) { *alpha = al[K]; *dalpha = dal[K]; *A = Ae[K]; *Kth = Kt[K]; return; }
    while (hi - lo > 1) { int mid = (lo + hi) / 2; if (Re[mid] <= R) lo = mid; else hi = mid; }
    double w = (R - Re[lo]) / (Re[hi] - Re[lo]);
    *alpha = al[lo] + w * (al[hi] - al[lo]);
    *dalpha = dal[lo] + w * (dal[hi] - dal[lo]);
    *A = Ae[lo] + w * (Ae[hi] - Ae[lo]);
    *Kth = Kt[lo] + w * (Kt[hi] - Kt[lo]);
}

static void warm_rhs(int K, const double *Re, const double *al, const double *dal, const double *Ae, const double *Kt,
                     int cosmo, double t_i, double Lsq, double x, double Pm, double t,
                     double *dx, double *dP, double *dtau) {
    double a = 1.0, H = 0.0, Kc = 0.0;
    if (cosmo) { a = pow(t / t_i, 2.0 / 3.0); H = 2.0 / (3.0 * t); Kc = -3.0 * H; }
    double sgn = (x >= 0.0) ? 1.0 : -1.0;
    double R = a * fabs(x);
    double Rs = (R > 1e-300) ? R : 1e-300;
    double alpha, dalpha, A, Kth;
    metric_interp(K, Re, al, dal, Ae, Kt, R, &alpha, &dalpha, &A, &Kth);
    double W = sqrt(1.0 + Pm * Pm + Lsq / (Rs * Rs));
    double dRdt = alpha * sgn * Pm / (A * W) - alpha * R * Kth;
    *dx = sgn * (dRdt - H * R) / a;
    *dP = sgn * (-(W / A) * dalpha + alpha * Lsq / (W * A * Rs * Rs * Rs)) + alpha * Pm * (Kc - 2.0 * Kth);
    *dtau = alpha / W;
}

int stiff_substeps(int n, int K, const double *Re, const double *al, const double *dal, const double *Ae, const double *Kt,
                   int cosmo, double t_i, const double *Lsq, const double *x0, const double *P0, const double *tau0,
                   double t0, double dt, const int *nsub,
                   double *x_half, double *P_half, double *x_end, double *P_end, double *tau_end) {
    int nbad = 0;
    for (int i = 0; i < n; i++) {
        int ns = nsub[i]; if (ns < 2) ns = 2; if (ns % 2) ns++;
        double h = dt / ns, x = x0[i], P = P0[i], tau = tau0[i], t = t0, L = Lsq[i];
        double k1x, k1P, k1t, k2x, k2P, k2t, k3x, k3P, k3t, k4x, k4P, k4t;
        for (int s = 0; s < ns; s++) {
            warm_rhs(K, Re, al, dal, Ae, Kt, cosmo, t_i, L, x, P, t, &k1x, &k1P, &k1t);
            warm_rhs(K, Re, al, dal, Ae, Kt, cosmo, t_i, L, x + 0.5 * h * k1x, P + 0.5 * h * k1P, t + 0.5 * h, &k2x, &k2P, &k2t);
            warm_rhs(K, Re, al, dal, Ae, Kt, cosmo, t_i, L, x + 0.5 * h * k2x, P + 0.5 * h * k2P, t + 0.5 * h, &k3x, &k3P, &k3t);
            warm_rhs(K, Re, al, dal, Ae, Kt, cosmo, t_i, L, x + h * k3x, P + h * k3P, t + h, &k4x, &k4P, &k4t);
            x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            P += h / 6.0 * (k1P + 2.0 * k2P + 2.0 * k3P + k4P);
            tau += h / 6.0 * (k1t + 2.0 * k2t + 2.0 * k3t + k4t);
            t += h;
            if (s == ns / 2 - 1) { x_half[i] = x; P_half[i] = P; }
            if (fabs(x) < 1e-12) nbad++;
        }
        x_end[i] = x; P_end[i] = P; tau_end[i] = tau;
    }
    return nbad;
}

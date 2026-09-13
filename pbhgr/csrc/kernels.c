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

/* PBHVlasov example for GRTeclyn (pbhgr project, stage 2). Based on Examples/ScalarField (GRTeclyn fd76177). */
#include "PBHVlasovLevel.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include "AlgebraicConstraintsEnforcer.hpp"
#include "CCZ4RHSWithMatter.hpp"
#include "ConstraintsWithMatter.hpp"
#include "EMTensor.hpp"
#include "FixedGridsTagger.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GRParmParse.hpp"
#include "GammaCalculator.hpp"
#include "IntegratedMovingPunctureGauge.hpp"
#include "Interval.hpp"
#include "PBHGaugeTeclyn.hpp"
#include "PBHAHFinder.hpp"
#include "LineExtraction.hpp"
#include "PositiveChiAndLapse.hpp"
#include "StateTypes.hpp"
#include <AMReX_MultiFabUtil.H>
#include <AMReX_iMultiFab.H>
#include <fstream>
#include <sstream>

using VlasovConstraints = ConstraintsWithMatter<PBHVlasovLevel::Matter<>>;
using VlasovEnergyDensity = EMTensor<PBHVlasovLevel::Matter<>, EMTensorOptions::justEnergyDensity>;

namespace
{
struct InitParams
{
    std::string mode{"flrw"}; // flrw | table
    amrex::Real H0{1.0};      // FLRW: initial Hubble rate (K = -3 H0, rho = 3 H0^2 / 8 pi)
    std::string table;        // table mode: r chi K Arr rho ur (uniform r, isotropic coordinates)
    int n_per_dir{2};         // particles per cell per direction
    int ah_interval{0};       // coarse steps between apparent-horizon searches (0 = off)
    amrex::Real ah_start{0.0}; // search only for t >= ah_start
    amrex::Real ah_guess{1.0}; // initial guess radius (coordinate)
    int ah_num_particles{2000};
    int ah_max_iter{2000};
    int lattice_buffer{4};    // level-l cells beyond a level's grids that still get that level's particle lattice
    amrex::Real max_disp_cells{0.8}; // cap on the particle displacement per push (cells)
    int theta_rays{1};        // spherical-expansion diagnostic along the 6 coordinate rays every coarse step
    int theta_ray_n{256};     // points per ray
    amrex::Real theta_ray_dr{0.5}; // spacing in finest-level cells
    int theta_profile_interval{20}; // coarse steps between full Theta(r) profile dumps (0 = never)
    amrex::Real tau0{0.0};    // initial proper time of the particles (the LTB time of the initial slice)
    amrex::Real sigma{0.0};   // warm particles: 1D velocity dispersion at injection (table columns 7-8 scale it to t_0)
    long sigma_seed{12345};
    // yoo mode: long-wavelength CMC data at a_i = 1 (H0 = H_i) for zeta = mu exp(-k^2 r^2/6) [1 + (k^2/6)(p (2X^2 - Y^2
    // - Z^2) + 3 e (Y^2 - Z^2))]; grid coordinates x' = a_ref x (comoving x in units of 1/k)
    amrex::Real mu{0.0}, ell_e{0.0}, ell_p{0.0}, kp{1.0}, a_ref{1.0};
    int sphere_diag{0};       // expansion of the coordinate spheres on sphere_ndir directions (non-spherical runs)
    int sphere_ndir{96}, sphere_nr{160};
    amrex::Real sphere_dr{0.75}; // radial spacing in finest-level cells
    int dt_mode{0};           // 0: fixed dt = dt_multiplier dx; 1: expansion-aware step with adaptive subcycling
    amrex::Real dt_frac{0.03}; // dt_mode 1: coarse step <= dt_frac (t + tau0)
    void read()
    {
        GRParmParse pp("pbh_vlasov");
        pp.query("init", mode);
        pp.query("H0", H0);
        pp.query("table", table);
        pp.query("n_per_dir", n_per_dir);
        pp.query("ah_interval", ah_interval);
        pp.query("ah_start", ah_start);
        pp.query("ah_guess", ah_guess);
        pp.query("ah_num_particles", ah_num_particles);
        pp.query("ah_max_iter", ah_max_iter);
        pp.query("lattice_buffer", lattice_buffer);
        pp.query("max_disp_cells", max_disp_cells);
        pp.query("theta_rays", theta_rays);
        pp.query("theta_ray_n", theta_ray_n);
        pp.query("theta_ray_dr", theta_ray_dr);
        pp.query("theta_profile_interval", theta_profile_interval);
        pp.query("tau0", tau0);
        pp.query("sigma", sigma);
        pp.query("sigma_seed", sigma_seed);
        pp.query("mu", mu);
        pp.query("ell_e", ell_e);
        pp.query("ell_p", ell_p);
        pp.query("kp", kp);
        pp.query("a_ref", a_ref);
        pp.query("sphere_diag", sphere_diag);
        pp.query("sphere_ndir", sphere_ndir);
        pp.query("sphere_nr", sphere_nr);
        pp.query("sphere_dr", sphere_dr);
        pp.query("dt_mode", dt_mode);
        pp.query("dt_frac", dt_frac);
    }
};

// Long-wavelength CMC initial data of Yoo et al. 2026 (eqs. 3.4-3.11, profile 5.1) at a_i = 1 for comoving position
// (X, Y, Z): conformal factor psi, conformal metric gt_ij = delta_ij - (4/5) p_ij / H_i^2 and At_ij = (2/5) p_ij / H_i
// (K = -3 H_i); p_ij = Psi^-4 [-(2/Psi)(d_i d_j Psi - delta_ij lap Psi / 3) + (6/Psi^2)(d_i Psi d_j Psi - delta_ij
// |d Psi|^2 / 3)], psi = Psi [1 + (2/9) lap Psi / (Psi^5 H_i^2)], Psi = exp(zeta / 2).
struct YooData
{
    amrex::Real psi;
    amrex::Real g[3][3], A[3][3];
};
AMREX_GPU_DEVICE inline YooData yoo_data(amrex::Real X, amrex::Real Y, amrex::Real Z, amrex::Real mu, amrex::Real e,
                                         amrex::Real pp, amrex::Real k, amrex::Real Hi)
{
    const amrex::Real x[3] = {X, Y, Z};
    const amrex::Real k2 = k * k, r2 = X * X + Y * Y + Z * Z;
    const amrex::Real gs = std::exp(-k2 * r2 / 6.0);
    const amrex::Real Q = 1.0 + (k2 / 6.0) * (pp * (2 * X * X - Y * Y - Z * Z) + 3.0 * e * (Y * Y - Z * Z));
    const amrex::Real dQ[3] = {(k2 / 3.0) * 2.0 * pp * X, (k2 / 3.0) * (3.0 * e - pp) * Y, -(k2 / 3.0) * (3.0 * e + pp) * Z};
    const amrex::Real ddQ[3] = {(k2 / 3.0) * 2.0 * pp, (k2 / 3.0) * (3.0 * e - pp), -(k2 / 3.0) * (3.0 * e + pp)};
    amrex::Real dz[3], ddz[3][3];
    for (int i = 0; i < 3; ++i)
        dz[i] = mu * (-(k2 / 3.0) * x[i] * gs * Q + gs * dQ[i]);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const amrex::Real ddg = ((i == j ? -(k2 / 3.0) : 0.0) + (k2 * k2 / 9.0) * x[i] * x[j]) * gs;
            ddz[i][j] = mu * (ddg * Q - (k2 / 3.0) * gs * (x[i] * dQ[j] + x[j] * dQ[i]) + (i == j ? gs * ddQ[i] : 0.0));
        }
    const amrex::Real Psi = std::exp(0.5 * mu * gs * Q);
    amrex::Real dP[3], ddP[3][3], lap = 0.0, grad2 = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        dP[i] = 0.5 * Psi * dz[i];
        grad2 += dP[i] * dP[i];
        for (int j = 0; j < 3; ++j)
            ddP[i][j] = Psi * (0.5 * ddz[i][j] + 0.25 * dz[i] * dz[j]);
        lap += ddP[i][i];
    }
    YooData D;
    const amrex::Real Psi4 = Psi * Psi * Psi * Psi;
    const amrex::Real q = -(4.0 / 3.0) * lap / (Psi4 * Psi);
    D.psi = Psi * (1.0 - q / (6.0 * Hi * Hi));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const amrex::Real dl = (i == j) ? 1.0 : 0.0;
            const amrex::Real pij = (-(2.0 / Psi) * (ddP[i][j] - dl * lap / 3.0) +
                                     (6.0 / (Psi * Psi)) * (dP[i] * dP[j] - dl * grad2 / 3.0)) / Psi4;
            D.g[i][j] = dl - 0.8 * pij / (Hi * Hi);
            D.A[i][j] = 0.4 * pij / Hi;
        }
    return D;
}

// Spherical expansion along the six coordinate rays from the box centre, from the grid fields interpolated at
// r_k = (k + 1/2) dr (apparent-horizon diagnostic for a nearly spherical configuration).  Along a ray with unit
// direction n: R = r sqrt(h~_T / chi) with h~_T = (tr h~ - h~_nn) / 2, gamma_nn = h~_nn / chi,
// Theta = 2 (dR/dr) / (R sqrt(gamma_nn)) - 2 K / 3 + A~_nn / h~_nn  (uses A^th_th = -A^r_r / 2),
// M_MS = (R / 2) (1 - (dR/dr)^2 / gamma_nn + (R K_T)^2) with K_T = K / 3 - A~_nn / (2 h~_nn).
// The outermost zero crossing of Theta (negative inside, positive outside) is the horizon candidate; its mean
// coordinate radius over the rays is stored in amr->ah_ray_radius (-1 if fewer than 3 rays find one).
void theta_ray_diagnostic(PBHVlasovAmr *amr, const amrex::Geometry &geom, amrex::Real time, long step,
                          const InitParams &ip, bool first_step)
{
    constexpr int NR = 6, NC = 14;
    static const amrex::Real dirs[NR][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const int comps[NC] = {c_chi, c_h11, c_h12, c_h13, c_h22, c_h23, c_h33, c_K, c_A11, c_A12, c_A13, c_A22, c_A23, c_A33};
    const amrex::Real dx_f = geom.CellSize(0) / (1 << amr->finestLevel());
    const amrex::Real dr = ip.theta_ray_dr * dx_f;
    const auto plo = geom.ProbLoArray(), phi = geom.ProbHiArray();
    // rays stay inside the domain (and clear of the periodic images): r_max < 0.45 L
    const int n = std::max(4, std::min(ip.theta_ray_n, static_cast<int>(0.45 * (phi[0] - plo[0]) / dr)));
    const amrex::Real c0[3] = {0.5 * (plo[0] + phi[0]), 0.5 * (plo[1] + phi[1]), 0.5 * (plo[2] + phi[2])};
    const bool io = amrex::ParallelDescriptor::IOProcessor();
    const int npts = io ? NR * n : 0;
    std::vector<amrex::ParticleReal> x(npts), y(npts), z(npts);
    for (int ray = 0; ray < NR && io; ++ray)
        for (int k = 0; k < n; ++k)
        {
            const amrex::Real r = (k + 0.5) * dr;
            x[ray * n + k] = c0[0] + r * dirs[ray][0];
            y[ray * n + k] = c0[1] + r * dirs[ray][1];
            z[ray * n + k] = c0[2] + r * dirs[ray][2];
        }
    std::vector<amrex::ParticleReal> data(static_cast<size_t>(npts) * NC, 0.0);
    InterpolationQueryParticle query(npts);
    query.setCoords(0, x.data()).setCoords(1, y.data()).setCoords(2, z.data());
    for (int c = 0; c < NC; ++c)
        query.addComp(comps[c], data.data() + static_cast<size_t>(c) * npts);
    amr->theta_interpolator.interp(query, false); // collective
    amr->ah_ray_radius = -1.0;
    if (io)
    {
    const bool write_prof = ip.theta_profile_interval > 0 && step % ip.theta_profile_interval == 0;
    std::ofstream prof;
    if (write_prof)
    {
        std::filesystem::create_directory("theta_profiles");
        prof.open("theta_profiles/theta_" + std::to_string(step) + ".dat");
        prof.precision(8);
        prof << "# time " << time << "\n# ray r R Theta M_MS\n";
    }
    std::vector<double> r(n), R(n), gnn(n), Kv(n), Annh(n), Th(n), MMS(n);
    double r_ah[NR], R_ah[NR], M_ah[NR];
    int nfound = 0;
    double sum_r = 0, sum_R = 0, sum_M = 0, min_R = 1e300, max_R = -1e300;
    for (int ray = 0; ray < NR; ++ray)
    {
        const amrex::Real *nn = dirs[ray];
        for (int k = 0; k < n; ++k)
        {
            auto g = [&](int c) { return static_cast<double>(data[static_cast<size_t>(c) * npts + ray * n + k]); };
            const double chi = g(0);
            const double h[3][3] = {{g(1), g(2), g(3)}, {g(2), g(4), g(5)}, {g(3), g(5), g(6)}};
            const double A[3][3] = {{g(8), g(9), g(10)}, {g(9), g(11), g(12)}, {g(10), g(12), g(13)}};
            double hnn = 0, Ann = 0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    hnn += nn[i] * nn[j] * h[i][j];
                    Ann += nn[i] * nn[j] * A[i][j];
                }
            const double hT = 0.5 * (h[0][0] + h[1][1] + h[2][2] - hnn);
            r[k] = (k + 0.5) * dr;
            R[k] = r[k] * std::sqrt(std::max(hT / chi, 0.0));
            gnn[k] = hnn / chi;
            Kv[k] = g(7);
            Annh[k] = Ann / hnn;
        }
        for (int k = 0; k < n; ++k)
        {
            double dRdr;
            if (k == 0) dRdr = (R[1] - R[0]) / dr;
            else if (k == n - 1) dRdr = (R[n - 1] - R[n - 2]) / dr;
            else dRdr = 0.5 * (R[k + 1] - R[k - 1]) / dr;
            Th[k] = 2.0 * dRdr / (R[k] * std::sqrt(gnn[k])) - 2.0 * Kv[k] / 3.0 + Annh[k];
            const double KT = Kv[k] / 3.0 - 0.5 * Annh[k];
            MMS[k] = 0.5 * R[k] * (1.0 - dRdr * dRdr / gnn[k] + (R[k] * KT) * (R[k] * KT));
            if (write_prof)
                prof << ray << " " << r[k] << " " << R[k] << " " << Th[k] << " " << MMS[k] << "\n";
        }
        int kc = -1;
        for (int k = n - 2; k >= 1; --k)
            if (Th[k] <= 0.0 && Th[k + 1] > 0.0 && Th[k - 1] <= 0.0)
            {
                kc = k;
                break;
            }
        r_ah[ray] = -1.0; R_ah[ray] = 0.0; M_ah[ray] = 0.0;
        if (kc >= 0)
        {
            const double sfrac = -Th[kc] / (Th[kc + 1] - Th[kc]);
            r_ah[ray] = r[kc] + sfrac * dr;
            R_ah[ray] = R[kc] + sfrac * (R[kc + 1] - R[kc]);
            M_ah[ray] = 0.5 * R_ah[ray];
            ++nfound;
            sum_r += r_ah[ray]; sum_R += R_ah[ray]; sum_M += M_ah[ray];
            min_R = std::min(min_R, R_ah[ray]); max_R = std::max(max_R, R_ah[ray]);
        }
    }
    if (nfound >= 3)
        amr->ah_ray_radius = sum_r / nfound;
    std::ofstream out("pbh_ah_rays.dat", first_step ? std::ios::out : std::ios::app);
    if (first_step)
        out << "# time n_rays_found <r_AH> <R_AH> <M_AH>=<R_AH>/2 R_min R_max r_AH(+x -x +y -y +z -z)\n";
    out.precision(10);
    out << time << " " << nfound << " " << (nfound ? sum_r / nfound : 0.0) << " " << (nfound ? sum_R / nfound : 0.0)
        << " " << (nfound ? sum_M / nfound : 0.0) << " " << (nfound ? min_R : 0.0) << " " << (nfound ? max_R : 0.0);
    for (int ray = 0; ray < NR; ++ray)
        out << " " << r_ah[ray];
    out << "\n";
    }
    amrex::ParallelDescriptor::Bcast(&amr->ah_ray_radius, 1, amrex::ParallelDescriptor::IOProcessorNumber());
}

// Expansion of the coordinate spheres r = const about the box centre (horizon diagnostic without symmetry).
// With n_i = x_i / r, N = sqrt(gamma^{ij} n_i n_j), s^i = gamma^{ij} n_j / N (unit normal of the sphere):
//   Theta = D_i s^i + K_ij s^i s^j - K = d_i s^i - (3/2) s^i d_i chi / chi + A~_ij s^i s^j / chi - (2/3) K,
// area element sqrt(gamma) N r^2 dOmega = chi^{-3/2} N r^2 dOmega.  On sphere_ndir quasi-uniform directions and radii
// r_k = (k + 1/2) dr the minimum, mean and maximum of Theta and the area are formed.  r_in = largest radius whose
// sphere is trapped everywhere (max Theta <= 0: a horizon exists and encloses it); r_out = radius above which no
// sphere has a trapped point (min Theta > 0).  M = sqrt(A / 16 pi).  Output: pbh_ah_spheres.dat, sphere_profiles/.
// Reduces to the ray diagnostic in spherical symmetry; r_in replaces the ray radius for the tau statistics.
void sphere_diagnostic(PBHVlasovAmr *amr, const amrex::Geometry &geom, amrex::Real time, long step, const InitParams &ip,
                       bool first_step)
{
    constexpr int NV = 14, NC = 35;
    const int vcomp[NV] = {c_chi, c_h11, c_h12, c_h13, c_h22, c_h23, c_h33, c_K, c_A11, c_A12, c_A13, c_A22, c_A23, c_A33};
    static constexpr int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    const int ndir = ip.sphere_ndir;
    const amrex::Real dx_f = geom.CellSize(0) / (1 << amr->finestLevel());
    const amrex::Real dr = ip.sphere_dr * dx_f;
    const auto plo = geom.ProbLoArray(), phi = geom.ProbHiArray();
    const int nr = std::max(4, std::min(ip.sphere_nr, static_cast<int>(0.45 * (phi[0] - plo[0]) / dr)));
    const amrex::Real c0[3] = {0.5 * (plo[0] + phi[0]), 0.5 * (plo[1] + phi[1]), 0.5 * (plo[2] + phi[2])};
    const bool io = amrex::ParallelDescriptor::IOProcessor();
    const int npts = io ? ndir * nr : 0;
    // Fibonacci directions
    std::vector<std::array<double, 3>> dirs(ndir);
    for (int i = 0; i < ndir; ++i)
    {
        const double zz = 1.0 - (2.0 * i + 1.0) / ndir, rho = std::sqrt(std::max(0.0, 1.0 - zz * zz));
        const double ph = i * M_PI * (3.0 - std::sqrt(5.0));
        dirs[i] = {rho * std::cos(ph), rho * std::sin(ph), zz};
    }
    std::vector<amrex::ParticleReal> x(npts), y(npts), z(npts);
    for (int i = 0; i < ndir && io; ++i)
        for (int k = 0; k < nr; ++k)
        {
            const amrex::Real r = (k + 0.5) * dr;
            x[i * nr + k] = c0[0] + r * dirs[i][0];
            y[i * nr + k] = c0[1] + r * dirs[i][1];
            z[i * nr + k] = c0[2] + r * dirs[i][2];
        }
    std::vector<amrex::ParticleReal> data(static_cast<size_t>(npts) * NC, 0.0);
    InterpolationQueryParticle query(npts);
    query.setCoords(0, x.data()).setCoords(1, y.data()).setCoords(2, z.data());
    for (int c = 0; c < NV; ++c)
        query.addComp(vcomp[c], data.data() + static_cast<size_t>(c) * npts);
    for (int f = 0; f < 7; ++f) // first derivatives of chi and h~_ab: slot NV + 3 f + d
        for (int d = 0; d < 3; ++d)
        {
            Derivative dd;
            dd[d] = 1;
            query.addComp(vcomp[f], data.data() + static_cast<size_t>(NV + 3 * f + d) * npts, VariableType::state,
                          BCParity::undefined, dd);
        }
    amr->sphere_interpolator.interp(query, false); // collective
    double r_in = -1.0;
    if (io)
    {
        std::vector<double> th_min(nr, 1e300), th_max(nr, -1e300), th_mean(nr, 0.0), area(nr, 0.0);
        for (int i = 0; i < ndir; ++i)
            for (int k = 0; k < nr; ++k)
            {
                const size_t idx = static_cast<size_t>(i) * nr + k;
                auto g = [&](int c) { return static_cast<double>(data[static_cast<size_t>(c) * npts + idx]); };
                const double chi = g(0), Kt = g(7), r = (k + 0.5) * dr;
                double h[3][3], A[3][3], dh[3][3][3], dchi[3];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                    {
                        h[a][b] = g(1 + sym[a][b]);
                        A[a][b] = g(8 + sym[a][b]);
                        for (int d = 0; d < 3; ++d)
                            dh[d][a][b] = g(NV + 3 * (1 + sym[a][b]) + d);
                    }
                for (int d = 0; d < 3; ++d)
                    dchi[d] = g(NV + d);
                const double det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1]) -
                                   h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0]) +
                                   h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
                double hU[3][3];
                hU[0][0] = (h[1][1] * h[2][2] - h[1][2] * h[2][1]) / det;
                hU[0][1] = (h[0][2] * h[2][1] - h[0][1] * h[2][2]) / det;
                hU[0][2] = (h[0][1] * h[1][2] - h[0][2] * h[1][1]) / det;
                hU[1][1] = (h[0][0] * h[2][2] - h[0][2] * h[2][0]) / det;
                hU[1][2] = (h[0][2] * h[1][0] - h[0][0] * h[1][2]) / det;
                hU[2][2] = (h[0][0] * h[1][1] - h[0][1] * h[1][0]) / det;
                hU[1][0] = hU[0][1]; hU[2][0] = hU[0][2]; hU[2][1] = hU[1][2];
                const double *n = dirs[i].data();
                double gU[3][3], dgU[3][3][3]; // dgU[k][i][j] = d_k gamma^{ij}
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                    {
                        gU[a][b] = chi * hU[a][b];
                        for (int d = 0; d < 3; ++d)
                        {
                            double dhU = 0.0;
                            for (int l = 0; l < 3; ++l)
                                for (int m = 0; m < 3; ++m)
                                    dhU -= hU[a][l] * hU[b][m] * dh[d][l][m];
                            dgU[d][a][b] = dchi[d] * hU[a][b] + chi * dhU;
                        }
                    }
                double gn[3] = {0, 0, 0}, N2 = 0.0; // gamma^{ij} n_j and N^2
                for (int a = 0; a < 3; ++a)
                {
                    for (int b = 0; b < 3; ++b)
                        gn[a] += gU[a][b] * n[b];
                    N2 += gn[a] * n[a];
                }
                const double N = std::sqrt(N2);
                double div = 0.0;
                for (int a = 0; a < 3; ++a)
                {
                    double dN = 0.0; // d_a N
                    for (int b = 0; b < 3; ++b)
                    {
                        div += dgU[a][a][b] * n[b] / N;                                   // (d_i gamma^{ij}) n_j / N
                        div += gU[a][b] * ((a == b ? 1.0 : 0.0) - n[a] * n[b]) / (r * N); // gamma^{ij} d_i n_j / N
                        for (int c = 0; c < 3; ++c)
                            dN += dgU[a][b][c] * n[b] * n[c];
                        dN += 2.0 * gn[b] * ((a == b ? 1.0 : 0.0) - n[a] * n[b]) / r;
                    }
                    dN /= 2.0 * N;
                    div -= gn[a] * dN / N2;
                }
                double Ass = 0.0, sdchi = 0.0;
                for (int a = 0; a < 3; ++a)
                {
                    sdchi += gn[a] / N * dchi[a];
                    for (int b = 0; b < 3; ++b)
                        Ass += A[a][b] * gn[a] * gn[b] / N2;
                }
                const double Theta = div - 1.5 * sdchi / chi + Ass / chi - 2.0 * Kt / 3.0;
                th_min[k] = std::min(th_min[k], Theta);
                th_max[k] = std::max(th_max[k], Theta);
                th_mean[k] += Theta / ndir;
                area[k] += 4.0 * M_PI * r * r / ndir * std::pow(chi, -1.5) * N;
            }
        auto mass = [&](int k) { return std::sqrt(area[k] / (16.0 * M_PI)); };
        // r_in: outermost zero of max Theta below which the spheres are trapped; r_out: innermost radius of the
        // untrapped exterior
        int kin = -1, kout = -1;
        for (int k = nr - 2; k >= 1; --k)
            if (th_max[k] <= 0.0 && th_max[k + 1] > 0.0 && th_max[k - 1] <= 0.0)
            {
                kin = k;
                break;
            }
        for (int k = nr - 1; k >= 0 && th_min[k] > 0.0; --k)
            kout = k;
        double R_in = 0, M_in = 0, r_out = -1.0, M_out = 0, spread = 0;
        if (kin >= 0)
        {
            const double f = -th_max[kin] / (th_max[kin + 1] - th_max[kin]);
            r_in = (kin + 0.5 + f) * dr;
            M_in = mass(kin) + f * (mass(kin + 1) - mass(kin));
            R_in = 2.0 * M_in;
            spread = th_max[kin] - th_min[kin];
        }
        if (kout >= 1 && kout < nr - 1)
        {
            const double f = (th_min[kout - 1] < 0.0) ? th_min[kout] / (th_min[kout] - th_min[kout - 1]) : 0.0;
            r_out = (kout + 0.5 - f) * dr;
            M_out = mass(kout) - f * (mass(kout) - mass(kout - 1));
        }
        std::ofstream out("pbh_ah_spheres.dat", first_step ? std::ios::out : std::ios::app);
        if (first_step)
            out << "# time r_in R_in=2M_in M_in r_out M_out Theta_spread(r_in)  (r_in: all-trapped sphere, r_out: no "
                   "trapped point outside)\n";
        out.precision(10);
        out << time << " " << r_in << " " << R_in << " " << M_in << " " << r_out << " " << M_out << " " << spread << "\n";
        if (ip.theta_profile_interval > 0 && step % ip.theta_profile_interval == 0)
        {
            std::filesystem::create_directory("sphere_profiles");
            std::ofstream pf("sphere_profiles/sph_" + std::to_string(step) + ".dat");
            pf.precision(8);
            pf << "# time " << time << "\n# r Theta_min Theta_mean Theta_max area M=sqrt(A/16pi)\n";
            for (int k = 0; k < nr; ++k)
                pf << (k + 0.5) * dr << " " << th_min[k] << " " << th_mean[k] << " " << th_max[k] << " " << area[k] << " "
                   << mass(k) << "\n";
        }
        if (r_in > 0.0)
            amr->ah_ray_radius = r_in;
    }
    amrex::ParallelDescriptor::Bcast(&amr->ah_ray_radius, 1, amrex::ParallelDescriptor::IOProcessorNumber());
}

// proper time of the matter at the ray horizon (shell |r - r_AH| < max(dx_f, 0.05 r_AH)), inside it and in the core
// (r < 2 dx_f): pbh_ah_tau.dat
void tau_diagnostic(PBHVlasovAmr *amr, const amrex::Geometry &geom, amrex::Real time, bool first_step, long step,
                    int profile_interval)
{
    const amrex::Real dx_f = geom.CellSize(0) / (1 << amr->finestLevel());
    const auto plo = geom.ProbLoArray(), phi = geom.ProbHiArray();
    const amrex::Real c0[3] = {0.5 * (plo[0] + phi[0]), 0.5 * (plo[1] + phi[1]), 0.5 * (plo[2] + phi[2])};
    const amrex::Real r_ah = amr->ah_ray_radius;
    VlasovParticles::tau_stats_t shell, inside;
    if (r_ah > 0.0)
    {
        const amrex::Real w = std::max(dx_f, 0.05 * r_ah);
        shell = amr->particles.tau_in_shell(c0, std::max(0.0, r_ah - w), r_ah + w);
        inside = amr->particles.tau_in_shell(c0, 0.0, r_ah);
    }
    const VlasovParticles::tau_stats_t core = amr->particles.tau_in_shell(c0, 0.0, 2.0 * dx_f);
    // radial profile of the particle proper time (bins of 2 dx_f): tau_profiles/tau_<step>.dat
    if (profile_interval > 0 && step % profile_interval == 0)
    {
        constexpr int nb = 96;
        std::vector<double> n, mass, mtau, mgam;
        amr->particles.tau_profile(c0, 2.0 * dx_f, nb, n, mass, mtau, mgam);
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            std::filesystem::create_directory("tau_profiles");
            std::ofstream pf("tau_profiles/tau_" + std::to_string(step) + ".dat");
            pf.precision(10);
            pf << "# time " << time << "\n# k r_lo r_hi n mass M_in(rest, cumulative) <tau> <Gamma>\n";
            double cum = 0.0;
            for (int k = 0; k < nb; ++k)
            {
                cum += mass[k];
                pf << k << " " << 2.0 * dx_f * k << " " << 2.0 * dx_f * (k + 1) << " " << n[k] << " " << mass[k] << " "
                   << cum << " " << (mass[k] > 0 ? mtau[k] / mass[k] : 0.0) << " "
                   << (mass[k] > 0 ? mgam[k] / mass[k] : 0.0) << "\n";
            }
        }
    }
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ofstream out("pbh_ah_tau.dat", first_step ? std::ios::out : std::ios::app);
        if (first_step)
            out << "# time r_AH n_shell <tau>_shell tau_min tau_max <Gamma>_shell n_in M_in <tau>_in n_core <tau>_core\n";
        out.precision(10);
        out << time << " " << r_ah << " " << shell.n << " " << shell.tau_mean << " " << shell.tau_min << " "
            << shell.tau_max << " " << shell.gamma_mean << " " << inside.n << " " << inside.mass << " "
            << inside.tau_mean << " " << core.n << " " << core.tau_mean << "\n";
    }
}
} // namespace

PBHVlasovAmr *PBHVlasovLevel::get_vlasov_amr_ptr() { return dynamic_cast<PBHVlasovAmr *>(get_gr_amr_ptr()); }

void PBHVlasovLevel::variableSetUp()
{
    BL_PROFILE("PBHVlasovLevel::variableSetUp()");
    state_variable_set_up();
    VlasovConstraints::set_up(state_index);
    VlasovEnergyDensity::set_up(state_index);
}

void PBHVlasovLevel::load_table()
{
    if (m_table_loaded)
        return;
    InitParams ip;
    ip.read();
    std::ifstream in(ip.table);
    if (!in)
        amrex::Abort("PBHVlasov: cannot open table " + ip.table);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ss(line);
        amrex::Real r, chi, K, Arr, rho, ur;
        if (!(ss >> r >> chi >> K >> Arr >> rho >> ur))
            continue;
        m_tab_r.push_back(r); m_tab_chi.push_back(chi); m_tab_K.push_back(K); m_tab_Arr.push_back(Arr);
        m_tab_rho.push_back(rho); m_tab_ur.push_back(ur);
        amrex::Real sr = 0.0, st = 0.0; // optional warm-start columns: sigma_r, sigma_t ratios at t_0
        if (ss >> sr >> st)
        {
            m_tab_sr.push_back(sr); m_tab_st.push_back(st);
        }
    }
    if (m_tab_r.size() < 4)
        amrex::Abort("PBHVlasov: table too short");
    m_table_loaded = true;
    amrex::Print() << "PBHVlasov: table " << ip.table << " with " << m_tab_r.size() << " rows, r_max = "
                   << m_tab_r.back() << ", chi(0) = " << m_tab_chi.front() << "\n";
}

void PBHVlasovLevel::initData()
{
    BL_PROFILE("PBHVlasovLevel::initData()");
    InitParams ip;
    ip.read();
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays = state_new.arrays();
    const auto dx = Geom().CellSizeArray();
    const auto plo = Geom().ProbLoArray();
    const auto phi = Geom().ProbHiArray();
    const amrex::Real cx = 0.5 * (plo[0] + phi[0]), cy = 0.5 * (plo[1] + phi[1]), cz = 0.5 * (plo[2] + phi[2]);

    if (ip.mode == "flrw")
    {
        const amrex::Real K0 = -3.0 * ip.H0;
        amrex::ParallelFor(state_new, state_new.nGrowVect(), [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
            const amrex::CellData<amrex::Real> cell = state_arrays[box_no].cellData(ix, iy, iz);
            for (int c = 0; c < cell.nComp(); ++c)
                cell[c] = 0.0;
            cell[c_chi] = 1.0; cell[c_h11] = 1.0; cell[c_h22] = 1.0; cell[c_h33] = 1.0;
            cell[c_K] = K0; cell[c_lapse] = 1.0; cell[c_fref] = 1.0;
        });
    }
    else if (ip.mode == "yoo")
    {
        const amrex::Real mu = ip.mu, ee = ip.ell_e, ep = ip.ell_p, kp = ip.kp, Hi = ip.H0, aref = ip.a_ref;
        const amrex::Real Lb[3] = {phi[0] - plo[0], phi[1] - plo[1], phi[2] - plo[2]};
        amrex::ParallelFor(state_new, state_new.nGrowVect(), [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
            const amrex::CellData<amrex::Real> cell = state_arrays[box_no].cellData(ix, iy, iz);
            for (int c = 0; c < cell.nComp(); ++c)
                cell[c] = 0.0;
            // comoving position relative to the centre (minimum image for the ghost cells of the periodic box)
            amrex::Real xp[3] = {plo[0] + (ix + 0.5) * dx[0] - cx, plo[1] + (iy + 0.5) * dx[1] - cy,
                                 plo[2] + (iz + 0.5) * dx[2] - cz};
            for (int d = 0; d < 3; ++d)
            {
                if (xp[d] > 0.5 * Lb[d]) xp[d] -= Lb[d];
                if (xp[d] < -0.5 * Lb[d]) xp[d] += Lb[d];
            }
            const YooData D = yoo_data(xp[0] / aref, xp[1] / aref, xp[2] / aref, mu, ee, ep, kp, Hi);
            const amrex::Real det = D.g[0][0] * (D.g[1][1] * D.g[2][2] - D.g[1][2] * D.g[2][1]) -
                                    D.g[0][1] * (D.g[1][0] * D.g[2][2] - D.g[1][2] * D.g[2][0]) +
                                    D.g[0][2] * (D.g[1][0] * D.g[2][1] - D.g[1][1] * D.g[2][0]);
            const amrex::Real d13 = std::cbrt(det), psi4 = D.psi * D.psi * D.psi * D.psi;
            cell[c_chi] = aref * aref / (psi4 * d13);
            cell[c_h11] = D.g[0][0] / d13; cell[c_h12] = D.g[0][1] / d13; cell[c_h13] = D.g[0][2] / d13;
            cell[c_h22] = D.g[1][1] / d13; cell[c_h23] = D.g[1][2] / d13; cell[c_h33] = D.g[2][2] / d13;
            cell[c_A11] = D.A[0][0] / d13; cell[c_A12] = D.A[0][1] / d13; cell[c_A13] = D.A[0][2] / d13;
            cell[c_A22] = D.A[1][1] / d13; cell[c_A23] = D.A[1][2] / d13; cell[c_A33] = D.A[2][2] / d13;
            cell[c_K] = -3.0 * Hi; cell[c_lapse] = 1.0; cell[c_fref] = 1.0;
        });
    }
    else
    {
        load_table();
        // copy the table to device-accessible vectors
        const int nt = static_cast<int>(m_tab_r.size());
        amrex::Gpu::DeviceVector<amrex::Real> d_r(nt), d_chi(nt), d_K(nt), d_Arr(nt);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_tab_r.begin(), m_tab_r.end(), d_r.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_tab_chi.begin(), m_tab_chi.end(), d_chi.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_tab_K.begin(), m_tab_K.end(), d_K.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_tab_Arr.begin(), m_tab_Arr.end(), d_Arr.begin());
        const amrex::Real *pr = d_r.data(), *pchi = d_chi.data(), *pK = d_K.data(), *pArr = d_Arr.data();
        const int n = static_cast<int>(m_tab_r.size());
        const amrex::Real dr = m_tab_r[1] - m_tab_r[0];
        amrex::ParallelFor(state_new, state_new.nGrowVect(), [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
            const amrex::CellData<amrex::Real> cell = state_arrays[box_no].cellData(ix, iy, iz);
            for (int c = 0; c < cell.nComp(); ++c)
                cell[c] = 0.0;
            const amrex::Real x = plo[0] + (ix + 0.5) * dx[0] - cx, y = plo[1] + (iy + 0.5) * dx[1] - cy,
                              z = plo[2] + (iz + 0.5) * dx[2] - cz;
            const amrex::Real r = std::sqrt(x * x + y * y + z * z);
            auto interp = [&](const amrex::Real *tab) {
                if (r >= pr[n - 1]) return tab[n - 1];
                const amrex::Real s = r / dr;
                int i = static_cast<int>(s); if (i > n - 2) i = n - 2;
                const amrex::Real t = s - i;
                return (1 - t) * tab[i] + t * tab[i + 1];
            };
            const amrex::Real chi = interp(pchi), K = interp(pK), Arr = interp(pArr), Ath = -0.5 * Arr;
            amrex::Real nx = 0, ny = 0, nz = 0;
            if (r > 1e-12 * dx[0]) { nx = x / r; ny = y / r; nz = z / r; }
            cell[c_chi] = chi; cell[c_h11] = 1.0; cell[c_h22] = 1.0; cell[c_h33] = 1.0;
            cell[c_K] = K; cell[c_lapse] = 1.0;
            cell[c_A11] = Ath + (Arr - Ath) * nx * nx; cell[c_A22] = Ath + (Arr - Ath) * ny * ny;
            cell[c_A33] = Ath + (Arr - Ath) * nz * nz; cell[c_A12] = (Arr - Ath) * nx * ny;
            cell[c_A13] = (Arr - Ath) * nx * nz; cell[c_A23] = (Arr - Ath) * ny * nz;
            cell[c_fref] = K / pK[n - 1]; // 1 far away, 0 at the turned-around centre
        });
    }
    const GammaCalculator<FourthOrderDerivatives> gamma_calculator(Geom().CellSize(0));
    const IntegratedMovingPunctureGauge<FourthOrderDerivatives> gauge(Geom().CellSize(0));
    amrex::ParallelFor(state_new, [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        gamma_calculator(ix, iy, iz, state_arrays[box_no]);
        gauge.set_initial_B_to_Gamma(ix, iy, iz, state_arrays[box_no]);
    });
    amrex::Gpu::streamSynchronize();
}

void PBHVlasovLevel::specific_post_init()
{
    BL_PROFILE("PBHVlasovLevel::specific_post_init()");
    // particles are created once, on the finest level present at init (single-level runs: level 0)
    InitParams ip;
    ip.read();
    auto *amr = get_vlasov_amr_ptr();
    if (Level() == 0)
        amr->particles.setup(amr);
    if (Level() == amr->finestLevel())
    {
        const amrex::Geometry &geom0 = amr->getLevel(0).Geom();
        amr->particles.set_lattice_buffer(ip.lattice_buffer);
        amr->particles.set_max_disp_cells(ip.max_disp_cells);
        {
            int vd = 0;
            GRParmParse("pbh_vlasov").query("verbose_deposit", vd);
            amr->particles.set_verbose_deposit(vd != 0);
            int dg = 1, df = 0, dv = 1;
            GRParmParse("pbh_vlasov").query("dep_ghosts", dg);
            GRParmParse("pbh_vlasov").query("dep_fine_to_coarse", df);
            GRParmParse("pbh_vlasov").query("dep_virtuals", dv);
            amr->particles.set_deposit_options(dg != 0, df != 0, dv != 0);
        }
        amr->particles.set_tau0(ip.tau0);
        if (ip.mode == "flrw")
        {
            const amrex::Real rho = 3.0 * ip.H0 * ip.H0 / (8.0 * M_PI);
            amr->particles.init_uniform(geom0, rho, ip.n_per_dir);
        }
        else if (ip.mode == "yoo")
        {
            amr->particles.init_lattice(geom0, ip.n_per_dir);
            for (int lev = 0; lev <= amr->finestLevel(); ++lev)
                dynamic_cast<PBHVlasovLevel &>(amr->getLevel(lev)).init_particles_from_constraints();
            // match the deposited sources to the constraints (see build_deposit_correction)
            int n_match = 8;
            GRParmParse("pbh_vlasov").query("yoo_match_iter", n_match);
            for (int it = 0; it <= n_match; ++it)
            {
                amr->deposit_all_levels(state_index);
                amrex::Real res = 0.0;
                for (int lev = 0; lev <= amr->finestLevel(); ++lev)
                    res = std::max(res, dynamic_cast<PBHVlasovLevel &>(amr->getLevel(lev)).build_deposit_correction());
                amrex::Print() << "PBHVlasov: deposit vs constraints after " << it << " corrections: max |D_target/D_dep - 1| = "
                               << res << "\n";
                if (it == n_match)
                    break;
                for (int lev = 0; lev <= amr->finestLevel(); ++lev)
                    dynamic_cast<PBHVlasovLevel &>(amr->getLevel(lev)).apply_deposit_correction();
            }
            for (int lev = 0; lev <= amr->finestLevel(); ++lev)
                dynamic_cast<PBHVlasovLevel &>(amr->getLevel(lev)).release_deposit_targets();
        }
        else
        {
            load_table();
            if (ip.sigma > 0.0 && m_tab_sr.size() != m_tab_r.size())
                amrex::Abort("PBHVlasov: sigma > 0 needs the table columns sigma_r, sigma_t");
            amr->particles.init_from_table(geom0, m_tab_r, m_tab_rho, m_tab_ur, m_tab_chi, ip.n_per_dir, m_tab_sr,
                                           m_tab_st, ip.sigma, static_cast<std::uint64_t>(ip.sigma_seed));
            if (ip.sigma > 0.0)
            {
                const auto plo0 = geom0.ProbLoArray(), phi0 = geom0.ProbHiArray();
                const amrex::Real c0[3] = {0.5 * (plo0[0] + phi0[0]), 0.5 * (plo0[1] + phi0[1]), 0.5 * (plo0[2] + phi0[2])};
                const amrex::Real dxf = geom0.CellSize(0) / (1 << amr->finestLevel());
                const auto core = amr->particles.tau_in_shell(c0, 0.0, 16.0 * dxf);
                amrex::Print() << "PBHVlasov: warm start, sigma = " << ip.sigma << " at injection; <Gamma> - 1 within 16 dx_f = "
                               << core.gamma_mean - 1.0 << " (" << core.n << " particles)\n";
            }
        }
        amr->particles_ready = true;
        amrex::Print() << "PBHVlasov: " << amr->particles.TotalNumberOfParticles() << " particles on "
                       << amr->finestLevel() + 1 << " levels, total rest mass " << amr->particles.total_mass(0) << "\n";
        amr->deposit_all_levels(state_index);
        amr->K_far = amr->getLevel(0).get_new_data(state_index).min(c_K); // far-field reference before the first RHS
        amr->K_far_time = get_state_data(state_index).curTime();
    }
}

void PBHVlasovLevel::init_particles_from_constraints()
{
    BL_PROFILE("PBHVlasovLevel::init_particles_from_constraints()");
    amrex::MultiFab &state_new = get_new_data(state_index);
    const amrex::Real time = get_state_data(state_index).curTime();
    amrex::MultiFab state_gh(state_new.boxArray(), state_new.DistributionMap(), state_new.nComp(), 3);
    FillPatch(*this, state_gh, 3, time, state_index, 0, state_new.nComp());
    // vacuum constraints on the valid cells grown by one: E = Ham / 16 pi, J_i = Mom_i / 8 pi
    amrex::MultiFab con(state_new.boxArray(), state_new.DistributionMap(), 4, 1);
    m_yoo_target = std::make_unique<amrex::MultiFab>(state_new.boxArray(), state_new.DistributionMap(), 11, 1);
    amrex::MultiFab &fld = *m_yoo_target;
    const Constraints constraints(Geom().CellSize(0), 0, Interval(1, 3));
    const auto &gh = state_gh.const_arrays();
    const auto &ca = con.arrays();
    const auto &fa = fld.arrays();
    amrex::ParallelFor(con, amrex::IntVect(1), [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        constraints(ix, iy, iz, ca[box_no], gh[box_no]);
        fa[box_no](ix, iy, iz, 0) = ca[box_no](ix, iy, iz, 0) / (16.0 * M_PI);
        for (int i = 0; i < 3; ++i)
            fa[box_no](ix, iy, iz, 1 + i) = ca[box_no](ix, iy, iz, 1 + i) / (8.0 * M_PI);
        fa[box_no](ix, iy, iz, 4) = gh[box_no](ix, iy, iz, c_chi);
        const int hc[6] = {c_h11, c_h12, c_h13, c_h22, c_h23, c_h33};
        for (int i = 0; i < 6; ++i)
            fa[box_no](ix, iy, iz, 5 + i) = gh[box_no](ix, iy, iz, hc[i]);
    });
    amrex::Gpu::streamSynchronize();
    get_vlasov_amr_ptr()->particles.set_from_fields(Level(), fld, Geom());
    amrex::Print() << "PBHVlasov: level " << Level() << " particles from the constraints, E in [" << fld.min(0, 0) << ", "
                   << fld.max(0, 0) << "], max |J| " << std::max({fld.norm0(1), fld.norm0(2), fld.norm0(3)}) << "\n";
}

// At a super-horizon start the binding energy of a shell is a small difference of large terms (2E against U^2 and
// 2m/R, 10^2-10^3 times larger), so a 10^-3 error of the deposited density (CIC smoothing of the O(1) variation of
// sqrt(gamma) E, level interfaces) changes E by tens of per cent.  The particle masses are therefore iterated until
// the deposited coordinate density equals the constraint one, and u_i is shifted to match the momentum density.
amrex::Real PBHVlasovLevel::build_deposit_correction()
{
    const amrex::MultiFab &state_new = get_new_data(state_index);
    m_yoo_corr = std::make_unique<amrex::MultiFab>(state_new.boxArray(), state_new.DistributionMap(), 4, 1);
    m_yoo_corr->setVal(1.0, 0, 1, 1);
    m_yoo_corr->setVal(0.0, 1, 3, 1);
    amrex::MultiFab dev(state_new.boxArray(), state_new.DistributionMap(), 1, 0);
    {
        const auto &sa = state_new.const_arrays();
        const auto &ta = m_yoo_target->const_arrays();
        const auto &ca = m_yoo_corr->arrays();
        const auto &da = dev.arrays();
        amrex::ParallelFor(dev, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) {
            const amrex::Real sg = std::pow(ta[b](i, j, k, 4), -1.5); // sqrt(gamma)
            const amrex::Real Dt = ta[b](i, j, k, 0) * sg;            // target coordinate energy density
            ca[b](i, j, k, 0) = Dt / sa[b](i, j, k, c_rho_p);
            for (int d = 0; d < 3; ++d)
                ca[b](i, j, k, 1 + d) = (ta[b](i, j, k, 1 + d) * sg - sa[b](i, j, k, c_S1 + d)) / Dt;
            da[b](i, j, k) = ca[b](i, j, k, 0) - 1.0;
        });
        amrex::Gpu::streamSynchronize();
    }
    // residual on the cells not covered by a finer level (covered cells hold averaged-down fine values, whose
    // difference from the coarse target is a discretisation effect that no particle carries)
    if (Level() < parent->finestLevel())
    {
        const amrex::MultiFab &fine = parent->getLevel(Level() + 1).get_new_data(state_index);
        const amrex::iMultiFab mask = amrex::makeFineMask(dev, fine.boxArray(), parent->refRatio(Level()), 1, 0);
        const auto &ma = mask.const_arrays();
        const auto &da = dev.arrays();
        amrex::ParallelFor(dev, [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) { da[b](i, j, k) *= ma[b](i, j, k); });
        amrex::Gpu::streamSynchronize();
    }
    const amrex::Real res = dev.norm0(0, 0);
    amrex::Print() << "    level " << Level() << ": max " << res << ", rms " << dev.norm2(0) / std::sqrt(dev.boxArray().d_numPts())
                   << "\n";
    if (Level() > 0)
    {
        // ghost cells outside the level: piecewise-constant values of the coarser level's correction
        auto &crse = dynamic_cast<PBHVlasovLevel &>(parent->getLevel(Level() - 1));
        const int rr = parent->refRatio(Level() - 1)[0];
        amrex::BoxArray cba = m_yoo_corr->boxArray();
        cba.grow(1);
        cba.coarsen(rr);
        amrex::MultiFab ctmp(cba, m_yoo_corr->DistributionMap(), 4, 0);
        ctmp.setVal(1.0, 0, 1, 0);
        ctmp.setVal(0.0, 1, 3, 0);
        ctmp.ParallelCopy(*crse.m_yoo_corr, 0, 0, 4, 0, 0, parent->Geom(Level() - 1).periodicity());
        for (amrex::MFIter mfi(*m_yoo_corr); mfi.isValid(); ++mfi)
        {
            const amrex::Box vbx = mfi.validbox();
            const amrex::Box gbx = amrex::grow(vbx, 1);
            auto const &f = m_yoo_corr->array(mfi);
            auto const &c = ctmp.const_array(mfi);
            amrex::LoopOnCpu(gbx, [&](int i, int j, int k) {
                if (!vbx.contains(amrex::IntVect(i, j, k)))
                    for (int n = 0; n < 4; ++n)
                        f(i, j, k, n) = c(amrex::coarsen(i, rr), amrex::coarsen(j, rr), amrex::coarsen(k, rr), n);
            });
        }
    }
    m_yoo_corr->FillBoundary(Geom().periodicity());
    return res;
}

void PBHVlasovLevel::apply_deposit_correction()
{
    get_vlasov_amr_ptr()->particles.apply_correction(Level(), *m_yoo_corr, Geom());
}

// Expansion-aware time step: the coordinate light speed of the far field is sqrt(chi_max), so the coarse step is
// dt_multiplier dx_0 / sqrt(chi_max), capped by dt_frac (t + tau0) for the accuracy of the background expansion.
// A finer level takes its parent's step while that satisfies its own limit (n_cycle = 1), otherwise half of it.
void PBHVlasovLevel::set_time_steps(int finest_level, amrex::Vector<int> &n_cycle, amrex::Vector<amrex::Real> &dt_level)
{
    InitParams ip;
    ip.read();
    GRParmParse pp;
    amrex::Real mult{};
    pp.get("evolution.dt_multiplier", mult);
    auto *amr = get_vlasov_amr_ptr();
    const amrex::Real time = get_state_data(state_index).curTime();
    const amrex::Real chi_max = amr->getLevel(0).get_new_data(state_index).max(c_chi);
    amr->dt_scale = 1.0 / std::sqrt(chi_max);
    dt_level[0] = std::min(mult * parent->Geom(0).CellSize(0) * amr->dt_scale, ip.dt_frac * (time + ip.tau0));
    n_cycle[0] = 1;
    for (int l = 1; l <= finest_level; ++l)
    {
        const amrex::Real limit = mult * parent->Geom(l).CellSize(0) * amr->dt_scale;
        if (dt_level[l - 1] <= limit * (1.0 + 1e-12))
        {
            n_cycle[l] = 1;
            dt_level[l] = dt_level[l - 1];
        }
        else
        {
            n_cycle[l] = 2;
            dt_level[l] = 0.5 * dt_level[l - 1];
        }
    }
}

void PBHVlasovLevel::computeInitialDt(int finest_level, int sub_cycle, amrex::Vector<int> &n_cycle,
                                      const amrex::Vector<amrex::IntVect> &ref_ratio,
                                      amrex::Vector<amrex::Real> &dt_level, amrex::Real stop_time)
{
    InitParams ip;
    ip.read();
    if (ip.dt_mode != 1)
    {
        GRAmrLevel::computeInitialDt(finest_level, sub_cycle, n_cycle, ref_ratio, dt_level, stop_time);
        return;
    }
    if (Level() == 0)
        set_time_steps(finest_level, n_cycle, dt_level);
}

void PBHVlasovLevel::computeNewDt(int finest_level, int sub_cycle, amrex::Vector<int> &n_cycle,
                                  const amrex::Vector<amrex::IntVect> &ref_ratio, amrex::Vector<amrex::Real> &dt_min,
                                  amrex::Vector<amrex::Real> &dt_level, amrex::Real stop_time, int post_regrid_flag)
{
    InitParams ip;
    ip.read();
    if (ip.dt_mode != 1)
    {
        GRAmrLevel::computeNewDt(finest_level, sub_cycle, n_cycle, ref_ratio, dt_min, dt_level, stop_time,
                                 post_regrid_flag);
        return;
    }
    if (Level() == 0)
    {
        set_time_steps(finest_level, n_cycle, dt_level);
        for (int l = 0; l <= finest_level; ++l)
            dt_min[l] = dt_level[l];
    }
}

void PBHVlasovLevel::deposit_particles()
{
    get_vlasov_amr_ptr()->deposit_one_level(Level(), state_index);
}

void PBHVlasovLevel::specific_advance()
{
    BL_PROFILE("PBHVlasovLevel::specific_advance()");
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays = state_new.arrays();
    const AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    const PositiveChiAndLapse positive_chi_and_lapse;
    amrex::ParallelFor(state_new, [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        algebraic_constraints_enforcer(ix, iy, iz, state_arrays[box_no]);
        positive_chi_and_lapse(ix, iy, iz, state_arrays[box_no]);
    });
    amrex::Gpu::streamSynchronize();

    // push the particles with the new metric (ghost-filled copy), then deposit the sources for the next step
    auto *amr = get_vlasov_amr_ptr();
    if (!amr->particles_ready)
        return;
    const amrex::Real t_new = get_state_data(state_index).curTime();
    const amrex::Real dt = get_gr_amr_ptr()->dtLevel(Level());
    amrex::MultiFab state_gh(state_new.boxArray(), state_new.DistributionMap(), state_new.nComp(), 3);
    FillPatch(*this, state_gh, 3, t_new, state_index, 0, state_new.nComp());
    amr->particles.push(Level(), dt, state_gh, Geom());
    deposit_particles();
}

void PBHVlasovLevel::specific_eval_rhs(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs, const amrex::Real a_time)
{
    BL_PROFILE("PBHVlasovLevel::specific_eval_rhs()");
    const auto &soln_arrays = a_soln.arrays();
    const auto &const_soln_arrays = a_soln.const_arrays();
    const auto &rhs_arrays = a_rhs.arrays();
    const AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    const PositiveChiAndLapse positive_chi_and_lapse;
    amrex::ParallelFor(a_soln, a_soln.nGrowVect(), [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        algebraic_constraints_enforcer(ix, iy, iz, soln_arrays[box_no]);
        positive_chi_and_lapse(ix, iy, iz, soln_arrays[box_no]);
    });
    if (m_evolution_spatial_derivative_order != 4)
        amrex::Abort("PBHVlasov: spatial_derivative_order must be 4");
    const CCZ4RHSWithMatter<Matter<FourthOrderDerivatives>, FourthOrderDerivatives> ccz4_rhs(Geom().CellSize(0));
    // far-field K at the stage time: dust FLRW from the last measurement, dK/dt = K^2/2 (otherwise the reference lags
    // by up to one coarse step and the far-field lapse drifts below 1)
    amrex::Real K_ref = get_vlasov_amr_ptr()->K_far;
    {
        int k_far_extrap = 1;
        GRParmParse("pbh_vlasov").query("k_far_extrap", k_far_extrap);
        if (k_far_extrap)
            K_ref /= 1.0 - 0.5 * K_ref * (a_time - get_vlasov_amr_ptr()->K_far_time);
    }
    PBHGaugeTeclyn<FourthOrderDerivatives> gauge(Geom().CellSize(0), K_ref);
    // expansion-aware step: Kreiss-Oliger coefficient and Gamma-driver speed follow the coordinate light speed
    InitParams ip_rhs;
    ip_rhs.read();
    amrex::Real sigma_ko{};
    GRParmParse("evolution").get("sigma", sigma_ko);
    if (ip_rhs.dt_mode == 1)
    {
        const amrex::Real c_far = 1.0 / get_vlasov_amr_ptr()->dt_scale; // sqrt(chi_max)
        sigma_ko *= c_far;
        gauge.scale_shift_Gamma(c_far * c_far);
    }
    const FourthOrderDerivatives diss_deriv(Geom().CellSize(0));
    amrex::ParallelFor(a_rhs, [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        ccz4_rhs.compute_chi_and_h_ij(ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
    });
    amrex::ParallelFor(a_rhs, [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        ccz4_rhs.compute_A_ij_and_Theta_and_Gamma(ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
    });
    amrex::ParallelFor(a_rhs, [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        ccz4_rhs.add_emtensor_rhs(ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
        gauge.calculate_rhs(ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
        ccz4_rhs.add_matter_rhs(ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
        diss_deriv.add_dissipation(ix, iy, iz, rhs_arrays[box_no].cellData(ix, iy, iz), const_soln_arrays[box_no],
                                   sigma_ko, NUM_CCZ4_VARS);
        // no dissipation on the deposited sources (they are not evolved)
        for (int c = c_rho_p; c < NUM_VARS; ++c)
            rhs_arrays[box_no](ix, iy, iz, c) = 0.0;
    });
    amrex::Gpu::streamSynchronize();
}

void PBHVlasovLevel::specific_update_ode(amrex::MultiFab &a_soln)
{
    BL_PROFILE("PBHVlasovLevel::specific_update_ode()");
    const auto &soln_arrays = a_soln.arrays();
    const AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    amrex::ParallelFor(a_soln, amrex::IntVect(0), [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        algebraic_constraints_enforcer(ix, iy, iz, soln_arrays[box_no]);
    });
    amrex::Gpu::streamSynchronize();
}

void PBHVlasovLevel::specific_post_timestep()
{
    BL_PROFILE("PBHVlasovLevel::specific_post_timestep()");
    if (Level() != 0)
        return;
    const amrex::Real time = get_state_data(state_index).curTime();
    // with the variable step the outputs are indexed by the coarse step count (dt = time / steps keeps the
    // time / dt arithmetic of the extraction files and of the profile dumps)
    InitParams ip_step;
    ip_step.read();
    const long nstep = parent->levelSteps(0);
    const amrex::Real dt = (ip_step.dt_mode == 1) ? time / std::max<long>(nstep, 1) : get_gr_amr_ptr()->dtLevel(0);
    const amrex::Real restart_time = get_gr_amr_ptr()->get_restart_time();
    const bool first_step = (ip_step.dt_mode == 1) ? (nstep <= 1) : (time <= dt);
    auto *amr = get_vlasov_amr_ptr();
    amrex::MultiFab &state_new = get_new_data(state_index);
    // far-field reference for the lapse driver: the most negative K on the coarse level (FLRW region)
    amr->K_far = state_new.min(c_K);
    amr->K_far_time = time;
    const amrex::Real chi_mean = state_new.sum(c_chi) / Geom().Domain().numPts();
    const amrex::Real K_mean = state_new.sum(c_K) / Geom().Domain().numPts();
    // physical energy density = densitised grid source times chi^{3/2}
    amrex::MultiFab rho_phys(state_new.boxArray(), state_new.DistributionMap(), 1, 0);
    {
        const auto &sa = state_new.const_arrays();
        const auto &ra = rho_phys.arrays();
        amrex::ParallelFor(rho_phys, [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
            ra[box_no](ix, iy, iz) = sa[box_no](ix, iy, iz, c_rho_p) * std::pow(sa[box_no](ix, iy, iz, c_chi), 1.5);
        });
        amrex::Gpu::streamSynchronize();
    }
    const amrex::Real rho_mean = rho_phys.sum(0) / Geom().Domain().numPts();
    const amrex::Real rho_max = rho_phys.max(0);
    const amrex::Real chi_min = state_new.min(c_chi);
    const amrex::Real lapse_min = state_new.min(c_lapse);
    const amrex::Real mass = amr->particles.total_mass(0);
    const long np_total = amr->particles.TotalNumberOfParticles(); // collective: outside the IOProcessor block
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ofstream out("pbh_vlasov_out.dat", first_step ? std::ios::out : std::ios::app);
        if (first_step)
            out << "# time <chi> <K> <rho_p> rho_max chi_min lapse_min total_mass N_particles max_disp_cells "
                   "n_clamped n_capped n_frozen\n";
        out.precision(10);
        const auto &pd = amr->particles.push_diag();
        out << time << " " << chi_mean << " " << K_mean << " " << rho_mean << " " << rho_max << " " << chi_min << " "
            << lapse_min << " " << mass << " " << np_total << " " << pd.max_disp_cells << " " << pd.n_clamped << " "
            << pd.n_capped << " " << pd.n_frozen << "\n";
    }
    amr->particles.reset_push_diag();
    const LineExtraction<1> rho_extraction("rho_line_extraction", 0, dt, time, restart_time, first_step);
    const LineExtraction<1>::derived_vars_t rho_vars{VlasovEnergyDensity::name, {"rho"}, {BCParity::even}};
    rho_extraction.execute_query(&amr->rho_interpolator, rho_vars);
    const LineExtraction<1> chi_extraction("chi_line_extraction", c_chi, dt, time, restart_time, first_step);
    chi_extraction.execute_query(&amr->chi_interpolator);
    const LineExtraction<1> K_extraction("K_line_extraction", c_K, dt, time, restart_time, first_step);
    K_extraction.execute_query(&amr->K_interpolator);
    const LineExtraction<1> lapse_extraction("lapse_line_extraction", c_lapse, dt, time, restart_time, first_step);
    lapse_extraction.execute_query(&amr->lapse_interpolator);

    InitParams ip;
    ip.read();
    const long step = lround(time / dt);
    if (ip.theta_rays)
    {
        theta_ray_diagnostic(amr, Geom(), time, step, ip, first_step);
        if (ip.sphere_diag)
            sphere_diagnostic(amr, Geom(), time, step, ip, first_step);
        tau_diagnostic(amr, Geom(), time, first_step, step, ip.theta_profile_interval);
    }

    // apparent-horizon search (capped flow finder from the GRTeclyn AHFinder branch), seeded by the ray estimate
    if (ip.ah_interval > 0 && time >= ip.ah_start && step % ip.ah_interval == 0)
    {
        const auto plo = Geom().ProbLoArray(), phi = Geom().ProbHiArray();
        const std::array<double, AMREX_SPACEDIM> center = {0.5 * (plo[0] + phi[0]), 0.5 * (plo[1] + phi[1]),
                                                           0.5 * (plo[2] + phi[2])};
        const double guess = (amr->ah_ray_radius > 0.0)    ? amr->ah_ray_radius
                             : (amr->ah_last_radius > 0.0) ? 1.3 * amr->ah_last_radius
                                                           : ip.ah_guess;
        PBHAHFinder<21> finder(ip.ah_num_particles, center, guess);
        finder.set_max_iter(ip.ah_max_iter);
        finder.set_h_bounds(0.5 * Geom().CellSize(0) / (1 << amr->finestLevel()), 0.4 * (phi[0] - plo[0]));
        finder.init(amr);
        finder.find();
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            std::ofstream out("pbh_ah.dat", std::ios::app);
            out.precision(10);
            out << time << " " << (finder.converged() ? 1 : 0) << " " << finder.area() << " " << finder.mass() << " "
                << finder.mean_radius() << " " << finder.iterations() << " " << finder.theta_norm() << "\n";
        }
        amr->ah_last_radius = finder.converged() ? finder.mean_radius() : -1.0;
    }
}

void PBHVlasovLevel::specific_post_checkpoint(const std::string &a_dir, std::ostream & /*os*/)
{
    if (Level() == 0 && get_vlasov_amr_ptr()->particles_ready)
        get_vlasov_amr_ptr()->particles.write_checkpoint(a_dir);
}

void PBHVlasovLevel::specific_post_restart()
{
    auto *amr = get_vlasov_amr_ptr();
    if (Level() == 0)
    {
        InitParams ip;
        ip.read();
        amr->particles.setup(amr);
        amr->particles.set_max_disp_cells(ip.max_disp_cells);
        amr->particles.read_checkpoint(amr->theRestartFile());
        amr->particles_ready = true;
        amrex::Print() << "PBHVlasov: restarted " << amr->particles.TotalNumberOfParticles() << " particles\n";
    }
    if (Level() == amr->finestLevel())
    {
        amr->deposit_all_levels(state_index);
        amr->K_far = amr->getLevel(0).get_new_data(state_index).min(c_K);
        amr->K_far_time = get_state_data(state_index).curTime();
        amr->dt_scale = 1.0 / std::sqrt(amr->getLevel(0).get_new_data(state_index).max(c_chi));
    }
}

void PBHVlasovLevel::specific_post_regrid(int /*a_lbase*/, int /*a_new_finest*/)
{
    auto *amr = get_vlasov_amr_ptr();
    if (!amr->particles_ready) // called during Amr::FinalizeInit before the particles exist
        return;
    amr->particles.Redistribute();
    amr->deposit_all_levels(state_index);
}

void PBHVlasovLevel::tag_cells(amrex::TagBoxArray &a_tag_box_array, const amrex::Real /*a_regrid_threshold*/)
{
    BL_PROFILE("PBHVlasovLevel::tag_cells()");
    const auto &tag_arrays = a_tag_box_array.arrays();
    const FixedGridsTagger tagger(Geom().CellSize(0), Level());
    amrex::ParallelFor(a_tag_box_array, [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        tagger(ix, iy, iz, tag_arrays[box_no]);
    });
    amrex::Gpu::streamSynchronize();
}

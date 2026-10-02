/* PBHVlasov example for GRTeclyn (pbhgr project, stage 2). Based on Examples/ScalarField (GRTeclyn fd76177). */
#include "PBHVlasovLevel.hpp"
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
#include "PBHGaugeTeclyn.hpp"
#include "PBHAHFinder.hpp"
#include "LineExtraction.hpp"
#include "PositiveChiAndLapse.hpp"
#include "StateTypes.hpp"
#include <AMReX_MultiFabUtil.H>
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
    }
};

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
        else
        {
            load_table();
            amr->particles.init_from_table(geom0, m_tab_r, m_tab_rho, m_tab_ur, m_tab_chi, ip.n_per_dir);
        }
        amr->particles_ready = true;
        amrex::Print() << "PBHVlasov: " << amr->particles.TotalNumberOfParticles() << " particles on "
                       << amr->finestLevel() + 1 << " levels, total rest mass " << amr->particles.total_mass(0) << "\n";
        amr->deposit_all_levels(state_index);
        amr->K_far = amr->getLevel(0).get_new_data(state_index).min(c_K); // far-field reference before the first RHS
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

void PBHVlasovLevel::specific_eval_rhs(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs, const amrex::Real /*a_time*/)
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
    const PBHGaugeTeclyn<FourthOrderDerivatives> gauge(Geom().CellSize(0), get_vlasov_amr_ptr()->K_far);
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
        ccz4_rhs.apply_dissipation(ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
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
    const amrex::Real dt = get_gr_amr_ptr()->dtLevel(0);
    const amrex::Real restart_time = get_gr_amr_ptr()->get_restart_time();
    const bool first_step = (time <= dt);
    auto *amr = get_vlasov_amr_ptr();
    amrex::MultiFab &state_new = get_new_data(state_index);
    // far-field reference for the lapse driver: the most negative K on the coarse level (FLRW region)
    amr->K_far = state_new.min(c_K);
    const amrex::Real chi_mean = state_new.sum(c_chi) / Geom().Domain().numPts();
    const amrex::Real K_mean = state_new.sum(c_K) / Geom().Domain().numPts();
    const amrex::Real rho_mean = state_new.sum(c_rho_p) / Geom().Domain().numPts();
    const amrex::Real rho_max = state_new.max(c_rho_p);
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

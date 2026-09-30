/* PBHVlasov example for GRTeclyn (pbhgr project, stage 2). Based on Examples/ScalarField (GRTeclyn fd76177). */
#include "PBHVlasovLevel.hpp"
#include "AlgebraicConstraintsEnforcer.hpp"
#include "CCZ4RHSWithMatter.hpp"
#include "ConstraintsWithMatter.hpp"
#include "EMTensor.hpp"
#include "FixedGridsTagger.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GRParmParse.hpp"
#include "GammaCalculator.hpp"
#include "IntegratedMovingPunctureGauge.hpp"
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
    void read()
    {
        GRParmParse pp("pbh_vlasov");
        pp.query("init", mode);
        pp.query("H0", H0);
        pp.query("table", table);
        pp.query("n_per_dir", n_per_dir);
    }
};
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
            cell[c_K] = K0; cell[c_lapse] = 1.0;
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
        if (ip.mode == "flrw")
        {
            const amrex::Real rho = 3.0 * ip.H0 * ip.H0 / (8.0 * M_PI);
            amr->particles.init_uniform(Level(), Geom(), rho, ip.n_per_dir);
        }
        else
        {
            load_table();
            amr->particles.init_from_table(Level(), Geom(), m_tab_r, m_tab_rho, m_tab_ur, m_tab_chi, ip.n_per_dir);
        }
        amrex::Print() << "PBHVlasov: " << amr->particles.TotalNumberOfParticles() << " particles, total rest mass "
                       << amr->particles.total_mass(Level()) << "\n";
    }
    deposit_particles();
}

void PBHVlasovLevel::deposit_particles()
{
    auto *amr = get_vlasov_amr_ptr();
    amrex::MultiFab &state_new = get_new_data(state_index);
    amr->particles.deposit(Level(), state_new, Geom());
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
    const IntegratedMovingPunctureGauge<FourthOrderDerivatives> gauge(Geom().CellSize(0));
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
    // box averages of chi, K, rho_p (level 0 only: single-level tests)
    amrex::MultiFab &state_new = get_new_data(state_index);
    const amrex::Real chi_mean = state_new.sum(c_chi) / Geom().Domain().numPts();
    const amrex::Real K_mean = state_new.sum(c_K) / Geom().Domain().numPts();
    const amrex::Real rho_mean = state_new.sum(c_rho_p) / Geom().Domain().numPts();
    const amrex::Real rho_max = state_new.max(c_rho_p);
    const amrex::Real chi_min = state_new.min(c_chi);
    const amrex::Real lapse_min = state_new.min(c_lapse);
    const amrex::Real mass = amr->particles.total_mass(0);
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ofstream out("pbh_vlasov_out.dat", first_step ? std::ios::out : std::ios::app);
        if (first_step)
            out << "# time <chi> <K> <rho_p> rho_max chi_min lapse_min total_mass N_particles\n";
        out.precision(10);
        out << time << " " << chi_mean << " " << K_mean << " " << rho_mean << " " << rho_max << " " << chi_min << " "
            << lapse_min << " " << mass << " " << amr->particles.TotalNumberOfParticles() << "\n";
    }
    const LineExtraction<1> rho_extraction("rho_line_extraction", 0, dt, time, restart_time, first_step);
    const LineExtraction<1>::derived_vars_t rho_vars{VlasovEnergyDensity::name, {"rho"}, {BCParity::even}};
    rho_extraction.execute_query(&amr->rho_interpolator, rho_vars);
    const LineExtraction<1> chi_extraction("chi_line_extraction", c_chi, dt, time, restart_time, first_step);
    chi_extraction.execute_query(&amr->chi_interpolator);
    const LineExtraction<1> K_extraction("K_line_extraction", c_K, dt, time, restart_time, first_step);
    K_extraction.execute_query(&amr->K_interpolator);
    const LineExtraction<1> lapse_extraction("lapse_line_extraction", c_lapse, dt, time, restart_time, first_step);
    lapse_extraction.execute_query(&amr->lapse_interpolator);
}

void PBHVlasovLevel::specific_post_regrid(int /*a_lbase*/, int /*a_new_finest*/)
{
    get_vlasov_amr_ptr()->particles.Redistribute();
    deposit_particles();
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

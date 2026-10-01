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
            out << "# time <chi> <K> <rho_p> rho_max chi_min lapse_min total_mass N_particles\n";
        out.precision(10);
        out << time << " " << chi_mean << " " << K_mean << " " << rho_mean << " " << rho_max << " " << chi_min << " "
            << lapse_min << " " << mass << " " << np_total << "\n";
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

    // apparent-horizon search (capped flow finder from the GRTeclyn AHFinder branch)
    InitParams ip;
    ip.read();
    const long step = lround(time / dt);
    if (ip.ah_interval > 0 && time >= ip.ah_start && step % ip.ah_interval == 0)
    {
        const auto plo = Geom().ProbLoArray(), phi = Geom().ProbHiArray();
        const std::array<double, AMREX_SPACEDIM> center = {0.5 * (plo[0] + phi[0]), 0.5 * (plo[1] + phi[1]),
                                                           0.5 * (plo[2] + phi[2])};
        const double guess = (amr->ah_last_radius > 0.0) ? 1.3 * amr->ah_last_radius : ip.ah_guess;
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
        amr->particles.setup(amr);
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

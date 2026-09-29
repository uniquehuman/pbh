/* PBHCosmo example for GRChombo (pbhgr project). Based on Examples/ScalarFieldCosmo (GRChombo 37e6595). */

#include "PBHCosmoLevel.hpp"
#include "BoxLoops.hpp"
#include "NanCheck.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "SixthOrderDerivatives.hpp"
#include "TraceARemoval.hpp"

#include "MatterCCZ4RHS.hpp"
#include "NewMatterConstraints.hpp"
#include "SphereTaggingCriterion.hpp"

#include "AMRReductions.hpp"
#include "ComputePack.hpp"
#include "CosmoDiagnostics.hpp"
#include "PBHGauge.hpp"

double PBHGauge::s_K_ref = 0.0;
#include "GammaCalculator.hpp"
#include "InitialLTBData.hpp"
#include "Potential.hpp"
#include "ScalarField.hpp"
#include "SetValue.hpp"

#include "ConstraintsExtraction.hpp"
#include "CustomExtraction.hpp"

void PBHCosmoLevel::specificAdvance()
{
    BoxLoops::loop(make_compute_pack(TraceARemoval(), PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse)),
                   m_state_new, m_state_new, INCLUDE_GHOST_CELLS);
    if (m_p.nan_check)
        BoxLoops::loop(NanCheck(m_dx, m_p.center, "NaNCheck in specific Advance"), m_state_new, m_state_new,
                       EXCLUDE_GHOST_CELLS, disable_simd());
}

void PBHCosmoLevel::initialData()
{
    CH_TIME("PBHCosmoLevel::initialData");
    if (m_verbosity)
        pout() << "PBHCosmoLevel::initialData " << m_level << endl;

    // zero everything, then the tabulated LTB slice (table look-up: no simd)
    BoxLoops::loop(SetValue(0.), m_state_new, m_state_new, INCLUDE_GHOST_CELLS);
    InitialLTBData ltb(m_p.initial_params, m_dx);
    BoxLoops::loop(ltb, m_state_new, m_state_new, INCLUDE_GHOST_CELLS, disable_simd());

    fillAllGhosts();
    BoxLoops::loop(GammaCalculator(m_dx), m_state_new, m_state_new, EXCLUDE_GHOST_CELLS);

    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    BoxLoops::loop(MatterConstraints<ScalarFieldWithPotential>(scalar_field, m_dx, m_p.G_Newton, c_Ham,
                                                              Interval(c_Mom, c_Mom), c_Ham_abs_sum,
                                                              Interval(c_Mom_abs_sum, c_Mom_abs_sum)),
                   m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
    CosmoDiagnostics<ScalarFieldWithPotential> cosmo_diagnostics(scalar_field, m_dx, m_p.G_Newton);
    BoxLoops::loop(cosmo_diagnostics, m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);

    // background density for the tagging criterion (the proper-volume average is recomputed every coarse step)
    double rho_far = (m_p.ltb_rho_far > 0.) ? m_p.ltb_rho_far : ltb.rho_far();
    m_cosmo_amr.set_rho_mean(rho_far);
}

void PBHCosmoLevel::postRestart()
{
    if (m_time == 0.0)
    {
        fillAllGhosts();
        Potential potential(m_p.potential_params);
        ScalarFieldWithPotential scalar_field(potential);
        BoxLoops::loop(MatterConstraints<ScalarFieldWithPotential>(scalar_field, m_dx, m_p.G_Newton, c_Ham,
                                                                  Interval(c_Mom, c_Mom), c_Ham_abs_sum,
                                                                  Interval(c_Mom_abs_sum, c_Mom_abs_sum)),
                       m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
        CosmoDiagnostics<ScalarFieldWithPotential> cosmo_diagnostics(scalar_field, m_dx, m_p.G_Newton);
        BoxLoops::loop(cosmo_diagnostics, m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);

        AMRReductions<VariableType::diagnostic> amr_reductions_diagnostic(m_cosmo_amr);
        double phys_vol = amr_reductions_diagnostic.sum(c_sqrt_gamma);
        m_cosmo_amr.set_rho_mean(amr_reductions_diagnostic.sum(c_rho_scaled) / phys_vol);
        m_cosmo_amr.set_S_mean(amr_reductions_diagnostic.sum(c_S_scaled) / phys_vol);
        m_cosmo_amr.set_K_mean(amr_reductions_diagnostic.sum(c_K_scaled) / phys_vol);
        pout() << "postRestart: rho_mean = " << m_cosmo_amr.get_rho_mean() << ", K_mean = "
               << m_cosmo_amr.get_K_mean() << " on level " << m_level << endl;
    }
}

#ifdef CH_USE_HDF5
void PBHCosmoLevel::prePlotLevel()
{
    fillAllGhosts();
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    BoxLoops::loop(MatterConstraints<ScalarFieldWithPotential>(scalar_field, m_dx, m_p.G_Newton, c_Ham,
                                                              Interval(c_Mom, c_Mom)),
                   m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
    CosmoDiagnostics<ScalarFieldWithPotential> cosmo_diagnostics(scalar_field, m_dx, m_p.G_Newton);
    BoxLoops::loop(cosmo_diagnostics, m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
}
#endif

void PBHCosmoLevel::specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs, const double a_time)
{
    BoxLoops::loop(make_compute_pack(TraceARemoval(), PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse)), a_soln,
                   a_soln, INCLUDE_GHOST_CELLS);
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    PBHGauge::set_K_ref(m_cosmo_amr.get_K_mean());
    MatterCCZ4RHS<ScalarFieldWithPotential, PBHGauge, FourthOrderDerivatives> my_ccz4_matter(
        scalar_field, m_p.ccz4_params, m_dx, m_p.sigma, m_p.formulation, m_p.G_Newton);
    BoxLoops::loop(my_ccz4_matter, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
}

void PBHCosmoLevel::specificUpdateODE(GRLevelData &a_soln, const GRLevelData &a_rhs, Real a_dt)
{
    BoxLoops::loop(TraceARemoval(), a_soln, a_soln, INCLUDE_GHOST_CELLS);
}

void PBHCosmoLevel::preTagCells()
{
    fillAllGhosts();
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    BoxLoops::loop(MatterConstraints<ScalarFieldWithPotential>(scalar_field, m_dx, m_p.G_Newton, c_Ham,
                                                              Interval(c_Mom, c_Mom), c_Ham_abs_sum,
                                                              Interval(c_Mom_abs_sum, c_Mom_abs_sum)),
                   m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
    CosmoDiagnostics<ScalarFieldWithPotential> cosmo_diagnostics(scalar_field, m_dx, m_p.G_Newton);
    BoxLoops::loop(cosmo_diagnostics, m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
}

void PBHCosmoLevel::computeTaggingCriterion(FArrayBox &tagging_criterion, const FArrayBox &current_state,
                                            const FArrayBox &current_state_diagnostics)
{
    double radius = (m_level < (int)m_p.tagging_radii.size()) ? m_p.tagging_radii[m_level] : 0.0;
    double rho_thr = (m_level < (int)m_p.tagging_rho_thr.size()) ? m_p.tagging_rho_thr[m_level] : 0.0;
    BoxLoops::loop(SphereTaggingCriterion(m_dx, m_p.tagging_center, radius, rho_thr), current_state_diagnostics,
                   tagging_criterion);
}

void PBHCosmoLevel::specificPostTimeStep()
{
    int min_level = 0;
    // without subcycling every level steps with the coarse dt: diagnostics every step on every level
    bool calculate_diagnostics = (m_p.use_subcycling == 0) ? true : at_level_timestep_multiple(min_level);
    bool first_step = (m_time == 0.);
    // K_ref must be refreshed every coarse step; the heavier output only every diag_interval coarse steps
    if (calculate_diagnostics && m_p.diag_interval > 1 && !first_step)
    {
        long step = lround(m_time / m_p.coarsest_dt);
        if (step % m_p.diag_interval != 0)
        {
            if (m_level == min_level)
            {
                AMRInterpolator<Lagrange<4>> interpolator(m_cosmo_amr, m_p.origin, m_p.dx, m_p.boundary_params,
                                                          m_p.verbosity);
                interpolator.refresh();
                double cx = 0.5 * m_p.coarsest_dx, cy = cx, cz = cx, K_corner = 0.;
                InterpolationQuery query(1);
                query.setCoords(0, &cx).setCoords(1, &cy).setCoords(2, &cz)
                    .addComp(c_K, &K_corner, Derivative::LOCAL, VariableType::evolution);
                interpolator.interp(query);
                if (m_p.gauge_K_ref == 1)
                    m_cosmo_amr.set_K_mean(K_corner);
            }
            return;
        }
    }

    if (calculate_diagnostics)
    {
        fillAllGhosts();
        Potential potential(m_p.potential_params);
        ScalarFieldWithPotential scalar_field(potential);
        BoxLoops::loop(MatterConstraints<ScalarFieldWithPotential>(scalar_field, m_dx, m_p.G_Newton, c_Ham,
                                                                  Interval(c_Mom, c_Mom), c_Ham_abs_sum,
                                                                  Interval(c_Mom_abs_sum, c_Mom_abs_sum)),
                       m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
        CosmoDiagnostics<ScalarFieldWithPotential> cosmo_diagnostics(scalar_field, m_dx, m_p.G_Newton);
        BoxLoops::loop(cosmo_diagnostics, m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);

        if (m_level == min_level)
        {
            AMRReductions<VariableType::diagnostic> amr_reductions_diagnostic(m_cosmo_amr);
            double phys_vol = amr_reductions_diagnostic.sum(c_sqrt_gamma);
            double L2_Ham = amr_reductions_diagnostic.norm(c_Ham);
            double L2_Mom = amr_reductions_diagnostic.norm(c_Mom);
            double K_total = amr_reductions_diagnostic.sum(c_K_scaled);
            m_cosmo_amr.set_rho_mean(amr_reductions_diagnostic.sum(c_rho_scaled) / phys_vol);
            m_cosmo_amr.set_S_mean(amr_reductions_diagnostic.sum(c_S_scaled) / phys_vol);
            double K_mean = K_total / phys_vol;

            AMRReductions<VariableType::evolution> amr_reductions_evolution(m_cosmo_amr);
            double chi_mean = amr_reductions_evolution.sum(c_chi) / phys_vol;
            double chi_min = amr_reductions_evolution.min(c_chi);
            double lapse_min = amr_reductions_evolution.min(c_lapse);
            double rho_max = amr_reductions_diagnostic.max(c_rho);

            // far-field reference: K and lapse at the box corner (FLRW region; periodic box)
            AMRInterpolator<Lagrange<4>> interpolator(m_cosmo_amr, m_p.origin, m_p.dx, m_p.boundary_params,
                                                      m_p.verbosity);
            interpolator.refresh();
            double corner_x = 0.5 * m_p.coarsest_dx, corner_y = 0.5 * m_p.coarsest_dx, corner_z = 0.5 * m_p.coarsest_dx;
            double K_corner = 0., lapse_corner = 0.;
            {
                InterpolationQuery query(1);
                query.setCoords(0, &corner_x).setCoords(1, &corner_y).setCoords(2, &corner_z)
                    .addComp(c_K, &K_corner, Derivative::LOCAL, VariableType::evolution)
                    .addComp(c_lapse, &lapse_corner, Derivative::LOCAL, VariableType::evolution);
                interpolator.interp(query);
            }
            m_cosmo_amr.set_K_mean(m_p.gauge_K_ref == 1 ? K_corner : K_mean);

            SmallDataIO data_out_file(m_p.data_path + "data_out", m_dt, m_time, m_restart_time,
                                      SmallDataIO::APPEND, first_step);
            data_out_file.remove_duplicate_time_data();
            if (first_step)
                data_out_file.write_header_line({"L^2_Ham", "L^2_Mom", "<chi>", "<rho>", "<K>", "chi_min",
                                                 "lapse_min", "rho_max", "phys_vol", "K_corner", "lapse_corner",
                                                 "K_ref"});
            data_out_file.write_time_data_line({L2_Ham, L2_Mom, chi_mean, m_cosmo_amr.get_rho_mean(), K_mean,
                                                chi_min, lapse_min, rho_max, phys_vol, K_corner, lapse_corner,
                                                m_cosmo_amr.get_K_mean()});

            // lineouts along the x axis through the centre: rho, chi, lapse, K
            std::array<double, CH_SPACEDIM> extraction_origin = {0., m_p.L / 2, m_p.L / 2};
            CustomExtraction rho_extraction(c_rho, m_p.lineout_num_points, m_p.L, extraction_origin, m_dt, m_time);
            rho_extraction.execute_query(&interpolator, m_p.data_path + "rho_lineout");
            CustomExtraction chi_extraction(c_chi, m_p.lineout_num_points, m_p.L, extraction_origin, m_dt, m_time,
                                            VariableType::evolution);
            chi_extraction.execute_query(&interpolator, m_p.data_path + "chi_lineout");
            CustomExtraction lapse_extraction(c_lapse, m_p.lineout_num_points, m_p.L, extraction_origin, m_dt,
                                              m_time, VariableType::evolution);
            lapse_extraction.execute_query(&interpolator, m_p.data_path + "lapse_lineout");
            CustomExtraction K_extraction(c_K, m_p.lineout_num_points, m_p.L, extraction_origin, m_dt, m_time,
                                          VariableType::evolution);
            K_extraction.execute_query(&interpolator, m_p.data_path + "K_lineout");
        }
    }
}

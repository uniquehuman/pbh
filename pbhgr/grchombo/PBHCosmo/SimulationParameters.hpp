/* PBHCosmo example for GRChombo (pbhgr project). Based on Examples/ScalarFieldCosmo. */
#ifndef SIMULATIONPARAMETERS_HPP_
#define SIMULATIONPARAMETERS_HPP_

#include "GRParmParse.hpp"
#include "SimulationParametersBase.hpp"

#include "InitialLTBData.hpp"
#include "Potential.hpp"

class SimulationParameters : public SimulationParametersBase
{
  public:
    SimulationParameters(GRParmParse &pp) : SimulationParametersBase(pp)
    {
        read_params(pp);
        check_params();
    }

    void read_params(GRParmParse &pp)
    {
        initial_params.center = center; // already read in SimulationParametersBase
        pp.load("G_Newton", G_Newton, 1.0);
        pp.load("scalar_mass", potential_params.scalar_mass, 0.1);
        pp.load("ltb_table", initial_params.table);
        pp.load("ltb_rho_far", ltb_rho_far, -1.0);
        pp.load("lineout_num_points", lineout_num_points, 10);
        pp.load("tagging_center", tagging_center, center);
        // nested spheres: level l+1 inside tagging_radii[l] (coordinate radius); max_level entries
        if (max_level > 0)
        {
            pp.load("tagging_radii", tagging_radii, max_level);
            // density gate per level (rho > tagging_rho_thr[l] needed to create level l+1); 0 = always
            pp.load("tagging_rho_thr", tagging_rho_thr, max_level, 0.0);
        }
        pp.load("diag_interval", diag_interval, 1); // coarse steps between data_out / lineout outputs
        // lapse condition d_t alpha = -2 alpha (K - K_ref): K_ref = K at the box corner (1, FLRW far field)
        // or the proper-volume mean <K> (0, as in ScalarFieldCosmo)
        pp.load("gauge_K_ref", gauge_K_ref, 1);
        // time stepping: 0 = all levels advance with the same (fixed) dt — for runs whose dt is set by the
        // scalar-field oscillation rather than the CFL condition (needs fixed_dt <= dt_multiplier * finest dx)
        pp.load("use_subcycling", use_subcycling, 1);
        pp.load("fixed_dt", fixed_dt, -1.0);
#ifdef USE_AHFINDER
        pp.load("AH_initial_guess", AH_initial_guess, 0.05 * L);
#endif
    }

    void check_params()
    {
        warn_parameter("scalar_mass", potential_params.scalar_mass,
                       potential_params.scalar_mass < 0.2 / coarsest_dx / dt_multiplier,
                       "oscillations of scalar field do not appear to be resolved on coarsest level");
    }

    double G_Newton, ltb_rho_far;
    int lineout_num_points, gauge_K_ref, use_subcycling;
    double fixed_dt;
    std::array<double, CH_SPACEDIM> tagging_center;
    std::vector<double> tagging_radii, tagging_rho_thr;
    int diag_interval;
    InitialLTBData::params_t initial_params;
    Potential::params_t potential_params;
#ifdef USE_AHFINDER
    double AH_initial_guess;
#endif
};

#endif /* SIMULATIONPARAMETERS_HPP_ */

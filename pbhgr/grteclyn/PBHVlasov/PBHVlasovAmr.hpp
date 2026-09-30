/* PBHVlasov example for GRTeclyn (pbhgr project). GRAmr with the particle container and line interpolators. */
#ifndef PBHVLASOVAMR_HPP_
#define PBHVLASOVAMR_HPP_

#include "GRAmr.hpp"
#include "ParticleInterpolator.hpp"
#include "VlasovParticles.hpp"
#include <AMReX_MultiFabUtil.H>

class PBHVlasovAmr : public GRAmr
{
  public:
    VlasovParticles particles;
    bool particles_ready{false};
    //! deposit the particles of every level and average the sources down to the coarser levels
    void deposit_all_levels(int state_index)
    {
        amrex::Vector<amrex::MultiFab *> states;
        amrex::Vector<amrex::Geometry> geoms;
        for (int lev = 0; lev <= finestLevel(); ++lev)
        {
            states.push_back(&getLevel(lev).get_new_data(state_index));
            geoms.push_back(getLevel(lev).Geom());
        }
        particles.deposit_all(states, geoms);
    }
    // last apparent-horizon result (for the guess radius of the next search)
    double ah_last_radius{-1.0};
    void average_down_sources(int state_index)
    {
        for (int lev = finestLevel(); lev >= 1; --lev)
            amrex::average_down(getLevel(lev).get_new_data(state_index), getLevel(lev - 1).get_new_data(state_index),
                                getLevel(lev).Geom(), getLevel(lev - 1).Geom(), c_rho_p, 10, refRatio(lev - 1));
    }
    amrex::Real K_far{0.0}; // far-field K for PBHGaugeTeclyn, updated every coarse step // set once the container is defined and populated (post_init)
    ParticleInterpolator<1> rho_interpolator;
    ParticleInterpolator<1> chi_interpolator;
    ParticleInterpolator<1> K_interpolator;
    ParticleInterpolator<1> lapse_interpolator;

    explicit PBHVlasovAmr(amrex::LevelBld *a_level_bld) : GRAmr(a_level_bld) {}

    void init(amrex::Real a_start_time, amrex::Real a_stop_time) override
    {
        GRAmr::init(a_start_time, a_stop_time);
        rho_interpolator.setup(this);
        chi_interpolator.setup(this);
        K_interpolator.setup(this);
        lapse_interpolator.setup(this);
    }
};

#endif /* PBHVLASOVAMR_HPP_ */

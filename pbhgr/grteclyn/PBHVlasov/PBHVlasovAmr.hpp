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
    amrex::Vector<amrex::MultiFab *> states(int state_index)
    {
        amrex::Vector<amrex::MultiFab *> v;
        for (int lev = 0; lev <= finestLevel(); ++lev)
            v.push_back(&getLevel(lev).get_new_data(state_index));
        return v;
    }
    amrex::Vector<amrex::Geometry> geoms()
    {
        amrex::Vector<amrex::Geometry> v;
        for (int lev = 0; lev <= finestLevel(); ++lev)
            v.push_back(getLevel(lev).Geom());
        return v;
    }
    //! deposit every level (initialisation / regrid) with the covered coarse cells averaged down
    void deposit_all_levels(int state_index) { particles.deposit_all(states(state_index), geoms()); }
    //! deposit one level (after its particle push)
    void deposit_one_level(int lev, int state_index) { particles.deposit_level(lev, states(state_index), geoms()); }
    // last apparent-horizon result (for the guess radius of the next search)
    double ah_last_radius{-1.0};
    // 1 / sqrt(max chi on level 0): inverse coordinate light speed of the far field (expansion-aware time step)
    amrex::Real dt_scale{1.0};
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
    ParticleInterpolator<14> theta_interpolator; // chi, h~_ij, K, A~_ij along the rays
    ParticleInterpolator<35> sphere_interpolator; // the same plus d_k chi, d_k h~_ij on the coordinate spheres
    double ah_ray_radius{-1.0};                  // horizon radius from the ray diagnostic (-1 = none)

    explicit PBHVlasovAmr(amrex::LevelBld *a_level_bld) : GRAmr(a_level_bld) {}

    void init(amrex::Real a_start_time, amrex::Real a_stop_time) override
    {
        GRAmr::init(a_start_time, a_stop_time);
        rho_interpolator.setup(this);
        chi_interpolator.setup(this);
        K_interpolator.setup(this);
        lapse_interpolator.setup(this);
        theta_interpolator.setup(this);
        sphere_interpolator.setup(this);
    }
};

#endif /* PBHVLASOVAMR_HPP_ */

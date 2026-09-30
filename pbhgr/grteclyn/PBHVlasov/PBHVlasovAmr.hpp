/* PBHVlasov example for GRTeclyn (pbhgr project). GRAmr with the particle container and line interpolators. */
#ifndef PBHVLASOVAMR_HPP_
#define PBHVLASOVAMR_HPP_

#include "GRAmr.hpp"
#include "ParticleInterpolator.hpp"
#include "VlasovParticles.hpp"

class PBHVlasovAmr : public GRAmr
{
  public:
    VlasovParticles particles;
    bool particles_ready{false}; // set once the container is defined and populated (post_init)
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

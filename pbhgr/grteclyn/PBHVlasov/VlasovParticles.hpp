/* PBHVlasov example for GRTeclyn (pbhgr project, stage 2).
 * Collisionless particles (Vlasov matter) on the AMReX particle infrastructure.
 *   per particle (AoS reals): m (rest mass, comoving weight), u_1 u_2 u_3 (covariant spatial 4-velocity),
 *                             x0 y0 z0 u01 u02 u03 (saved state for the midpoint step), Gamma
 *   SoA reals: NFLD = 44 interpolated grid fields (lapse, shift(3), chi, h~(6) and their 3 derivatives each)
 * deposit(): rho = sum m Gamma W / (sqrt(gamma) dV), S_i = sum m u_i W / (sqrt(gamma) dV),
 *            S_ij = sum m u_i u_j W / (Gamma sqrt(gamma) dV)   (CIC kernel W, normal-observer densities)
 * push():    dx^i/dt = alpha gamma^{ij} u_j / Gamma - beta^i
 *            du_i/dt = -alpha Gamma d_i alpha + u_j d_i beta^j - (alpha / 2 Gamma) u_j u_k d_i gamma^{jk}
 *            (midpoint RK2; the fields are interpolated linearly from a MultiFab of values and derivatives) */
#ifndef VLASOVPARTICLES_HPP_
#define VLASOVPARTICLES_HPP_

#include <AMReX_MultiFab.H>
#include <AMReX_ParticleInterpolators.H>
#include <AMReX_ParticleMesh.H>
#include <AMReX_AmrParGDB.H>
#include <AMReX_Particles.H>

#include "FourthOrderDerivatives.hpp"
#include "GRAmr.hpp"
#include "StateVariables.hpp"

namespace VlasovIdx
{
// AoS real components
enum
{
    m = 0, u1, u2, u3, px0, py0, pz0, u01, u02, u03, Gamma, NAOS
};
// interpolated field layout (SoA): value then d/dx, d/dy, d/dz for each of the 11 metric fields
constexpr int NMET = 11;      // lapse, shift1..3, chi, h11 h12 h13 h22 h23 h33
constexpr int NFLD = 4 * NMET; // 44
constexpr int f_lapse = 0, f_shift = 1, f_chi = 4, f_h = 5;
} // namespace VlasovIdx

class VlasovParticles : public amrex::ParticleContainer<VlasovIdx::NAOS, 0, VlasovIdx::NFLD, 0>
{
  public:
    using PC = amrex::ParticleContainer<VlasovIdx::NAOS, 0, VlasovIdx::NFLD, 0>;
    using ParticleType = PC::ParticleType;

    VlasovParticles() = default;
    void setup(GRAmr *a_gr_amr) { this->Define(static_cast<amrex::ParGDBBase *>(a_gr_amr->GetParGDB())); }

    //! uniform lattice of nppc^(1/3) per cell per direction with mass = rho * dV / nppc and u_i = 0 (FLRW dust)
    void init_uniform(int lev, const amrex::Geometry &geom, amrex::Real rho, int n_per_dir);

    //! spherical tabulated data: rho(r) (normal-observer density) and radial u_r(r) about the box centre,
    //! particles on a uniform lattice with mass = rho(r_p) sqrt(gamma) dV / nppc (sqrt(gamma) from chi(r))
    void init_from_table(int lev, const amrex::Geometry &geom, const std::vector<amrex::Real> &r_tab,
                         const std::vector<amrex::Real> &rho_tab, const std::vector<amrex::Real> &ur_tab,
                         const std::vector<amrex::Real> &chi_tab, int n_per_dir);

    //! deposit rho_p, S_i, S_ij into the matter components of a_state (interior cells) on level lev
    void deposit(int lev, amrex::MultiFab &a_state, const amrex::Geometry &geom);

    //! interpolate lapse, shift, chi, h~ and their first derivatives from a ghost-filled state
    void interpolate_fields(int lev, const amrex::MultiFab &a_state_gh, const amrex::Geometry &geom);

    //! midpoint RK2 push by dt using the interpolated fields (calls interpolate_fields twice)
    void push(int lev, amrex::Real dt, const amrex::MultiFab &a_state_gh, const amrex::Geometry &geom);

    //! total rest mass and particle count on a level (diagnostics)
    amrex::Real total_mass(int lev);
};

#endif /* VLASOVPARTICLES_HPP_ */

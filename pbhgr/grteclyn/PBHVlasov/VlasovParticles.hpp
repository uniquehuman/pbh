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
#include <AMReX_AmrParticles.H> // provides ParticleContainer::AssignDensity (multilevel)
#include <AMReX_Particles.H>

#include "FourthOrderDerivatives.hpp"
#include "GRAmr.hpp"
#include "StateVariables.hpp"

namespace VlasovIdx
{
// AoS real components: 10 deposit slots first (AMReX AssignDensity uses rdata(0) as the mass and deposits
// rdata(0)*rdata(k) for k = 1..9), then the particle state
enum
{
    dep0 = 0, // = m Gamma (the "mass" of the deposit); dep1..dep9 = w_k / (m Gamma) with w_k = m u_i, m u_i u_j / Gamma
    m = 10, u1, u2, u3, px0, py0, pz0, u01, u02, u03, Gamma, NAOS
};
constexpr int NDEP = 10;
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
    void setup(GRAmr *a_gr_amr)
    {
        this->Define(static_cast<amrex::ParGDBBase *>(a_gr_amr->GetParGDB()));
        this->reserveData();
        this->resizeData();
    }

    //! finest level whose grids (grown by m_lattice_buffer level-0 cells) cover level-0 cell iv0, so that the
    //! level-l particle lattice extends a buffer beyond the level-l grids and the coarse–fine interface sees
    //! particles of uniform mass (AMReX ghost/virtual particles then deposit smoothly)
    int finest_level_covering(const amrex::IntVect &iv0) const;
    void set_lattice_buffer(int n) { m_lattice_buffer = n; }

    //! uniform lattice created on level 0 over the whole domain with n_per_dir * 2^l particles per direction in
    //! every level-0 cell covered by level l (so that each level has n_per_dir^3 particles per own cell),
    //! mass = rho * dV_fine / n_per_dir^3 and u_i = 0 (FLRW dust); particles are then redistributed to their levels
    void init_uniform(const amrex::Geometry &geom, amrex::Real rho, int n_per_dir);

    //! spherical tabulated data: rho(r) (normal-observer density) and radial u_r(r) about the box centre,
    //! particles on a uniform lattice with mass = rho(r_p) sqrt(gamma) dV / nppc (sqrt(gamma) from chi(r))
    void init_from_table(const amrex::Geometry &geom, const std::vector<amrex::Real> &r_tab,
                         const std::vector<amrex::Real> &rho_tab, const std::vector<amrex::Real> &ur_tab,
                         const std::vector<amrex::Real> &chi_tab, int n_per_dir);

    //! multilevel deposit (AMReX AssignDensity: consistent coarse–fine interfaces) of rho_p, S_i, S_ij into the
    //! matter components of the states of all levels; a_states[lev] and a_geoms[lev] for lev = 0..finest
    void deposit_all(const amrex::Vector<amrex::MultiFab *> &a_states, const amrex::Vector<amrex::Geometry> &a_geoms);

    //! interpolate lapse, shift, chi, h~ and their first derivatives from a ghost-filled state
    void interpolate_fields(int lev, const amrex::MultiFab &a_state_gh, const amrex::Geometry &geom);

    //! midpoint RK2 push by dt using the interpolated fields (calls interpolate_fields twice)
    void push(int lev, amrex::Real dt, const amrex::MultiFab &a_state_gh, const amrex::Geometry &geom);

    //! total rest mass over all levels (diagnostics)
    amrex::Real total_mass(int lev);

  private:
    int m_lattice_buffer{2};
};

#endif /* VLASOVPARTICLES_HPP_ */

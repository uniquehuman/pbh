/* PBHVlasov example for GRTeclyn (pbhgr project).
 * matter_t for CCZ4RHSWithMatter / ConstraintsWithMatter: the energy-momentum tensor is read from the grid
 * variables deposited from the particles (rho_p, S_i, S_ij, all measured by the normal observer); the matter
 * variables have no RHS (the particles are pushed separately, see VlasovParticles). */
#ifndef PARTICLEMATTER_HPP_
#define PARTICLEMATTER_HPP_

#include "CCZ4Geometry.hpp"
#include "CCZ4RHSWithMatter.hpp"
#include "DimensionDefinitions.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GRParmParse.hpp"
#include "ParticleMatterVars.hpp"
#include "StateVariables.hpp"
#include "TensorAlgebra.hpp"

template <class deriv_t = FourthOrderDerivatives> class ParticleMatter
{
  protected:
    amrex::Real m_G_Newton{1.0};

  public:
    struct params_t
    {
        amrex::Real G_Newton{1.0};
        static void check_params()
        {
            GRParmParse pp("particle_matter");
            amrex::Real G_Newton{1.0};
            pp.queryAdd("G_Newton", G_Newton);
            if (G_Newton < 0.0)
                pp.error("G_Newton", "must be >= 0.0");
        }
        void fill_params()
        {
            GRParmParse pp("particle_matter");
            pp.query("G_Newton", G_Newton);
        }
    };

    ParticleMatter()
    {
        params_t params;
        params.fill_params();
        m_G_Newton = params.G_Newton;
    }

    using Vars = ParticleMatterVars;

    [[nodiscard]] AMREX_GPU_DEVICE emtensor_t compute_emtensor(const int ix, const int iy, const int iz,
                                                              const amrex::Array4<const amrex::Real> &state,
                                                              const deriv_t & /*a_deriv*/,
                                                              const Tensor::Rank2 &h_UU) const
    {
        emtensor_t out;
        const Vars vars(state.cellData(ix, iy, iz));
        out.rho = vars.rho_p();
        FOR (i) { out.j(i) = vars.S_i(i); }
        FOR (i, j) { out.S(i, j) = vars.S_ij(i, j); }
        out.trS = vars.chi() * TensorAlgebra::compute_trace(out.S, h_UU);
        return out;
    }

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE einstein_sources_t
    compute_einstein_sources(int ix, int iy, int iz, const amrex::Array4<const amrex::Real> &state,
                             const deriv_t &a_deriv, const Tensor::Rank2 &h_UU) const
    {
        const emtensor_t emtensor = compute_emtensor(ix, iy, iz, state, a_deriv, h_UU);
        const amrex::Real coupling = 8.0 * M_PI * m_G_Newton;
        einstein_sources_t out;
        out.rho = coupling * emtensor.rho;
        out.trS = coupling * emtensor.trS;
        FOR (i) { out.j(i) = coupling * emtensor.j(i); }
        FOR (i, j) { out.S(i, j) = coupling * emtensor.S(i, j); }
        return out;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void add_matter_rhs(const int ix, const int iy, const int iz,
                                                            const amrex::Array4<amrex::Real> &rhs_state,
                                                            const amrex::Array4<const amrex::Real> & /*state*/,
                                                            const deriv_t & /*a_deriv*/) const
    {
        const amrex::CellData<amrex::Real> rhs = rhs_state.cellData(ix, iy, iz);
        for (int c = c_rho_p; c < NUM_VARS; ++c)
            rhs[c] = 0.0;
    }
};

#endif /* PARTICLEMATTER_HPP_ */

/* PBHVlasov example (pbhgr project).  Weak profile-referenced 1+log lapse (devlog 29–30 Sep):
 *   d_t alpha = advec - eps alpha^p (K - fref(x) K_far(t) - 2 Theta),
 * fref = K_0(x)/K_0(far) carried as a static state variable (c_fref), K_far = far-field (FLRW) K set by the
 * level every coarse step (static member).  Shift: integrated Gamma-driver as in IntegratedMovingPunctureGauge. */
#ifndef PBHGAUGETECLYN_HPP_
#define PBHGAUGETECLYN_HPP_

#include "CCZ4Vars.hpp"
#include "DimensionDefinitions.hpp"
#include "IntegratedMovingPunctureGauge.hpp"
#include "StateVariables.hpp"

template <class deriv_t = FourthOrderDerivatives> class PBHGaugeTeclyn : public IntegratedMovingPunctureGauge<deriv_t>
{
  public:
    using base_t = IntegratedMovingPunctureGauge<deriv_t>;
    using params_t = typename base_t::params_t;
    PBHGaugeTeclyn(amrex::Real a_dx, amrex::Real a_K_far) : base_t(a_dx), m_K_far(a_K_far) {}

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void calculate_rhs(int ix, int iy, int iz, const amrex::Array4<amrex::Real> &rhs,
                                                           const amrex::Array4<const amrex::Real> &state) const
    {
        const amrex::CellData<amrex::Real> &rhs_cell_data = rhs.cellData(ix, iy, iz);
        const amrex::CellData<const amrex::Real> &state_cell_data = state.cellData(ix, iy, iz);
        const CCZ4Vars vars(state_cell_data);
        const Tensor::Rank1 shift_vector({vars.shift(0), vars.shift(1), vars.shift(2)});
        const amrex::Real advec_lapse = this->m_deriv.advec_scalar(ix, iy, iz, state, shift_vector, c_lapse);
        const Tensor::Rank1 advec_shift = this->m_deriv.advec_vector(ix, iy, iz, state, shift_vector, c_shift1);
        amrex::Real eta_of_x{};
        this->compute_eta(eta_of_x, ix, iy, iz);
        const amrex::Real fref = state_cell_data[c_fref];
        rhs_cell_data[c_lapse] = this->m_params.lapse_advec_coeff * advec_lapse -
                                 this->m_params.lapse_coeff * pow(vars.lapse(), this->m_params.lapse_power) *
                                     (vars.K() - fref * m_K_far - 2.0 * vars.Theta());
        FOR (i)
        {
            rhs_cell_data[c_shift1 + i] = this->m_params.shift_advec_coeff * advec_shift(i) +
                                          this->m_params.shift_Gamma_coeff * vars.Gamma(i) - eta_of_x * vars.shift(i) -
                                          vars.B(i);
            rhs_cell_data[c_B1 + i] = 0.0;
        }
        rhs_cell_data[c_fref] = 0.0;
    }

  private:
    amrex::Real m_K_far;
};

#endif /* PBHGAUGETECLYN_HPP_ */

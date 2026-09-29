/* PBHCosmo example (pbhgr project).  Moving-puncture gauge with the lapse driven by K - K_ref,
 * K_ref = far-field (FLRW) value set by the level every coarse step (static, so that the gauge object
 * constructed inside MatterCCZ4RHS sees it; the upstream CosmoMovingPunctureGauge::set_K_mean is called on a
 * local object that never reaches the RHS).  Shift: integrated Gamma-driver as in MovingPunctureGauge. */
#ifndef PBHGAUGE_HPP_
#define PBHGAUGE_HPP_

#include "DimensionDefinitions.hpp"
#include "MovingPunctureGauge.hpp"
#include "Tensor.hpp"

class PBHGauge
{
  public:
    using params_t = MovingPunctureGauge::params_t;
    static double s_K_ref;
    static void set_K_ref(double a_K_ref) { s_K_ref = a_K_ref; }
    static double get_K_ref() { return s_K_ref; }

  protected:
    params_t m_params;

  public:
    PBHGauge(const params_t &a_params) : m_params(a_params) {}

    template <class data_t, template <typename> class vars_t, template <typename> class diff2_vars_t>
    inline void rhs_gauge(vars_t<data_t> &rhs, const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
                          const diff2_vars_t<Tensor<2, data_t>> &d2, const vars_t<data_t> &advec) const
    {
        rhs.lapse = m_params.lapse_advec_coeff * advec.lapse -
                    m_params.lapse_coeff * pow(vars.lapse, m_params.lapse_power) *
                        (vars.K - vars.fref * s_K_ref - 2 * vars.Theta);
        FOR(i)
        {
            rhs.shift[i] = m_params.shift_advec_coeff * advec.shift[i] + m_params.shift_Gamma_coeff * vars.B[i];
            rhs.B[i] = m_params.shift_advec_coeff * advec.B[i] - m_params.shift_advec_coeff * advec.Gamma[i] +
                       rhs.Gamma[i] - m_params.eta * vars.B[i];
        }
    }

    template <class data_t, template <typename> class vars_t>
    inline void rhs_gauge_add_matter_terms(vars_t<data_t> &matter_rhs, const vars_t<data_t> &matter_vars,
                                           Tensor<2, data_t, 3> h_UU, const emtensor_t<data_t> emtensor,
                                           const double G_Newton) const
    {
        FOR(i)
        {
            data_t matter_term_Gamma = 0.0;
            FOR(j)
            {
                matter_term_Gamma += -16.0 * M_PI * G_Newton * matter_vars.lapse * h_UU[i][j] * emtensor.Si[j];
            }
            matter_rhs.B[i] += matter_term_Gamma;
        }
    }
};

#endif /* PBHGAUGE_HPP_ */

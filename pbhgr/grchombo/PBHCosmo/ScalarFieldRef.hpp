/* PBHCosmo example (pbhgr project).  Massive real scalar field exactly as GRChombo's ScalarField<Potential>
 * (Source/Matter/ScalarField.impl.hpp, GRChombo 37e6595) plus one static grid variable fref = K_0(x)/K_0(far),
 * the gauge reference profile used by PBHGauge: no RHS, carried along on regrid like any evolved variable. */
#ifndef SCALARFIELDREF_HPP_
#define SCALARFIELDREF_HPP_

#include "CCZ4Geometry.hpp"
#include "DimensionDefinitions.hpp"
#include "FourthOrderDerivatives.hpp"
#include "Potential.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
#include "VarsTools.hpp"

class ScalarFieldRef
{
  protected:
    Potential my_potential;

  public:
    ScalarFieldRef(const Potential a_potential) : my_potential(a_potential) {}

    template <class data_t> struct Vars
    {
        data_t phi;
        data_t Pi;
        data_t fref;
        template <typename mapping_function_t> void enum_mapping(mapping_function_t mapping_function)
        {
            VarsTools::define_enum_mapping(mapping_function, c_phi, phi);
            VarsTools::define_enum_mapping(mapping_function, c_Pi, Pi);
            VarsTools::define_enum_mapping(mapping_function, c_fref, fref);
        }
    };

    template <class data_t> struct Diff2Vars
    {
        data_t phi;
        template <typename mapping_function_t> void enum_mapping(mapping_function_t mapping_function)
        {
            VarsTools::define_enum_mapping(mapping_function, c_phi, phi);
        }
    };

    template <class data_t, template <typename> class vars_t>
    emtensor_t<data_t> compute_emtensor(const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
                                        const Tensor<2, data_t> &h_UU, const Tensor<3, data_t> &chris_ULL) const
    {
        emtensor_t<data_t> out;
        data_t Vt = -vars.Pi * vars.Pi;
        FOR(i, j) { Vt += vars.chi * h_UU[i][j] * d1.phi[i] * d1.phi[j]; }
        FOR(i, j) { out.Sij[i][j] = -0.5 * vars.h[i][j] * Vt / vars.chi + d1.phi[i] * d1.phi[j]; }
        out.S = vars.chi * TensorAlgebra::compute_trace(out.Sij, h_UU);
        FOR(i) { out.Si[i] = -d1.phi[i] * vars.Pi; }
        out.rho = vars.Pi * vars.Pi + 0.5 * Vt;
        data_t V_of_phi = 0.0, dVdphi = 0.0;
        my_potential.compute_potential(V_of_phi, dVdphi, vars);
        out.rho += V_of_phi;
        out.S += -3.0 * V_of_phi;
        FOR(i, j) { out.Sij[i][j] += -vars.h[i][j] * V_of_phi / vars.chi; }
        return out;
    }

    template <class data_t, template <typename> class vars_t, template <typename> class diff2_vars_t,
              template <typename> class rhs_vars_t>
    void add_matter_rhs(rhs_vars_t<data_t> &rhs, const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
                        const diff2_vars_t<Tensor<2, data_t>> &d2, const vars_t<data_t> &advec) const
    {
        using namespace TensorAlgebra;
        const auto h_UU = compute_inverse_sym(vars.h);
        const auto chris = compute_christoffel(d1.h, h_UU);
        rhs.phi = vars.lapse * vars.Pi + advec.phi;
        rhs.Pi = vars.lapse * vars.K * vars.Pi + advec.Pi;
        FOR(i, j)
        {
            rhs.Pi += h_UU[i][j] * (-0.5 * d1.chi[j] * vars.lapse * d1.phi[i] +
                                    vars.chi * vars.lapse * d2.phi[i][j] + vars.chi * d1.lapse[i] * d1.phi[j]);
            FOR(k) { rhs.Pi += -vars.chi * vars.lapse * h_UU[i][j] * chris.ULL[k][i][j] * d1.phi[k]; }
        }
        data_t V_of_phi = 0.0, dVdphi = 0.0;
        my_potential.compute_potential(V_of_phi, dVdphi, vars);
        rhs.Pi += -vars.lapse * dVdphi;
        rhs.fref = 0.0; // static profile (not advected: it is a coordinate-space reference)
    }
};

#endif /* SCALARFIELDREF_HPP_ */

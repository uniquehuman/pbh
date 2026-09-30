/* PBHVlasov example for GRTeclyn (pbhgr project). */
#ifndef SIMULATIONPARAMETERS_HPP_
#define SIMULATIONPARAMETERS_HPP_

#include "BaseParameterChecker.hpp"
#include "CCZ4RHS.hpp"
#include "FixedGridsTagger.hpp"
#include "IntegratedMovingPunctureGauge.hpp"
#include "LineExtractionParameters.hpp"
#include "ParticleMatter.hpp"

class SimulationParameters
{
  public:
    SimulationParameters() = delete;
    static void check_params()
    {
        BaseParameterChecker::check_params();
        CCZ4_params_t::check_params();
        IntegratedMovingPunctureGauge<FourthOrderDerivatives>::params_t::check_params();
        FixedGridsTagger::check_params();
        ParticleMatter<>::params_t::check_params();
        line_extraction_params_t::check_params("rho_line_extraction");
        line_extraction_params_t::check_params("chi_line_extraction");
        line_extraction_params_t::check_params("K_line_extraction");
        line_extraction_params_t::check_params("lapse_line_extraction");
    }
};

#endif /* SIMULATIONPARAMETERS_HPP_ */

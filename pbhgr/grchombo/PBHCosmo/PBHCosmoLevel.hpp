/* PBHCosmo example for GRChombo (pbhgr project). Based on Examples/ScalarFieldCosmo. */
#ifndef PBHCOSMOLEVEL_HPP_
#define PBHCOSMOLEVEL_HPP_

#include "CosmoAMR.hpp"
#include "DefaultLevelFactory.hpp"
#include "GRAMRLevel.hpp"
#include "Potential.hpp"
#include "ScalarField.hpp"

/*! Massive real scalar field (dust proxy) on a late-start spherical slice from the LTB solution,
 *  evolved with BSSN/CCZ4 and the cosmological moving-puncture gauge (K - <K>); AH finder on. */
class PBHCosmoLevel : public GRAMRLevel
{
    friend class DefaultLevelFactory<PBHCosmoLevel>;
    using GRAMRLevel::GRAMRLevel;

    CosmoAMR &m_cosmo_amr = dynamic_cast<CosmoAMR &>(m_gr_amr);
    typedef ScalarField<Potential> ScalarFieldWithPotential;

    virtual void specificAdvance() override;
    virtual void initialData() override;
    virtual void postRestart() override;
#ifdef CH_USE_HDF5
    virtual void prePlotLevel() override;
#endif
    virtual void specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs, const double a_time) override;
    virtual void specificUpdateODE(GRLevelData &a_soln, const GRLevelData &a_rhs, Real a_dt) override;
    virtual void preTagCells() override;
    virtual void computeTaggingCriterion(FArrayBox &tagging_criterion, const FArrayBox &current_state,
                                         const FArrayBox &current_state_diagnostics) override;
    virtual void specificPostTimeStep() override;
};

#endif /* PBHCOSMOLEVEL_HPP_ */

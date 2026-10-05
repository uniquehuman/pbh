/* PBHVlasov example for GRTeclyn (pbhgr project, stage 2): BSSN/CCZ4 + Vlasov particles. */
#ifndef PBHVLASOVLEVEL_HPP_
#define PBHVLASOVLEVEL_HPP_

#include "DefaultLevelBld.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GRAmrLevel.hpp"
#include "PBHVlasovAmr.hpp"
#include "ParticleMatter.hpp"

class PBHVlasovLevel : public GRAmrLevel
{
  public:
    using GRAmrLevel::GRAmrLevel;
    template <class deriv_t = FourthOrderDerivatives> using Matter = ParticleMatter<deriv_t>;

    PBHVlasovAmr *get_vlasov_amr_ptr();
    static void variableSetUp();

    void initData() override;
    void specific_post_init() override;
    void specific_advance() override;
    void specific_eval_rhs(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs, amrex::Real a_time) override;
    void specific_update_ode(amrex::MultiFab &a_soln) override;
    void specific_post_timestep() override;
    void specific_post_regrid(int a_lbase, int a_new_finest) override;
    void specific_post_checkpoint(const std::string &a_dir, std::ostream &os) override;
    void specific_post_restart() override;
    void tag_cells(amrex::TagBoxArray &a_tag_box_array, amrex::Real a_regrid_threshold) final;

  private:
    void deposit_particles();
    void load_table();
    // tabulated spherical initial data (isotropic radius): chi, K, A^r_r, rho, u_r
    std::vector<amrex::Real> m_tab_r, m_tab_chi, m_tab_K, m_tab_Arr, m_tab_rho, m_tab_ur;
    std::vector<amrex::Real> m_tab_sr, m_tab_st; // optional: sigma_r, sigma_t ratios (warm start)
    bool m_table_loaded{false};
};

#endif /* PBHVLASOVLEVEL_HPP_ */

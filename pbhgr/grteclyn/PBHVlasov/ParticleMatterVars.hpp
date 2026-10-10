/* PBHVlasov example for GRTeclyn (pbhgr project). Cell accessors for the deposited particle sources. */
#ifndef PARTICLEMATTERVARS_HPP_
#define PARTICLEMATTERVARS_HPP_

#include "CCZ4Vars.hpp"
#include "StateVariables.hpp"
#include "Tensor.hpp"

class ParticleMatterVars : public CCZ4Vars
{
  public:
    AMREX_GPU_DEVICE
    ParticleMatterVars(const amrex::CellData<const amrex::Real> &input_cell_data) : CCZ4Vars(input_cell_data) {}

    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE const amrex::Real &rho_p() const { return cell_data[c_rho_p]; }
    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE const amrex::Real &S_i(int i) const { return cell_data[c_S1 + i]; }
    [[nodiscard]] AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real S_ij(int i, int j) const
    {
        // symmetric storage: 11 12 13 22 23 33
        static constexpr int idx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        return cell_data[c_S11 + idx[i][j]];
    }
};

#endif /* PARTICLEMATTERVARS_HPP_ */

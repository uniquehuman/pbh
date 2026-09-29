/* PBHCosmo example (pbhgr project): nested-sphere tagging — level l+1 is created inside radius radii[l]
 * around the centre (coordinate radius).  criterion = 100 inside, 0 outside; use regrid_threshold < 100. */
#ifndef SPHERETAGGINGCRITERION_HPP_
#define SPHERETAGGINGCRITERION_HPP_

#include "Cell.hpp"
#include "DiagnosticVariables.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "Tensor.hpp"
#include <array>

class SphereTaggingCriterion
{
  protected:
    const double m_dx;
    const std::array<double, CH_SPACEDIM> m_center;
    const double m_radius;
    const double m_rho_thr; // additionally require rho > rho_thr (0: no density gate)

  public:
    SphereTaggingCriterion(double dx, std::array<double, CH_SPACEDIM> center, double radius, double rho_thr = 0.)
        : m_dx(dx), m_center(center), m_radius(radius), m_rho_thr(rho_thr){};

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        const Coordinates<data_t> coords(current_cell, m_dx, m_center);
        const data_t r = coords.get_radius();
        const data_t rho = current_cell.load_vars(c_rho);
        auto inside = simd_compare_lt(r, m_radius);
        auto dense = simd_compare_gt(rho, m_rho_thr);
        data_t criterion = simd_conditional(inside, data_t(100.0), data_t(0.0));
        criterion = simd_conditional(dense, criterion, data_t(0.0));
        current_cell.store_vars(criterion, 0);
    }
};

#endif /* SPHERETAGGINGCRITERION_HPP_ */

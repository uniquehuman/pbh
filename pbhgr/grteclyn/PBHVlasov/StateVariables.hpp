/* PBHVlasov example for GRTeclyn (pbhgr project, stage 2).
 * State = CCZ4 variables + 10 grid variables holding the particle (Vlasov) matter sources deposited
 * from the particles: rho_p, S_i (momentum density), S_ij (stress).  They carry no RHS. */
#ifndef STATEVARIABLES_HPP
#define STATEVARIABLES_HPP

#include "ArrayTools.hpp"
#include "BCParity.hpp"
#include "CCZ4StateVariables.hpp"

enum
{
    c_rho_p = NUM_CCZ4_VARS,
    c_S1,
    c_S2,
    c_S3,
    c_S11,
    c_S12,
    c_S13,
    c_S22,
    c_S23,
    c_S33,
    c_fref, // gauge reference profile K_0(x)/K_0(far), static
    NUM_VARS
};

namespace StateVariables
{
static const amrex::Vector<std::string> particle_matter_names = {
    "rho_p", "S1", "S2", "S3", "S11", "S12", "S13", "S22", "S23", "S33", "fref"};
static const amrex::Vector<std::string> names =
    ArrayTools::concatenate(CCZ4StateVariables::names, particle_matter_names);
static constexpr int NUM_MATTER_VARS = NUM_VARS - NUM_CCZ4_VARS;
static const std::array<BCParity, NUM_MATTER_VARS> particle_matter_parities = {
    BCParity::even,  BCParity::odd_x,  BCParity::odd_y,  BCParity::odd_z,   BCParity::even,
    BCParity::odd_xy, BCParity::odd_xz, BCParity::even,   BCParity::odd_yz, BCParity::even, BCParity::even};
static const std::array<BCParity, NUM_VARS> parities =
    ArrayTools::concatenate(CCZ4StateVariables::parities, particle_matter_parities);
static const std::array<amrex::Real, NUM_MATTER_VARS> particle_matter_asymptotic_values = {
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
static const std::array<amrex::Real, NUM_VARS> asymptotic_values =
    ArrayTools::concatenate(CCZ4StateVariables::asymptotic_values, particle_matter_asymptotic_values);
} // namespace StateVariables

#endif /* STATEVARIABLES_HPP */

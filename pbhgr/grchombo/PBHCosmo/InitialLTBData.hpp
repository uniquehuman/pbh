/* PBHCosmo example for GRChombo (pbhgr project, plan 6.12 stage 1).
 * Initial data from a tabulated spherically symmetric slice in isotropic coordinates
 * (produced by scripts/grchombo/ltb_to_grchombo.py from the exact LTB solution):
 *   gamma_ij = psi^4 delta_ij  (chi = psi^-4),  K,  A^rho_rho (traceless part, A^th_th = -A^rho_rho/2),
 *   real scalar field with phi = 0 and Pi = sqrt(2 rho)  ->  rho_sf = rho, J_i = 0 (constraints as in LTB).
 * The table is uniform in the isotropic radius; beyond its end the last row (FLRW far field) is used.
 * compute() must be called with disable_simd() (table look-up per point).
 */
#ifndef INITIALLTBDATA_HPP_
#define INITIALLTBDATA_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "UserVariables.hpp"
#include "parstream.H"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class InitialLTBData
{
  public:
    struct params_t
    {
        std::string table;
        std::array<double, CH_SPACEDIM> center;
    };

    InitialLTBData(params_t a_params, double a_dx) : m_dx(a_dx), m_params(a_params)
    {
        read_table(a_params.table);
    }

    double rho_far() const { return 0.5 * m_Pi.back() * m_Pi.back(); }
    double chi_centre() const { return m_chi.front(); }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        Coordinates<data_t> coords(current_cell, m_dx, m_params.center);
        const double x = coords.x, y = coords.y, z = coords.z;
        const double rr = std::sqrt(x * x + y * y + z * z);
        const double chi = interp(m_chi, rr);
        const double K = interp(m_K, rr);
        const double Arr = interp(m_Arr, rr);
        const double Pi = interp(m_Pi, rr);
        const double Ath = -0.5 * Arr;
        double nx = 0., ny = 0., nz = 0.;
        if (rr > 1e-12 * m_dx)
        {
            nx = x / rr; ny = y / rr; nz = z / rr;
        }
        // conformal traceless A~_ij = chi * A_ij = A^rho_rho n_i n_j + A^th_th (delta_ij - n_i n_j)
        current_cell.store_vars(chi, c_chi);
        current_cell.store_vars(1.0, c_h11);
        current_cell.store_vars(1.0, c_h22);
        current_cell.store_vars(1.0, c_h33);
        current_cell.store_vars(K, c_K);
        current_cell.store_vars(Ath + (Arr - Ath) * nx * nx, c_A11);
        current_cell.store_vars(Ath + (Arr - Ath) * ny * ny, c_A22);
        current_cell.store_vars(Ath + (Arr - Ath) * nz * nz, c_A33);
        current_cell.store_vars((Arr - Ath) * nx * ny, c_A12);
        current_cell.store_vars((Arr - Ath) * nx * nz, c_A13);
        current_cell.store_vars((Arr - Ath) * ny * nz, c_A23);
        current_cell.store_vars(1.0, c_lapse);
        current_cell.store_vars(0.0, c_phi);
        current_cell.store_vars(Pi, c_Pi);
    }

  protected:
    double m_dx;
    params_t m_params;
    std::vector<double> m_r, m_chi, m_K, m_Arr, m_Pi;
    double m_dr = 0.;

    void read_table(const std::string &file)
    {
        std::ifstream in(file);
        if (!in)
            MayDay::Error(("InitialLTBData: cannot open table " + file).c_str());
        std::string line;
        while (std::getline(in, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream ss(line);
            double r, chi, K, Arr, Pi;
            if (!(ss >> r >> chi >> K >> Arr >> Pi))
                continue;
            m_r.push_back(r); m_chi.push_back(chi); m_K.push_back(K); m_Arr.push_back(Arr); m_Pi.push_back(Pi);
        }
        if (m_r.size() < 8)
            MayDay::Error("InitialLTBData: table too short");
        m_dr = m_r[1] - m_r[0];
        pout() << "InitialLTBData: read " << m_r.size() << " rows from " << file << ", dr = " << m_dr
               << ", r_max = " << m_r.back() << ", chi(0) = " << m_chi.front() << ", rho_far = " << rho_far() << std::endl;
    }

    // 4-point Lagrange interpolation on the uniform table (clamped; constant beyond the end)
    double interp(const std::vector<double> &f, double r) const
    {
        const int n = static_cast<int>(f.size());
        const double s = r / m_dr;
        if (s >= n - 1)
            return f[n - 1];
        int i0 = static_cast<int>(std::floor(s)) - 1;
        i0 = std::max(0, std::min(i0, n - 4));
        const double t = s - i0;
        const double L0 = -(t - 1) * (t - 2) * (t - 3) / 6.0;
        const double L1 = t * (t - 2) * (t - 3) / 2.0;
        const double L2 = -t * (t - 1) * (t - 3) / 2.0;
        const double L3 = t * (t - 1) * (t - 2) / 6.0;
        return L0 * f[i0] + L1 * f[i0 + 1] + L2 * f[i0 + 2] + L3 * f[i0 + 3];
    }
};

#endif /* INITIALLTBDATA_HPP_ */

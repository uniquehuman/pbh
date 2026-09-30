/* PBHVlasov example for GRTeclyn (pbhgr project, stage 2). See VlasovParticles.hpp. */
#include "VlasovParticles.hpp"
#include "DimensionDefinitions.hpp"
#include "TensorAlgebra.hpp"
#include <AMReX_MultiFabUtil.H>
#include <cmath>

using namespace VlasovIdx;

namespace
{
// metric quantities from the interpolated field block f[NFLD] (values + derivatives)
struct MetricAtParticle
{
    amrex::Real alpha, chi;
    amrex::Real beta[3], dalpha[3], dchi[3];
    amrex::Real h[3][3], dh[3][3][3], dbeta[3][3]; // dbeta[i][j] = d_i beta^j, dh[i][j][k] = d_i h_jk
    amrex::Real gUU[3][3];                          // gamma^{ij} = chi h~^{ij}
    amrex::Real dgUU[3][3][3];                      // d_i gamma^{jk}
    AMREX_GPU_DEVICE static MetricAtParticle from(const amrex::Real *f)
    {
        MetricAtParticle M;
        auto val = [&](int field, int d) { return f[4 * field + d]; }; // d = 0 value, 1..3 derivative
        M.alpha = val(f_lapse, 0);
        M.chi = val(f_chi, 0);
        static constexpr int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
        for (int i = 0; i < 3; ++i)
        {
            M.beta[i] = val(f_shift + i, 0);
            M.dalpha[i] = val(f_lapse, 1 + i);
            M.dchi[i] = val(f_chi, 1 + i);
            for (int j = 0; j < 3; ++j)
            {
                M.dbeta[i][j] = val(f_shift + j, 1 + i);
                M.h[i][j] = val(f_h + sym[i][j], 0);
                for (int k = 0; k < 3; ++k)
                    M.dh[k][i][j] = val(f_h + sym[i][j], 1 + k);
            }
        }
        // inverse of the symmetric conformal metric (det h~ = 1 up to numerical error; use the full inverse)
        const amrex::Real det = M.h[0][0] * (M.h[1][1] * M.h[2][2] - M.h[1][2] * M.h[2][1]) -
                                M.h[0][1] * (M.h[1][0] * M.h[2][2] - M.h[1][2] * M.h[2][0]) +
                                M.h[0][2] * (M.h[1][0] * M.h[2][1] - M.h[1][1] * M.h[2][0]);
        amrex::Real hUU[3][3];
        hUU[0][0] = (M.h[1][1] * M.h[2][2] - M.h[1][2] * M.h[2][1]) / det;
        hUU[0][1] = (M.h[0][2] * M.h[2][1] - M.h[0][1] * M.h[2][2]) / det;
        hUU[0][2] = (M.h[0][1] * M.h[1][2] - M.h[0][2] * M.h[1][1]) / det;
        hUU[1][1] = (M.h[0][0] * M.h[2][2] - M.h[0][2] * M.h[2][0]) / det;
        hUU[1][2] = (M.h[0][2] * M.h[1][0] - M.h[0][0] * M.h[1][2]) / det;
        hUU[2][2] = (M.h[0][0] * M.h[1][1] - M.h[0][1] * M.h[1][0]) / det;
        hUU[1][0] = hUU[0][1]; hUU[2][0] = hUU[0][2]; hUU[2][1] = hUU[1][2];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                M.gUU[i][j] = M.chi * hUU[i][j];
                for (int k = 0; k < 3; ++k)
                {
                    // d_k h~^{ij} = - h~^{il} h~^{jm} d_k h~_lm
                    amrex::Real dhUU = 0.0;
                    for (int l = 0; l < 3; ++l)
                        for (int m = 0; m < 3; ++m)
                            dhUU -= hUU[i][l] * hUU[j][m] * M.dh[k][l][m];
                    M.dgUU[k][i][j] = M.dchi[k] * hUU[i][j] + M.chi * dhUU;
                }
            }
        return M;
    }
    AMREX_GPU_DEVICE amrex::Real Gamma(const amrex::Real *u) const
    {
        amrex::Real s = 1.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                s += gUU[i][j] * u[i] * u[j];
        return std::sqrt(s);
    }
    AMREX_GPU_DEVICE void rhs(const amrex::Real *u, amrex::Real *dx, amrex::Real *du) const
    {
        const amrex::Real G = Gamma(u);
        for (int i = 0; i < 3; ++i)
        {
            amrex::Real gu = 0.0;
            for (int j = 0; j < 3; ++j)
                gu += gUU[i][j] * u[j];
            dx[i] = alpha * gu / G - beta[i];
            amrex::Real t = -alpha * G * dalpha[i];
            for (int j = 0; j < 3; ++j)
            {
                t += u[j] * dbeta[i][j];
                for (int k = 0; k < 3; ++k)
                    t -= 0.5 * alpha / G * u[j] * u[k] * dgUU[i][j][k];
            }
            du[i] = t;
        }
    }
};
} // namespace

void VlasovParticles::init_uniform(int lev, const amrex::Geometry &geom, amrex::Real rho, int n_per_dir)
{
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const amrex::Real dV = dx[0] * dx[1] * dx[2];
    const amrex::Real mass = rho * dV / (n_per_dir * n_per_dir * n_per_dir);
    const int n = n_per_dir;
    for (amrex::MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const amrex::Box &box = mfi.tilebox();
        auto &ptile = GetParticles(lev)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        amrex::Gpu::HostVector<ParticleType> host;
        for (amrex::IntVect iv = box.smallEnd(); iv <= box.bigEnd(); box.next(iv))
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b)
                    for (int c = 0; c < n; ++c)
                    {
                        ParticleType p;
                        p.id() = ParticleType::NextID();
                        p.cpu() = amrex::ParallelDescriptor::MyProc();
                        p.pos(0) = plo[0] + (iv[0] + (a + 0.5) / n) * dx[0];
                        p.pos(1) = plo[1] + (iv[1] + (b + 0.5) / n) * dx[1];
                        p.pos(2) = plo[2] + (iv[2] + (c + 0.5) / n) * dx[2];
                        for (int r = 0; r < NAOS; ++r)
                            p.rdata(r) = 0.0;
                        p.rdata(m) = mass;
                        p.rdata(Gamma) = 1.0;
                        host.push_back(p);
                    }
        auto old_size = ptile.numParticles();
        ptile.resize(old_size + host.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), ptile.GetArrayOfStructs().begin() + old_size);
    }
    Redistribute();
}

void VlasovParticles::init_from_table(int lev, const amrex::Geometry &geom, const std::vector<amrex::Real> &r_tab,
                                      const std::vector<amrex::Real> &rho_tab, const std::vector<amrex::Real> &ur_tab,
                                      const std::vector<amrex::Real> &chi_tab, int n_per_dir)
{
    const auto dx = geom.CellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto phi = geom.ProbHiArray();
    const amrex::Real cx = 0.5 * (plo[0] + phi[0]), cy = 0.5 * (plo[1] + phi[1]), cz = 0.5 * (plo[2] + phi[2]);
    const amrex::Real dV = dx[0] * dx[1] * dx[2];
    const int n = n_per_dir;
    auto interp = [&](const std::vector<amrex::Real> &tab, amrex::Real r) {
        if (r >= r_tab.back()) return tab.back();
        const amrex::Real s = r / (r_tab[1] - r_tab[0]);
        const int i = std::min(static_cast<int>(s), static_cast<int>(r_tab.size()) - 2);
        const amrex::Real t = s - i;
        return (1 - t) * tab[i] + t * tab[i + 1];
    };
    for (amrex::MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
        const amrex::Box &box = mfi.tilebox();
        auto &ptile = GetParticles(lev)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        amrex::Gpu::HostVector<ParticleType> host;
        for (amrex::IntVect iv = box.smallEnd(); iv <= box.bigEnd(); box.next(iv))
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b)
                    for (int c = 0; c < n; ++c)
                    {
                        ParticleType p;
                        p.id() = ParticleType::NextID();
                        p.cpu() = amrex::ParallelDescriptor::MyProc();
                        p.pos(0) = plo[0] + (iv[0] + (a + 0.5) / n) * dx[0];
                        p.pos(1) = plo[1] + (iv[1] + (b + 0.5) / n) * dx[1];
                        p.pos(2) = plo[2] + (iv[2] + (c + 0.5) / n) * dx[2];
                        const amrex::Real X = p.pos(0) - cx, Y = p.pos(1) - cy, Z = p.pos(2) - cz;
                        const amrex::Real r = std::sqrt(X * X + Y * Y + Z * Z);
                        const amrex::Real rho = interp(rho_tab, r), chi = interp(chi_tab, r), ur = interp(ur_tab, r);
                        const amrex::Real sqrtg = std::pow(chi, -1.5);
                        for (int q = 0; q < NAOS; ++q)
                            p.rdata(q) = 0.0;
                        // rho is the normal-observer energy density = Gamma * rest-mass density; with radial u_r the
                        // covariant components are u_i = u_r n_i psi^2 (n_i = x_i/r, gamma_ij = psi^4 delta_ij)
                        const amrex::Real psi2 = 1.0 / std::sqrt(chi);
                        const amrex::Real G = std::sqrt(1.0 + ur * ur);
                        p.rdata(m) = rho / G * sqrtg * dV / (n * n * n);
                        if (r > 0.0)
                        {
                            p.rdata(u1) = ur * psi2 * X / r;
                            p.rdata(u2) = ur * psi2 * Y / r;
                            p.rdata(u3) = ur * psi2 * Z / r;
                        }
                        p.rdata(Gamma) = G;
                        host.push_back(p);
                    }
        auto old_size = ptile.numParticles();
        ptile.resize(old_size + host.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), ptile.GetArrayOfStructs().begin() + old_size);
    }
    Redistribute();
}

void VlasovParticles::deposit(int lev, amrex::MultiFab &a_state, const amrex::Geometry &geom)
{
    // 1) Gamma per particle from the current fields (interpolate chi and h~ only: reuse interpolate_fields)
    // 2) CIC deposition of the 10 weighted sums into a scratch MultiFab with one ghost cell
    amrex::MultiFab dep(a_state.boxArray(), a_state.DistributionMap(), StateVariables::NUM_MATTER_VARS, 1);
    dep.setVal(0.0);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    amrex::ParticleToMesh(
        *this, dep, lev,
        [=] AMREX_GPU_DEVICE(const PC::ParticleTileType::ConstParticleTileDataType &ptd, int ip,
                             amrex::Array4<amrex::Real> const &arr)
        {
            const auto &p = ptd.m_aos[ip];
            amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
            const amrex::Real mm = p.rdata(m), G = p.rdata(Gamma);
            const amrex::Real u[3] = {p.rdata(u1), p.rdata(u2), p.rdata(u3)};
            amrex::Real w[StateVariables::NUM_MATTER_VARS];
            w[0] = mm * G;
            for (int i = 0; i < 3; ++i)
                w[1 + i] = mm * u[i];
            static constexpr int ii[6] = {0, 0, 0, 1, 1, 2}, jj[6] = {0, 1, 2, 1, 2, 2};
            for (int s = 0; s < 6; ++s)
                w[4 + s] = mm * u[ii[s]] * u[jj[s]] / G;
            interp.ParticleToMesh(p, arr, 0, 0, StateVariables::NUM_MATTER_VARS,
                                  [=] AMREX_GPU_DEVICE(const ParticleType & /*part*/, int comp) { return w[comp]; });
        });
    // 3) divide by sqrt(gamma) dV and copy into the state (interior cells)
    const amrex::Real dV = 1.0 / (dxi[0] * dxi[1] * dxi[2]);
    const auto &dep_arrays = dep.arrays();
    const auto &st_arrays = a_state.arrays();
    amrex::ParallelFor(a_state, [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        const amrex::Real chi = st_arrays[box_no](ix, iy, iz, c_chi);
        const amrex::Real inv = std::pow(chi, 1.5) / dV; // 1/(sqrt(gamma) dV)
        for (int c = 0; c < StateVariables::NUM_MATTER_VARS; ++c)
            st_arrays[box_no](ix, iy, iz, c_rho_p + c) = dep_arrays[box_no](ix, iy, iz, c) * inv;
    });
    amrex::Gpu::streamSynchronize();
}

void VlasovParticles::interpolate_fields(int lev, const amrex::MultiFab &a_state_gh, const amrex::Geometry &geom)
{
    // fields MultiFab: value and 3 derivatives of lapse, shift(3), chi, h~(6): 44 components, 1 ghost cell
    amrex::MultiFab fld(a_state_gh.boxArray(), a_state_gh.DistributionMap(), NFLD, 1);
    const int comps[NMET] = {c_lapse, c_shift1, c_shift2, c_shift3, c_chi, c_h11, c_h12, c_h13, c_h22, c_h23, c_h33};
    const auto dx = geom.CellSizeArray();
    const auto &st = a_state_gh.const_arrays();
    const auto &fa = fld.arrays();
    amrex::GpuArray<int, NMET> comp_arr;
    for (int i = 0; i < NMET; ++i)
        comp_arr[i] = comps[i];
    amrex::ParallelFor(fld, amrex::IntVect(1), [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        const auto &s = st[box_no];
        for (int f = 0; f < NMET; ++f)
        {
            const int c = comp_arr[f];
            fa[box_no](ix, iy, iz, 4 * f) = s(ix, iy, iz, c);
            // fourth-order centred first derivatives
            fa[box_no](ix, iy, iz, 4 * f + 1) =
                (s(ix - 2, iy, iz, c) - 8 * s(ix - 1, iy, iz, c) + 8 * s(ix + 1, iy, iz, c) - s(ix + 2, iy, iz, c)) / (12 * dx[0]);
            fa[box_no](ix, iy, iz, 4 * f + 2) =
                (s(ix, iy - 2, iz, c) - 8 * s(ix, iy - 1, iz, c) + 8 * s(ix, iy + 1, iz, c) - s(ix, iy + 2, iz, c)) / (12 * dx[1]);
            fa[box_no](ix, iy, iz, 4 * f + 3) =
                (s(ix, iy, iz - 2, c) - 8 * s(ix, iy, iz - 1, c) + 8 * s(ix, iy, iz + 1, c) - s(ix, iy, iz + 2, c)) / (12 * dx[2]);
        }
    });
    amrex::Gpu::streamSynchronize();
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    amrex::MeshToParticle(*this, fld, lev,
                          [=] AMREX_GPU_DEVICE(const PC::ParticleTileType::ParticleTileDataType &ptd, int ip,
                                               amrex::Array4<const amrex::Real> const &arr)
                          {
                              auto &p = ptd.m_aos[ip];
                              amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
                              for (int c = 0; c < NFLD; ++c)
                                  ptd.m_rdata[c][ip] = 0.0;
                              interp.MeshToParticle(
                                  p, arr, 0, 0, NFLD,
                                  [=] AMREX_GPU_DEVICE(amrex::Array4<const amrex::Real> const &a, int i, int j, int k,
                                                       int comp) { return a(i, j, k, comp); },
                                  [=] AMREX_GPU_DEVICE(ParticleType & /*part*/, int comp, amrex::Real val) {
                                      ptd.m_rdata[comp][ip] += val;
                                  });
                              // update Gamma from the interpolated metric
                              amrex::Real f[NFLD];
                              for (int c = 0; c < NFLD; ++c)
                                  f[c] = ptd.m_rdata[c][ip];
                              const MetricAtParticle M = MetricAtParticle::from(f);
                              const amrex::Real u[3] = {p.rdata(u1), p.rdata(u2), p.rdata(u3)};
                              p.rdata(Gamma) = M.Gamma(u);
                          });
    amrex::Gpu::streamSynchronize();
}

void VlasovParticles::push(int lev, amrex::Real dt, const amrex::MultiFab &a_state_gh, const amrex::Geometry &geom)
{
    // stage 1: k1 at (x^n, u^n); move to the midpoint
    interpolate_fields(lev, a_state_gh, geom);
    for (PC::ParIterType pti(*this, lev); pti.isValid(); ++pti)
    {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        const int np = pti.numParticles();
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int ip) {
            auto &p = ptd.m_aos[ip];
            amrex::Real f[NFLD];
            for (int c = 0; c < NFLD; ++c)
                f[c] = ptd.m_rdata[c][ip];
            const MetricAtParticle M = MetricAtParticle::from(f);
            amrex::Real u[3] = {p.rdata(u1), p.rdata(u2), p.rdata(u3)}, kx[3], ku[3];
            M.rhs(u, kx, ku);
            p.rdata(px0) = p.pos(0); p.rdata(py0) = p.pos(1); p.rdata(pz0) = p.pos(2);
            p.rdata(u01) = u[0]; p.rdata(u02) = u[1]; p.rdata(u03) = u[2];
            for (int i = 0; i < 3; ++i)
                p.pos(i) += 0.5 * dt * kx[i];
            p.rdata(u1) = u[0] + 0.5 * dt * ku[0];
            p.rdata(u2) = u[1] + 0.5 * dt * ku[1];
            p.rdata(u3) = u[2] + 0.5 * dt * ku[2];
        });
    }
    amrex::Gpu::streamSynchronize();
    // stage 2: k2 at the midpoint, full step from the saved state
    interpolate_fields(lev, a_state_gh, geom);
    for (PC::ParIterType pti(*this, lev); pti.isValid(); ++pti)
    {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        const int np = pti.numParticles();
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int ip) {
            auto &p = ptd.m_aos[ip];
            amrex::Real f[NFLD];
            for (int c = 0; c < NFLD; ++c)
                f[c] = ptd.m_rdata[c][ip];
            const MetricAtParticle M = MetricAtParticle::from(f);
            amrex::Real u[3] = {p.rdata(u1), p.rdata(u2), p.rdata(u3)}, kx[3], ku[3];
            M.rhs(u, kx, ku);
            p.pos(0) = p.rdata(px0) + dt * kx[0];
            p.pos(1) = p.rdata(py0) + dt * kx[1];
            p.pos(2) = p.rdata(pz0) + dt * kx[2];
            p.rdata(u1) = p.rdata(u01) + dt * ku[0];
            p.rdata(u2) = p.rdata(u02) + dt * ku[1];
            p.rdata(u3) = p.rdata(u03) + dt * ku[2];
        });
    }
    amrex::Gpu::streamSynchronize();
    Redistribute();
}

amrex::Real VlasovParticles::total_mass(int lev)
{
    amrex::Real sum = 0.0;
    for (PC::ParIterType pti(*this, lev); pti.isValid(); ++pti)
    {
        const auto &aos = pti.GetArrayOfStructs();
        for (const auto &p : aos)
            sum += p.rdata(m);
    }
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    return sum;
}

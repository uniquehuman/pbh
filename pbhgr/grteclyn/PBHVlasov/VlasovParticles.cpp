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
            amrex::Real t = -G * dalpha[i]; // -Gamma d_i alpha (H = alpha Gamma - beta^j u_j)
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

namespace
{
// Particle lattice built level by level in each level's own index space: n_per_dir^3 particles per level-l cell
// inside the level-l grids grown by buffer_fine cells, excluding the (grown) region of level l+1.  Cells are
// visited through the level-0 tiles owned by this rank (refined to level l), so creation is distributed;
// the particles are added at level 0 and redistributed to their levels.
template <class F>
void make_lattice(VlasovParticles &pc, const amrex::Geometry &geom0, int n_per_dir, int buffer_fine, F &&fill)
{
    using ParticleType = VlasovParticles::ParticleType;
    const int finest = pc.finestLevel();
    const auto plo = geom0.ProbLoArray();
    const auto dx0 = geom0.CellSizeArray();
    // cumulative refinement ratio from level 0 to level l
    std::vector<int> ratio(finest + 1, 1);
    for (int l = 1; l <= finest; ++l)
        ratio[l] = ratio[l - 1] * pc.GetParGDB()->refRatio(l - 1)[0];
    // grown grids per level (level 0: the whole domain), and the region of each level = grown grids minus the
    // grown grids of the next level (coarsened to this level)
    std::vector<amrex::BoxArray> grown(finest + 1);
    for (int l = 0; l <= finest; ++l)
    {
        if (l == 0)
            grown[l] = amrex::BoxArray(geom0.Domain());
        else
        {
            grown[l] = pc.ParticleBoxArray(l);
            grown[l].grow(buffer_fine);
            grown[l] = amrex::intersect(grown[l], amrex::refine(geom0.Domain(), ratio[l]));
            grown[l].removeOverlap(); // grown boxes overlap each other: make the list disjoint
        }
    }
    std::vector<amrex::BoxList> region(finest + 1);
    for (int l = 0; l <= finest; ++l)
    {
        if (l == finest)
            region[l] = grown[l].boxList();
        else
        {
            amrex::BoxList next_bl = grown[l + 1].boxList();
            next_bl.coarsen(pc.GetParGDB()->refRatio(l)[0]);
            amrex::BoxArray next(next_bl); // plain array (coarsened lists cannot call removeOverlap)
            next.removeOverlap();
            for (int b = 0; b < grown[l].size(); ++b)
                region[l].join(next.complementIn(grown[l][b]));
        }
    }
    for (amrex::MFIter mfi = pc.MakeMFIter(0); mfi.isValid(); ++mfi)
    {
        const amrex::Box &tile0 = mfi.tilebox();
        amrex::Gpu::HostVector<ParticleType> host;
        for (int l = 0; l <= finest; ++l)
        {
            const amrex::Box tile_l = amrex::refine(tile0, ratio[l]);
            const int n = n_per_dir;
            const amrex::Real dxl[3] = {dx0[0] / ratio[l], dx0[1] / ratio[l], dx0[2] / ratio[l]};
            const amrex::Real dV_sub = dxl[0] * dxl[1] * dxl[2] / (static_cast<amrex::Real>(n) * n * n);
            for (const amrex::Box &rb : region[l])
            {
                const amrex::Box cells = tile_l & rb;
                if (cells.isEmpty())
                    continue;
                for (amrex::IntVect iv = cells.smallEnd(); iv <= cells.bigEnd(); cells.next(iv))
                    for (int a = 0; a < n; ++a)
                        for (int b = 0; b < n; ++b)
                            for (int c = 0; c < n; ++c)
                            {
                                ParticleType p;
                                p.id() = ParticleType::NextID();
                                p.cpu() = amrex::ParallelDescriptor::MyProc();
                                p.pos(0) = plo[0] + (iv[0] + (a + 0.5) / n) * dxl[0];
                                p.pos(1) = plo[1] + (iv[1] + (b + 0.5) / n) * dxl[1];
                                p.pos(2) = plo[2] + (iv[2] + (c + 0.5) / n) * dxl[2];
                                for (int q = 0; q < VlasovIdx::NAOS; ++q)
                                    p.rdata(q) = 0.0;
                                fill(p, dV_sub);
                                host.push_back(p);
                            }
            }
        }
        auto &ptile = pc.GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto old_size = ptile.numParticles();
        ptile.resize(old_size + host.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), ptile.GetArrayOfStructs().begin() + old_size);
    }
    pc.Redistribute();
}
} // namespace

void VlasovParticles::init_uniform(const amrex::Geometry &geom, amrex::Real rho, int n_per_dir)
{
    make_lattice(*this, geom, n_per_dir, m_lattice_buffer, [=](ParticleType &p, amrex::Real dV_sub) {
        p.rdata(m) = rho * dV_sub;
        p.rdata(Gamma) = 1.0;
        p.rdata(tau) = m_tau0;
    });
}

void VlasovParticles::init_from_table(const amrex::Geometry &geom, const std::vector<amrex::Real> &r_tab,
                                      const std::vector<amrex::Real> &rho_tab, const std::vector<amrex::Real> &ur_tab,
                                      const std::vector<amrex::Real> &chi_tab, int n_per_dir)
{
    const auto plo = geom.ProbLoArray();
    const auto phi = geom.ProbHiArray();
    const amrex::Real cx = 0.5 * (plo[0] + phi[0]), cy = 0.5 * (plo[1] + phi[1]), cz = 0.5 * (plo[2] + phi[2]);
    auto interp = [&](const std::vector<amrex::Real> &tab, amrex::Real r) {
        if (r >= r_tab.back()) return tab.back();
        const amrex::Real s = r / (r_tab[1] - r_tab[0]);
        const int i = std::min(static_cast<int>(s), static_cast<int>(r_tab.size()) - 2);
        const amrex::Real t = s - i;
        return (1 - t) * tab[i] + t * tab[i + 1];
    };
    make_lattice(*this, geom, n_per_dir, m_lattice_buffer, [&](ParticleType &p, amrex::Real dV_sub) {
        const amrex::Real X = p.pos(0) - cx, Y = p.pos(1) - cy, Z = p.pos(2) - cz;
        const amrex::Real r = std::sqrt(X * X + Y * Y + Z * Z);
        const amrex::Real rho = interp(rho_tab, r), chi = interp(chi_tab, r), ur = interp(ur_tab, r);
        // rho is the normal-observer energy density = Gamma * rest-mass density; with radial u_r the covariant
        // components are u_i = u_r n_i psi^2 (n_i = x_i/r, gamma_ij = psi^4 delta_ij)
        const amrex::Real sqrtg = std::pow(chi, -1.5), psi2 = 1.0 / std::sqrt(chi), G = std::sqrt(1.0 + ur * ur);
        p.rdata(m) = rho / G * sqrtg * dV_sub;
        if (r > 0.0)
        {
            p.rdata(u1) = ur * psi2 * X / r;
            p.rdata(u2) = ur * psi2 * Y / r;
            p.rdata(u3) = ur * psi2 * Z / r;
        }
        p.rdata(Gamma) = G;
        p.rdata(tau) = m_tau0;
    });
}

namespace
{
// CIC deposit functor: comp 0 <- dep0 = m Gamma, comp k <- dep0 * dep_k  (k = 1..9)
template <class PC>
struct DepositCIC
{
    amrex::GpuArray<amrex::Real, 3> plo, dxi;
    AMREX_GPU_DEVICE void operator()(const typename PC::ParticleTileType::ConstParticleTileDataType &ptd, int ip,
                                     amrex::Array4<amrex::Real> const &arr) const
    {
        const auto &p = ptd.m_aos[ip];
        amrex::ParticleInterpolator::Linear interp(p, plo, dxi);
        const amrex::Real w0 = p.rdata(VlasovIdx::dep0);
        interp.ParticleToMesh(p, arr, 0, 0, VlasovIdx::NDEP,
                              [=] AMREX_GPU_DEVICE(const typename PC::ParticleType &part, int comp) {
                                  return comp == 0 ? w0 : w0 * part.rdata(VlasovIdx::dep0 + comp);
                              });
    }
};

// fill the deposit slots of the particles of one level from (m, u_i, Gamma)
void fill_deposit_slots(VlasovParticles &pc, int lev)
{
    using PC = VlasovParticles::PC;
    for (PC::ParIterType pti(pc, lev); pti.isValid(); ++pti)
    {
        auto ptd = pti.GetParticleTile().getParticleTileData();
        const int np = pti.numParticles();
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int ip) {
            auto &p = ptd.m_aos[ip];
            const amrex::Real mm = p.rdata(m), G = p.rdata(Gamma);
            const amrex::Real u[3] = {p.rdata(u1), p.rdata(u2), p.rdata(u3)};
            p.rdata(dep0) = mm * G;
            for (int i = 0; i < 3; ++i)
                p.rdata(dep0 + 1 + i) = u[i] / G;
            static constexpr int ii[6] = {0, 0, 0, 1, 1, 2}, jj[6] = {0, 1, 2, 1, 2, 2};
            for (int s = 0; s < 6; ++s)
                p.rdata(dep0 + 4 + s) = u[ii[s]] * u[jj[s]] / (G * G);
        });
    }
    amrex::Gpu::streamSynchronize();
}

// metric fields and first derivatives on a level: NFLD components, 1 ghost cell, from a state with >= 3 ghosts
void metric_fields(const amrex::MultiFab &a_state_gh, const amrex::Geometry &geom, amrex::MultiFab &fld)
{
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
            fa[box_no](ix, iy, iz, 4 * f + 1) =
                (s(ix - 2, iy, iz, c) - 8 * s(ix - 1, iy, iz, c) + 8 * s(ix + 1, iy, iz, c) - s(ix + 2, iy, iz, c)) / (12 * dx[0]);
            fa[box_no](ix, iy, iz, 4 * f + 2) =
                (s(ix, iy - 2, iz, c) - 8 * s(ix, iy - 1, iz, c) + 8 * s(ix, iy + 1, iz, c) - s(ix, iy + 2, iz, c)) / (12 * dx[1]);
            fa[box_no](ix, iy, iz, 4 * f + 3) =
                (s(ix, iy, iz - 2, c) - 8 * s(ix, iy, iz - 1, c) + 8 * s(ix, iy, iz + 1, c) - s(ix, iy, iz + 2, c)) / (12 * dx[2]);
        }
    });
    amrex::Gpu::streamSynchronize();
}

// linear (CIC) interpolation of the NFLD fields to a particle position.  The stencil is clamped to the array box:
// the field MultiFab has one ghost cell, so a particle more than half a cell outside its box (possible at the
// midpoint of a step near a box boundary) would otherwise read out of bounds.  Returns 0 (stencil inside), 1
// (clamped) or -1 (non-finite position; f untouched).
template <class P>
AMREX_GPU_DEVICE int interp_fields_at(const P &p, amrex::Array4<const amrex::Real> const &arr,
                                      const amrex::GpuArray<amrex::Real, 3> &plo,
                                      const amrex::GpuArray<amrex::Real, 3> &dxi, amrex::Real *f)
{
    const auto lo = amrex::lbound(arr);
    const auto hi = amrex::ubound(arr);
    const int alo[3] = {lo.x, lo.y, lo.z}, ahi[3] = {hi.x, hi.y, hi.z};
    int i0[3], i1[3];
    amrex::Real w1[3];
    int status = 0;
    for (int d = 0; d < 3; ++d)
    {
        const amrex::Real l = (p.pos(d) - plo[d]) * dxi[d] - 0.5;
        if (!(l > -1.0e15 && l < 1.0e15))
            return -1;
        const int i = static_cast<int>(amrex::Math::floor(l));
        w1[d] = l - i;
        if (i < alo[d] || i + 1 > ahi[d])
            status = 1;
        i0[d] = amrex::min(amrex::max(i, alo[d]), ahi[d]);
        i1[d] = amrex::min(amrex::max(i + 1, alo[d]), ahi[d]);
    }
    const amrex::Real wx0 = 1.0 - w1[0], wy0 = 1.0 - w1[1], wz0 = 1.0 - w1[2];
    for (int c = 0; c < NFLD; ++c)
        f[c] = wx0 * wy0 * wz0 * arr(i0[0], i0[1], i0[2], c) + w1[0] * wy0 * wz0 * arr(i1[0], i0[1], i0[2], c) +
               wx0 * w1[1] * wz0 * arr(i0[0], i1[1], i0[2], c) + w1[0] * w1[1] * wz0 * arr(i1[0], i1[1], i0[2], c) +
               wx0 * wy0 * w1[2] * arr(i0[0], i0[1], i1[2], c) + w1[0] * wy0 * w1[2] * arr(i1[0], i0[1], i1[2], c) +
               wx0 * w1[1] * w1[2] * arr(i0[0], i1[1], i1[2], c) + w1[0] * w1[1] * w1[2] * arr(i1[0], i1[1], i1[2], c);
    return status;
}

AMREX_GPU_DEVICE inline bool finite_rhs(const amrex::Real *kx, const amrex::Real *ku)
{
    for (int i = 0; i < 3; ++i)
        if (!(std::isfinite(kx[i]) && std::isfinite(ku[i])))
            return false;
    return true;
}
} // namespace

void VlasovParticles::deposit_level(int lev, const amrex::Vector<amrex::MultiFab *> &a_states,
                                    const amrex::Vector<amrex::Geometry> &a_geoms)
{
    const int finest = this->finestLevel();
    fill_deposit_slots(*this, lev);
    if (lev > 0)
        fill_deposit_slots(*this, lev - 1);
    if (lev < finest)
        fill_deposit_slots(*this, lev + 1);
    // 3 ghost cells: ghost particles may sit 2 cells outside the grids and their CIC clouds reach one more
    amrex::MultiFab dep(this->ParticleBoxArray(lev), this->ParticleDistributionMap(lev), NDEP, 3);
    const DepositCIC<PC> f{a_geoms[lev].ProbLoArray(), a_geoms[lev].InvCellSizeArray()};
    if (m_verbose_deposit) amrex::Print() << "  deposit: level " << lev << " own particles\n";
    amrex::ParticleToMesh(*this, dep, lev, f, true);
    if (lev > 0 && m_dep_ghosts)
    {
        // coarse (lev-1) particles within 1 coarse cell of the level-lev grids: their inward clouds
        VlasovParticles ghosts;
        ghosts.Define(this->GetParGDB());
        ghosts.reserveData();
        ghosts.resizeData();
        PC::ParticleTileType ghost_tile;
        this->CreateGhostParticles(lev - 1, 1, ghost_tile);
        if (m_verbose_deposit) amrex::Print() << "  deposit: level " << lev << " ghosts " << ghost_tile.numParticles() << " (rank 0)\n";
        ghosts.AddParticlesAtLevel(ghost_tile, lev, 2);
        amrex::ParticleToMesh(ghosts, dep, lev, f, false);
    }
    if (lev < finest && m_dep_virtuals)
    {
        // copies of the level-(lev+1) particles near the boundary of the fine grids, deposited here with this
        // level's kernel: the cells just outside the fine grids see the same clouds as from own particles
        const int lf = lev + 1;
        const amrex::BoxArray &ba = this->ParticleBoxArray(lf);
        const auto plo = a_geoms[lf].ProbLoArray();
        const auto dxi = a_geoms[lf].InvCellSizeArray();
        const int margin = 2 * this->GetParGDB()->refRatio(lev)[0]; // one coarse CIC cloud, in fine cells
        amrex::Gpu::HostVector<ParticleType> host;
        for (PC::ParIterType pti(*this, lf); pti.isValid(); ++pti)
        {
            const auto &aos = pti.GetArrayOfStructs();
            for (const auto &p : aos)
            {
                const amrex::IntVect iv(static_cast<int>(std::floor((p.pos(0) - plo[0]) * dxi[0])),
                                        static_cast<int>(std::floor((p.pos(1) - plo[1]) * dxi[1])),
                                        static_cast<int>(std::floor((p.pos(2) - plo[2]) * dxi[2])));
                const amrex::Box nb(iv - amrex::IntVect(margin), iv + amrex::IntVect(margin));
                if (!ba.contains(nb))
                    host.push_back(p);
            }
        }
        if (m_verbose_deposit) amrex::Print() << "  deposit: level " << lev << " virtual copies from level " << lf << ": " << host.size() << " (rank 0)\n";
        PC::ParticleTileType copies;
        copies.resize(host.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host.begin(), host.end(), copies.GetArrayOfStructs().begin());
        VlasovParticles virt;
        virt.Define(this->GetParGDB());
        virt.reserveData();
        virt.resizeData();
        virt.AddParticlesAtLevel(copies, lev, 0);
        amrex::ParticleToMesh(virt, dep, lev, f, false);
    }
    // divide by sqrt(gamma) dV and copy into the state (interior cells)
    const auto dx = a_geoms[lev].CellSizeArray();
    const amrex::Real dV = dx[0] * dx[1] * dx[2];
    amrex::MultiFab &st = *a_states[lev];
    amrex::MultiFab dep_on_state(st.boxArray(), st.DistributionMap(), NDEP, 0);
    dep_on_state.ParallelCopy(dep, 0, 0, NDEP);
    const auto &dep_arrays = dep_on_state.const_arrays();
    const auto &st_arrays = st.arrays();
    amrex::ParallelFor(st, [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) {
        const amrex::Real chi = st_arrays[box_no](ix, iy, iz, c_chi);
        const amrex::Real inv = std::pow(chi, 1.5) / dV; // 1/(sqrt(gamma) dV)
        for (int c = 0; c < NDEP; ++c)
            st_arrays[box_no](ix, iy, iz, c_rho_p + c) = dep_arrays[box_no](ix, iy, iz, c) * inv;
    });
    amrex::Gpu::streamSynchronize();
}

void VlasovParticles::deposit_all(const amrex::Vector<amrex::MultiFab *> &a_states,
                                  const amrex::Vector<amrex::Geometry> &a_geoms)
{
    const int finest = this->finestLevel();
    for (int lev = 0; lev <= finest; ++lev)
        deposit_level(lev, a_states, a_geoms);
    for (int lev = finest; lev >= 1; --lev)
        amrex::average_down(*a_states[lev], *a_states[lev - 1], a_geoms[lev], a_geoms[lev - 1], c_rho_p, NDEP,
                            this->GetParGDB()->refRatio(lev - 1));
}

void VlasovParticles::update_gamma(int lev, const amrex::MultiFab &a_state_gh, const amrex::Geometry &geom)
{
    amrex::MultiFab fld(a_state_gh.boxArray(), a_state_gh.DistributionMap(), NFLD, 1);
    metric_fields(a_state_gh, geom, fld);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    amrex::MeshToParticle(*this, fld, lev,
                          [=] AMREX_GPU_DEVICE(ParticleType &p, amrex::Array4<const amrex::Real> const &arr) {
                              amrex::Real f[NFLD];
                              interp_fields_at(p, arr, plo, dxi, f);
                              const MetricAtParticle M = MetricAtParticle::from(f);
                              const amrex::Real u[3] = {p.rdata(u1), p.rdata(u2), p.rdata(u3)};
                              p.rdata(Gamma) = M.Gamma(u);
                          });
    amrex::Gpu::streamSynchronize();
}

void VlasovParticles::push(int lev, amrex::Real dt, const amrex::MultiFab &a_state_gh, const amrex::Geometry &geom)
{
    amrex::MultiFab fld(a_state_gh.boxArray(), a_state_gh.DistributionMap(), NFLD, 1);
    metric_fields(a_state_gh, geom, fld);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const amrex::Real dx0 = geom.CellSize(0);
    const amrex::Real cap_full = m_max_disp_cells * dx0; // cap on the displacement per step
    const amrex::Real cap_half = 0.5 * cap_full;
    // diagnostics: [0] clamped stencils, [1] capped displacements, [2] frozen particles; max displacement
    amrex::Gpu::DeviceVector<long> cnt(3, 0L);
    amrex::Gpu::DeviceVector<amrex::Real> dmax(1, 0.0);
    long *pcnt = cnt.data();
    amrex::Real *pdmax = dmax.data();
    // stage 1: k1 at (x^n, u^n); move to the midpoint.  The state is saved first so that stage 2 can restore it.
    amrex::MeshToParticle(*this, fld, lev,
                          [=] AMREX_GPU_DEVICE(ParticleType &p, amrex::Array4<const amrex::Real> const &arr) {
                              p.rdata(px0) = p.pos(0); p.rdata(py0) = p.pos(1); p.rdata(pz0) = p.pos(2);
                              p.rdata(u01) = p.rdata(u1); p.rdata(u02) = p.rdata(u2); p.rdata(u03) = p.rdata(u3);
                              amrex::Real f[NFLD];
                              const int st = interp_fields_at(p, arr, plo, dxi, f);
                              if (st == 1) amrex::Gpu::Atomic::Add(pcnt + 0, 1L);
                              if (st < 0) { amrex::Gpu::Atomic::Add(pcnt + 2, 1L); return; }
                              const MetricAtParticle M = MetricAtParticle::from(f);
                              const amrex::Real u[3] = {p.rdata(u1), p.rdata(u2), p.rdata(u3)};
                              amrex::Real kx[3], ku[3];
                              M.rhs(u, kx, ku);
                              if (!finite_rhs(kx, ku)) { amrex::Gpu::Atomic::Add(pcnt + 2, 1L); return; }
                              amrex::Real d[3] = {0.5 * dt * kx[0], 0.5 * dt * kx[1], 0.5 * dt * kx[2]};
                              const amrex::Real dn = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                              amrex::Gpu::Atomic::Max(pdmax, 2.0 * dn);
                              if (dn > cap_half)
                              {
                                  const amrex::Real sc = cap_half / dn;
                                  for (int i = 0; i < 3; ++i) d[i] *= sc;
                                  amrex::Gpu::Atomic::Add(pcnt + 1, 1L);
                              }
                              for (int i = 0; i < 3; ++i)
                                  p.pos(i) += d[i];
                              p.rdata(u1) = u[0] + 0.5 * dt * ku[0];
                              p.rdata(u2) = u[1] + 0.5 * dt * ku[1];
                              p.rdata(u3) = u[2] + 0.5 * dt * ku[2];
                          });
    amrex::Gpu::streamSynchronize();
    // stage 2: k2 at the midpoint, full step from the saved state; Gamma at the new state.  A particle whose
    // fields or rhs are not finite is restored to its saved state (frozen for this step) and counted.
    amrex::MeshToParticle(*this, fld, lev,
                          [=] AMREX_GPU_DEVICE(ParticleType &p, amrex::Array4<const amrex::Real> const &arr) {
                              amrex::Real f[NFLD];
                              const int st = interp_fields_at(p, arr, plo, dxi, f);
                              if (st == 1) amrex::Gpu::Atomic::Add(pcnt + 0, 1L);
                              bool ok = st >= 0;
                              MetricAtParticle M;
                              amrex::Real kx[3] = {0.0, 0.0, 0.0}, ku[3] = {0.0, 0.0, 0.0}, dtau = 0.0;
                              if (ok)
                              {
                                  M = MetricAtParticle::from(f);
                                  const amrex::Real u[3] = {p.rdata(u1), p.rdata(u2), p.rdata(u3)};
                                  M.rhs(u, kx, ku);
                                  dtau = dt * M.alpha / M.Gamma(u); // d tau / dt = alpha / Gamma at the midpoint
                                  ok = finite_rhs(kx, ku) && std::isfinite(dtau);
                              }
                              if (!ok)
                              {
                                  p.pos(0) = p.rdata(px0); p.pos(1) = p.rdata(py0); p.pos(2) = p.rdata(pz0);
                                  p.rdata(u1) = p.rdata(u01); p.rdata(u2) = p.rdata(u02); p.rdata(u3) = p.rdata(u03);
                                  amrex::Gpu::Atomic::Add(pcnt + 2, 1L);
                                  return;
                              }
                              amrex::Real d[3] = {dt * kx[0], dt * kx[1], dt * kx[2]};
                              const amrex::Real dn = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                              amrex::Gpu::Atomic::Max(pdmax, dn);
                              if (dn > cap_full)
                              {
                                  const amrex::Real sc = cap_full / dn;
                                  for (int i = 0; i < 3; ++i) d[i] *= sc;
                                  amrex::Gpu::Atomic::Add(pcnt + 1, 1L);
                              }
                              p.pos(0) = p.rdata(px0) + d[0];
                              p.pos(1) = p.rdata(py0) + d[1];
                              p.pos(2) = p.rdata(pz0) + d[2];
                              p.rdata(u1) = p.rdata(u01) + dt * ku[0];
                              p.rdata(u2) = p.rdata(u02) + dt * ku[1];
                              p.rdata(u3) = p.rdata(u03) + dt * ku[2];
                              const amrex::Real un[3] = {p.rdata(u1), p.rdata(u2), p.rdata(u3)};
                              p.rdata(Gamma) = M.Gamma(un);
                              p.rdata(tau) += dtau;
                          });
    amrex::Gpu::streamSynchronize();
    // diagnostics, reduced over the ranks
    std::vector<long> c(3, 0L);
    std::vector<amrex::Real> dm(1, 0.0);
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, cnt.begin(), cnt.end(), c.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, dmax.begin(), dmax.end(), dm.begin());
    amrex::ParallelDescriptor::ReduceLongSum(c.data(), 3);
    amrex::ParallelDescriptor::ReduceRealMax(dm[0]);
    const amrex::Real dm_cells = dm[0] / dx0;
    m_push_diag.max_disp_cells = std::max(m_push_diag.max_disp_cells, static_cast<double>(dm_cells));
    m_push_diag.n_clamped += c[0];
    m_push_diag.n_capped += c[1];
    m_push_diag.n_frozen += c[2];
    if (c[0] + c[1] + c[2] > 0 || dm_cells > 0.5)
        amrex::Print() << "PBHVlasov push L" << lev << ": max displacement " << dm_cells << " cells, clamped " << c[0]
                       << ", capped " << c[1] << ", frozen " << c[2] << "\n";
    Redistribute();
}

VlasovParticles::tau_stats_t VlasovParticles::tau_in_shell(const amrex::Real *center, amrex::Real r_lo, amrex::Real r_hi)
{
    tau_stats_t s;
    double sum_mt = 0.0, sum_mg = 0.0, tmin = 1e300, tmax = -1e300;
    for (int l = 0; l <= this->finestLevel(); ++l)
        for (PC::ParIterType pti(*this, l); pti.isValid(); ++pti)
        {
            const auto &aos = pti.GetArrayOfStructs();
            for (const auto &p : aos)
            {
                const amrex::Real X = p.pos(0) - center[0], Y = p.pos(1) - center[1], Z = p.pos(2) - center[2];
                const amrex::Real r = std::sqrt(X * X + Y * Y + Z * Z);
                if (r < r_lo || r >= r_hi)
                    continue;
                const double mm = p.rdata(m), t = p.rdata(tau);
                ++s.n;
                s.mass += mm;
                sum_mt += mm * t;
                sum_mg += mm * p.rdata(Gamma);
                tmin = std::min(tmin, t);
                tmax = std::max(tmax, t);
            }
        }
    amrex::ParallelDescriptor::ReduceLongSum(s.n);
    amrex::ParallelDescriptor::ReduceRealSum(s.mass);
    amrex::ParallelDescriptor::ReduceRealSum(sum_mt);
    amrex::ParallelDescriptor::ReduceRealSum(sum_mg);
    amrex::ParallelDescriptor::ReduceRealMin(tmin);
    amrex::ParallelDescriptor::ReduceRealMax(tmax);
    if (s.n > 0 && s.mass > 0.0)
    {
        s.tau_mean = sum_mt / s.mass;
        s.gamma_mean = sum_mg / s.mass;
        s.tau_min = tmin;
        s.tau_max = tmax;
    }
    return s;
}

void VlasovParticles::tau_profile(const amrex::Real *center, amrex::Real dr, int nbins, std::vector<double> &n,
                                  std::vector<double> &mass, std::vector<double> &mtau, std::vector<double> &mgamma)
{
    n.assign(nbins, 0.0); mass.assign(nbins, 0.0); mtau.assign(nbins, 0.0); mgamma.assign(nbins, 0.0);
    for (int l = 0; l <= this->finestLevel(); ++l)
        for (PC::ParIterType pti(*this, l); pti.isValid(); ++pti)
        {
            const auto &aos = pti.GetArrayOfStructs();
            for (const auto &p : aos)
            {
                const amrex::Real X = p.pos(0) - center[0], Y = p.pos(1) - center[1], Z = p.pos(2) - center[2];
                const int k = static_cast<int>(std::sqrt(X * X + Y * Y + Z * Z) / dr);
                if (k >= nbins)
                    continue;
                const double mm = p.rdata(m);
                n[k] += 1.0; mass[k] += mm; mtau[k] += mm * p.rdata(tau); mgamma[k] += mm * p.rdata(Gamma);
            }
        }
    amrex::ParallelDescriptor::ReduceRealSum(n.data(), nbins);
    amrex::ParallelDescriptor::ReduceRealSum(mass.data(), nbins);
    amrex::ParallelDescriptor::ReduceRealSum(mtau.data(), nbins);
    amrex::ParallelDescriptor::ReduceRealSum(mgamma.data(), nbins);
}

amrex::Real VlasovParticles::total_mass(int lev)
{
    amrex::Real sum = 0.0;
    (void)lev;
    for (int l = 0; l <= this->finestLevel(); ++l)
    for (PC::ParIterType pti(*this, l); pti.isValid(); ++pti)
    {
        const auto &aos = pti.GetArrayOfStructs();
        for (const auto &p : aos)
            sum += p.rdata(m);
    }
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    return sum;
}

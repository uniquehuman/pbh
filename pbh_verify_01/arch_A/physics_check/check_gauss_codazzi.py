"""Independent check of the constraint signs (F8, F9) from the 4D Einstein tensor:
   2 G_ab n^a n^b = 3R + K^2 - K_ij K^ij  and  -G_ab n^a gamma^b_r = D_j K^j_r - D_r K  (= 8 pi j_r),
for the general spherical 3+1 metric with lapse and shift, K from the 4-metric (K_ab = -gamma gamma nabla n)."""
import sympy as sp
exec(open(__file__.replace('check_gauss_codazzi.py', 'check_contract.py')).read().split('# ---------------------------------------------------------------- stress tensor')[0])

def ricci4(G):
    Ric = sp.zeros(4, 4)
    for b in range(4):
        for d in range(b, 4):
            Ric[b, d] = sum(sp.diff(G[a][b][d], X[a]) - sp.diff(G[a][b][a], X[d])
                            + sum(G[a][a][e] * G[e][b][d] - G[a][d][e] * G[e][b][a] for e in range(4))
                            for a in range(4))
            Ric[d, b] = Ric[b, d]
    return Ric

Ric = ricci4(G4)
Rs4 = sum(gi[a, b] * Ric[a, b] for a in range(4) for b in range(4))
Gein = Ric - g * Rs4 / 2
Gnn = (n_up.T * Gein * n_up)[0]
Kt2 = (KA + 2 * KB)**2 - (KA**2 + 2 * KB**2)
Gm = sp.diff(R, r) / A
R3 = 2 * (1 - Gm**2) / R**2 - 4 * sp.diff(Gm, r) / (A * R)
check("Hamiltonian: 2 G_nn = 3R + K^2 - KK", 2 * Gnn - (R3 + Kt2))
# momentum: -G_ab n^a gamma^b_r
Gnr = -(n_up.T * Gein * proj)[1]
# D_j K^j_r - D_r K for diagonal mixed K = diag(KA,KB,KB) on gamma = diag(A^2,R^2,R^2 sin^2)
divK = sp.diff(KA, r) + 2 * sp.diff(R, r) / R * (KA - KB) - sp.diff(KA + 2 * KB, r)
check("Momentum: -G(n,e_r) = D_j K^j_r - D_r K", Gnr - divK)
print("Gauss-Codazzi sign checks passed")

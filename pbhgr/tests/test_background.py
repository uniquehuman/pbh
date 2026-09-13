import numpy as np
from pbhgr.background import evolve, conservation_residual


def test_scaling_no_decay():
    res = evolve(rho_X0=1.0, rho_R0=1.0, Gamma=0.0, t_end=50.0, t0=0.1)
    a, rX, rR = res["a"], res["rho_X"], res["rho_R"]
    assert np.allclose(rX * a**3, rX[0], rtol=1e-7)
    assert np.allclose(rR * a**4, rR[0], rtol=1e-7)


def test_radiation_dominated_power_law():
    # start radiation dominated at Hubble time t0 = 1/(2H0)
    rR0 = 1.0
    H0 = np.sqrt(8 * np.pi / 3 * rR0)
    t0 = 1 / (2 * H0)
    res = evolve(rho_X0=0.0, rho_R0=rR0, Gamma=0.0, t_end=100 * t0, t0=t0)
    assert np.allclose(res["a"], (res["t"] / t0) ** 0.5, rtol=1e-6)


def test_decay_transfers_energy_and_conserves_total():
    G = 0.3
    res = evolve(rho_X0=1.0, rho_R0=1e-3, Gamma=G, t_end=30.0, t0=0.1)
    a, rX, rR, t = res["a"], res["rho_X"], res["rho_R"], res["t"]
    # comoving X density decays as exp(-Gamma t)
    assert np.allclose(rX * a**3, rX[0] * np.exp(-G * (t - t[0])), rtol=1e-6)
    # integral form of total covariant conservation: (rho a^3)(t) - (rho a^3)(0) = -int H rho_R a^3 dt
    res = evolve(rho_X0=1.0, rho_R0=1e-3, Gamma=G, t_end=30.0, t0=0.1, n_out=20000)
    a, rX, rR, t, H = res["a"], res["rho_X"], res["rho_R"], res["t"], res["H"]
    tot = (rX + rR) * a**3
    work = np.concatenate([[0.0], np.cumsum(0.5 * (H * rR * a**3)[1:] + 0.5 * (H * rR * a**3)[:-1]) * np.diff(t)])
    assert np.max(np.abs(tot - tot[0] + work)) < 1e-5 * tot[0]
    # at late times the energy is in radiation
    assert rR[-1] > 10 * rX[-1]

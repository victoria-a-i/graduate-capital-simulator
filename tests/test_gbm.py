import numpy as np

from gradsim.calibrate import estimate_gbm_params
from gradsim.gbm import simulate_euler, simulate_exact, theoretical_moments


S0, MU, SIG, T = 100.0, 0.08, 0.2, 5.0


def test_shape_and_start():
    p = simulate_exact(S0, MU, SIG, T, 60, 10, seed=1)
    assert p.shape == (10, 61)
    assert np.all(p[:, 0] == S0)
    assert np.all(p > 0)


def test_exact_mean_matches_theory():
    p = simulate_exact(S0, MU, SIG, T, 60, 200_000, seed=2)
    mean, var = theoretical_moments(S0, MU, SIG, T)
    se = np.sqrt(var / p.shape[0])
    assert abs(p[:, -1].mean() - mean) < 4 * se


def test_log_returns_distribution():
    n = 120
    p = simulate_exact(S0, MU, SIG, T, n, 50_000, seed=3)
    r = np.diff(np.log(p), axis=1)
    dt = T / n
    assert np.isclose(r.mean(), (MU - 0.5 * SIG**2) * dt, atol=2e-4)
    assert np.isclose(r.std(), SIG * np.sqrt(dt), rtol=0.01)


def test_zero_vol_is_deterministic():
    p = simulate_exact(S0, MU, 0.0, T, 10, 3, seed=4)
    assert np.allclose(p[:, -1], S0 * np.exp(MU * T))


def test_euler_strong_error_shrinks_with_dt():
    rng = np.random.default_rng(5)
    errs = []
    for n in (10, 40, 160):
        z = rng.standard_normal((20_000, n))
        e = simulate_exact(S0, MU, SIG, T, n, 0, z=z)[:, -1]
        m = simulate_euler(S0, MU, SIG, T, n, 0, z=z)[:, -1]
        errs.append(np.mean(np.abs(e - m)))
    assert errs[0] > errs[1] > errs[2]


def test_calibration_recovers_parameters():
    p = simulate_exact(S0, MU, SIG, 2000, 24_000, 1, seed=6)[0]  # 2000 "years" of monthly data
    est = estimate_gbm_params(p, periods_per_year=12)
    assert abs(est["sigma"] - SIG) < 0.01
    assert abs(est["mu"] - MU) < 3 * est["mu_se"]

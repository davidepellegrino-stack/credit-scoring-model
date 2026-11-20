import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "ifrs9"))

from satellite import (build_dataset, durbin_watson, expanding_oos, fit, inv_logit, ljung_box,  # noqa: E402
                       logit, long_run_multiplier, ols_hac, oos_scores, simulate_path)


def synthetic_panel(n=140, seed=0, a=-0.4, rho=0.85, bu=0.05, bg=-0.01):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1985Q1", periods=n, freq="Q")
    du = rng.normal(0, 1, n)
    g = rng.normal(2, 2, n)
    y = np.empty(n)
    y[0] = -3.5
    for t in range(1, n):
        y[t] = a + rho * y[t - 1] + bu * du[t] + bg * g[t] + rng.normal(0, 0.03)
    return pd.DataFrame({"co_card": inv_logit(y), "d_unrate_4q": du, "gdp_growth_yoy": g}, index=idx)


def test_logit_roundtrip():
    r = np.array([0.5, 3.0, 12.0])
    assert inv_logit(logit(r)) == pytest.approx(r)


def test_ols_hac_matches_textbook_ols_and_zero_lag_white():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = 1.0 + 2.0 * X[:, 1] + rng.normal(size=200)
    res = ols_hac(y, X, lags=0)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    assert res["beta"] == pytest.approx(beta)
    # lags=0 is White's HC0 estimator
    e = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    white = bread @ (X.T * e ** 2) @ X @ bread
    assert res["se"] == pytest.approx(np.sqrt(np.diag(white)))


def test_hac_se_larger_than_naive_with_autocorrelated_errors():
    rng = np.random.default_rng(2)
    n = 400
    x = np.cumsum(rng.normal(size=n)) * 0.1 + rng.normal(size=n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + rng.normal()
    X = np.column_stack([np.ones(n), x])
    y = 1 + 0.5 * x + e
    hac = ols_hac(y, X, lags=8)["se"][1]
    naive = ols_hac(y, X, lags=0)["se"][1]
    assert hac > naive


def test_fit_recovers_parameters():
    panel = synthetic_panel()
    res = fit(build_dataset(panel, exclude=None))
    b = dict(zip(res["names"], res["beta"]))
    assert b["y_lag"] == pytest.approx(0.85, abs=0.05)
    assert b["d_unrate_4q"] == pytest.approx(0.05, abs=0.01)
    assert b["gdp_growth_yoy"] == pytest.approx(-0.01, abs=0.005)
    assert long_run_multiplier(res, "d_unrate_4q") == pytest.approx(0.05 / 0.15, rel=0.35)


def test_build_dataset_drops_missing_and_covid_window():
    panel = synthetic_panel(160)
    panel.loc[panel.index[100], "d_unrate_4q"] = np.nan
    d = build_dataset(panel)
    assert panel.index[100] not in d.index
    assert not any(pd.Period("2020Q1") <= q <= pd.Period("2021Q4") for q in d.index)
    # lag is taken before the window is removed, so 2022Q1 keeps its true 2021Q4 lag
    q = pd.Period("2022Q1")
    assert d.loc[q, "y_lag"] == pytest.approx(logit(panel.loc[pd.Period("2021Q4"), "co_card"]))


def test_diagnostics_on_white_noise_and_autocorrelated_series():
    rng = np.random.default_rng(3)
    wn = rng.normal(size=500)
    assert 1.8 < durbin_watson(wn) < 2.2
    assert ljung_box(wn, 4)[1] > 0.01
    ar = np.zeros(500)
    for t in range(1, 500):
        ar[t] = 0.7 * ar[t - 1] + rng.normal()
    assert ljung_box(ar, 4)[1] < 1e-6


def test_simulate_path_rises_with_unemployment_shock():
    panel = synthetic_panel()
    res = fit(build_dataset(panel, exclude=None))
    quarters = pd.period_range("2030Q1", periods=8, freq="Q")
    calm = pd.DataFrame({"d_unrate_4q": 0.0, "gdp_growth_yoy": 2.0}, index=quarters)
    shock = calm.copy()
    shock["d_unrate_4q"] = 3.0
    base = simulate_path(res, 3.0, calm)
    stress = simulate_path(res, 3.0, shock)
    assert (stress > base).all()
    assert (stress.diff().dropna() > 0).all()  # keeps rising towards the new long-run level


def test_expanding_oos_beats_ar1_when_macro_matters():
    panel = synthetic_panel(160, bu=0.12)
    oos = expanding_oos(panel, start="2005Q1", exclude=None)
    s = oos_scores(oos)
    assert s["n"] == len(oos) > 30
    assert s["rmse_arx"] < s["rmse_ar1"]
    assert s["ratio"] < 1

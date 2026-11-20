"""Macro satellite model: link the US bank credit-card charge-off rate to the economy.

Specification (logit of the quarterly charge-off rate, an autoregressive model with macro drivers):

    logit(c_t) = a + rho * logit(c_{t-1}) + b_u * d_unrate_4q_t + b_g * gdp_growth_yoy_t + e_t

* The logit keeps the fitted rate inside (0, 100) and makes the macro effect multiplicative.
* The lagged dependent variable captures how persistent credit losses are; without it the
  residual autocorrelation is very large and the macro coefficients are meaningless.
* Standard errors are Newey-West (Bartlett kernel, 4 lags). OLS is implemented in numpy so the
  project needs no extra dependency; tests compare it with the textbook formulas.
* The COVID window 2020Q1-2021Q4 is excluded from the fit by default: stimulus and payment
  holidays pushed charge-offs down while unemployment spiked, a break that would flip the sign
  of the unemployment effect. The exclusion is a documented choice, not a hidden one;
  `run_satellite.py` also prints the full-sample fit.
* Quarters with a missing regressor (for example unrate 2025Q4) are dropped, never interpolated.

Usage:
    python ifrs9/run_satellite.py
"""
import numpy as np
import pandas as pd
from scipy import stats

TARGET = "co_card"
REGRESSORS = ("d_unrate_4q", "gdp_growth_yoy")
COVID_WINDOW = (pd.Period("2020Q1"), pd.Period("2021Q4"))


def logit(rate_pct):
    """Percent rate -> logit of the proportion."""
    p = np.asarray(rate_pct, dtype=float) / 100.0
    return np.log(p / (1.0 - p))


def inv_logit(x):
    """Logit -> rate in percent."""
    return 100.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def ols_hac(y, X, lags=4):
    """OLS with Newey-West (Bartlett) heteroskedasticity- and autocorrelation-robust SEs.

    X must already contain the constant column. Returns a dict with coefficients, robust
    standard errors, t-stats, two-sided p-values (normal), residuals and R^2.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    u = X * resid[:, None]
    S = u.T @ u / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = u[lag:].T @ u[:-lag] / n
        S += w * (gamma + gamma.T)
    bread = np.linalg.inv(X.T @ X / n)
    cov = bread @ S @ bread / n
    se = np.sqrt(np.diag(cov))
    t = beta / se
    return {"beta": beta, "se": se, "t": t, "p": 2 * stats.norm.sf(np.abs(t)),
            "resid": resid, "r2": 1.0 - resid.var() / y.var(), "n": n}


def durbin_watson(resid):
    resid = np.asarray(resid, dtype=float)
    return float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))


def ljung_box(resid, lags=4):
    """Ljung-Box Q statistic and p-value for residual autocorrelation up to `lags`."""
    r = np.asarray(resid, dtype=float) - np.mean(resid)
    n = len(r)
    denom = np.sum(r ** 2)
    acf = np.array([np.sum(r[k:] * r[:-k]) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


def build_dataset(panel, target=TARGET, regressors=REGRESSORS, exclude=COVID_WINDOW):
    """Dependent variable, lagged dependent variable and regressors, complete rows only."""
    d = pd.DataFrame(index=panel.index)
    d["y"] = logit(panel[target])
    d["y_lag"] = d["y"].shift(1)
    for r in regressors:
        d[r] = panel[r]
    d = d.dropna()
    if exclude is not None:
        d = d[(d.index < exclude[0]) | (d.index > exclude[1])]
    return d


def fit(data, regressors=REGRESSORS, lags=4, use_macro=True):
    """Fit the ARX model (or the AR(1)-only benchmark with use_macro=False)."""
    cols = ["y_lag"] + (list(regressors) if use_macro else [])
    X = np.column_stack([np.ones(len(data))] + [data[c].to_numpy() for c in cols])
    res = ols_hac(data["y"].to_numpy(), X, lags=lags)
    res["names"] = ["const"] + cols
    res["index"] = data.index
    return res


def coef_table(res):
    return pd.DataFrame({"coef": res["beta"], "nw_se": res["se"], "t": res["t"], "p": res["p"]},
                        index=res["names"])


def long_run_multiplier(res, name):
    """Effect on the long-run logit of a permanent one-unit change in a regressor: b / (1 - rho)."""
    b = dict(zip(res["names"], res["beta"]))
    return b[name] / (1.0 - b["y_lag"])


def predict_one_step(res, row):
    """One-step-ahead prediction of the logit for a data row (uses the actual lagged value)."""
    b = dict(zip(res["names"], res["beta"]))
    return b["const"] + sum(b[c] * row[c] for c in res["names"][1:])


def simulate_path(res, y0_rate_pct, macro_path):
    """Dynamic path of the rate (percent) given the starting rate and a macro DataFrame.

    `macro_path` has one row per future quarter and the model's regressor columns; the model's
    own previous prediction feeds the lag, so errors accumulate as they would in a scenario run.
    """
    b = dict(zip(res["names"], res["beta"]))
    y = float(logit(y0_rate_pct))
    out = []
    for _, row in macro_path.iterrows():
        y = b["const"] + b["y_lag"] * y + sum(b[c] * row[c] for c in res["names"][2:])
        out.append(float(inv_logit(y)))
    return pd.Series(out, index=macro_path.index, name="rate_pct")


def expanding_oos(panel, start="2005Q1", target=TARGET, regressors=REGRESSORS, lags=4,
                  exclude=COVID_WINDOW):
    """One-step-ahead out-of-sample errors from an expanding window, ARX vs AR(1) benchmark.

    Each test quarter is forecast with a model fitted only on earlier quarters (COVID window
    removed from fitting). Test quarters inside the COVID window are skipped in the scores.
    """
    data = build_dataset(panel, target, regressors, exclude=None)
    fit_data_all = build_dataset(panel, target, regressors, exclude=exclude)
    rows = []
    for q in data.index[data.index >= pd.Period(start)]:
        if exclude is not None and exclude[0] <= q <= exclude[1]:
            continue
        train = fit_data_all[fit_data_all.index < q]
        arx, ar = fit(train, regressors, lags), fit(train, regressors, lags, use_macro=False)
        row = data.loc[q]
        rows.append({"quarter": q, "actual": float(inv_logit(row["y"])),
                     "arx": float(inv_logit(predict_one_step(arx, row))),
                     "ar1": float(inv_logit(predict_one_step(ar, row)))})
    return pd.DataFrame(rows).set_index("quarter")


def oos_scores(oos):
    """RMSE (percentage points of the rate) and the ARX / AR(1) ratio; Diebold-Mariano on squared errors."""
    e_arx, e_ar = oos["actual"] - oos["arx"], oos["actual"] - oos["ar1"]
    d = e_ar ** 2 - e_arx ** 2  # positive when the macro model is better
    dm = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return {"n": len(oos), "rmse_arx": float(np.sqrt((e_arx ** 2).mean())),
            "rmse_ar1": float(np.sqrt((e_ar ** 2).mean())),
            "ratio": float(np.sqrt((e_arx ** 2).mean() / (e_ar ** 2).mean())),
            "dm_t": float(dm), "dm_p": float(2 * stats.norm.sf(abs(dm)))}

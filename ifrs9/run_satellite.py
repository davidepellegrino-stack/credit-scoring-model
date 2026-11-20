"""Fit the macro satellite model on FRED data and print coefficients, diagnostics and the OOS check."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fred import load_macro_panel  # noqa: E402
from satellite import (REGRESSORS, TARGET, build_dataset, coef_table, durbin_watson,  # noqa: E402
                       expanding_oos, fit, ljung_box, long_run_multiplier, oos_scores)


def describe(label, data):
    res = fit(data)
    print(f"\n{label}: {data.index.min()} to {data.index.max()}, n = {res['n']}")
    print(coef_table(res).round(4).to_string())
    q, p = ljung_box(res["resid"], 4)
    print(f"R2 {res['r2']:.3f}  residual sd {res['resid'].std():.3f}  DW {durbin_watson(res['resid']):.2f}  "
          f"Ljung-Box(4) {q:.1f} (p {p:.2f})")
    return res


if __name__ == "__main__":
    panel = load_macro_panel()
    print(f"Target: logit of {TARGET} (credit-card charge-off rate, all US banks); regressors {REGRESSORS}")
    describe("Estimation sample (COVID 2020Q1-2021Q4 excluded)", build_dataset(panel))
    describe("Full sample (COVID included, for comparison)", build_dataset(panel, exclude=None))

    res = fit(build_dataset(panel))
    print("\nLong-run effect of a permanent +1 pp rise in the 4-quarter unemployment change on the "
          f"charge-off rate: x{np.exp(long_run_multiplier(res, 'd_unrate_4q')):.2f}")

    oos = expanding_oos(panel)
    s = oos_scores(oos)
    print(f"\nExpanding-window one-step-ahead forecasts, {oos.index.min()} to {oos.index.max()} (n = {s['n']}):")
    print(f"RMSE ARX {s['rmse_arx']:.3f} pp, AR(1) only {s['rmse_ar1']:.3f} pp, ratio {s['ratio']:.3f}, "
          f"Diebold-Mariano t = {s['dm_t']:.2f} (p = {s['dm_p']:.2f})")

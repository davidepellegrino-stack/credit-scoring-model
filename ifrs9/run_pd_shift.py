"""Champion loan-level PDs shifted by the macro scenarios: 12-month and lifetime PD per scenario.

Fits the champion (HistGB with bureau + previous-application features) on the 80% training split
(about 3 minutes), predicts the 61,503 test loans, and applies `pd_shift` with the multipliers
from the satellite model. Writes outputs/tables/pd_by_scenario.csv; test PDs are cached in
outputs/tables/champion_test_pd.csv so later steps do not refit.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from config import OUTPUT_TABLES, RANDOM_STATE, TEST_SIZE  # noqa: E402
from data_home_credit import ID_COL, TARGET_COL, load_application, load_table  # noqa: E402
from features_history import (BUREAU_COLS, HISTORY_FEATURES, PREV_COLS,  # noqa: E402
                              add_history_features, bureau_aggregates, previous_aggregates)
from features_home_credit import NUMERIC, build_features  # noqa: E402
from run_home_credit import build_home_credit_pipelines  # noqa: E402

from fred import load_macro_panel  # noqa: E402
from pd_shift import pd_term_structure, scenario_multipliers  # noqa: E402
from satellite import build_dataset, fit, simulate_path  # noqa: E402
from scenarios import build_scenarios  # noqa: E402

CACHE = os.path.join(OUTPUT_TABLES, "champion_test_pd.csv")


def champion_test_pds():
    """Test-set PDs and defaults of the champion; cached after the first run."""
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    df = load_application("application_train")
    X_app, y = build_features(df), df[TARGET_COL]
    bureau = bureau_aggregates(load_table("bureau", BUREAU_COLS))
    prev = previous_aggregates(load_table("previous_application", PREV_COLS))
    X = add_history_features(X_app, df[ID_COL], bureau, prev)
    idx_tr, idx_te = train_test_split(X_app.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    pipe = build_home_credit_pipelines(NUMERIC + HISTORY_FEATURES)["Gradient Boosting"]
    pipe.fit(X.loc[idx_tr], y.loc[idx_tr])
    out = pd.DataFrame({"pd": pipe.predict_proba(X.loc[idx_te])[:, 1], "default": y.loc[idx_te].values},
                       index=df.loc[idx_te, ID_COL].values)
    out.index.name = ID_COL
    out.reset_index().to_csv(CACHE, index=False)
    return out.reset_index()


def scenario_pd_table():
    """Per-scenario DataFrame of loan PDs plus the multipliers used."""
    test = champion_test_pds()
    panel = load_macro_panel()
    res = fit(build_dataset(panel))
    last = panel["co_card"].dropna().index.max()
    y0 = float(panel.loc[last, "co_card"])
    scen = build_scenarios(panel, last)
    paths = {n: simulate_path(res, y0, m) for n, m in scen.items()}
    mult = scenario_multipliers(paths)
    ts = pd_term_structure(test["pd"].to_numpy(), {n: m.to_numpy() for n, m in mult.items()})
    return test, mult, ts


if __name__ == "__main__":
    test, mult, ts = scenario_pd_table()
    print(f"{len(test):,} test loans, observed default rate {test['default'].mean():.2%}, "
          f"mean champion PD {test['pd'].mean():.2%}")
    print("\nCharge-off multiplier vs baseline, quarters 1, 4, 8, 12:")
    print(pd.DataFrame({n: m.iloc[[0, 3, 7, 11]].values for n, m in mult.items()},
                       index=[1, 4, 8, 12]).round(3))
    rows = []
    for n, d in ts.items():
        rows.append({"scenario": n, "mean_pd12": d["pd12"].mean(), "mean_pd_life_3y": d["pd_life"].mean(),
                     "p99_pd12": np.quantile(d["pd12"], 0.99)})
    table = pd.DataFrame(rows).set_index("scenario")
    print("\nPortfolio PD by scenario:")
    print((table * 100).round(2).to_string())
    table.to_csv(os.path.join(OUTPUT_TABLES, "pd_by_scenario.csv"))

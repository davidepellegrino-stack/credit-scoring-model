"""Calibrate the champion PD model (HistGB + bureau/previous-application history).

    python src/run_calibration.py

The champion is refit on 75% of the training loans; the other 25% is the calibration split
used to fit Platt scaling and isotonic regression. The 20% test set (same split as
run_home_credit.py) is untouched until the final comparison.
Outputs: outputs/tables/home_credit_calibration.csv, _reliability.csv and
outputs/figures/home_credit_reliability.png.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from calibration import (IsotonicCalibrator, PlattCalibrator, calibration_summary,  # noqa: E402
                         reliability_table)
from config import OUTPUT_FIGS, OUTPUT_TABLES, RANDOM_STATE, TEST_SIZE  # noqa: E402
from data_home_credit import ID_COL, TARGET_COL, load_application, load_table  # noqa: E402
from features_history import (BUREAU_COLS, HISTORY_FEATURES, PREV_COLS,  # noqa: E402
                              add_history_features, bureau_aggregates, previous_aggregates)
from features_home_credit import NUMERIC, build_features  # noqa: E402
from run_home_credit import build_home_credit_pipelines  # noqa: E402

CAL_SIZE = 0.25


def main():
    df = load_application("application_train")
    X_app, y = build_features(df), df[TARGET_COL]
    X = add_history_features(
        X_app, df[ID_COL],
        bureau_aggregates(load_table("bureau", BUREAU_COLS)),
        previous_aggregates(load_table("previous_application", PREV_COLS)))
    idx_tr, idx_te = train_test_split(
        X.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    idx_fit, idx_cal = train_test_split(
        idx_tr, test_size=CAL_SIZE, random_state=RANDOM_STATE, stratify=y.loc[idx_tr])
    print(f"fit {len(idx_fit):,} / calibration {len(idx_cal):,} / test {len(idx_te):,}")

    pipe = build_home_credit_pipelines(NUMERIC + HISTORY_FEATURES)["Gradient Boosting"]
    pipe.fit(X.loc[idx_fit], y.loc[idx_fit])
    p_cal = pipe.predict_proba(X.loc[idx_cal])[:, 1]
    p_te = pipe.predict_proba(X.loc[idx_te])[:, 1]
    y_cal, y_te = y.loc[idx_cal], y.loc[idx_te]

    versions = {"Raw HistGB": p_te}
    for name, cal in [("Platt scaling", PlattCalibrator()), ("Isotonic", IsotonicCalibrator())]:
        versions[name] = cal.fit(p_cal, y_cal).predict(p_te)

    summary = pd.DataFrame(
        [{"Version": k, **calibration_summary(y_te, v)} for k, v in versions.items()])
    summary.round(5).to_csv(os.path.join(OUTPUT_TABLES, "home_credit_calibration.csv"), index=False)
    rel = pd.concat([reliability_table(y_te, v).assign(version=k) for k, v in versions.items()])
    rel.round(5).to_csv(os.path.join(OUTPUT_TABLES, "home_credit_reliability.csv"), index=False)
    print(summary.round(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for k, v in versions.items():
        t = reliability_table(y_te, v)
        ax.plot(t["mean_pd"], t["default_rate"], marker="o", ms=4, label=k)
    top = max(rel["mean_pd"].max(), rel["default_rate"].max()) * 1.05
    ax.plot([0, top], [0, top], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted PD (test decile)")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability curve, test set (10 equal-count bins)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIGS, "home_credit_reliability.png"), dpi=150)


if __name__ == "__main__":
    main()

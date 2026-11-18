"""Out-of-time check of the champion PD model (HistGB + bureau/previous-application history).

    python src/run_out_of_time.py

Train on loans applied before 2020-01-01 (2018-2019), test on 2020. For reference the same
model is fitted on a random subset of the same size and tested on the same 2020 loans, so any
gap comes from the time split and not from the sample size.
Outputs: outputs/tables/home_credit_oot.csv and home_credit_oot_quarterly.csv.
"""
import os
import sys

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_TABLES, RANDOM_STATE  # noqa: E402
from data_home_credit import DATE_COL, ID_COL, TARGET_COL, load_application, load_table  # noqa: E402
from features_history import (BUREAU_COLS, HISTORY_FEATURES, PREV_COLS,  # noqa: E402
                              add_history_features, bureau_aggregates, previous_aggregates)
from features_home_credit import NUMERIC, build_features  # noqa: E402
from metrics import gini, ks_statistic  # noqa: E402
from out_of_time import date_id_rank_correlation, quarterly_table, time_split  # noqa: E402
from run_home_credit import build_home_credit_pipelines  # noqa: E402

CUTOFF = "2020-01-01"


def main():
    df = load_application("application_train")
    dates, y = df[DATE_COL], df[TARGET_COL]
    print(f"Spearman(application_date, SK_ID_CURR) = {date_id_rank_correlation(dates, df[ID_COL]):.5f}")
    X = add_history_features(
        build_features(df), df[ID_COL],
        bureau_aggregates(load_table("bureau", BUREAU_COLS)),
        previous_aggregates(load_table("previous_application", PREV_COLS)))
    idx_tr, idx_te = time_split(dates, CUTOFF)
    idx_rand = pd.Index(pd.Series(df.index.difference(idx_te)).sample(
        len(idx_tr), random_state=RANDOM_STATE))
    print(f"train {len(idx_tr):,} (< {CUTOFF}) / test {len(idx_te):,}")

    rows, p_oot = [], None
    for name, idx in [("Out-of-time train (2018-2019)", idx_tr), ("Random train, same size", idx_rand)]:
        pipe = build_home_credit_pipelines(NUMERIC + HISTORY_FEATURES)["Gradient Boosting"]
        pipe.fit(X.loc[idx], y.loc[idx])
        p = pipe.predict_proba(X.loc[idx_te])[:, 1]
        p_oot = p if p_oot is None else p_oot
        rows.append({"Training set": name, "AUC": roc_auc_score(y.loc[idx_te], p),
                     "Gini": gini(y.loc[idx_te], p), "KS": ks_statistic(y.loc[idx_te], p),
                     "Brier": brier_score_loss(y.loc[idx_te], p),
                     "Mean PD": p.mean(), "Default rate": y.loc[idx_te].mean()})
    out = pd.DataFrame(rows)
    out.round(5).to_csv(os.path.join(OUTPUT_TABLES, "home_credit_oot.csv"), index=False)
    print(out.round(4).to_string(index=False))

    quarterly = quarterly_table(dates.loc[idx_te], y.loc[idx_te], p_oot)
    quarterly.round(5).to_csv(os.path.join(OUTPUT_TABLES, "home_credit_oot_quarterly.csv"), index=False)
    print(quarterly.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""WoE / IV scorecard on Home Credit, compared with the champion gradient-boosting model.

    python src/run_scorecard.py

Same 80/20 stratified split as run_home_credit.py. Bins are fitted on the training loans only.
Variables with IV >= MIN_IV on the training set enter the scorecard (history features included).
Outputs: outputs/tables/home_credit_iv.csv, home_credit_scorecard_points.csv,
home_credit_scorecard_metrics.csv.
"""
import os
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_TABLES, RANDOM_STATE, TEST_SIZE  # noqa: E402
from data_home_credit import ID_COL, TARGET_COL, load_application, load_table  # noqa: E402
from features_history import (BUREAU_COLS, HISTORY_FEATURES, PREV_COLS,  # noqa: E402
                              add_history_features, bureau_aggregates, previous_aggregates)
from features_home_credit import CATEGORICAL, NUMERIC, build_features  # noqa: E402
from metrics import gini, ks_statistic  # noqa: E402
from scorecard import Scorecard, WoEBinner  # noqa: E402

MIN_IV = 0.02
MAX_VARS = 15


def main():
    df = load_application("application_train")
    X_app, y = build_features(df), df[TARGET_COL]
    X = add_history_features(
        X_app, df[ID_COL],
        bureau_aggregates(load_table("bureau", BUREAU_COLS)),
        previous_aggregates(load_table("previous_application", PREV_COLS)))
    idx_tr, idx_te = train_test_split(
        X.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    Xtr, Xte, ytr, yte = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]

    num = [c for c in NUMERIC + HISTORY_FEATURES if c in X.columns]
    iv = WoEBinner(num, CATEGORICAL).fit(Xtr, ytr).iv_
    print("Information value (train):\n", iv.round(3).head(20).to_string())
    iv.rename("iv").round(4).to_csv(os.path.join(OUTPUT_TABLES, "home_credit_iv.csv"))

    keep = iv[iv >= MIN_IV].head(MAX_VARS).index.tolist()
    binner = WoEBinner([c for c in keep if c in num], [c for c in keep if c in CATEGORICAL])
    sc = Scorecard(binner, pdo=20, base_score=600, base_odds=50).fit(Xtr, ytr)

    p, s = sc.predict_proba(Xte), sc.score(Xte)
    row = {"variables": len(keep), "test_auc": roc_auc_score(yte, p), "gini": gini(yte, p),
           "ks": ks_statistic(yte, p), "score_min": s.min(), "score_median": float(pd.Series(s).median()),
           "score_max": s.max()}
    m = pd.DataFrame([row])
    print(m.round(4).to_string(index=False))
    m.round(4).to_csv(os.path.join(OUTPUT_TABLES, "home_credit_scorecard_metrics.csv"), index=False)
    pts = sc.points_table()
    pts.round(4).to_csv(os.path.join(OUTPUT_TABLES, "home_credit_scorecard_points.csv"), index=False)

    bands = pd.DataFrame({"score": s, "y": yte.to_numpy()})
    bands["band"] = pd.qcut(bands["score"], 10, duplicates="drop")
    print(bands.groupby("band", observed=True)["y"].agg(["count", "mean"]).round(4).to_string())
    print(pts.groupby("variable")["points"].agg(lambda v: v.max() - v.min()).sort_values(
        ascending=False).round(1).to_string())


if __name__ == "__main__":
    main()

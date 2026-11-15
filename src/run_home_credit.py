"""Train and compare PD models on the real Home Credit application table.

    python src/run_home_credit.py

Each model is trained twice on the same stratified 80/20 split: on application-form
features only, and with bureau + previous_application aggregates added. The AUC gain
comes with a paired bootstrap interval over the test set.

Three models:
  * Logistic Regression: median imputation + missing-value indicators + scaling
  * Random Forest: median imputation + missing-value indicators
  * Gradient boosting (HistGradientBoostingClassifier): handles NaN natively

Imputers are fitted inside the Pipeline, i.e. on training data only. Metrics are
computed on the held-out 20% and written to outputs/tables/home_credit_models.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_TABLES, RANDOM_STATE, TEST_SIZE  # noqa: E402
from data_home_credit import ID_COL, TARGET_COL, load_application, load_table  # noqa: E402
from features_history import (BUREAU_COLS, HISTORY_FEATURES, PREV_COLS,  # noqa: E402
                              add_history_features, bureau_aggregates, previous_aggregates)
from features_home_credit import CATEGORICAL, NUMERIC, build_features  # noqa: E402
from metrics import decile_table, gini, ks_statistic  # noqa: E402


def _onehot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=50)


def build_home_credit_pipelines(numeric=None) -> dict:
    """Named sklearn Pipelines; `numeric` defaults to the application-form features."""
    numeric = NUMERIC if numeric is None else numeric

    def pre(scale: bool, impute: bool = True):
        num_steps = []
        if impute:
            num_steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True)))
        if scale:
            num_steps.append(("scale", StandardScaler()))
        num = Pipeline(num_steps) if num_steps else "passthrough"
        return ColumnTransformer([("num", num, numeric), ("cat", _onehot(), CATEGORICAL)])

    return {
        "Logistic Regression": Pipeline([
            ("pre", pre(scale=True)),
            ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("pre", pre(scale=False)),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=100,
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("pre", pre(scale=False, impute=False)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                min_samples_leaf=100, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.1,
                random_state=RANDOM_STATE)),
        ]),
    }


def paired_bootstrap_auc_gain(y, p_base, p_new, n_boot=300, seed=0):
    """AUC(new) - AUC(base) with a 95% percentile interval, resampling the test set."""
    rng = np.random.default_rng(seed)
    y, p_base, p_new = np.asarray(y), np.asarray(p_base), np.asarray(p_new)
    gains = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        gains.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_base[idx]))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return roc_auc_score(y, p_new) - roc_auc_score(y, p_base), lo, hi


def main():
    df = load_application("application_train")
    X_app, y = build_features(df), df[TARGET_COL]
    bureau = bureau_aggregates(load_table("bureau", BUREAU_COLS))
    prev = previous_aggregates(load_table("previous_application", PREV_COLS))
    X_full = add_history_features(X_app, df[ID_COL], bureau, prev)
    print(f"bureau history for {X_full['has_bureau'].mean():.1%} of loans, "
          f"previous applications for {X_full['has_prev'].mean():.1%}")

    idx_tr, idx_te = train_test_split(
        X_app.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    y_tr, y_te = y.loc[idx_tr], y.loc[idx_te]
    print(f"train {len(idx_tr):,} / test {len(idx_te):,}; default rate {y.mean():.2%}")

    sets = {"Application only": (X_app, NUMERIC),
            "+ Bureau & previous": (X_full, NUMERIC + HISTORY_FEATURES)}
    preds, rows = {}, []
    for set_name, (X, numeric) in sets.items():
        for name, pipe in build_home_credit_pipelines(numeric).items():
            pipe.fit(X.loc[idx_tr], y_tr)
            p = pipe.predict_proba(X.loc[idx_te])[:, 1]
            preds[(set_name, name)] = p
            rows.append({
                "Features": set_name, "Model": name,
                "Test AUC": round(roc_auc_score(y_te, p), 4),
                "Gini": round(gini(y_te, p), 4),
                "KS": round(ks_statistic(y_te, p), 4),
                "Avg Precision": round(average_precision_score(y_te, p), 4),
                "Brier": round(brier_score_loss(y_te, p), 4),
            })
            print(rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTPUT_TABLES, "home_credit_models.csv"), index=False)

    gains = []
    for name in build_home_credit_pipelines():
        g, lo, hi = paired_bootstrap_auc_gain(
            y_te, preds[("Application only", name)], preds[("+ Bureau & previous", name)])
        gains.append({"Model": name, "AUC gain": round(g, 4),
                      "95% CI low": round(lo, 4), "95% CI high": round(hi, 4)})
    gain_df = pd.DataFrame(gains)
    gain_df.to_csv(os.path.join(OUTPUT_TABLES, "home_credit_history_gain.csv"), index=False)
    champion = preds[("+ Bureau & previous", "Gradient Boosting")]
    deciles = decile_table(y_te, champion)
    deciles.round(4).to_csv(os.path.join(OUTPUT_TABLES, "home_credit_deciles.csv"), index=False)
    print("\nDecile gains/lift, HistGB with history (decile 1 = riskiest):\n"
          + deciles.round(4).to_string(index=False))
    print("\n" + res.to_string(index=False) + "\n\n" + gain_df.to_string(index=False))


if __name__ == "__main__":
    main()

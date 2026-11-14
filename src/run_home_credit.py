"""Train and compare PD models on the real Home Credit application table.

    python src/run_home_credit.py

Three models on the same stratified 80/20 split:
  * Logistic Regression: median imputation + missing-value indicators + scaling
  * Random Forest: median imputation + missing-value indicators
  * Gradient boosting (HistGradientBoostingClassifier): handles NaN natively

Imputers are fitted inside the Pipeline, i.e. on training data only. Metrics are
computed on the held-out 20% and written to outputs/tables/home_credit_models.csv.
"""
import os
import sys

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
from data_home_credit import TARGET_COL, load_application  # noqa: E402
from features_home_credit import CATEGORICAL, NUMERIC, build_features  # noqa: E402


def _onehot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=50)


def build_home_credit_pipelines() -> dict:
    """Named sklearn Pipelines for the engineered application features."""
    def pre(scale: bool, impute: bool = True):
        num_steps = []
        if impute:
            num_steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True)))
        if scale:
            num_steps.append(("scale", StandardScaler()))
        num = Pipeline(num_steps) if num_steps else "passthrough"
        return ColumnTransformer([("num", num, NUMERIC), ("cat", _onehot(), CATEGORICAL)])

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


def main():
    df = load_application("application_train")
    X, y = build_features(df), df[TARGET_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"train {len(X_tr):,} / test {len(X_te):,}; default rate {y.mean():.2%}")

    rows = []
    for name, pipe in build_home_credit_pipelines().items():
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        rows.append({
            "Model": name,
            "Test AUC": round(roc_auc_score(y_te, p), 4),
            "Avg Precision": round(average_precision_score(y_te, p), 4),
            "Brier": round(brier_score_loss(y_te, p), 4),
        })
        print(rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTPUT_TABLES, "home_credit_models.csv"), index=False)
    print("\n" + res.to_string(index=False))


if __name__ == "__main__":
    main()

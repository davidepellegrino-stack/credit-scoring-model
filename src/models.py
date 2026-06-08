"""
Credit scoring models using scikit-learn pipelines.

Models:
  1. Logistic Regression (L2 regularisation) — interpretable baseline
  2. Random Forest — non-linear, handles interactions, feature importance
  3. Gradient Boosting (GBM) — state-of-the-art for tabular credit data

Each model uses a full sklearn Pipeline:
  Numeric: StandardScaler
  Categorical: OneHotEncoder
  → Classifier

This ensures no data leakage and makes the pipeline deployable.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    average_precision_score, brier_score_loss
)
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
                    TEST_SIZE, CV_FOLDS, RANDOM_STATE, OUTPUT_TABLES)


def build_preprocessor() -> ColumnTransformer:
    """Preprocessing: scale numerics, one-hot encode categoricals."""
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])


def build_pipelines() -> dict:
    """Return a dict of named sklearn Pipelines."""
    pre = build_preprocessor()
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(
                C=0.1, max_iter=1000, random_state=RANDOM_STATE,
                class_weight="balanced"
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=20,
                random_state=RANDOM_STATE, class_weight="balanced",
                n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, subsample=0.8,
                random_state=RANDOM_STATE
            )),
        ]),
    }


def split_data(df: pd.DataFrame):
    """Stratified train/test split preserving default rate."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )


def evaluate(model, X_test, y_test, model_name: str) -> dict:
    """Compute all evaluation metrics for a fitted model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc   = roc_auc_score(y_test, y_proba)
    ap    = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    cm    = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  ROC-AUC:              {auc:.4f}")
    print(f"  Average Precision:    {ap:.4f}")
    print(f"  Brier Score:          {brier:.4f}")
    print(f"  Accuracy:             {(tp+tn)/(tp+tn+fp+fn):.4f}")
    print(f"  Sensitivity (Recall): {tp/(tp+fn):.4f}")
    print(f"  Specificity:          {tn/(tn+fp):.4f}")
    print(f"  Precision:            {tp/(tp+fp):.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp:4d}  FP={fp:4d}")
    print(f"    FN={fn:4d}  TN={tn:4d}")

    return {
        "model": model_name,
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "brier_score": round(brier, 4),
        "sensitivity": round(tp/(tp+fn), 4),
        "specificity": round(tn/(tn+fp), 4),
        "precision": round(tp/(tp+fp), 4),
        "y_test": y_test,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "cm": cm,
    }


def cross_validate_all(pipelines: dict, X_train, y_train) -> pd.DataFrame:
    """Stratified k-fold CV on training set for all models."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        rows.append({
            "Model": name,
            f"CV AUC (mean)": round(scores.mean(), 4),
            f"CV AUC (std)":  round(scores.std(), 4),
        })
        print(f"  {name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_TABLES, "cv_results.csv"), index=False)
    return df


def get_feature_importance(model, feature_names: list, model_name: str) -> pd.DataFrame:
    """Extract feature importances (RF/GBM) or coefficients (LR)."""
    clf = model.named_steps["classifier"]
    pre = model.named_steps["preprocessor"]

    # Get feature names after preprocessing
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_names = list(NUMERIC_FEATURES) + list(cat_names)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Feature": all_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    df.to_csv(os.path.join(OUTPUT_TABLES,
              f"feature_importance_{model_name.replace(' ','_').lower()}.csv"),
              index=False)
    return df

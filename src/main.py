"""
Credit Scoring Model — end-to-end pipeline.

1. Generate / load data
2. Exploratory data analysis
3. Train 3 models (Logistic Regression, Random Forest, Gradient Boosting)
4. Cross-validate on training set
5. Evaluate on held-out test set (ROC-AUC, PR, confusion matrix)
6. Extract feature importances
7. Save all charts and tables
"""
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_DIR, OUTPUT_TABLES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from generate_data import generate
from models import build_pipelines, split_data, evaluate, cross_validate_all, get_feature_importance
from visualizations import (
    plot_eda, plot_roc_curves, plot_precision_recall,
    plot_confusion_matrices, plot_feature_importance, plot_score_distribution
)


def main():
    print("=" * 60)
    print("  CREDIT SCORING MODEL")
    print("=" * 60)

    # ── 1. DATA ───────────────────────────────────────────────────────
    data_path = os.path.join(DATA_DIR, "credit_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded dataset: {len(df)} applicants, "
              f"{df[TARGET].mean()*100:.1f}% default rate")
    else:
        print("Generating dataset...")
        df = generate()

    print(f"\nFeatures: {len(NUMERIC_FEATURES)} numeric, "
          f"{len(CATEGORICAL_FEATURES)} categorical")

    # ── 2. EDA ────────────────────────────────────────────────────────
    print("\nGenerating EDA charts...")
    plot_eda(df)

    # ── 3. SPLIT ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train default rate: {y_train.mean()*100:.1f}%  "
          f"| Test: {y_test.mean()*100:.1f}%")

    # ── 4. CROSS-VALIDATION ───────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("  Cross-validation (5-fold stratified, ROC-AUC)")
    print(f"{'-'*60}")
    pipelines = build_pipelines()
    cross_validate_all(pipelines, X_train, y_train)

    # ── 5. TRAIN + EVALUATE ───────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("  Test Set Evaluation")
    print(f"{'-'*60}")

    results = []
    fitted_models = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        res = evaluate(pipe, X_test, y_test, name)
        results.append(res)
        fitted_models[name] = pipe

    # ── 6. SUMMARY TABLE ──────────────────────────────────────────────
    summary = pd.DataFrame([{
        "Model":           r["model"],
        "ROC-AUC":         r["roc_auc"],
        "Avg Precision":   r["avg_precision"],
        "Brier Score":     r["brier_score"],
        "Sensitivity":     r["sensitivity"],
        "Specificity":     r["specificity"],
        "Precision":       r["precision"],
    } for r in results])

    print(f"\n{'='*60}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_TABLES, "model_comparison.csv"), index=False)

    # Best model
    best = summary.loc[summary["ROC-AUC"].idxmax(), "Model"]
    best_auc = summary["ROC-AUC"].max()
    print(f"\n  Best model: {best} (AUC = {best_auc:.4f})")

    # ── 7. FEATURE IMPORTANCE ─────────────────────────────────────────
    print("\nExtracting feature importances...")
    for name in ["Random Forest", "Gradient Boosting", "Logistic Regression"]:
        imp = get_feature_importance(fitted_models[name],
                                     NUMERIC_FEATURES + CATEGORICAL_FEATURES, name)
        plot_feature_importance(imp, name)

    # ── 8. VISUALISATIONS ─────────────────────────────────────────────
    print("\nGenerating charts...")
    plot_roc_curves(results)
    plot_precision_recall(results)
    plot_confusion_matrices(results)
    plot_score_distribution(results)

    print(f"\nAll outputs saved to outputs/")
    print("  figures/ — EDA, ROC, PR, confusion matrices, feature importance, score distributions")
    print("  tables/  — model_comparison.csv, cv_results.csv, feature_importance_*.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()

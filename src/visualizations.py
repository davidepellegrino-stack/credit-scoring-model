"""
Visualizations for the credit scoring model.
  1. Default rate by feature (EDA)
  2. ROC curves — all models overlaid
  3. Precision-Recall curves
  4. Confusion matrices
  5. Feature importance (top 15)
  6. Score distribution (calibration)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_FIGS, TARGET

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

PALETTE = {"Logistic Regression": "#e74c3c",
           "Random Forest":       "#2c3e50",
           "Gradient Boosting":   "#3498db"}


def plot_eda(df: pd.DataFrame):
    """4-panel EDA: default rate by key features."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Exploratory Data Analysis — Default Rate by Feature",
                 fontweight="bold", fontsize=13, y=1.01)

    # 1. Default rate by loan-to-income bucket
    ax = axes[0, 0]
    df["lti_bucket"] = pd.cut(df["loan_to_income"], bins=5)
    dr = df.groupby("lti_bucket", observed=True)[TARGET].mean() * 100
    dr.plot(kind="bar", ax=ax, color="#e74c3c", edgecolor="white", width=0.7)
    ax.set_title("Default Rate by Loan-to-Income Ratio")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)

    # 2. Default rate by employment type
    ax = axes[0, 1]
    dr = df.groupby("employment_type")[TARGET].mean() * 100
    dr.sort_values(ascending=False).plot(
        kind="bar", ax=ax, color="#3498db", edgecolor="white", width=0.7)
    ax.set_title("Default Rate by Employment Type")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)

    # 3. Income distribution: defaulted vs not
    ax = axes[1, 0]
    for label, color, ls in [(0, "#2c3e50", "-"), (1, "#e74c3c", "--")]:
        subset = df[df[TARGET] == label]["income_eur"] / 1000
        subset.plot(kind="kde", ax=ax, color=color, linestyle=ls,
                    linewidth=2,
                    label="No Default" if label == 0 else "Default")
    ax.set_title("Income Distribution by Default Status")
    ax.set_xlabel("Annual Income (€ thousands)")
    ax.legend()

    # 4. Credit utilization vs default
    ax = axes[1, 1]
    df.groupby(pd.cut(df["credit_utilization_pct"], bins=10), observed=True)[TARGET].mean() \
      .mul(100).plot(kind="bar", ax=ax, color="#f39c12", edgecolor="white", width=0.7)
    ax.set_title("Default Rate by Credit Utilization (%)")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Credit Utilization (%)")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "eda.png"), bbox_inches="tight")
    plt.close()
    print("Saved: eda.png")


def plot_roc_curves(results: list):
    """ROC curves for all models on a single chart."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for res in results:
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        roc_auc = auc(fpr, tpr)
        color = PALETTE.get(res["model"], "gray")
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{res['model']} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "roc_curves.png"), bbox_inches="tight")
    plt.close()
    print("Saved: roc_curves.png")


def plot_precision_recall(results: list):
    """Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for res in results:
        prec, rec, _ = precision_recall_curve(res["y_test"], res["y_proba"])
        ap = res["avg_precision"]
        color = PALETTE.get(res["model"], "gray")
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{res['model']} (AP = {ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "precision_recall.png"), bbox_inches="tight")
    plt.close()
    print("Saved: precision_recall.png")


def plot_confusion_matrices(results: list):
    """Confusion matrices side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Confusion Matrices (Test Set)", fontweight="bold", fontsize=13)

    for ax, res in zip(axes, results):
        cm = res["cm"].astype(float)
        cm_pct = cm / cm.sum() * 100
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=["Pred: No Default", "Pred: Default"],
                    yticklabels=["True: No Default", "True: Default"],
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_title(res["model"], fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "confusion_matrices.png"), bbox_inches="tight")
    plt.close()
    print("Saved: confusion_matrices.png")


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str, top_n: int = 15):
    """Horizontal bar chart of top N feature importances."""
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    color = PALETTE.get(model_name, "#2c3e50")

    ax.barh(top["Feature"][::-1], top["Importance"][::-1],
            color=color, edgecolor="white", alpha=0.85)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fname = f"feature_importance_{model_name.replace(' ','_').lower()}.png"
    plt.savefig(os.path.join(OUTPUT_FIGS, fname), bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_score_distribution(results: list):
    """Predicted probability distribution by true class, per model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    fig.suptitle("Predicted Default Probability Distribution",
                 fontweight="bold", fontsize=13)

    for ax, res in zip(axes, results):
        y_test   = np.array(res["y_test"])
        y_proba  = res["y_proba"]

        ax.hist(y_proba[y_test == 0], bins=40, alpha=0.6,
                color="#2c3e50", label="No Default", density=True)
        ax.hist(y_proba[y_test == 1], bins=40, alpha=0.6,
                color="#e74c3c", label="Default", density=True)
        ax.axvline(0.5, color="black", linestyle="--", lw=1.2, label="Threshold = 0.5")
        ax.set_title(res["model"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Default Probability")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "score_distribution.png"), bbox_inches="tight")
    plt.close()
    print("Saved: score_distribution.png")

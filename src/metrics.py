"""Rank-ordering metrics used in credit scoring: Gini, KS and a decile gains/lift table.

All functions take the true 0/1 outcome and a predicted PD (higher = riskier).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def gini(y, p) -> float:
    """Gini coefficient (accuracy ratio) = 2 * AUC - 1."""
    return 2.0 * roc_auc_score(y, p) - 1.0


def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov: max gap between the cumulative score distributions of
    defaulters and non-defaulters, i.e. max(TPR - FPR) along the ROC curve."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def decile_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Gains/lift table. Loans are ranked by predicted PD and cut into equal-size
    bins; bin 1 holds the riskiest loans.

    Columns: loans, defaults, default_rate, mean_pd, lift (bin rate / overall rate),
    cum_default_share (share of all defaults captured up to and including the bin;
    the 'gains' curve) and cum_lift.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) < n_bins:
        raise ValueError("fewer observations than bins")
    if y.sum() == 0:
        raise ValueError("no defaults in y")
    # stable sort on descending PD; ties are split by position, so bins stay equal-sized
    order = np.argsort(-p, kind="mergesort")
    bins = np.array_split(order, n_bins)
    overall = y.mean()
    rows, cum_def, cum_n = [], 0.0, 0
    for i, idx in enumerate(bins, start=1):
        d = y[idx].sum()
        cum_def += d
        cum_n += len(idx)
        rate = d / len(idx)
        rows.append({
            "decile": i, "loans": len(idx), "defaults": int(d),
            "default_rate": rate, "mean_pd": p[idx].mean(),
            "lift": rate / overall,
            "cum_default_share": cum_def / y.sum(),
            "cum_lift": (cum_def / cum_n) / overall,
        })
    return pd.DataFrame(rows)

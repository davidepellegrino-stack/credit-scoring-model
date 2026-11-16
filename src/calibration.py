"""Probability calibration for PD models: ECE, reliability table and Platt / isotonic recalibration.

IFRS 9 expected credit loss multiplies PD by LGD and EAD, so the PD has to be a probability,
not just a ranking score. These helpers measure how well predicted PDs match observed default
rates and fit a monotone map from raw score to calibrated PD on a held-out calibration split.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins on predicted PD: mean PD, observed default rate and loans per bin."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) < n_bins:
        raise ValueError("fewer observations than bins")
    bins = np.array_split(np.argsort(p, kind="mergesort"), n_bins)
    return pd.DataFrame({
        "bin": range(1, n_bins + 1),
        "loans": [len(i) for i in bins],
        "mean_pd": [p[i].mean() for i in bins],
        "default_rate": [y[i].mean() for i in bins],
    })


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE: loan-weighted mean |observed rate - mean PD| over equal-count bins."""
    t = reliability_table(y, p, n_bins)
    w = t["loans"] / t["loans"].sum()
    return float((w * (t["default_rate"] - t["mean_pd"]).abs()).sum())


def calibration_summary(y, p) -> dict:
    """Brier, log loss, ECE and the calibration-in-the-large gap (mean PD - default rate)."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    return {
        "Brier": brier_score_loss(y, p),
        "Log loss": log_loss(y, np.clip(p, EPS, 1 - EPS)),
        "ECE": expected_calibration_error(y, p),
        "Mean PD - default rate": float(p.mean() - y.mean()),
    }


class PlattCalibrator:
    """Logistic regression on logit(score): p = sigmoid(a * logit(score) + b)."""

    def fit(self, p, y):
        self._lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self._lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Monotone non-parametric map from score to default rate; clipped outside the fitted range."""

    def fit(self, p, y):
        self._iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
        self._iso.fit(np.asarray(p, dtype=float), y)
        return self

    def predict(self, p):
        return self._iso.predict(np.asarray(p, dtype=float))

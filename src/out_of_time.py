"""Out-of-time validation helpers for the Home Credit PD model.

An out-of-time (OOT) split trains on earlier applications and tests on later ones,
which is how a bank validates a PD model before it goes live: the model must hold up
on loans originated after the ones it was fitted on.

Caveat for this dataset: the Hugging Face mirror's `application_date` is a synthetic
timestamp, not Home Credit's real origination date (Kaggle's original has none).
`date_id_rank_correlation` measures how that column was built, and the default rate
per quarter shows whether it carries any drift. The OOT numbers are therefore a
mechanical check of the pipeline, not evidence of stability over time.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def time_split(dates: pd.Series, cutoff) -> tuple:
    """Index of loans dated before `cutoff` (train) and on/after it (test)."""
    cutoff = pd.Timestamp(cutoff)
    if dates.isna().any():
        raise ValueError("dates contain missing values")
    train, test = dates.index[dates < cutoff], dates.index[dates >= cutoff]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"cutoff {cutoff.date()} leaves an empty train or test set")
    return train, test


def date_id_rank_correlation(dates: pd.Series, ids: pd.Series) -> float:
    """Spearman correlation between application date and loan id.

    Close to 1 means the dates were assigned in id order, i.e. they say nothing
    about when the loan was really originated.
    """
    return float(pd.Series(dates.to_numpy().astype("int64")).corr(
        pd.Series(ids.to_numpy()), method="spearman"))


def quarterly_table(dates: pd.Series, y: pd.Series, p: np.ndarray) -> pd.DataFrame:
    """Per calendar quarter: loans, default rate, mean PD and AUC (NaN if one class only)."""
    d = pd.DataFrame({"q": dates.dt.to_period("Q").astype(str).to_numpy(),
                      "y": y.to_numpy(), "p": np.asarray(p)})
    rows = []
    for q, g in d.groupby("q"):
        auc = roc_auc_score(g["y"], g["p"]) if g["y"].nunique() == 2 else np.nan
        rows.append({"quarter": q, "n": len(g), "default_rate": g["y"].mean(),
                     "mean_pd": g["p"].mean(), "auc": auc})
    return pd.DataFrame(rows)

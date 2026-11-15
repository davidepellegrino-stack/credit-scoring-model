"""Credit-history features: aggregates of the bureau and previous_application tables.

Both tables hold many rows per applicant (SK_ID_CURR). They are collapsed to one
row per applicant and left-joined onto the application features.

Point-in-time safety: in these tables all DAYS_* dates are relative to the
current application date and negative for past events. Rows dated after the
application (positive DAYS_CREDIT / DAYS_DECISION) are dropped, so nothing from
after the loan decision can leak into the features.

Applicants with no history get count features = 0 and an explicit has_* flag;
ratios stay NaN (unknown, not zero) and are imputed inside the model pipelines.
"""
import numpy as np
import pandas as pd

ID = "SK_ID_CURR"
DAYS_PER_YEAR = 365.25

BUREAU_COLS = [
    ID, "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE",
    "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
]
PREV_COLS = [
    ID, "NAME_CONTRACT_STATUS", "DAYS_DECISION", "AMT_APPLICATION", "AMT_CREDIT",
    "AMT_ANNUITY", "CNT_PAYMENT",
]

BUREAU_FEATURES = [
    "has_bureau", "bureau_n_loans", "bureau_n_active", "bureau_n_closed", "bureau_n_bad_debt",
    "bureau_share_active", "bureau_n_overdue_now", "bureau_max_overdue_days",
    "bureau_max_overdue_amt", "bureau_overdue_amt_sum", "bureau_credit_sum_active",
    "bureau_debt_sum_active", "bureau_debt_to_credit_active", "bureau_prolongations",
    "bureau_oldest_credit_years", "bureau_last_credit_years", "bureau_n_last_year",
]
PREV_FEATURES = [
    "has_prev", "prev_n", "prev_n_approved", "prev_n_refused", "prev_n_canceled",
    "prev_refused_share", "prev_approved_credit_sum", "prev_approved_annuity_mean",
    "prev_credit_to_application", "prev_mean_cnt_payment", "prev_last_decision_years",
    "prev_n_last_year",
]
HISTORY_FEATURES = BUREAU_FEATURES + PREV_FEATURES


def _past_only(df: pd.DataFrame, days_col: str) -> pd.DataFrame:
    """Keep rows dated on or before the application date (days <= 0)."""
    return df[df[days_col] <= 0]


def bureau_aggregates(bureau: pd.DataFrame) -> pd.DataFrame:
    """One row per applicant summarising credits reported to the credit bureau."""
    b = _past_only(bureau, "DAYS_CREDIT").copy()
    active = b["CREDIT_ACTIVE"] == "Active"
    b["is_active"] = active.astype(float)
    b["is_closed"] = (b["CREDIT_ACTIVE"] == "Closed").astype(float)
    b["is_bad_debt"] = (b["CREDIT_ACTIVE"] == "Bad debt").astype(float)
    b["overdue_now"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(float)
    b["credit_active"] = b["AMT_CREDIT_SUM"].where(active)
    b["debt_active"] = b["AMT_CREDIT_SUM_DEBT"].clip(lower=0).where(active)  # negative = overpaid
    b["recent"] = (b["DAYS_CREDIT"] > -DAYS_PER_YEAR).astype(float)

    g = b.groupby(ID)
    out = pd.DataFrame({
        "bureau_n_loans": g.size().astype(float),
        "bureau_n_active": g["is_active"].sum(),
        "bureau_n_closed": g["is_closed"].sum(),
        "bureau_n_bad_debt": g["is_bad_debt"].sum(),
        "bureau_n_overdue_now": g["overdue_now"].sum(),
        "bureau_max_overdue_days": g["CREDIT_DAY_OVERDUE"].max(),
        "bureau_max_overdue_amt": g["AMT_CREDIT_MAX_OVERDUE"].max(),
        "bureau_overdue_amt_sum": g["AMT_CREDIT_SUM_OVERDUE"].sum(),
        "bureau_credit_sum_active": g["credit_active"].sum(min_count=1),
        "bureau_debt_sum_active": g["debt_active"].sum(min_count=1),
        "bureau_prolongations": g["CNT_CREDIT_PROLONG"].sum(),
        "bureau_oldest_credit_years": -g["DAYS_CREDIT"].min() / DAYS_PER_YEAR,
        "bureau_last_credit_years": -g["DAYS_CREDIT"].max() / DAYS_PER_YEAR,
        "bureau_n_last_year": g["recent"].sum(),
    })
    out["bureau_share_active"] = out["bureau_n_active"] / out["bureau_n_loans"]
    out["bureau_debt_to_credit_active"] = (
        out["bureau_debt_sum_active"] / out["bureau_credit_sum_active"].where(
            out["bureau_credit_sum_active"] > 0)).clip(upper=2.0)  # debt can exceed the limit; cap outliers
    out["has_bureau"] = 1.0
    return out[BUREAU_FEATURES]


def previous_aggregates(prev: pd.DataFrame) -> pd.DataFrame:
    """One row per applicant summarising earlier Home Credit applications."""
    p = _past_only(prev, "DAYS_DECISION").copy()
    status = p["NAME_CONTRACT_STATUS"]
    approved = status == "Approved"
    p["is_approved"] = approved.astype(float)
    p["is_refused"] = (status == "Refused").astype(float)
    p["is_canceled"] = (status == "Canceled").astype(float)
    p["credit_appr"] = p["AMT_CREDIT"].where(approved)
    p["annuity_appr"] = p["AMT_ANNUITY"].where(approved)
    p["ratio_appr"] = (p["AMT_CREDIT"] / p["AMT_APPLICATION"].where(p["AMT_APPLICATION"] > 0)).where(approved)
    p["recent"] = (p["DAYS_DECISION"] > -DAYS_PER_YEAR).astype(float)

    g = p.groupby(ID)
    out = pd.DataFrame({
        "prev_n": g.size().astype(float),
        "prev_n_approved": g["is_approved"].sum(),
        "prev_n_refused": g["is_refused"].sum(),
        "prev_n_canceled": g["is_canceled"].sum(),
        "prev_approved_credit_sum": g["credit_appr"].sum(min_count=1),
        "prev_approved_annuity_mean": g["annuity_appr"].mean(),
        "prev_credit_to_application": g["ratio_appr"].mean(),
        "prev_mean_cnt_payment": g["CNT_PAYMENT"].mean(),
        "prev_last_decision_years": -g["DAYS_DECISION"].max() / DAYS_PER_YEAR,
        "prev_n_last_year": g["recent"].sum(),
    })
    out["prev_refused_share"] = out["prev_n_refused"] / out["prev_n"]
    out["has_prev"] = 1.0
    return out[PREV_FEATURES]


# Heavy-tailed money amounts are log1p-transformed (monotone, so trees are unaffected;
# it keeps the logistic regression from being dominated by a few huge values).
AMOUNT_FEATURES = [
    "bureau_max_overdue_amt", "bureau_overdue_amt_sum", "bureau_credit_sum_active",
    "bureau_debt_sum_active", "prev_approved_credit_sum", "prev_approved_annuity_mean",
]
COUNT_FEATURES = [
    "has_bureau", "bureau_n_loans", "bureau_n_active", "bureau_n_closed", "bureau_n_bad_debt",
    "bureau_n_overdue_now", "bureau_n_last_year", "bureau_prolongations",
    "has_prev", "prev_n", "prev_n_approved", "prev_n_refused", "prev_n_canceled",
    "prev_n_last_year",
]


def add_history_features(X: pd.DataFrame, ids: pd.Series, bureau_agg: pd.DataFrame,
                         prev_agg: pd.DataFrame) -> pd.DataFrame:
    """Left-join the aggregates onto the application features (aligned on `ids`).

    Applicants without history: counts and has_* flags become 0; ratios, amounts and
    ages stay NaN so the pipeline's imputer treats them as unknown.
    """
    hist = pd.DataFrame(index=X.index)
    for agg in (bureau_agg, prev_agg):
        hist = hist.join(agg.reindex(ids.to_numpy()).set_axis(X.index))
    hist[COUNT_FEATURES] = hist[COUNT_FEATURES].fillna(0.0)
    hist[AMOUNT_FEATURES] = np.log1p(hist[AMOUNT_FEATURES])
    return pd.concat([X, hist[HISTORY_FEATURES]], axis=1)

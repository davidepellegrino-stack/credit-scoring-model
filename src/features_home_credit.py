"""Application-table features for the Home Credit PD model.

Every feature is computed from the application form alone, so there is no
leakage from the outcome window. Missing values are kept as NaN here; the
imputation strategy lives in the model pipelines (`run_home_credit.py`), so it
is fitted on the training fold only.

Notes on Home Credit conventions:
  * DAYS_* columns are negative days relative to the application date.
  * DAYS_EMPLOYED == 365243 is a sentinel for "not employed" (pensioners,
    unemployed): 18% of loans. It is turned into NaN plus an explicit flag.
"""
import numpy as np
import pandas as pd

DAYS_EMPLOYED_SENTINEL = 365243

RAW_NUMERIC = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_INCOME_TOTAL",
    "AMT_CREDIT", "AMT_ANNUITY", "CNT_CHILDREN", "REGION_RATING_CLIENT",
]
ENGINEERED_NUMERIC = [
    "age_years", "employed_years", "unemployed_flag", "credit_income_ratio",
    "annuity_income_ratio", "goods_credit_ratio", "credit_term_months",
    "income_per_family_member", "ext_source_mean", "ext_source_min",
    "ext_source_prod", "ext_source_n_missing", "registration_years",
    "id_publish_years", "phone_change_years",
]
CATEGORICAL = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
]
NUMERIC = RAW_NUMERIC + ENGINEERED_NUMERIC


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den with 0 or missing denominators mapped to NaN, never inf."""
    return num / den.where(den > 0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix (NUMERIC + CATEGORICAL columns) for an application table."""
    out = pd.DataFrame(index=df.index)
    for c in RAW_NUMERIC:
        out[c] = df[c].astype(float)

    out["age_years"] = -df["DAYS_BIRTH"] / 365.25
    employed = df["DAYS_EMPLOYED"].astype(float)
    sentinel = employed == DAYS_EMPLOYED_SENTINEL
    out["unemployed_flag"] = sentinel.astype(float)
    out["employed_years"] = (-employed).where(~sentinel) / 365.25

    out["credit_income_ratio"] = _ratio(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    out["annuity_income_ratio"] = _ratio(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    out["goods_credit_ratio"] = _ratio(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    out["credit_term_months"] = _ratio(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    out["income_per_family_member"] = _ratio(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])

    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].astype(float)
    out["ext_source_mean"] = ext.mean(axis=1)  # NaN only if all three are missing
    out["ext_source_min"] = ext.min(axis=1)
    out["ext_source_prod"] = ext.prod(axis=1, min_count=3)
    out["ext_source_n_missing"] = ext.isna().sum(axis=1).astype(float)

    out["registration_years"] = -df["DAYS_REGISTRATION"] / 365.25
    out["id_publish_years"] = -df["DAYS_ID_PUBLISH"] / 365.25
    out["phone_change_years"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25

    for c in CATEGORICAL:
        out[c] = df[c].astype("object").where(df[c].notna(), "missing")
    return out[NUMERIC + CATEGORICAL]

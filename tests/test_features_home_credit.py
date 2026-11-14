"""Tests for the Home Credit feature builder: tiny in-memory fixture, no network."""
import numpy as np
import pandas as pd
import pytest

import features_home_credit as fh
from run_home_credit import build_home_credit_pipelines


@pytest.fixture
def raw():
    n = 6
    return pd.DataFrame({
        "AMT_INCOME_TOTAL": [100_000, 200_000, 0, 150_000, 80_000, 120_000],
        "AMT_CREDIT": [400_000, 600_000, 300_000, 150_000, 240_000, 360_000],
        "AMT_ANNUITY": [20_000, 30_000, 15_000, 0, 12_000, np.nan],
        "AMT_GOODS_PRICE": [360_000, 600_000, 300_000, 150_000, np.nan, 300_000],
        "CNT_CHILDREN": [0, 1, 2, 0, 1, 0],
        "CNT_FAM_MEMBERS": [2, 3, 4, 1, 2, 2],
        "REGION_RATING_CLIENT": [2, 2, 3, 1, 2, 2],
        "DAYS_BIRTH": [-365.25 * 40] * n,
        "DAYS_EMPLOYED": [-365.25 * 5, -365.25 * 2, 365243, -100, -200, -300],
        "DAYS_REGISTRATION": [-3_652.5] * n,
        "DAYS_ID_PUBLISH": [-1_826.25] * n,
        "DAYS_LAST_PHONE_CHANGE": [-730.5] * n,
        "EXT_SOURCE_1": [0.5, np.nan, 0.2, 0.9, np.nan, 0.4],
        "EXT_SOURCE_2": [0.6, 0.5, 0.3, 0.8, np.nan, 0.4],
        "EXT_SOURCE_3": [0.4, 0.5, np.nan, 0.7, np.nan, 0.4],
        "NAME_CONTRACT_TYPE": "Cash loans", "CODE_GENDER": "F",
        "NAME_EDUCATION_TYPE": "Higher education", "NAME_INCOME_TYPE": "Working",
        "NAME_FAMILY_STATUS": "Married", "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "Y",
    })


def test_columns_and_row_alignment(raw):
    out = fh.build_features(raw)
    assert list(out.columns) == fh.NUMERIC + fh.CATEGORICAL
    assert len(out) == len(raw)


def test_unit_conversions(raw):
    out = fh.build_features(raw)
    assert out["age_years"].iloc[0] == pytest.approx(40.0)
    assert out["employed_years"].iloc[0] == pytest.approx(5.0)
    assert out["registration_years"].iloc[0] == pytest.approx(10.0)
    assert out["credit_income_ratio"].iloc[0] == pytest.approx(4.0)
    assert out["credit_term_months"].iloc[0] == pytest.approx(20.0)


def test_employment_sentinel_becomes_nan_with_flag(raw):
    out = fh.build_features(raw)
    assert out["unemployed_flag"].tolist() == [0, 0, 1, 0, 0, 0]
    assert np.isnan(out["employed_years"].iloc[2])
    assert (out["employed_years"].dropna() > 0).all()


def test_zero_denominators_give_nan_not_inf(raw):
    out = fh.build_features(raw)
    assert np.isnan(out["annuity_income_ratio"].iloc[2])  # zero income
    assert np.isnan(out["credit_term_months"].iloc[3])    # zero annuity
    assert not np.isinf(out[fh.NUMERIC].to_numpy(dtype=float)).any()


def test_ext_source_aggregates(raw):
    out = fh.build_features(raw)
    assert out["ext_source_mean"].iloc[0] == pytest.approx(0.5)
    assert out["ext_source_min"].iloc[0] == pytest.approx(0.4)
    assert out["ext_source_n_missing"].tolist() == [0, 1, 1, 0, 3, 0]
    assert np.isnan(out["ext_source_mean"].iloc[4])  # all three missing
    assert np.isnan(out["ext_source_prod"].iloc[1])  # product needs all three


def test_missing_categorical_labelled(raw):
    raw.loc[0, "NAME_EDUCATION_TYPE"] = None
    assert fh.build_features(raw)["NAME_EDUCATION_TYPE"].iloc[0] == "missing"


@pytest.mark.parametrize("name", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
def test_pipelines_fit_with_missing_values(raw, name):
    X = fh.build_features(raw)
    y = pd.Series([0, 1, 0, 1, 0, 1])
    pipe = build_home_credit_pipelines()[name]
    pipe.set_params(**({"clf__min_samples_leaf": 1} if name != "Logistic Regression" else {}))
    if name == "Gradient Boosting":
        pipe.set_params(clf__early_stopping=False)
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (6,) and ((p >= 0) & (p <= 1)).all()

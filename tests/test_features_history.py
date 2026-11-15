"""Tests for bureau / previous_application aggregates and the sharded table loader."""
import numpy as np
import pandas as pd
import pytest

import data_home_credit as hc
import features_history as fh


@pytest.fixture
def bureau():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1, 2, 3],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Bad debt", "Active"],
        "DAYS_CREDIT": [-730, -3_652, -100, -400, 50],  # id 3's only row is dated after the application
        "CREDIT_DAY_OVERDUE": [0, 0, 30, 0, 99],
        "AMT_CREDIT_MAX_OVERDUE": [np.nan, 500.0, 1_000.0, 0.0, 0.0],
        "CNT_CREDIT_PROLONG": [0, 0, 1, 0, 0],
        "AMT_CREDIT_SUM": [10_000.0, 5_000.0, 20_000.0, 8_000.0, 1_000.0],
        "AMT_CREDIT_SUM_DEBT": [4_000.0, 0.0, 30_000.0, 8_000.0, 0.0],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 250.0, 0.0, 0.0],
    })


@pytest.fixture
def prev():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1, 2],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved", "Canceled"],
        "DAYS_DECISION": [-365, -200, -1_000, -30],
        "AMT_APPLICATION": [100.0, 50.0, 200.0, 10.0],
        "AMT_CREDIT": [110.0, 0.0, 180.0, 0.0],
        "AMT_ANNUITY": [10.0, np.nan, 20.0, np.nan],
        "CNT_PAYMENT": [12.0, np.nan, 24.0, np.nan],
    })


def test_bureau_counts_and_ages(bureau):
    a = fh.bureau_aggregates(bureau)
    r = a.loc[1]
    assert (r["bureau_n_loans"], r["bureau_n_active"], r["bureau_n_closed"]) == (3, 2, 1)
    assert r["bureau_n_overdue_now"] == 1 and r["bureau_max_overdue_days"] == 30
    assert r["bureau_prolongations"] == 1 and r["bureau_overdue_amt_sum"] == 250
    assert r["bureau_oldest_credit_years"] == pytest.approx(3_652 / 365.25)
    assert r["bureau_last_credit_years"] == pytest.approx(100 / 365.25)
    assert r["bureau_n_last_year"] == 1  # only the -100 days credit is within a year
    assert a.loc[2, "bureau_n_bad_debt"] == 1


def test_bureau_active_exposure_ratio_is_capped(bureau):
    a = fh.bureau_aggregates(bureau)
    assert a.loc[1, "bureau_credit_sum_active"] == 30_000
    assert a.loc[1, "bureau_debt_sum_active"] == 34_000
    assert a.loc[1, "bureau_debt_to_credit_active"] == pytest.approx(34_000 / 30_000)
    capped = bureau.copy()
    capped.loc[2, "AMT_CREDIT_SUM_DEBT"] = 1e9
    assert fh.bureau_aggregates(capped).loc[1, "bureau_debt_to_credit_active"] == 2.0


def test_rows_after_application_are_dropped(bureau, prev):
    assert 3 not in fh.bureau_aggregates(bureau).index  # its only record is in the future
    future = pd.concat([prev, prev.iloc[[0]].assign(DAYS_DECISION=40, NAME_CONTRACT_STATUS="Refused")])
    assert fh.previous_aggregates(future).loc[1, "prev_n"] == 3


def test_previous_aggregates(prev):
    r = fh.previous_aggregates(prev).loc[1]
    assert (r["prev_n"], r["prev_n_approved"], r["prev_n_refused"]) == (3, 2, 1)
    assert r["prev_refused_share"] == pytest.approx(1 / 3)
    assert r["prev_approved_credit_sum"] == 290 and r["prev_approved_annuity_mean"] == 15
    assert r["prev_credit_to_application"] == pytest.approx((1.1 + 0.9) / 2)
    assert r["prev_last_decision_years"] == pytest.approx(200 / 365.25)
    assert r["prev_n_last_year"] == 2  # decisions at -200 and -365 (365 < 365.25)


def test_join_fills_counts_but_keeps_unknowns_nan(bureau, prev):
    ids = pd.Series([1, 2, 99], index=[10, 11, 12])
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]}, index=ids.index)
    out = fh.add_history_features(X, ids, fh.bureau_aggregates(bureau), fh.previous_aggregates(prev))
    assert list(out.index) == [10, 11, 12] and list(out.columns) == ["x"] + fh.HISTORY_FEATURES
    novel = out.loc[12]  # applicant with no history anywhere
    assert novel["has_bureau"] == 0 and novel["bureau_n_loans"] == 0 and novel["prev_n"] == 0
    assert np.isnan(novel["bureau_debt_to_credit_active"]) and np.isnan(novel["prev_refused_share"])
    assert out.loc[10, "has_bureau"] == 1 and out.loc[10, "has_prev"] == 1
    assert out.loc[10, "bureau_credit_sum_active"] == pytest.approx(np.log1p(30_000))


def test_sharded_loader_concatenates_and_rejects_single_path(tmp_path):
    for i, ids in enumerate([[1, 2], [3]]):
        pd.DataFrame({"SK_ID_CURR": ids}).to_parquet(tmp_path / f"previous_application_{i}.parquet")
    df = hc.load_table("previous_application", cache_dir=str(tmp_path))
    assert df["SK_ID_CURR"].tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        hc.fetch_table("previous_application", cache_dir=str(tmp_path))

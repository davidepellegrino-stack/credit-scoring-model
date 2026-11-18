import numpy as np
import pandas as pd
import pytest

from out_of_time import date_id_rank_correlation, quarterly_table, time_split


def _dates():
    return pd.Series(pd.to_datetime(["2018-01-05", "2018-06-01", "2019-12-31", "2020-01-01", "2020-05-05"]),
                     index=[10, 11, 12, 13, 14])


def test_time_split_respects_cutoff():
    tr, te = time_split(_dates(), "2020-01-01")
    assert list(tr) == [10, 11, 12] and list(te) == [13, 14]


def test_time_split_rejects_empty_side():
    with pytest.raises(ValueError):
        time_split(_dates(), "2030-01-01")
    with pytest.raises(ValueError):
        time_split(_dates(), "2000-01-01")


def test_rank_correlation_detects_id_ordered_dates():
    ids = pd.Series([5, 9, 20, 31, 40])
    assert date_id_rank_correlation(_dates(), ids) == pytest.approx(1.0)
    assert date_id_rank_correlation(_dates(), ids[::-1].reset_index(drop=True)) == pytest.approx(-1.0)


def test_quarterly_table_values_and_single_class_auc():
    dates = pd.Series(pd.to_datetime(["2020-01-10", "2020-02-10", "2020-03-10", "2020-04-10", "2020-05-10"]))
    y = pd.Series([0, 1, 0, 0, 0])
    p = np.array([0.1, 0.9, 0.2, 0.3, 0.4])
    t = quarterly_table(dates, y, p).set_index("quarter")
    assert t.loc["2020Q1", "n"] == 3 and t.loc["2020Q1", "auc"] == 1.0
    assert t.loc["2020Q1", "default_rate"] == pytest.approx(1 / 3)
    assert t.loc["2020Q2", "mean_pd"] == pytest.approx(0.35)
    assert np.isnan(t.loc["2020Q2", "auc"])

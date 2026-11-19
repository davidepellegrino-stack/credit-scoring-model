import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "ifrs9"))

from fred import (SERIES, add_transforms, build_quarterly_panel, coverage,  # noqa: E402
                  fetch_series, parse_fred_csv, to_quarterly)


def test_parse_handles_missing_marker_and_sorts():
    text = "observation_date,DGS10\n2020-01-03,1.8\n2020-01-02,.\n2020-01-01,1.9\n"
    s = parse_fred_csv(text, "DGS10")
    assert list(s.values) == [1.9, 1.8]  # '.' dropped, sorted by date
    assert s.index.is_monotonic_increasing


def test_parse_rejects_wrong_series_or_shape():
    with pytest.raises(ValueError):
        parse_fred_csv("observation_date,OTHER\n2020-01-01,1\n", "UNRATE")
    with pytest.raises(ValueError):
        parse_fred_csv("a,b,c\n1,2,3\n", "UNRATE")


def test_monthly_quarter_is_mean_and_partial_quarter_dropped():
    idx = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"])
    s = pd.Series([3.0, 4.0, 5.0, 6.0, 7.0], index=idx)
    q = to_quarterly(s, "M")
    assert q[pd.Period("2020Q1")] == pytest.approx(4.0)
    assert np.isnan(q[pd.Period("2020Q2")])  # only 2 of 3 months


def test_daily_needs_enough_observations():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")  # about 65 business days
    full = to_quarterly(pd.Series(2.0, index=idx), "D")
    assert full[pd.Period("2020Q1")] == pytest.approx(2.0)
    short = to_quarterly(pd.Series(2.0, index=idx[:20]), "D")
    assert np.isnan(short[pd.Period("2020Q1")])


def test_quarterly_series_kept_as_is():
    s = pd.Series([1.5, 2.5], index=pd.to_datetime(["2020-01-01", "2020-04-01"]))
    q = to_quarterly(s, "Q")
    assert list(q.values) == [1.5, 2.5]
    assert list(q.index) == [pd.Period("2020Q1"), pd.Period("2020Q2")]


def test_panel_alignment_and_transforms():
    qidx = pd.date_range("2018-01-01", periods=8, freq="QS")
    gdp = pd.Series(100 * 1.01 ** np.arange(8), index=qidx)
    months = pd.date_range("2018-01-01", periods=24, freq="MS")
    unrate = pd.Series(4.0 + 0.1 * np.arange(24), index=months)
    panel = add_transforms(build_quarterly_panel({"GDPC1": gdp, "UNRATE": unrate}))
    assert list(panel.index[:2]) == [pd.Period("2018Q1"), pd.Period("2018Q2")]
    assert panel["gdp_growth_yoy"].iloc[:4].isna().all()
    assert panel["gdp_growth_yoy"].iloc[4] == pytest.approx(100 * (1.01 ** 4 - 1))
    assert panel["gdp_growth_qoq_ann"].iloc[1] == pytest.approx(100 * (1.01 ** 4 - 1))
    # unemployment drifts +0.1 per month, so the 4-quarter change is 1.2 pp
    assert panel["d_unrate_4q"].iloc[4] == pytest.approx(1.2)


def test_term_spread_and_coverage():
    panel = pd.DataFrame({"dgs10": [3.0, 2.0], "fedfunds": [1.0, 2.5]},
                         index=pd.PeriodIndex(["2020Q1", "2020Q2"], freq="Q"))
    out = add_transforms(panel)
    assert list(out["term_spread"]) == [2.0, -0.5]
    cov = coverage(out)
    assert cov.loc["dgs10", "n_quarters"] == 2 and cov.loc["dgs10", "first"] == "2020Q1"


def test_fetch_uses_cache_without_network(tmp_path):
    (tmp_path / "UNRATE.csv").write_text("observation_date,UNRATE\n2020-01-01,3.5\n")
    s = fetch_series("UNRATE", cache_dir=str(tmp_path))  # would raise if it tried the network
    assert s.iloc[0] == 3.5


def test_series_registry_is_consistent():
    names = [v[0] for v in SERIES.values()]
    assert len(names) == len(set(names))
    assert all(v[1] in ("M", "Q", "D") for v in SERIES.values())


def test_all_nan_quarters_are_dropped():
    months = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"])
    panel = build_quarterly_panel({"UNRATE": pd.Series([1.0, 2.0, 3.0, 4.0], index=months)})
    assert list(panel.index) == [pd.Period("2020Q1")]  # 2020Q2 has 1 of 3 months

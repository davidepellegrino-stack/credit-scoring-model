import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "ifrs9"))

from scenarios import HORIZON, SCENARIOS, build_scenarios, interpolate_knots, observed_unrate  # noqa: E402


def make_panel(n=12, last="2026Q2", u=4.0):
    idx = pd.period_range(end=last, periods=n, freq="Q")
    unrate = np.full(n, u)
    unrate[-4:] = [4.0, 4.2, 4.4, 4.6]
    return pd.DataFrame({"unrate": unrate, "co_card": np.full(n, 3.0)}, index=idx)


def test_interpolate_knots_linear_and_flat():
    p = interpolate_knots({0: 0.0, 4: 4.0}, horizon=6, start=1)
    assert p == pytest.approx([1, 2, 3, 4, 4, 4])


def test_missing_quarter_interpolated_not_dropped():
    panel = make_panel()
    panel.iloc[-3, 0] = np.nan
    u = observed_unrate(panel, panel.index[-1])
    assert not u.isna().any()
    assert u.iloc[-3] == pytest.approx((u.iloc[-4] + u.iloc[-2]) / 2)


def test_scenarios_start_after_last_observed_quarter_and_have_regressors():
    panel = make_panel()
    scen = build_scenarios(panel)
    assert set(scen) == set(SCENARIOS)
    for df in scen.values():
        assert len(df) == HORIZON
        assert df.index[0] == pd.Period("2026Q3")
        assert list(df.columns) == ["unrate", "gdp_growth_yoy", "d_unrate_4q"]
        assert not df.isna().any().any()


def test_d_unrate_4q_mixes_history_and_scenario():
    panel = make_panel()                      # u: ..., 4.0, 4.2, 4.4, 4.6 (last 4 quarters)
    base = build_scenarios(panel)["baseline"]  # flat at 4.6 afterwards
    # 2026Q3: 4.6 - u(2025Q3) ; 2025Q3 is the 4th-last observed quarter -> 4.0 .. check against the index
    u_hist = panel["unrate"]
    assert base["d_unrate_4q"].iloc[0] == pytest.approx(4.6 - u_hist.loc["2025Q3"])
    assert base["d_unrate_4q"].iloc[-1] == pytest.approx(0.0)


def test_severity_ordering():
    scen = build_scenarios(make_panel())
    assert scen["baseline"]["unrate"].max() < scen["adverse"]["unrate"].max() < scen["severely_adverse"]["unrate"].max()
    assert scen["baseline"]["gdp_growth_yoy"].min() > scen["adverse"]["gdp_growth_yoy"].min() \
        > scen["severely_adverse"]["gdp_growth_yoy"].min()


def test_scenario_peaks_match_knots():
    panel = make_panel()
    scen = build_scenarios(panel)
    u0 = panel["unrate"].iloc[-1]
    assert scen["adverse"]["unrate"].max() == pytest.approx(u0 + 2.5)
    assert scen["severely_adverse"]["unrate"].max() == pytest.approx(u0 + 5.5)

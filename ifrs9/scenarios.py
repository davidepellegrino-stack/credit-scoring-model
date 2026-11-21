"""Macro scenarios for the IFRS 9 satellite model: baseline, adverse and severely adverse paths.

The paths are defined by this project, not copied from a supervisor. They are shaped like the
supervisory stress tests (a sharp GDP contraction, unemployment rising by several points over
about two years, then a partial recovery) but the numbers below are our own transparent
assumptions. Replace the knots with the Federal Reserve's published scenario tables if you
need a regulatory run.

Each scenario is a set of piecewise-linear knots over a 12-quarter horizon (3 years), starting
the quarter after the last observed charge-off rate:

* unemployment is specified as a change from its last observed level (percentage points);
* real GDP growth is specified directly as the year-on-year rate (%).

`d_unrate_4q`, the regressor the satellite model uses, is then computed from the unemployment
level path joined to the observed history, so the first four quarters mix actual and
scenario values exactly as the model's definition requires.

Usage:
    python ifrs9/run_scenarios.py
"""
import numpy as np
import pandas as pd

HORIZON = 12

# knots: {quarter ahead h: value}; h = 0 is the last observed quarter
SCENARIOS = {
    "baseline": {
        "unrate_change": {0: 0.0, 12: 0.0},
        "gdp_growth_yoy": {1: 2.0, 12: 2.0},
    },
    "adverse": {
        "unrate_change": {0: 0.0, 6: 2.5, 12: 1.5},
        "gdp_growth_yoy": {1: 1.5, 3: -0.5, 5: -1.5, 8: 0.0, 12: 1.5},
    },
    "severely_adverse": {
        "unrate_change": {0: 0.0, 7: 5.5, 12: 3.0},
        "gdp_growth_yoy": {1: 0.5, 3: -3.5, 5: -6.0, 8: -2.0, 12: 1.5},
    },
}


def interpolate_knots(knots, horizon=HORIZON, start=1):
    """Piecewise-linear path for h = start..horizon from a {h: value} dict (flat outside the knots)."""
    h = np.arange(start, horizon + 1)
    ks = sorted(knots)
    return np.interp(h, ks, [knots[k] for k in ks])


def observed_unrate(panel, last_quarter):
    """Observed unemployment up to `last_quarter`; a missing quarter is linearly interpolated.

    Only used to build the four-quarter change at the start of a scenario (2025Q4 is missing in
    the FRED series because the October 2025 survey was never published).
    """
    u = panel["unrate"].loc[:last_quarter]
    return u.interpolate(limit_area="inside")


def build_scenarios(panel, last_quarter=None, scenarios=SCENARIOS, horizon=HORIZON):
    """Return {name: DataFrame} with unrate, gdp_growth_yoy and d_unrate_4q for the next `horizon` quarters.

    `last_quarter` defaults to the last quarter with an observed charge-off rate.
    """
    if last_quarter is None:
        last_quarter = panel["co_card"].dropna().index.max()
    hist = observed_unrate(panel, last_quarter)
    future_index = pd.period_range(last_quarter + 1, periods=horizon, freq="Q")
    u0 = float(hist.iloc[-1])
    out = {}
    for name, spec in scenarios.items():
        u_future = u0 + interpolate_knots(spec["unrate_change"], horizon)
        u_all = pd.concat([hist, pd.Series(u_future, index=future_index)])
        d4 = (u_all - u_all.shift(4)).loc[future_index]
        gdp = interpolate_knots(spec["gdp_growth_yoy"], horizon)
        out[name] = pd.DataFrame({"unrate": u_future, "gdp_growth_yoy": gdp, "d_unrate_4q": d4.values},
                                 index=future_index)
    return out

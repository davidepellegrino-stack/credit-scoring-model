import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "ifrs9"))

from ecl import weighted_ecl  # noqa: E402
from pd_shift import pd_term_structure  # noqa: E402
from scenarios import SCENARIOS, build_scenarios  # noqa: E402
from sensitivity import ecl_row, shift_unemployment  # noqa: E402

W = {"baseline": 0.5, "adverse": 0.3, "severely_adverse": 0.2}


@pytest.fixture
def panel():
    idx = pd.period_range("2020Q1", periods=20, freq="Q")
    return pd.DataFrame({"unrate": np.full(20, 4.0), "co_card": np.full(20, 3.0)}, index=idx)


@pytest.fixture
def ts():
    pds = np.random.default_rng(2).beta(1.2, 12, 2_000)
    return pd_term_structure(pds, {"baseline": np.ones(12), "adverse": np.linspace(1, 1.8, 12),
                                   "severely_adverse": np.linspace(1, 3.0, 12)})


def test_zero_shift_changes_nothing(panel):
    a = build_scenarios(panel, scenarios=SCENARIOS)
    b = build_scenarios(panel, scenarios=shift_unemployment(SCENARIOS, 0.0))
    for n in a:
        pd.testing.assert_frame_equal(a[n], b[n])


def test_shift_ramps_in_then_holds(panel):
    base = build_scenarios(panel, scenarios=SCENARIOS)
    moved = build_scenarios(panel, scenarios=shift_unemployment(SCENARIOS, 1.0))
    for n in base:
        diff = (moved[n]["unrate"] - base[n]["unrate"]).to_numpy()
        assert diff == pytest.approx(np.minimum(np.arange(1, 13) / 4, 1.0))
        assert (moved[n]["gdp_growth_yoy"] == base[n]["gdp_growth_yoy"]).all()


def test_shift_keeps_the_shape_of_the_original_path(panel):
    base = build_scenarios(panel, scenarios=SCENARIOS)["adverse"]["unrate"].to_numpy()
    moved = build_scenarios(panel, scenarios=shift_unemployment(SCENARIOS, 2.0))["adverse"]["unrate"].to_numpy()
    assert moved.argmax() == base.argmax()
    assert moved.max() - base.max() == pytest.approx(2.0)


def test_origination_pd_override_changes_staging(ts):
    ead = np.full(2_000, 1000.0)
    default = weighted_ecl(ts, W, 0.6, ead, 0.1, ratio=1.25)
    low_orig = weighted_ecl(ts, W, 0.6, ead, 0.1, ratio=1.25, pd_life_orig=ts["baseline"]["pd_life"] * 0.5)
    assert (low_orig["stage"] == 2).mean() > (default["stage"] == 2).mean()
    same = weighted_ecl(ts, W, 0.6, ead, 0.1, ratio=1.25, pd_life_orig=ts["baseline"]["pd_life"])
    assert (same["stage"] == default["stage"]).all()


def test_ecl_row_scales_linearly_with_lgd(ts):
    ead = np.full(2_000, 1000.0)
    orig = ts["baseline"]["pd_life"]
    a = ecl_row(ts, W, 0.4, ead, 0.1, 2.0, orig, "a")
    b = ecl_row(ts, W, 0.8, ead, 0.1, 2.0, orig, "b")
    assert b["ecl_m"] == pytest.approx(2 * a["ecl_m"])
    assert b["ecl_pct_ead"] == pytest.approx(2 * a["ecl_pct_ead"])

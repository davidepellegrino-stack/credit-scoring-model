import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "ifrs9"))

from pd_shift import (cumulative_pd, hazard_matrix, logit_shift_to_mean, pd_term_structure,  # noqa: E402
                      scenario_multipliers, shifted_pds)


@pytest.fixture
def pds():
    return np.random.default_rng(0).beta(1.2, 12, 5_000)


def test_shift_hits_target_mean(pds):
    for target in (0.02, pds.mean(), 0.2):
        d = logit_shift_to_mean(pds, target)
        x = np.log(pds / (1 - pds)) + d
        assert (1 / (1 + np.exp(-x))).mean() == pytest.approx(target, abs=1e-9)


def test_multiplier_one_is_identity(pds):
    assert shifted_pds(pds, 1.0) == pytest.approx(pds, abs=1e-8)


def test_shift_keeps_rank_and_bounds(pds):
    s = shifted_pds(pds, 1.8)
    assert (s > 0).all() and (s < 1).all()
    assert (np.argsort(s) == np.argsort(pds)).all()
    assert s.mean() == pytest.approx(pds.mean() * 1.8, rel=1e-6)


def test_high_pd_loans_move_more_in_absolute_terms(pds):
    s = shifted_pds(pds, 1.5)
    hi, lo = pds > np.quantile(pds, 0.9), pds < np.quantile(pds, 0.1)
    assert (s - pds)[hi].mean() > (s - pds)[lo].mean() > 0


def test_invalid_target_raises(pds):
    with pytest.raises(ValueError):
        logit_shift_to_mean(pds, 1.2)


def test_scenario_multipliers_ratio():
    idx = pd.period_range("2026Q3", periods=3, freq="Q")
    paths = {"baseline": pd.Series([2.0, 2.0, 2.0], index=idx), "adverse": pd.Series([2.0, 3.0, 4.0], index=idx)}
    m = scenario_multipliers(paths)
    assert m["baseline"].tolist() == [1.0, 1.0, 1.0]
    assert m["adverse"].tolist() == pytest.approx([1.0, 1.5, 2.0])


def test_constant_hazard_cumulative_matches_closed_form():
    h = np.full((3, 12), 0.01)
    assert cumulative_pd(h, 4) == pytest.approx(1 - 0.99 ** 4)
    assert cumulative_pd(h, 12) == pytest.approx(1 - 0.99 ** 12)


def test_baseline_pd12_equals_input_pd(pds):
    """With multiplier 1 in every quarter the 12-month PD is the original PD (hazard conversion round-trips)."""
    h = hazard_matrix(pds, np.ones(12))
    assert cumulative_pd(h, 4) == pytest.approx(pds, abs=1e-7)


def test_term_structure_orders_scenarios(pds):
    ts = pd_term_structure(pds, {"baseline": np.ones(12), "adverse": np.linspace(1, 1.6, 12)})
    assert ts["adverse"]["pd12"].mean() > ts["baseline"]["pd12"].mean()
    assert (ts["adverse"]["pd_life"] > ts["adverse"]["pd12"]).all()
    assert ts["baseline"]["pd_life"].mean() > ts["baseline"]["pd12"].mean()

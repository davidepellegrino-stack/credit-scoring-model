import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from metrics import decile_table, gini, ks_statistic


def test_perfect_and_random_scores():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert gini(y, y.astype(float)) == pytest.approx(1.0)
    assert ks_statistic(y, y.astype(float)) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    yr = rng.integers(0, 2, 20_000)
    pr = rng.random(20_000)
    assert abs(gini(yr, pr)) < 0.03
    assert ks_statistic(yr, pr) < 0.03


def test_gini_is_two_auc_minus_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = rng.random(500) + 0.3 * y
    assert gini(y, p) == pytest.approx(2 * roc_auc_score(y, p) - 1)


def test_ks_matches_direct_cdf_gap():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    p = rng.random(400) + 0.4 * y
    grid = np.unique(p)
    gap = max(abs((p[y == 1] <= t).mean() - (p[y == 0] <= t).mean()) for t in grid)
    assert ks_statistic(y, p) == pytest.approx(gap)


def test_decile_table_structure():
    rng = np.random.default_rng(3)
    y = (rng.random(1_003) < 0.1).astype(int)
    p = rng.random(1_003) * 0.5 + 0.5 * y
    t = decile_table(y, p)
    assert len(t) == 10 and t["loans"].sum() == 1_003
    assert t["defaults"].sum() == y.sum()
    assert t["cum_default_share"].iloc[-1] == pytest.approx(1.0)
    assert t["cum_lift"].iloc[-1] == pytest.approx(1.0)
    assert t["lift"].iloc[0] > t["lift"].iloc[-1]
    # weighted mean of bin lifts is 1
    assert np.average(t["lift"], weights=t["loans"]) == pytest.approx(1.0)


def test_decile_table_riskiest_first_and_errors():
    y = np.array([1] * 5 + [0] * 95)
    p = np.linspace(1, 0, 100)
    t = decile_table(y, p)
    assert t["defaults"].iloc[0] == 5 and t["cum_default_share"].iloc[0] == 1.0
    with pytest.raises(ValueError):
        decile_table(np.zeros(100), p)
    with pytest.raises(ValueError):
        decile_table(y[:5], p[:5])

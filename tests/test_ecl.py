import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "ifrs9"))

from ecl import (assign_stage, check_weights, discount_factors, ecl_over_horizon,  # noqa: E402
                 marginal_default_prob, weighted_ecl)
from pd_shift import pd_term_structure  # noqa: E402


def test_marginal_probs_sum_to_cumulative_pd():
    h = np.array([[0.01, 0.02, 0.03, 0.04]])
    assert marginal_default_prob(h).sum() == pytest.approx(1 - np.prod(1 - h))


def test_constant_hazard_marginals():
    h = np.full((1, 3), 0.1)
    assert marginal_default_prob(h)[0] == pytest.approx([0.1, 0.09, 0.081])


def test_ecl_closed_form_without_discounting():
    h = np.full((2, 4), 0.01)
    ecl = ecl_over_horizon(h, lgd=0.5, ead=np.array([1000.0, 2000.0]), eir=0.0, quarters=4)
    assert ecl == pytest.approx(np.array([1000.0, 2000.0]) * 0.5 * (1 - 0.99 ** 4))


def test_discounting_lowers_ecl_and_is_mid_quarter():
    assert discount_factors(4, 0.10)[0] == pytest.approx(1.10 ** -0.125)
    h = np.full((1, 4), 0.02)
    assert ecl_over_horizon(h, 0.6, 100.0, 0.15, 4) < ecl_over_horizon(h, 0.6, 100.0, 0.0, 4)


def test_ecl_linear_in_lgd_and_ead():
    h = np.full((1, 12), 0.015)
    base = ecl_over_horizon(h, 0.4, 100.0, 0.1, 12)
    assert ecl_over_horizon(h, 0.8, 100.0, 0.1, 12) == pytest.approx(2 * base)
    assert ecl_over_horizon(h, 0.4, 300.0, 0.1, 12) == pytest.approx(3 * base)


def test_sicr_needs_both_ratio_and_absolute_increase():
    orig = np.array([0.20, 0.01, 0.20])
    now = np.array([0.45, 0.025, 0.30])   # 2.25x and +25pp; 2.5x but only +1.5pp; 1.5x
    assert assign_stage(now, orig, ratio=2.0, abs_floor=0.02).tolist() == [2, 1, 1]
    assert assign_stage(now, orig, ratio=2.0, abs_floor=0.01).tolist() == [2, 2, 1]


def test_impaired_flag_forces_stage3():
    s = assign_stage(np.array([0.1, 0.1]), np.array([0.1, 0.1]), impaired=[False, True])
    assert s.tolist() == [1, 3]


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        check_weights({"a": 0.5, "b": 0.4})


@pytest.fixture
def ts():
    pds = np.random.default_rng(1).beta(1.2, 12, 3_000)
    return pd_term_structure(pds, {"baseline": np.ones(12), "adverse": np.linspace(1, 1.8, 12),
                                   "severely_adverse": np.linspace(1, 3.0, 12)})


def test_weighted_ecl_is_the_weighted_average_of_scenarios(ts):
    w = {"baseline": 0.5, "adverse": 0.3, "severely_adverse": 0.2}
    ead = np.full(3_000, 1000.0)
    r = weighted_ecl(ts, w, 0.6, ead, 0.1)
    for key, idx in (("ecl12", 0), ("ecl_life", 1)):
        manual = sum(w[n] * r["by_scenario"][n][idx] for n in w)
        assert r[key] == pytest.approx(manual)
    assert np.all(np.isin(r["stage"], [1, 2]))
    assert np.all(r["ecl"] >= r["ecl12"] - 1e-12)          # stage 2 uses the longer horizon


def test_single_scenario_weight_reproduces_that_scenario(ts):
    ead = np.full(3_000, 500.0)
    r = weighted_ecl(ts, {"baseline": 1.0, "adverse": 0.0, "severely_adverse": 0.0}, 0.5, ead, 0.1)
    assert r["ecl12"] == pytest.approx(r["by_scenario"]["baseline"][0])
    assert (r["stage"] == 1).all()                          # no increase vs origination


def test_heavier_stress_weight_raises_ecl(ts):
    ead = np.full(3_000, 1000.0)
    mild = weighted_ecl(ts, {"baseline": 0.8, "adverse": 0.2, "severely_adverse": 0.0}, 0.6, ead, 0.1)
    harsh = weighted_ecl(ts, {"baseline": 0.2, "adverse": 0.3, "severely_adverse": 0.5}, 0.6, ead, 0.1)
    assert harsh["ecl"].sum() > mild["ecl"].sum()


def test_impaired_loans_get_lgd_times_ead(ts):
    ead = np.full(3_000, 1000.0)
    imp = np.zeros(3_000, dtype=bool)
    imp[:5] = True
    r = weighted_ecl(ts, {"baseline": 1.0, "adverse": 0.0, "severely_adverse": 0.0}, 0.6, ead, 0.1, impaired=imp)
    assert r["ecl"][:5] == pytest.approx(600.0)
    assert (r["stage"][:5] == 3).all()

import numpy as np
import pytest

from calibration import (IsotonicCalibrator, PlattCalibrator, calibration_summary,
                         expected_calibration_error, reliability_table)


def _overconfident(n=40_000, seed=0):
    """True PD p; the 'model' reports p**0.6 * 1.8 clipped, i.e. systematically too high."""
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 12, n)
    y = (rng.random(n) < p).astype(int)
    score = np.clip(1.8 * p ** 0.8, 1e-4, 0.99)
    return y, score


def test_perfectly_calibrated_has_small_ece():
    rng = np.random.default_rng(1)
    p = rng.beta(1, 10, 50_000)
    y = (rng.random(50_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.005


def test_ece_detects_miscalibration_and_is_zero_for_exact_rates():
    y, s = _overconfident()
    assert expected_calibration_error(y, s) > 0.03
    y2 = np.array([0, 0, 0, 1] * 5)  # every bin of 5 has rate 0.25
    assert expected_calibration_error(y2, np.full(20, 0.25), n_bins=5) == pytest.approx(0.0)


def test_reliability_table_bins_are_equal_and_sorted():
    y, s = _overconfident(n=1_003)
    t = reliability_table(y, s, 10)
    assert t["loans"].sum() == 1_003 and t["loans"].max() - t["loans"].min() <= 1
    assert t["mean_pd"].is_monotonic_increasing


@pytest.mark.parametrize("cls", [PlattCalibrator, IsotonicCalibrator])
def test_calibrators_fix_bias_out_of_sample(cls):
    y, s = _overconfident()
    half = len(y) // 2
    cal = cls().fit(s[:half], y[:half])
    q = cal.predict(s[half:])
    before, after = calibration_summary(y[half:], s[half:]), calibration_summary(y[half:], q)
    assert after["ECE"] < 0.3 * before["ECE"]
    assert after["Brier"] < before["Brier"]
    assert abs(after["Mean PD - default rate"]) < 0.005


@pytest.mark.parametrize("cls", [PlattCalibrator, IsotonicCalibrator])
def test_calibrators_are_monotone_and_valid_probabilities(cls):
    y, s = _overconfident(n=10_000)
    q = cls().fit(s, y).predict(np.linspace(0, 1, 200))
    assert np.all(np.diff(q) >= -1e-12) and q.min() > 0 and q.max() < 1

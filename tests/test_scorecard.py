import numpy as np
import pandas as pd
import pytest

from scorecard import Scorecard, WoEBinner, woe_iv


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    z = -2.5 - 1.0 * x + np.where(cat == "c", 0.8, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    X = pd.DataFrame({"x": x, "noise": noise, "cat": cat})
    X.loc[rng.random(n) < 0.1, "x"] = np.nan
    return X, y


def test_woe_iv_known_values():
    # two bins, 100 bad / 900 good in total: bin 0 has 80 bad of 400 loans
    woe, iv = woe_iv(np.array([80.0, 20.0]), np.array([320.0, 580.0]))
    assert woe[0] < 0 < woe[1]  # riskier bin has negative WoE
    assert (iv >= 0).all()
    assert np.isfinite(woe_iv(np.array([0.0, 5.0]), np.array([10.0, 5.0]))[0]).all()


def test_informative_variable_has_higher_iv_than_noise(data):
    X, y = data
    b = WoEBinner(["x", "noise"], ["cat"]).fit(X, y)
    assert b.iv_["x"] > 0.1
    assert b.iv_["noise"] < 0.02
    assert b.iv_.index[0] == "x"


def test_numeric_bins_monotone_and_big_enough(data):
    X, y = data
    b = WoEBinner(["x"], min_bin_frac=0.05).fit(X, y)
    t = b.table_["x"]
    reg = t[t["bin"] != "missing"]
    assert (np.diff(reg["bad_rate"]) <= 1e-12).all()  # bad rate falls as x rises
    assert (t["loans"] >= 0.05 * len(X) - 1).all()
    assert t["loans"].sum() == len(X)


def test_missing_has_own_bin_and_unseen_category_is_neutral(data):
    X, y = data
    b = WoEBinner(["x"], ["cat"]).fit(X, y)
    assert "missing" in set(b.table_["x"]["bin"])
    new = X.head(3).copy()
    new["cat"] = ["zzz", "a", "b"]
    assert list(b.bin_labels(new, 'cat'))[0] == 'other'
    W = b.transform(new)
    assert np.isfinite(W.to_numpy()).all()


def test_transform_matches_table_woe(data):
    X, y = data
    b = WoEBinner(["x"], ["cat"]).fit(X, y)
    W, lab = b.transform(X), b.bin_labels(X, "cat")
    m = b.table_["cat"].set_index("bin")["woe"]
    assert np.allclose(W["cat"], lab.map(m))


def test_scorecard_scaling_and_pd_consistency(data):
    X, y = data
    sc = Scorecard(WoEBinner(["x", "noise"], ["cat"]), pdo=20, base_score=600, base_odds=50).fit(X, y)
    p, s = sc.predict_proba(X), sc.score(X)
    # score is an exact linear map of ln(odds of good): 20 points = doubling the odds
    odds = (1 - p) / p
    assert np.allclose(s, 600 + 20 / np.log(2) * (np.log(odds) - np.log(50)), atol=1e-6)
    assert np.corrcoef(s, p)[0, 1] < -0.5


def test_points_table_adds_up_to_score(data):
    X, y = data
    sc = Scorecard(WoEBinner(["x", "noise"], ["cat"])).fit(X, y)
    pt = sc.points_table().set_index(["variable", "bin"])["points"]
    row = X.head(50)
    total = sum(
        sc.binner.bin_labels(row, c).map(lambda b, c=c: pt[(c, b)]) for c in sc.cols_)
    assert np.allclose(total, sc.score(row), atol=1e-6)

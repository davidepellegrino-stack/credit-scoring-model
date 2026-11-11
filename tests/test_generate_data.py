"""Sanity checks on the synthetic credit dataset."""
import numpy as np

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from generate_data import generate


def test_schema(credit_df):
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    assert set(credit_df.columns) == expected
    assert len(credit_df) == 2_000
    assert not credit_df.isna().any().any()


def test_reproducible_with_same_seed():
    a = generate(n=300, seed=123)
    b = generate(n=300, seed=123)
    assert a.equals(b)


def test_value_ranges(credit_df):
    assert credit_df[TARGET].isin([0, 1]).all()
    assert credit_df["age"].between(20, 69).all()
    assert credit_df["income_eur"].between(12_000, 200_000).all()
    assert credit_df["credit_utilization_pct"].between(0, 100).all()
    # 999 is the "never defaulted" sentinel; otherwise 1-119 months.
    m = credit_df["months_since_last_default"]
    assert ((m == 999) | m.between(1, 119)).all()


def test_default_rate_is_plausible(credit_df):
    rate = credit_df[TARGET].mean()
    assert 0.10 < rate < 0.40, f"default rate {rate:.2%} outside a plausible band"


def test_risk_drivers_point_the_right_way(credit_df):
    """The data generator encodes known effects; check they survive in the sample."""
    df = credit_df
    prior_default = df["months_since_last_default"] != 999
    assert df.loc[prior_default, TARGET].mean() > df.loc[~prior_default, TARGET].mean()

    high_lti = df["loan_to_income"] > df["loan_to_income"].median()
    assert df.loc[high_lti, TARGET].mean() > df.loc[~high_lti, TARGET].mean()

    corr = np.corrcoef(df["credit_utilization_pct"], df[TARGET])[0, 1]
    assert corr > 0

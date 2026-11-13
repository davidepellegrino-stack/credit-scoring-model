"""Tests for the Home Credit loader. They use a tiny in-memory fixture: no network."""
import numpy as np
import pandas as pd
import pytest

import data_home_credit as hc


@pytest.fixture
def tiny():
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "TARGET": rng.integers(0, 2, n),
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "AMT_INCOME_TOTAL": rng.uniform(50_000, 300_000, n),
        "AMT_CREDIT": rng.uniform(100_000, 900_000, n),
        "AMT_ANNUITY": rng.uniform(5_000, 40_000, n),
        "AMT_GOODS_PRICE": rng.uniform(100_000, 900_000, n),
        "DAYS_BIRTH": -rng.integers(7_000, 24_000, n),
        "DAYS_EMPLOYED": -rng.integers(100, 5_000, n),
        "EXT_SOURCE_1": rng.uniform(0, 1, n),
        "EXT_SOURCE_2": rng.uniform(0, 1, n),
        "EXT_SOURCE_3": rng.uniform(0, 1, n),
        "application_date": pd.date_range("2018-01-01", periods=n).astype(str),
    })
    return df


def test_valid_table_passes(tiny):
    hc.validate_application(tiny)


def test_test_table_needs_no_target(tiny):
    hc.validate_application(tiny.drop(columns="TARGET"), with_target=False)


def test_missing_column_reported(tiny):
    with pytest.raises(ValueError, match="EXT_SOURCE_2"):
        hc.validate_application(tiny.drop(columns="EXT_SOURCE_2"))


def test_duplicate_ids_rejected(tiny):
    tiny.loc[1, "SK_ID_CURR"] = tiny.loc[0, "SK_ID_CURR"]
    with pytest.raises(ValueError, match="not unique"):
        hc.validate_application(tiny)


def test_non_binary_or_missing_target_rejected(tiny):
    tiny.loc[0, "TARGET"] = 2
    with pytest.raises(ValueError, match="binary"):
        hc.validate_application(tiny)
    tiny.loc[0, "TARGET"] = np.nan
    with pytest.raises(ValueError, match="binary"):
        hc.validate_application(tiny)


def test_positive_days_birth_rejected(tiny):
    tiny.loc[0, "DAYS_BIRTH"] = 100
    with pytest.raises(ValueError, match="DAYS_BIRTH"):
        hc.validate_application(tiny)


def test_describe_target(tiny):
    info = hc.describe_target(tiny)
    assert info["n_loans"] == 50
    assert info["n_defaults"] == int(tiny["TARGET"].sum())
    assert info["default_rate"] == pytest.approx(tiny["TARGET"].mean())


def test_load_uses_cache_without_network(tiny, tmp_path):
    tiny.to_parquet(tmp_path / "application_train.parquet")
    df = hc.load_application("application_train", cache_dir=str(tmp_path))
    assert len(df) == 50
    assert pd.api.types.is_datetime64_any_dtype(df["application_date"])


def test_unknown_table_name():
    with pytest.raises(KeyError):
        hc.fetch_table("bogus")

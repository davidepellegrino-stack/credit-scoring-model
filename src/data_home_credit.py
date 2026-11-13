"""Home Credit Default Risk: download, cache, validate and load the application table.

Source of the data: Kaggle competition "Home Credit Default Risk" (Home Credit
Group, 2018). Kaggle requires credentials, so this script fetches the same
tables from a public Hugging Face mirror (`mohameddhameem/home-credit-default-risk`).
The mirror's training table carries an extra `application_date` column
(2018-01-01 to 2020-12-31) that is not in the Kaggle original; it is used later
for out-of-time validation and should be read as a synthetic timestamp.

Raw data is never committed: files are cached under `data/home_credit/`
(git-ignored) and re-used on the next run.

Usage:
    python src/data_home_credit.py            # download + print a summary
"""
import os
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR  # noqa: E402

HF_REPO = "mohameddhameem/home-credit-default-risk"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
CACHE_DIR = os.path.join(DATA_DIR, "home_credit")

# table name -> path of the parquet file inside the mirror
TABLES = {
    "application_train": "application_train_dated/train-00000-of-00001.parquet",
    "application_test": "application_test/train-00000-of-00001.parquet",
}

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
DATE_COL = "application_date"

# Columns the downstream feature code relies on. Failing early with a clear
# message is better than a KeyError deep inside a model pipeline.
REQUIRED_COLUMNS = [
    ID_COL, "NAME_CONTRACT_TYPE", "CODE_GENDER", "AMT_INCOME_TOTAL",
    "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH",
    "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]


def _download(url: str, dest: str) -> None:
    """Download `url` to `dest` atomically (temp file, then rename)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    os.replace(tmp, dest)


def fetch_table(name: str, cache_dir: str = CACHE_DIR, force: bool = False) -> str:
    """Return the local path of a cached table, downloading it if needed."""
    if name not in TABLES:
        raise KeyError(f"unknown table {name!r}; choose from {sorted(TABLES)}")
    dest = os.path.join(cache_dir, f"{name}.parquet")
    if force or not os.path.exists(dest):
        print(f"Downloading {name} from the Hugging Face mirror ...")
        _download(f"{HF_BASE}/{TABLES[name]}", dest)
    return dest


def validate_application(df: pd.DataFrame, with_target: bool = True) -> None:
    """Schema and sanity checks; raises ValueError with every problem found."""
    problems = []
    required = REQUIRED_COLUMNS + ([TARGET_COL] if with_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if ID_COL in df.columns and not df[ID_COL].is_unique:
        problems.append(f"{ID_COL} is not unique")
    if with_target and TARGET_COL in df.columns:
        if not df[TARGET_COL].isin([0, 1]).all():
            problems.append(f"{TARGET_COL} must be binary (0/1) with no missing values")
    if "DAYS_BIRTH" in df.columns and (df["DAYS_BIRTH"] >= 0).any():
        problems.append("DAYS_BIRTH should be negative (days before application)")
    if "AMT_INCOME_TOTAL" in df.columns and (df["AMT_INCOME_TOTAL"] <= 0).any():
        problems.append("AMT_INCOME_TOTAL must be positive")
    if problems:
        raise ValueError("Home Credit table failed validation: " + "; ".join(problems))


def load_application(name: str = "application_train", cache_dir: str = CACHE_DIR,
                     force_download: bool = False) -> pd.DataFrame:
    """Load a validated application table. The test table has no target (Kaggle holds it out)."""
    path = fetch_table(name, cache_dir, force_download)
    df = pd.read_parquet(path)
    validate_application(df, with_target=(name == "application_train"))
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def describe_target(df: pd.DataFrame) -> dict:
    """Default definition and class balance.

    TARGET = 1: the client had payment difficulties, i.e. a late payment of
    more than X days on at least one of the first Y instalments of the loan
    (Home Credit's definition; X and Y are not disclosed). It is a
    default-like event inside the observation window, not a 90-dpd Basel default.
    """
    n = len(df)
    n_def = int(df[TARGET_COL].sum())
    return {"n_loans": n, "n_defaults": n_def, "default_rate": n_def / n}


if __name__ == "__main__":
    train = load_application("application_train")
    info = describe_target(train)
    print(f"application_train: {info['n_loans']:,} loans, {train.shape[1]} columns")
    print(f"defaults: {info['n_defaults']:,} ({info['default_rate']:.2%})")
    if DATE_COL in train.columns:
        print(f"application dates: {train[DATE_COL].min().date()} to {train[DATE_COL].max().date()}")
    test = load_application("application_test")
    print(f"application_test: {len(test):,} loans (target held out by the competition)")

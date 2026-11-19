"""FRED macro and credit series, aligned to quarters, for the IFRS 9 satellite model.

Series come from the public FRED CSV endpoint (no API key needed):
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>
Raw downloads are cached under `data/fred/` (git-ignored) and re-used on the next run.

Alignment rules (one row per calendar quarter, PeriodIndex "Q"):
  * monthly and daily series are averaged within the quarter, but a quarter is dropped
    (NaN) if it has fewer observations than the series needs (3 months, 50 business
    days), so a half-finished current quarter never looks like a full one;
  * a quarter missing in every series is dropped; a missing month inside a quarter makes
    that quarter NaN for that series only (for example UNRATE 2025Q4: October 2025 was
    never published), so the satellite model must handle gaps rather than interpolate them;
  * series published quarterly (GDP, bank delinquency / charge-off rates) are kept as is.

Usage:
    python ifrs9/fred.py            # download, align, print the tail and coverage
"""
import http.client
import io
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from config import DATA_DIR  # noqa: E402

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
CACHE_DIR = os.path.join(DATA_DIR, "fred")

# FRED id -> (column name, native frequency, description)
SERIES = {
    "UNRATE": ("unrate", "M", "Civilian unemployment rate, %"),
    "GDPC1": ("gdp_real", "Q", "Real GDP, billions of chained 2017 USD, SAAR"),
    "FEDFUNDS": ("fedfunds", "M", "Effective federal funds rate, %"),
    "DGS10": ("dgs10", "D", "10-year Treasury constant-maturity yield, %"),
    "CPIAUCSL": ("cpi", "M", "CPI all urban consumers, index 1982-84=100"),
    "CSUSHPINSA": ("hpi", "M", "S&P/Case-Shiller US National Home Price Index"),
    "DRCCLACBS": ("dr_card", "Q", "Delinquency rate, credit cards, all US banks, %"),
    "CORCACBS": ("co_card", "Q", "Charge-off rate, credit cards, all US banks, %"),
    "DRCLACBS": ("dr_consumer", "Q", "Delinquency rate, consumer loans, all US banks, %"),
    "DRSFRMACBS": ("dr_mortgage", "Q", "Delinquency rate, single-family mortgages, %"),
}
# observations a quarter needs before an average is trusted
MIN_OBS = {"M": 3, "D": 50, "Q": 1}


def _download(url, retries=3):
    """GET a URL as text. FRED drops connections that carry an unfamiliar User-Agent, so
    send a plain curl-style one, and retry because the endpoint is occasionally flaky."""
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8")
        except (OSError, http.client.HTTPException):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_series(series_id, refresh=False, cache_dir=None):
    """Download one FRED series (or read the cached CSV) as a float Series on its native dates."""
    cache_dir = cache_dir or CACHE_DIR
    path = os.path.join(cache_dir, f"{series_id}.csv")
    if refresh or not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        text = _download(FRED_URL.format(series_id))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    with open(path, encoding="utf-8") as fh:
        return parse_fred_csv(fh.read(), series_id)


def parse_fred_csv(text, series_id):
    """Parse FRED's two-column CSV; the missing-value marker '.' becomes NaN."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    if df.shape[1] != 2:
        raise ValueError(f"{series_id}: expected 2 columns (date, value), got {list(df.columns)}")
    date_col, value_col = df.columns
    if value_col != series_id:
        raise ValueError(f"{series_id}: FRED returned column '{value_col}'")
    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df[date_col]), name=series_id)
    return s.dropna().sort_index()


def to_quarterly(s, freq):
    """Collapse a series to quarters (see module docstring for the rules)."""
    if freq not in MIN_OBS:
        raise ValueError(f"unknown frequency '{freq}'")
    q = s.copy()
    q.index = q.index.to_period("Q")
    grouped = q.groupby(level=0)
    out = grouped.mean()
    out[grouped.count() < MIN_OBS[freq]] = np.nan
    return out


def build_quarterly_panel(raw, series=SERIES):
    """Align already-downloaded series ({fred_id: Series}) into one quarterly DataFrame."""
    cols = {}
    for fid, (name, freq, _) in series.items():
        if fid in raw:
            cols[name] = to_quarterly(raw[fid], freq)
    panel = pd.DataFrame(cols).sort_index().dropna(how="all")  # drop all-NaN trailing quarters
    panel.index.name = "quarter"
    return panel


def add_transforms(panel):
    """Add the regressors the satellite model uses (all in percent or percentage points).

    gdp_growth_yoy : 100 * (GDP_t / GDP_{t-4} - 1)
    gdp_growth_qoq_ann : annualised quarter-on-quarter growth, 100 * ((GDP_t / GDP_{t-1})^4 - 1)
    d_unrate_4q    : change in the unemployment rate over four quarters, pp
    inflation_yoy  : 100 * (CPI_t / CPI_{t-4} - 1)
    hpi_growth_yoy : 100 * (HPI_t / HPI_{t-4} - 1)
    term_spread    : 10y Treasury yield minus fed funds rate, pp
    """
    out = panel.copy()
    if "gdp_real" in out:
        out["gdp_growth_yoy"] = 100 * (out["gdp_real"] / out["gdp_real"].shift(4) - 1)
        out["gdp_growth_qoq_ann"] = 100 * ((out["gdp_real"] / out["gdp_real"].shift(1)) ** 4 - 1)
    if "unrate" in out:
        out["d_unrate_4q"] = out["unrate"] - out["unrate"].shift(4)
    if "cpi" in out:
        out["inflation_yoy"] = 100 * (out["cpi"] / out["cpi"].shift(4) - 1)
    if "hpi" in out:
        out["hpi_growth_yoy"] = 100 * (out["hpi"] / out["hpi"].shift(4) - 1)
    if {"dgs10", "fedfunds"} <= set(out.columns):
        out["term_spread"] = out["dgs10"] - out["fedfunds"]
    return out


def load_macro_panel(refresh=False, cache_dir=None):
    """Download (or read from cache) every series and return the quarterly panel with transforms."""
    raw = {fid: fetch_series(fid, refresh=refresh, cache_dir=cache_dir) for fid in SERIES}
    return add_transforms(build_quarterly_panel(raw))


def coverage(panel):
    """First quarter, last quarter and number of non-missing quarters for each column."""
    rows = []
    for col in panel.columns:
        s = panel[col].dropna()
        rows.append({"column": col, "first": str(s.index.min()), "last": str(s.index.max()),
                     "n_quarters": len(s)})
    return pd.DataFrame(rows).set_index("column")


if __name__ == "__main__":
    panel = load_macro_panel()
    print(f"Quarterly macro panel: {panel.shape[0]} quarters x {panel.shape[1]} columns\n")
    print(coverage(panel).to_string())
    print("\nLast 6 quarters:")
    print(panel[["unrate", "gdp_growth_yoy", "fedfunds", "dr_card", "co_card"]].tail(6).round(2).to_string())

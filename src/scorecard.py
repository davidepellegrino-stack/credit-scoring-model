"""Weight of Evidence (WoE) binning, Information Value and a points scorecard.

This is the format retail credit teams use: every variable is cut into a few bins,
each bin gets a WoE, a logistic regression on the WoE values gives the PD, and the
log-odds are rescaled to points so that a fixed number of points (PDO) doubles the odds.

Conventions
-----------
* WoE = ln(share of good loans in the bin / share of bad loans in the bin); a higher WoE
  means a safer bin. Counts are smoothed by 0.5 so a pure bin never gives an infinite WoE.
* IV = sum over bins of (good share - bad share) * WoE. Common reading: < 0.02 useless,
  0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, > 0.5 suspiciously strong.
* Numeric variables: quantile pre-bins, then adjacent bins are merged until the bad rate
  is monotone in the variable and every bin holds at least `min_bin_frac` of the loans.
  Missing values form their own bin. Categorical variables: one bin per category, with
  rare categories pooled into "other".
* Score = offset + factor * ln(odds of good), factor = PDO / ln 2.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SMOOTH = 0.5
MISSING = "missing"
OTHER = "other"


def woe_iv(bad: np.ndarray, good: np.ndarray):
    """WoE and per-bin IV contribution from bad/good counts per bin."""
    b = (bad + SMOOTH) / (bad.sum() + SMOOTH * len(bad))
    g = (good + SMOOTH) / (good.sum() + SMOOTH * len(good))
    woe = np.log(g / b)
    return woe, (g - b) * woe


def _merge_numeric(edges, bad, good, min_n, increasing):
    """Merge adjacent bins until bad rate is monotone (direction `increasing`) and
    every bin has >= min_n loans. `edges` are the interior cut points."""
    edges, bad, good = list(edges), list(bad), list(good)

    def rate(i):
        return bad[i] / max(bad[i] + good[i], 1)

    while len(bad) > 1:
        n = [b + g for b, g in zip(bad, good)]
        small = [i for i in range(len(n)) if n[i] < min_n]
        if small:
            i = min(small, key=lambda k: n[k])
            if i == 0:
                j = 1
            elif i == len(n) - 1:
                j = i - 1
            else:  # merge with the neighbour whose bad rate is closest
                j = i - 1 if abs(rate(i) - rate(i - 1)) <= abs(rate(i) - rate(i + 1)) else i + 1
            k = min(i, j)
        else:
            sign = 1 if increasing else -1
            viol = [i for i in range(len(bad) - 1) if sign * (rate(i + 1) - rate(i)) < 0]
            if not viol:
                break
            k = viol[0]
        bad[k] += bad.pop(k + 1)
        good[k] += good.pop(k + 1)
        edges.pop(k)  # the cut between bin k and k+1 disappears
    return edges, np.array(bad), np.array(good)


class WoEBinner:
    """Fit bins on a training set, then map any frame to bin labels or WoE values."""

    def __init__(self, numeric, categorical=(), n_prebins=20, min_bin_frac=0.05,
                 max_bins=8, min_cat_frac=0.01):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.n_prebins, self.min_bin_frac = n_prebins, min_bin_frac
        self.max_bins, self.min_cat_frac = max_bins, min_cat_frac

    # -- fitting ---------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y) -> "WoEBinner":
        y = pd.Series(np.asarray(y), index=X.index)
        min_n = int(self.min_bin_frac * len(X))
        self.edges_, self.cats_, self.table_ = {}, {}, {}
        for c in self.numeric:
            self._fit_numeric(c, X[c], y, min_n)
        for c in self.categorical:
            self._fit_categorical(c, X[c], y)
        self.iv_ = pd.Series({c: t["iv"].sum() for c, t in self.table_.items()}).sort_values(
            ascending=False)
        return self

    def _fit_numeric(self, c, x, y, min_n):
        ok = x.notna()
        qs = np.unique(np.quantile(x[ok], np.linspace(0, 1, self.n_prebins + 1)[1:-1]))
        codes = np.searchsorted(qs, x[ok].to_numpy(), side="right")
        yy = y[ok].to_numpy()
        k = len(qs) + 1
        bad = np.bincount(codes, weights=yy, minlength=k)
        good = np.bincount(codes, weights=1 - yy, minlength=k)
        # direction of the trend; a variable with a single pre-bin has none (default: increasing)
        increasing = len(qs) == 0 or np.cov(codes, yy)[0, 1] >= 0
        edges, bad, good = _merge_numeric(qs, bad, good, min_n, increasing)
        while len(bad) > self.max_bins:  # still too many: merge the most similar pair
            r = bad / (bad + good)
            k = int(np.argmin(np.abs(np.diff(r))))
            bad[k] += bad[k + 1]; good[k] += good[k + 1]
            bad, good = np.delete(bad, k + 1), np.delete(good, k + 1)
            edges.pop(k)
        labels = self._labels(edges)
        if (~ok).any():
            labels.append(MISSING)
            bad = np.append(bad, y[~ok].sum())
            good = np.append(good, (1 - y[~ok]).sum())
        self.edges_[c] = np.array(edges)
        self.table_[c] = self._table(labels, bad, good)

    @staticmethod
    def _labels(edges):
        e = [-np.inf] + list(edges) + [np.inf]
        return [f"({e[i]:.4g}, {e[i + 1]:.4g}]" for i in range(len(e) - 1)]

    def _fit_categorical(self, c, x, y):
        x = x.astype("object").where(x.notna(), MISSING)
        share = x.value_counts(normalize=True)
        keep = share[share >= self.min_cat_frac].index.tolist()
        self.cats_[c] = keep
        x = x.where(x.isin(keep), OTHER)
        g = pd.DataFrame({"x": x, "y": y.to_numpy()}).groupby("x")["y"].agg(["sum", "count"])
        self.table_[c] = self._table(g.index.tolist(), g["sum"].to_numpy(float),
                                     (g["count"] - g["sum"]).to_numpy(float))

    @staticmethod
    def _table(labels, bad, good):
        woe, iv = woe_iv(bad, good)
        n = bad + good
        return pd.DataFrame({"bin": labels, "loans": n.astype(int), "share": n / n.sum(),
                             "bad_rate": bad / n, "woe": woe, "iv": iv})

    # -- applying ----------------------------------------------------------------------
    def bin_labels(self, X: pd.DataFrame, c: str) -> pd.Series:
        x = X[c]
        if c in self.numeric:
            labels = np.array(self.table_[c]["bin"].tolist())
            codes = np.searchsorted(self.edges_[c], x.to_numpy(float), side="right")
            # NaN would sort last; send it to the missing bin (or the last bin if none was seen)
            miss_code = len(labels) - 1
            codes = np.where(x.isna().to_numpy(), miss_code, codes)
            return pd.Series(labels[np.minimum(codes, len(labels) - 1)], index=X.index)
        x = x.astype("object").where(x.notna(), MISSING)
        return x.where(x.isin(self.cats_[c]), OTHER)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """WoE value of every variable for every row. Unseen category -> 0 (neutral)."""
        out = {}
        for c in self.numeric + self.categorical:
            m = self.table_[c].set_index("bin")["woe"]
            out[c] = self.bin_labels(X, c).map(m).fillna(0.0).astype(float)
        return pd.DataFrame(out, index=X.index)


class Scorecard:
    """Logistic regression on WoE values, rescaled to points (base score at base odds,
    `pdo` points to double the odds of good)."""

    def __init__(self, binner: WoEBinner, pdo=20.0, base_score=600.0, base_odds=50.0, C=1.0):
        self.binner, self.pdo, self.base_score, self.base_odds, self.C = (
            binner, pdo, base_score, base_odds, C)
        self.factor = pdo / np.log(2.0)
        self.offset = base_score - self.factor * np.log(base_odds)

    def fit(self, X, y) -> "Scorecard":
        W = self.binner.fit(X, y).transform(X)
        self.cols_ = list(W.columns)
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000).fit(W, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.lr_.predict_proba(self.binner.transform(X)[self.cols_])[:, 1]

    def score(self, X) -> np.ndarray:
        """Points. Higher = safer. Equals offset + factor * ln(odds of good)."""
        z = self.lr_.decision_function(self.binner.transform(X)[self.cols_])  # ln(odds of bad)
        return self.offset - self.factor * z

    def points_table(self) -> pd.DataFrame:
        """One row per bin: its points. The intercept is spread evenly over the variables."""
        n = len(self.cols_)
        base = (self.offset - self.factor * self.lr_.intercept_[0]) / n
        rows = []
        for c, b in zip(self.cols_, self.lr_.coef_[0]):
            t = self.binner.table_[c]
            for _, r in t.iterrows():
                rows.append({"variable": c, "bin": r["bin"], "loans": r["loans"],
                             "bad_rate": r["bad_rate"], "woe": r["woe"],
                             "points": base - self.factor * b * r["woe"]})
        return pd.DataFrame(rows)

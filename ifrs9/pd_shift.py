"""Turn scenario charge-off paths into loan-level PD paths (IFRS 9 step between the satellite model and ECL).

The champion PDs from the Home Credit model are treated as through-the-cycle (TTC) 12-month PDs
that match the baseline scenario. For each future quarter q and scenario s the portfolio mean
12-month PD is scaled by the satellite model's relative multiplier

    m(s, q) = chargeoff(s, q) / chargeoff(baseline, q)

and every loan's PD is moved on the logit scale by one common shift delta(s, q), found by
bisection so the portfolio mean equals mean(PD) * m(s, q). A logit shift keeps PDs inside (0, 1),
keeps the rank order of loans and moves low-PD loans less in absolute terms than high-PD ones.

Each quarter's shifted PD is read as a one-year PD in force during that quarter; it is converted
to a quarterly hazard h = 1 - (1 - PD)^(1/4). The cumulative PD over k quarters is
1 - prod(1 - h_1..h_k). The 12-month PD at the reporting date uses k = 4 and the lifetime PD uses
the whole scenario horizon (12 quarters = 3 years, an assumption about remaining life stated in
the README: Home Credit loans are consumer instalment loans of that order of maturity).

Assumptions, stated plainly: the multiplier is estimated on US bank card charge-offs, not on the
Home Credit book; baseline equals the TTC PD; no prepayment or maturity amortisation (so lifetime
PD is an upper bound for loans shorter than 3 years).
"""
import numpy as np

QUARTERS_PER_YEAR = 4


def scenario_multipliers(paths, baseline="baseline"):
    """Ratio of each scenario's charge-off path to the baseline path (same index)."""
    base = paths[baseline]
    return {name: p / base for name, p in paths.items()}


def _shift(pd_vec, delta):
    x = np.log(pd_vec / (1.0 - pd_vec)) + delta
    return 1.0 / (1.0 + np.exp(-x))


def logit_shift_to_mean(pd_vec, target_mean, tol=1e-12, max_iter=200):
    """Common logit shift delta such that mean(sigmoid(logit(pd) + delta)) = target_mean."""
    pd_vec = np.clip(np.asarray(pd_vec, dtype=float), 1e-9, 1 - 1e-9)
    if not 0.0 < target_mean < 1.0:
        raise ValueError("target_mean must be in (0, 1)")
    lo, hi = -20.0, 20.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _shift(pd_vec, mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def shifted_pds(pd_vec, multiplier):
    """Loan PDs after the logit shift that scales the portfolio mean PD by `multiplier`."""
    pd_vec = np.clip(np.asarray(pd_vec, dtype=float), 1e-9, 1 - 1e-9)
    delta = logit_shift_to_mean(pd_vec, pd_vec.mean() * multiplier)
    return _shift(pd_vec, delta)


def hazard_matrix(pd_vec, multipliers):
    """(n_loans, n_quarters) quarterly default hazards for one scenario."""
    cols = [1.0 - (1.0 - shifted_pds(pd_vec, m)) ** (1.0 / QUARTERS_PER_YEAR) for m in multipliers]
    return np.column_stack(cols)


def cumulative_pd(hazards, quarters):
    """Cumulative PD over the first `quarters` quarters, per loan."""
    return 1.0 - np.prod(1.0 - hazards[:, :quarters], axis=1)


def pd_term_structure(pd_vec, mult_by_scenario):
    """{scenario: {'pd12': array, 'pd_life': array, 'hazards': matrix}} for the loan PDs."""
    out = {}
    for name, m in mult_by_scenario.items():
        h = hazard_matrix(pd_vec, np.asarray(m, dtype=float))
        out[name] = {"pd12": cumulative_pd(h, QUARTERS_PER_YEAR),
                     "pd_life": cumulative_pd(h, h.shape[1]), "hazards": h}
    return out

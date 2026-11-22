"""IFRS 9 expected credit loss: staging, discounted PD x LGD x EAD, probability-weighted scenarios.

Inputs come from `pd_shift.pd_term_structure` (quarterly hazards per loan and scenario).

* Stage 1: ECL over the next 4 quarters. Stage 2: lifetime ECL over the whole horizon (12 quarters).
  Stage 3 (credit-impaired): PD = 1, ECL = LGD x EAD. The scored book is treated as performing at the
  reporting date, so no loan is stage 3 unless an `impaired` flag is passed in; the realised default
  flag of the dataset is NOT used for staging (that would be look-ahead).
* SICR rule (significant increase in credit risk), a relative test with an absolute backstop: stage 2 if
  the probability-weighted lifetime PD is at least `ratio` (default 2.0) times the origination lifetime PD
  AND exceeds it by at least `abs_floor` (default 1 percentage point). Origination PD = baseline lifetime PD,
  i.e. the through-the-cycle PD the loan was priced on. A 30+ days-past-due backstop is not applicable:
  the application table has no payment history.
* ECL = sum over quarters q of marginal_PD(q) x LGD x EAD, discounted at the effective interest rate,
  with defaults assumed in the middle of the quarter. marginal_PD(q) = S(q-1) x hazard(q).
* Reported ECL is the probability-weighted average over scenarios, with staging decided on the weighted PD.
"""
import numpy as np

QUARTERS_PER_YEAR = 4
STAGE1_QUARTERS = 4


def marginal_default_prob(hazards):
    """P(default in quarter q) = survival to q-1 x hazard q; shape (n_loans, n_quarters)."""
    surv_before = np.cumprod(1.0 - hazards, axis=1) / (1.0 - hazards)
    return surv_before * hazards


def discount_factors(n_quarters, eir):
    """Mid-quarter discount factors at annual effective rate `eir`."""
    t = (np.arange(1, n_quarters + 1) - 0.5) / QUARTERS_PER_YEAR
    return (1.0 + eir) ** (-t)


def ecl_over_horizon(hazards, lgd, ead, eir, quarters):
    """Discounted ECL per loan over the first `quarters` quarters."""
    m = marginal_default_prob(hazards)[:, :quarters]
    return (m * discount_factors(quarters, eir)).sum(axis=1) * lgd * ead


def assign_stage(pd_life, pd_life_orig, ratio=2.0, abs_floor=0.01, impaired=None):
    """Stage 1/2/3 from the SICR rule; `impaired` (bool array) forces stage 3."""
    sicr = (pd_life >= ratio * pd_life_orig) & ((pd_life - pd_life_orig) >= abs_floor)
    stage = np.where(sicr, 2, 1)
    if impaired is not None:
        stage = np.where(np.asarray(impaired, dtype=bool), 3, stage)
    return stage


def check_weights(weights):
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("scenario weights must sum to 1")


def weighted_ecl(term_structure, weights, lgd, ead, eir, baseline="baseline", ratio=2.0, abs_floor=0.01,
                 impaired=None):
    """Per-loan reported ECL, stage and per-scenario ECL.

    Returns a dict with 'stage', 'ecl' (probability-weighted, stage-appropriate horizon),
    'ecl12' and 'ecl_life' (weighted over scenarios), and 'by_scenario' {name: (ecl12, ecl_life)}.
    """
    check_weights(weights)
    n_q = next(iter(term_structure.values()))["hazards"].shape[1]
    lgd = np.broadcast_to(np.asarray(lgd, dtype=float), np.shape(ead)) if np.ndim(lgd) else lgd
    by, e12, elife, pd_w = {}, 0.0, 0.0, 0.0
    for name, w in weights.items():
        h = term_structure[name]["hazards"]
        a = ecl_over_horizon(h, lgd, ead, eir, STAGE1_QUARTERS)
        b = ecl_over_horizon(h, lgd, ead, eir, n_q)
        by[name] = (a, b)
        e12, elife = e12 + w * a, elife + w * b
        pd_w = pd_w + w * term_structure[name]["pd_life"]
    stage = assign_stage(pd_w, term_structure[baseline]["pd_life"], ratio, abs_floor, impaired)
    ecl = np.where(stage == 1, e12, elife)
    if impaired is not None:
        ecl = np.where(stage == 3, np.asarray(lgd) * np.asarray(ead), ecl)
    return {"stage": stage, "ecl": ecl, "ecl12": e12, "ecl_life": elife, "by_scenario": by, "pd_life_weighted": pd_w}

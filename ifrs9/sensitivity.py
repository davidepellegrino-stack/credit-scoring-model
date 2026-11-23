"""Sensitivity of the IFRS 9 ECL to the macro path and to the policy parameters.

Two kinds of change, kept apart on purpose:

* macro: every scenario's unemployment path is moved up by `shift_pp` (ramped in over the first four
  quarters, then held), the satellite model is run again on the moved paths, and the PD shift and ECL are
  recomputed. The multiplier is measured against the ORIGINAL baseline charge-off path, so the baseline
  itself also gets worse, and the origination PD used by the SICR test stays the original one (a loan's
  origination PD does not move because the economy did).
* parameters: SICR ratio, LGD, discount rate and scenario weights only re-weigh or re-stage the same
  PD term structures, so they are cheap and exactly separable from the macro effect.
"""
import numpy as np
import pandas as pd

from ecl import weighted_ecl
from pd_shift import pd_term_structure
from satellite import simulate_path
from scenarios import SCENARIOS, build_scenarios

RAMP_QUARTERS = 4


def shift_unemployment(scenarios, shift_pp, ramp=RAMP_QUARTERS):
    """Copy of the scenario knots with unemployment raised by `shift_pp` (full effect from quarter `ramp`)."""
    out = {}
    for name, spec in scenarios.items():
        knots = spec["unrate_change"]
        hs = sorted(set(knots) | {ramp})
        old = np.interp(hs, sorted(knots), [knots[k] for k in sorted(knots)])
        new = {h: float(v + shift_pp * min(h / ramp, 1.0)) for h, v in zip(hs, old)}
        out[name] = {**spec, "unrate_change": new}
    return out


def term_structure_under(test_pd, res, panel, y0, scenarios, base_paths):
    """PD term structures after running `scenarios` through the satellite model.

    Multipliers are charge-off path / original baseline path (`base_paths['baseline']`).
    """
    last = panel["co_card"].dropna().index.max()
    macro = build_scenarios(panel, last, scenarios)
    paths = {n: simulate_path(res, y0, m) for n, m in macro.items()}
    mult = {n: (p / base_paths["baseline"]).to_numpy() for n, p in paths.items()}
    return pd_term_structure(test_pd, mult), paths


def ecl_row(ts, weights, lgd, ead, eir, ratio, pd_life_orig, label, **extra):
    """One sensitivity row: reported ECL (m and % of EAD), stage-2 loan share."""
    r = weighted_ecl(ts, weights, lgd, ead, eir, ratio=ratio, pd_life_orig=pd_life_orig)
    total = float(ead.sum())
    return {"case": label, "ecl_m": r["ecl"].sum() / 1e6, "ecl_pct_ead": 100 * r["ecl"].sum() / total,
            "ecl12_pct_ead": 100 * r["ecl12"].sum() / total, "stage2_pct_loans": 100 * float((r["stage"] == 2).mean()),
            **extra}


def sensitivity_table(test_pd, ead, res, panel, y0, base_paths, scenarios=SCENARIOS, weights=None, lgd=0.65, eir=0.15,
                      shifts=(0.0, 1.0, 2.0), ratios=(1.25, 1.5, 2.0), lgds=(0.55, 0.75), eirs=(0.10, 0.20),
                      weight_sets=None):
    """DataFrame of ECL under each one-at-a-time change from the base case (base = first row)."""
    weights = weights or {"baseline": 0.5, "adverse": 0.3, "severely_adverse": 0.2}
    weight_sets = weight_sets or {
        "weights 70/20/10": {"baseline": 0.7, "adverse": 0.2, "severely_adverse": 0.1},
        "weights 30/40/30": {"baseline": 0.3, "adverse": 0.4, "severely_adverse": 0.3},
        "weights 100/0/0": {"baseline": 1.0, "adverse": 0.0, "severely_adverse": 0.0}}
    ts0, _ = term_structure_under(test_pd, res, panel, y0, scenarios, base_paths)
    orig = ts0["baseline"]["pd_life"]
    rows = [ecl_row(ts0, weights, lgd, ead, eir, 2.0, orig, "base case", group="base")]
    for s in shifts:
        if s == 0:
            continue
        ts, _ = term_structure_under(test_pd, res, panel, y0, shift_unemployment(scenarios, s), base_paths)
        rows.append(ecl_row(ts, weights, lgd, ead, eir, 2.0, orig, f"unemployment +{s:g} pp", group="macro"))
    for k in ratios:
        if k != 2.0:
            rows.append(ecl_row(ts0, weights, lgd, ead, eir, k, orig, f"SICR ratio {k:g}x", group="SICR"))
    for g in lgds:
        rows.append(ecl_row(ts0, weights, g, ead, eir, 2.0, orig, f"LGD {g:.0%}", group="LGD"))
    for e in eirs:
        rows.append(ecl_row(ts0, weights, lgd, ead, e, 2.0, orig, f"discount rate {e:.0%}", group="discount"))
    for label, w in weight_sets.items():
        rows.append(ecl_row(ts0, w, lgd, ead, eir, 2.0, orig, label, group="weights"))
    t = pd.DataFrame(rows).set_index("case")
    t["change_vs_base_%"] = 100 * (t["ecl_m"] / t.loc["base case", "ecl_m"] - 1.0)
    return t

"""Probability-weighted IFRS 9 ECL on the Home Credit test loans under the three macro scenarios.

Assumptions (all stated, all changeable below): EAD = AMT_CREDIT (no amortisation), LGD 65% for unsecured
consumer credit, discounting at 15% a year, scenario weights 50 / 30 / 20, SICR = lifetime PD at least
2x origination and +1 pp. Prints the ECL table by stage and scenario and writes outputs/tables/ecl_summary.csv and ecl_by_scenario.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from config import OUTPUT_TABLES  # noqa: E402
from data_home_credit import ID_COL, load_application  # noqa: E402

from ecl import assign_stage, weighted_ecl  # noqa: E402
from run_pd_shift import scenario_pd_table  # noqa: E402

LGD = 0.65
EIR = 0.15
WEIGHTS = {"baseline": 0.5, "adverse": 0.3, "severely_adverse": 0.2}


def run(lgd=LGD, eir=EIR, weights=WEIGHTS):
    test, _, ts = scenario_pd_table()
    app = load_application("application_train").set_index(ID_COL)
    ead = app.loc[test[ID_COL], "AMT_CREDIT"].to_numpy(dtype=float)
    return test, ead, weighted_ecl(ts, weights, lgd, ead, eir), ts


if __name__ == "__main__":
    test, ead, r, ts = run()
    total_ead = ead.sum()
    stage = pd.Series(r["stage"], name="stage")
    t = pd.DataFrame({"stage": stage, "ead": ead, "ecl": r["ecl"]}).groupby("stage").agg(
        loans=("ead", "size"), ead=("ead", "sum"), ecl=("ecl", "sum"))
    t["coverage_%"] = 100 * t["ecl"] / t["ead"]
    t.loc["total"] = [t["loans"].sum(), t["ead"].sum(), t["ecl"].sum(), 100 * t["ecl"].sum() / t["ead"].sum()]
    print(f"{len(ead):,} loans, EAD {total_ead / 1e9:.2f} bn (AMT_CREDIT), LGD {LGD:.0%}, EIR {EIR:.0%}, "
          f"weights {WEIGHTS}")
    print("\nWeighted ECL by stage:")
    print(t.assign(ead=t["ead"] / 1e6, ecl=t["ecl"] / 1e6).rename(columns={"ead": "ead_m", "ecl": "ecl_m"})
          .round(2).to_string())
    rows = {n: {"12m": 100 * e12.sum() / total_ead, "lifetime": 100 * el.sum() / total_ead}
            for n, (e12, el) in r["by_scenario"].items()}
    rows["weighted"] = {"12m": 100 * r["ecl12"].sum() / total_ead, "lifetime": 100 * r["ecl_life"].sum() / total_ead}
    print("\nECL as % of EAD if every loan were measured on one horizon:")
    by_scenario = pd.DataFrame(rows).T.rename_axis("scenario")
    print(by_scenario.round(2).to_string())
    by_scenario.round(4).to_csv(os.path.join(OUTPUT_TABLES, "ecl_by_scenario.csv"))
    print(f"\nReported (stage-appropriate, weighted) ECL: {r['ecl'].sum() / 1e6:,.0f} m = "
          f"{100 * r['ecl'].sum() / total_ead:.2f}% of EAD")
    orig = ts["baseline"]["pd_life"]
    views = {"weighted": r["pd_life_weighted"], "adverse only": ts["adverse"]["pd_life"],
             "severe only": ts["severely_adverse"]["pd_life"]}
    print("\nShare of loans in stage 2 by SICR ratio (abs floor 1 pp) and scenario view:")
    print(pd.DataFrame({v: {f"{k}x": float((assign_stage(x, orig, ratio=k) == 2).mean() * 100)
                           for k in (1.25, 1.5, 2.0)} for v, x in views.items()}).round(1).to_string())
    print(f"\nRealised default rate {test['default'].mean():.2%}; baseline 12-month ECL "
          f"{100 * r['by_scenario']['baseline'][0].sum() / total_ead:.2f}% of EAD")
    out = t.copy()
    out.to_csv(os.path.join(OUTPUT_TABLES, "ecl_summary.csv"))

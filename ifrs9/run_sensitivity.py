"""ECL sensitivity table: unemployment +1 / +2 pp, SICR ratio, LGD, discount rate, scenario weights.

Needs the cached champion test PDs (run `ifrs9/run_pd_shift.py` once). Prints the table and writes
outputs/tables/ecl_sensitivity.csv. Each row changes one assumption from the base case.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from config import OUTPUT_TABLES  # noqa: E402
from data_home_credit import ID_COL, load_application  # noqa: E402

from fred import load_macro_panel  # noqa: E402
from run_pd_shift import champion_test_pds  # noqa: E402
from satellite import build_dataset, fit, simulate_path  # noqa: E402
from scenarios import build_scenarios  # noqa: E402
from sensitivity import sensitivity_table  # noqa: E402


def main():
    test = champion_test_pds()
    app = load_application("application_train").set_index(ID_COL)
    ead = app.loc[test[ID_COL], "AMT_CREDIT"].to_numpy(dtype=float)
    panel = load_macro_panel()
    res = fit(build_dataset(panel))
    last = panel["co_card"].dropna().index.max()
    y0 = float(panel.loc[last, "co_card"])
    base_paths = {n: simulate_path(res, y0, m) for n, m in build_scenarios(panel, last).items()}
    t = sensitivity_table(test["pd"].to_numpy(), ead, res, panel, y0, base_paths)
    print(f"{len(ead):,} loans, EAD {ead.sum() / 1e9:.2f} bn; base: LGD 65%, discount 15%, weights 50/30/20, SICR 2x")
    print(t.drop(columns="group").round(2).to_string())
    t.to_csv(os.path.join(OUTPUT_TABLES, "ecl_sensitivity.csv"))


if __name__ == "__main__":
    main()

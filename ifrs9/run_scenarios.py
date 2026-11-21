"""Build the three macro scenarios, run them through the satellite model and print / plot the charge-off paths."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from fred import load_macro_panel  # noqa: E402
from satellite import build_dataset, fit, simulate_path  # noqa: E402
from scenarios import build_scenarios  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "figures")

if __name__ == "__main__":
    panel = load_macro_panel()
    res = fit(build_dataset(panel))
    last = panel["co_card"].dropna().index.max()
    y0 = float(panel.loc[last, "co_card"])
    scen = build_scenarios(panel, last)
    paths = {n: simulate_path(res, y0, m) for n, m in scen.items()}
    table = pd.DataFrame(paths)
    print(f"Last observed card charge-off rate {last}: {y0:.2f}%")
    for n, m in scen.items():
        print(f"\n{n}: peak unemployment {m['unrate'].max():.1f}%, trough GDP growth {m['gdp_growth_yoy'].min():.1f}%, "
              f"max d_unrate_4q {m['d_unrate_4q'].max():.2f} pp")
    print("\nCard charge-off rate by scenario (%):")
    print(table.round(2).to_string())
    print("\nPeak and 3-year average:")
    print(pd.DataFrame({"peak": table.max(), "peak_quarter": table.idxmax().astype(str), "mean_12q": table.mean()}).round(2))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    hist = panel["co_card"].dropna().loc["2015Q1":]
    ax.plot(hist.index.to_timestamp(), hist.values, color="black", label="Observed")
    for n, s in paths.items():
        ax.plot(s.index.to_timestamp(), s.values, label=n.replace("_", " "))
    ax.set_ylabel("Card charge-off rate, %")
    ax.set_title("Satellite model: card charge-offs under macro scenarios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scenario_chargeoffs.png"), dpi=120)
    print("\nFigure written to outputs/figures/scenario_chargeoffs.png")

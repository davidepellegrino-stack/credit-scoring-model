"""README figure: 12-month vs lifetime ECL by scenario as a small SVG (standard library only).

Usage: python ifrs9/plot_ecl.py [ecl_by_scenario.csv] [out.svg]   (the CSV is written by run_ecl.py)
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES = (("12m", "12-month ECL", "#1f4e79"), ("lifetime", "3-year (lifetime) ECL", "#9dbbd9"))
LABELS = {"baseline": "Baseline", "adverse": "Adverse", "severely_adverse": "Severely adverse",
          "weighted": "Weighted 50/30/20"}


def read_rows(path):
    with open(path, newline="") as f:
        return {r["scenario"]: {k: float(r[k]) for k, _, _ in SERIES} for r in csv.DictReader(f)}


def render_svg(rows, width=800, height=420, base=360, plot_h=260):
    """Grouped bars (% of EAD) with a zero baseline; bar heights are proportional to the values."""
    top = max(v for r in rows.values() for v in r.values())
    if top <= 0:
        raise ValueError("ECL values must be positive")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           'font-family="Helvetica,Arial,sans-serif" font-size="13" text-anchor="middle">',
           f'<rect width="{width}" height="{height}" fill="#fff"/>',
           f'<text x="{width / 2:.0f}" y="28" font-size="17" font-weight="bold">'
           'IFRS 9 expected credit loss by macro scenario (% of EAD)</text>']
    for i, (_, label, colour) in enumerate(SERIES):
        x = 230 + 200 * i
        out.append(f'<rect x="{x}" y="46" width="14" height="14" fill="{colour}"/>'
                   f'<text x="{x + 20}" y="58" text-anchor="start">{label}</text>')
    slot = (width - 60) / len(rows)
    for j, (name, values) in enumerate(rows.items()):
        mid = 30 + slot * (j + 0.5)
        for i, (key, _, colour) in enumerate(SERIES):
            h = plot_h * values[key] / top
            x = mid - 62 + 64 * i
            out.append(f'<rect x="{x:.1f}" y="{base - h:.1f}" width="60" height="{h:.1f}" fill="{colour}"/>'
                       f'<text x="{x + 30:.1f}" y="{base - h - 6:.1f}">{values[key]:.2f}%</text>')
        out.append(f'<text x="{mid:.1f}" y="{base + 22}">{LABELS.get(name, name)}</text>')
    out.append(f'<line x1="30" y1="{base}" x2="{width - 30}" y2="{base}" stroke="#333"/>')
    out.append(f'<text x="{width / 2:.0f}" y="{height - 12}" font-size="11" fill="#555">'
               'Home Credit test loans; EAD, LGD and discount rate are stated assumptions (see README)</text>')
    return "\n".join(out + ["</svg>"]) + "\n"


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "outputs", "tables", "ecl_by_scenario.csv")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "assets", "ecl_by_scenario.svg")
    with open(dst, "w") as f:
        f.write(render_svg(read_rows(src)))
    print(f"wrote {dst}")

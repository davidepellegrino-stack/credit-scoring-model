import os
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ifrs9"))
from plot_ecl import read_rows, render_svg  # noqa: E402

ROWS = {"baseline": {"12m": 4.0, "lifetime": 10.0}, "adverse": {"12m": 5.0, "lifetime": 20.0}}
NS = "{http://www.w3.org/2000/svg}"


def test_bar_heights_are_proportional_to_ecl():
    root = ET.fromstring(render_svg(ROWS))
    h = [float(r.get("height")) for r in root.iter(f"{NS}rect") if r.get("width") == "60"]
    assert h == pytest.approx([52.0, 130.0, 65.0, 260.0])


def test_labels_and_values_are_written():
    text = render_svg(ROWS)
    assert "Adverse" in text and "20.00%" in text


def test_csv_round_trip(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("scenario,12m,lifetime\nbaseline,4.0,10.0\nadverse,5.0,20.0\n")
    assert read_rows(p) == ROWS


def test_rejects_non_positive_values():
    with pytest.raises(ValueError):
        render_svg({"baseline": {"12m": 0.0, "lifetime": 0.0}})

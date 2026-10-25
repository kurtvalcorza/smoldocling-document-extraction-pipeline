"""Prose numbers stay true across every recorded hosted run (Kurt, 2026-10-25: point to output + ranges).

Recorded runs: Kaggle T4 `aaf8fbf0` (2026-09-21, adapted test CER 0.782), Colab T4 `e3d9e0a5`
(2026-10-11, 0.782) and Colab T4 `5b9a54dc` (2026-10-25, 0.781). Timings are approximate.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "smoldocling_document_extraction_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_adapted_cer_is_a_range_not_one_run():
    md = _markdown()
    assert not re.search(r"(?<![–\d])0\.782(?!\d)", md.replace("was 0.782 in the first two", ""))
    assert md.count("0.781–0.782") >= 3


def test_recorded_runs_include_the_2026_10_25_run():
    md = _markdown()
    assert "`5b9a54dc` and `282a5afd` on 25 October 2026" in md and "`5b9a54dc` and `282a5afd`, 2026-10-25" in md
    assert "neither re-run for this revision" not in md


def test_timings_are_records_of_named_runs():
    md = _markdown()
    for exact in ("1112.3 s", "1084 s", "157 s", "337 s"):
        assert exact not in md, exact
    assert "1,103.7 s to 1,129.1 s" in md and "Colab Tesla T4" in md

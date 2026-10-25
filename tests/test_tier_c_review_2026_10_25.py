"""Tier C review findings T83-1..T83-5 (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "smoldocling_document_extraction_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    return "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")


def test_t83_1_cer_is_not_inverted():
    md = _markdown()
    assert "four characters in five" not in md and "22 % of the characters right" not in md
    assert "78 % of the reference characters" in md


def test_t83_2_dat9_overlap_statement():
    md = _markdown()
    assert "**Overlap with the training data:**" in md and "inferred to be unlikely, not verified" in md
    assert "has never seen cursive" not in md and "handwriting it was never trained on" not in md


def test_t83_3_no_cross_row_ranking():
    md = _markdown()
    assert "lands between GOT-OCR and Florence-2" not in md
    assert "0.759" not in md and "0.797" not in md and "are not a ranking" in md


def test_t83_4_numbers_name_their_run():
    md = _markdown()
    assert "build record" not in md
    assert "about 25 minutes" in md and "`aaf8fbf0`, 2026-09-21" in md


def test_t83_5_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown()
    assert "[1] A. Nassar" in md and "`DIMER_ISOLATED_ENV`" in md and "NOTEBOOK_SPEC 2.0" not in md

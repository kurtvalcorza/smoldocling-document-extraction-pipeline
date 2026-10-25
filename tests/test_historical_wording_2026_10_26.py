"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Runs: Kaggle T4 of blob aaf8fbf0 (2026-09-21); Colab T4 of blobs e3d9e0a5 (2026-10-11),
5b9a54dc and 282a5afd (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "smoldocling_document_extraction_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_expectations_and_run_status_sentences_are_gone():
    md = _markdown()
    for stale in ("the default path takes about 25 minutes", "none re-run for this revision",
                  "Expect the frozen model", "expect about 20 minutes on a T4", "No experiment run is recorded yet",
                  "the adapted rows should"):
        assert stale not in md, stale


def test_confirming_run_of_282a5afd_is_recorded():
    md = _markdown()
    assert "`5b9a54dc` and `282a5afd`, 2026-10-25" in md
    assert "1,633.5 s, 1,590.8 s and 1,618.9 s" in md

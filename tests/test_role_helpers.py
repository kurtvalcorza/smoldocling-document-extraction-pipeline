"""Offline tests for the public validation, DocTags and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from smoldocling_document_extraction_pipeline import (
    DEFAULT_INSTRUCTION,
    DEFAULT_MAX_NEW_TOKENS,
    INPUT_SCHEMA,
    INSTRUCTIONS,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    doctags_summary,
    doctags_to_text,
    evaluation_report,
    validate_inputs,
    word_error_rate,
)

DOCTAGS = (
    "<doctag><section_header_level_1><loc_32><loc_25><loc_231><loc_39>Report</section_header_level_1>\n"
    "<text><loc_32><loc_40><loc_319><loc_93>quarterly revenue by region</text>\n"
    "<text><loc_32><loc_100><loc_319><loc_150>notes on methodology</text>\n"
    "<otsl><loc_32><loc_151><loc_459><loc_303><ched>Region<ched>Q1<nl><fcel>North 1<fcel>21,132<nl></otsl>\n"
    "</doctag>"
)


def _image(width: int = 850, height: int = 1100) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _result(doctags: str = DOCTAGS, truncated: bool = False) -> dict:
    return {"doctags": doctags, "instruction": DEFAULT_INSTRUCTION, "truncated": truncated}


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), names=["page.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["instructions"] == list(INSTRUCTIONS)
    assert manifest["inputs"] == [{"id": "page.png", "mode": "RGB", "size": [850, 1100]}]
    assert manifest["instruction"] == DEFAULT_INSTRUCTION
    assert manifest["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy",
    }
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_request() -> None:
    manifest = validate_inputs(_image(), instruction=INSTRUCTIONS[4], max_new_tokens=512)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert (
        manifest["instruction"] == "Convert table to OTSL."
        and manifest["generation"]["max_new_tokens"] == 512
    )


def test_validate_inputs_rejects_like_convert() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(_image(MAX_IMAGE_SIDE + 1, 64))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8))
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="INSTRUCTIONS"):
        validate_inputs(_image(), instruction="Summarise this page.")
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_inputs(_image(), max_new_tokens=0)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_image(), names=["a", "b"])


def test_doctags_to_text_strips_markup_and_locations() -> None:
    assert (
        doctags_to_text(DOCTAGS)
        == "Report quarterly revenue by region notes on methodology Region Q1 North 1 21,132"
    )
    assert doctags_to_text("<text><loc_1><loc_2><loc_3><loc_4>  spaced   out </text>") == "spaced out"


def test_doctags_summary_counts_elements_and_cells() -> None:
    summary = doctags_summary(DOCTAGS)
    assert summary["counts"]["section_header_level_1"] == 1
    assert summary["counts"]["text"] == 2 and summary["counts"]["otsl"] == 1
    assert summary["n_elements"] == 4 and summary["n_loc_tokens"] == 16
    assert summary["n_table_cells"] == 6  # 2 ched + 2 fcel + 2 nl
    assert summary["wrapped_in_doctag"] is True
    assert doctags_summary("<text>loose</text>")["wrapped_in_doctag"] is False


def test_word_error_rate_counts_word_edits() -> None:
    assert word_error_rate("a b c", "a b c") == 0.0
    assert word_error_rate("a b c d", "a x c") == pytest.approx(0.5)
    assert word_error_rate("Hello World", "hello world") == 0.0
    with pytest.raises(ValueError, match="at least one word"):
        word_error_rate("   ", "x")


def test_evaluation_report_not_measurable_without_references() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["doctags_summary"]["counts"]["otsl"] == 1
    assert "TEDS" in report["needs"] and "word_error_rate" in report["needs"]
    assert report["baselines"] == []
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_reference_text_and_counts() -> None:
    reference = "Report quarterly revenue by region notes on methodology Region Q1 North 1 21,132"
    report = evaluation_report(
        _result(), reference, {"text": 2, "otsl": 1, "picture": 0}, sample_kind="synthetic"
    )
    assert report["verdict"] == "sample-sanity" and report["sample_kind"] == "synthetic"
    by_id = {(m["id"], m.get("tag")): m for m in report["metrics"]}
    assert by_id[("word_error_rate", None)]["value"] == 0.0
    assert by_id[("element_count", "text")] == {
        "id": "element_count",
        "tag": "text",
        "expected": 2,
        "observed": 2,
        "estimation": "one page, structural sanity only",
    }
    assert by_id[("element_count", "picture")]["observed"] == 0
    with pytest.raises(ValueError, match="unknown element tag"):
        evaluation_report(_result(), None, {"table": 1})


def test_evaluation_report_carries_truncation_flag() -> None:
    assert evaluation_report(_result(truncated=True))["truncated"] is True

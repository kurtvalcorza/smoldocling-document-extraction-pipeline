"""Document-to-DocTags conversion with the pinned ``docling-project/SmolDocling-256M-preview`` checkpoint.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the Idefics3 architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "docling-project/SmolDocling-256M-preview"
MODEL_REVISION = "ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8"
MODEL_LICENSE = "cdla-permissive-2.0"
MODEL_KEY = "smoldocling-256m-preview"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# The instructions the pinned README's "Supported Instructions" table lists. Any other instruction
# is refused: the model was trained on these forms and a paraphrase is undefined behaviour.
INSTRUCTIONS = (
    "Convert this page to docling.",
    "Convert chart to table.",
    "Convert formula to LaTeX.",
    "Convert code to text.",
    "Convert table to OTSL.",
    "Find all 'text' elements on the page, retrieve all section headers.",
    "Detect footer elements on the page.",
)
DEFAULT_INSTRUCTION = INSTRUCTIONS[0]
# Generation ceilings. 8192 is the max_new_tokens the pinned README's transformers example passes and
# the text model's max_position_embeddings (config.json); the default is a practical page budget.
MAX_NEW_TOKENS = 8192
DEFAULT_MAX_NEW_TOKENS = 2048
DECODING = "greedy"
# Input ceilings. The processor resizes so the longest edge is 2048 px and splits the page into
# 512-px tiles of 64 visual tokens each plus one global view (preprocessor_config.json), so image
# cost is bounded; the side ceiling only guards memory during decoding and resizing.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
# Tokens the decoder emits around the answer; stripped from the returned DocTags (the README example
# decodes with skip_special_tokens=False so the DocTags markup survives, then removes these).
_TERMINATORS = ("<end_of_utterance>", "<|im_end|>")
# DocTags element tags counted by doctags_summary (added_tokens.json names them; the loc grid is
# 0..500 per axis, four <loc_N> tokens per element box).
_ELEMENT_TAGS = (
    "section_header_level_1",
    "section_header_level_2",
    "section_header_level_3",
    "text",
    "paragraph",
    "list_item",
    "ordered_list",
    "unordered_list",
    "otsl",
    "picture",
    "caption",
    "formula",
    "code",
    "page_header",
    "page_footer",
    "footnote",
    "chart",
    "key_value_region",
)
_TAG_RE = re.compile(r"</?([a-z_]+(?:_[0-9]+)?)>")
_LOC_RE = re.compile(r"<loc_[0-9]+>")
_OTSL_CELL_RE = re.compile(r"<(?:fcel|ecel|ched|rhed|srow|lcel|ucel|xcel|nl)>")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def build_messages(instruction: str) -> list[dict[str, Any]]:
    """One user turn: an image placeholder then the instruction, in the snapshot chat-template shape."""
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]


def doctags_to_text(doctags: str) -> str:
    """Plain text carried by a DocTags string: tags and <loc_N> tokens removed, whitespace collapsed.

    OTSL table cells become space-separated words; structure is lost. This is the text a caller would
    compare with an OCR reference, not a document export (use docling_core for that).
    """
    text = _LOC_RE.sub(" ", doctags)
    text = _OTSL_CELL_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return " ".join(text.split())


def doctags_summary(doctags: str) -> dict[str, Any]:
    """Count the DocTags elements in a conversion: which structure the model claims the page has.

    Counts opening tags per element type, the number of <loc_N> tokens (four per located element),
    whether the string is wrapped in <doctag>…</doctag>, and how many OTSL table cells appear.
    """
    opened = Counter(
        match.group(1) for match in _TAG_RE.finditer(doctags) if not match.group(0).startswith("</")
    )
    counts = {tag: opened.get(tag, 0) for tag in _ELEMENT_TAGS}
    return {
        "counts": counts,
        "n_elements": sum(counts.values()),
        "n_loc_tokens": len(_LOC_RE.findall(doctags)),
        "n_table_cells": len(_OTSL_CELL_RE.findall(doctags)),
        "wrapped_in_doctag": doctags.lstrip().startswith("<doctag>")
        and doctags.rstrip().endswith("</doctag>"),
        "n_chars": len(doctags),
    }


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate of ``hypothesis`` against ``reference`` after lower-casing and whitespace tokenisation.

    Levenshtein edits over words divided by reference words; punctuation is **not** stripped, so a
    stray comma counts. The metric a caller would use to score ``doctags_to_text`` against a known page.
    """
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    if not ref:
        raise ValueError("reference must contain at least one word")
    previous = list(range(len(hyp) + 1))
    for row_index, ref_token in enumerate(ref, 1):
        current = [row_index]
        for column_index, hyp_token in enumerate(hyp, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1] / len(ref)


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one page image as PIL.Image.Image (any mode, converted to RGB) plus one supported instruction",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "instructions": list(INSTRUCTIONS),
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "decoding": f"{DECODING} (do_sample=False), deterministic on a fixed device and dtype",
    "preprocessing": (
        "image converted to RGB; the processor resizes so the longest edge is 2048 px (aspect ratio "
        "preserved) and splits it into 512-px tiles of 64 visual tokens each plus one global view; the "
        "instruction is wrapped in the snapshot's chat template as one user turn (see build_messages)"
    ),
    "output": "DocTags markup (docling_core-compatible), decoded without dropping the tag tokens",
}


def _check_inputs(image: Any, instruction: Any, max_new_tokens: Any) -> tuple[Image.Image, str, int]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``convert`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    if not isinstance(instruction, str):
        raise TypeError("instruction must be a str")
    if instruction not in INSTRUCTIONS:
        raise ValueError(f"instruction {instruction!r} is not one of the supported INSTRUCTIONS")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an int")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    return rgb, instruction, max_new_tokens


def validate_inputs(
    image: Image.Image,
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``convert`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, checked_instruction, checked_tokens = _check_inputs(image, instruction, max_new_tokens)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (convert takes one page image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "instruction": checked_instruction,
        "generation": {"max_new_tokens": checked_tokens, "do_sample": False, "decoding": DECODING},
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    reference_text: str | None = None,
    expected_counts: Mapping[str, int] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``reference_text`` (the words the page really carries) the report carries ``word_error_rate``
    of ``doctags_to_text`` against it; with ``expected_counts`` (element tag -> expected number) it
    carries one ``element_count`` entry per tag comparing expected and observed. Either makes the
    verdict ``sample-sanity``; without both it is ``not-measurable`` and the report says what labelled
    data would make the task measurable.
    """
    doctags = str(result["doctags"])
    summary = doctags_summary(doctags)
    base = {
        "task": "document page image -> DocTags (layout, reading order, OCR, tables)",
        "score_semantics": (
            "generated markup carries no score, no probability and no correctness signal; well-formed "
            "tags are not evidence that the text or layout is right. Greedy decoding makes the output "
            "reproducible on a fixed device and dtype, which is a reproducibility property, not a quality one"
        ),
        "instruction": result.get("instruction"),
        "sample_kind": sample_kind,
        "doctags_summary": summary,
        "truncated": result.get("truncated"),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    metrics: list[dict[str, Any]] = []
    if reference_text:
        metrics.append(
            {
                "id": "word_error_rate",
                "value": word_error_rate(reference_text, doctags_to_text(doctags)),
                "normalisation": (
                    "lower-cased, whitespace-tokenised, tags and <loc_N> removed; punctuation kept"
                ),
                "estimation": "one page, no dispersion estimate",
            }
        )
    for tag, expected in (expected_counts or {}).items():
        if tag not in _ELEMENT_TAGS:
            raise ValueError(f"unknown element tag {tag!r}; expected one of {_ELEMENT_TAGS}")
        metrics.append(
            {
                "id": "element_count",
                "tag": tag,
                "expected": int(expected),
                "observed": summary["counts"][tag],
                "estimation": "one page, structural sanity only",
            }
        )
    if not metrics:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no reference text or expected element counts were supplied for the evaluated page",
            "needs": (
                "pages with ground-truth text and layout (for example DocLayNet-style annotations or the "
                "publisher's source) scored with word_error_rate on the OCR and with layout/table metrics "
                "such as TEDS on the structure; no such labelled set ships with this repository"
            ),
        }
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(metrics)} sanity measure(s) on one tutorial page whose text and layout you rendered "
            "yourself; plumbing evidence, not a document-conversion benchmark"
        ),
        "needs": (
            "a labelled page set from the deployment domain (scans, publishers, layouts, tables) for any "
            "OCR accuracy, layout or table-structure claim"
        ),
    }


@dataclass
class SmolDoclingPipeline:
    """``_runner(image, instruction, max_new_tokens)`` returns ``{"doctags": str, "new_tokens": int}``."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> SmolDoclingPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        common: dict[str, Any] = {"trust_remote_code": False}
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, common["local_files_only"], source = str(root), True, "local-snapshot"
        elif allow_download:
            location, common["revision"], source = MODEL_ID, MODEL_REVISION, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
        processor = AutoProcessor.from_pretrained(location, **common)
        model = AutoModelForImageTextToText.from_pretrained(location, dtype=dtype, **common)
        model = model.eval().to(resolved_device)

        def runner(image: Image.Image, instruction: str, max_new_tokens: int) -> dict[str, Any]:
            text = processor.apply_chat_template(build_messages(instruction), add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            new_ids = generated[0, inputs["input_ids"].shape[1] :]
            # skip_special_tokens=False keeps the DocTags markup (the tags are added tokens); the
            # terminator tokens are removed afterwards, as the pinned README's example does.
            decoded = processor.batch_decode(new_ids.unsqueeze(0), skip_special_tokens=False)[0]
            return {"doctags": decoded, "new_tokens": int(new_ids.shape[0])}

        return cls(runner, resolved_device, str(dtype).removeprefix("torch."), source)

    def convert(
        self,
        image: Image.Image,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """Run one supported instruction over one page image; ``doctags`` is the decoded markup."""
        rgb, checked_instruction, checked_tokens = _check_inputs(image, instruction, max_new_tokens)
        raw = self._runner(rgb, checked_instruction, checked_tokens)
        if not isinstance(raw, dict) or "doctags" not in raw:
            raise RuntimeError("runner must return a dict with 'doctags'")
        doctags = str(raw["doctags"])
        for terminator in _TERMINATORS:
            doctags = doctags.replace(terminator, "")
        doctags = doctags.strip()
        new_tokens = int(raw.get("new_tokens", 0))
        return {
            "doctags": doctags,
            "text": doctags_to_text(doctags),
            "instruction": checked_instruction,
            "image_size": list(rgb.size),
            "new_tokens": new_tokens,
            "truncated": new_tokens >= checked_tokens,
            "generation": {"max_new_tokens": checked_tokens, "do_sample": False, "decoding": DECODING},
            "device": self.device,
            "dtype": self.dtype,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

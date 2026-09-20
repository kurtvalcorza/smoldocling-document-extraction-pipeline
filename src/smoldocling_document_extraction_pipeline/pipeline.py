"""Document-to-DocTags conversion with the pinned ``docling-project/SmolDocling-256M-preview`` checkpoint, plus the
adaptation contract for one instruction: corpus evaluation of the text a DocTags conversion carries against
transcribed lines, bounded fine-tuning of the last decoder layers on cached prefix hidden states with a DocTags line
target, and a verified adapter artifact.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the Idefics3 architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed.
"""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
# cost is bounded; the side and area ceilings only guard memory during decoding and resizing (a
# 9,000 px wide text line is accepted).
MAX_IMAGE_SIDE = 16_384
MAX_IMAGE_PIXELS = 4096 * 4096
MIN_IMAGE_SIDE = 16
WEIGHTS_FILE = "model.safetensors"
END_OF_UTTERANCE = "<end_of_utterance>"  # the assistant turn's terminator in the snapshot chat template

# Adaptation contract: the last `TRAINABLE_LAYERS` of the 30 SmolLM2 decoder layers and the final norm are the
# adapter; the SigLIP vision encoder, the connector, the embeddings, the output head and the earlier decoder
# layers stay frozen, so their output for every training line is computed once and cached. The target of one
# transcribed line is the DocTags the model itself emits for a single text element that fills the image.
PARAMETER_COUNT = 256_484_928
DECODER_LAYERS = 30
TRAINABLE_LAYERS = 8
ADAPTER_PARAMETERS = 28_321_344
DEFAULT_LINE_MAX_NEW_TOKENS = 160  # the corpus stages' budget per text line (the DocTags wrapper is 25 tokens)
LINE_DOCTAGS = "<doctag><text><loc_0><loc_0><loc_500><loc_500>{text}</text>\n</doctag>"
_TRAINABLE_FIRST_LAYER = DECODER_LAYERS - TRAINABLE_LAYERS
_TRAINABLE_PREFIXES = tuple(f"model.text_model.layers.{i}." for i in range(_TRAINABLE_FIRST_LAYER, DECODER_LAYERS)) + ("model.text_model.norm.",)
ARTIFACT_FORMAT = f"org.valcorza.{MODEL_KEY}.adapter.v1"
ARTIFACT_VERSION = 1
ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_MANIFEST = "manifest.json"
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
MAX_EVAL_RECORDS = 5_000
EVAL_BATCH_SIZE = 8
CACHE_BATCH_SIZE = 4
GRAD_CLIP = 1.0
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


def _weight_digest(root: Path) -> str | None:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as handle:
        entries = json.load(handle).get("files", [])
    return next((e["sha256"] for e in entries if e["path"] == WEIGHTS_FILE), None)


def normalise_text(text: str) -> str:
    """The transcript form the corpus measures use: whitespace runs collapsed to one space, ends stripped."""
    return " ".join(str(text).split())


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance (insertions + deletions + substitutions, unit cost) between two sequences."""
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_item in enumerate(reference, 1):
        current = [row_index]
        for column_index, hyp_item in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column_index] + 1, previous[column_index - 1] + (ref_item != hyp_item)))
        previous = current
    return previous[-1]


def line_doctags(text: str) -> str:
    """The DocTags a transcribed line is trained to produce: one `<text>` element whose box fills the image."""
    return LINE_DOCTAGS.format(text=normalise_text(text))


def _trainable_names(model: Any) -> list[str]:
    """The last `TRAINABLE_LAYERS` text-decoder layers and the final norm; the vision encoder, the connector, the
    embeddings, the output head and the earlier decoder layers stay frozen."""
    return [name for name, _ in model.named_parameters() if name.startswith(_TRAINABLE_PREFIXES)]


def _check_artifact_manifest(manifest: Mapping[str, Any], artifact_dir: Path, base_sha256: str) -> None:
    """Refuse an adapter that names another base, another format or a file that does not match its digest."""
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    base = manifest.get("base", {})
    if base.get("model_id") != MODEL_ID or base.get("revision") != MODEL_REVISION:
        raise ValueError(f"artifact was trained on {base.get('model_id')}@{base.get('revision')}, not {MODEL_ID}@{MODEL_REVISION}")
    if base.get("weight_sha256") != base_sha256:
        raise ValueError("artifact base weight digest does not match the verified snapshot")
    files = manifest.get("files") or []
    if len(files) != 1 or files[0].get("path") != ADAPTER_WEIGHTS:
        raise ValueError(f"artifact manifest must list exactly {ADAPTER_WEIGHTS}")
    weights = artifact_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"artifact weights missing: {weights}")
    size = weights.stat().st_size
    if size != files[0].get("bytes"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: size {size} != manifest {files[0].get('bytes')}")
    digest = _sha256(weights)
    if digest != files[0].get("sha256"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: sha256 {digest} != manifest {files[0].get('sha256')}")
    names = manifest.get("tensors") or []
    if not names or any(not str(n).startswith(_TRAINABLE_PREFIXES) for n in names):
        raise ValueError(f"artifact tensors must all belong to the last {TRAINABLE_LAYERS} decoder layers or the final norm")
    adapter = manifest.get("adapter") or {}
    if adapter.get("instruction") not in INSTRUCTIONS:
        raise ValueError("artifact manifest must record adapter.instruction, one of the supported INSTRUCTIONS")


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
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"image area {width * height} px > MAX_IMAGE_PIXELS {MAX_IMAGE_PIXELS}")
    return image.convert("RGB")


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one page image as PIL.Image.Image (any mode, converted to RGB) plus one supported instruction",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "image_max_pixels": MAX_IMAGE_PIXELS,
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
    _batch_runner: Callable[..., list[dict[str, Any]]] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)
    weight_sha256: str | None = None
    adapter: dict[str, Any] | None = None

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
        # float32 on every device: the adapter is trained in float32 and overlays without a cast, and CPU,
        # Tesla-class and consumer GPUs then run the same arithmetic.
        dtype = torch.float32
        processor = AutoProcessor.from_pretrained(location, **common)
        model = AutoModelForImageTextToText.from_pretrained(location, dtype=dtype, **common)
        model = model.eval().to(resolved_device)
        for param in model.parameters():
            param.requires_grad_(False)
        end_id = processor.tokenizer.convert_tokens_to_ids(END_OF_UTTERANCE)
        pad_id = processor.tokenizer.pad_token_id

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

        def batch_runner(images: Sequence[Image.Image], instruction: str, max_new_tokens: int) -> list[dict[str, Any]]:
            text = processor.apply_chat_template(build_messages(instruction), add_generation_prompt=True)
            processor.tokenizer.padding_side = "left"  # prompts differ in tile count; generation needs left padding
            inputs = processor(text=[text] * len(images), images=[[image] for image in images], return_tensors="pt", padding=True).to(resolved_device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = int(inputs["input_ids"].shape[1])
            out = []
            for row in generated[:, prompt_len:]:
                ids = row.tolist()
                n_new = len(ids)
                for position, token in enumerate(ids):
                    if token in (end_id, pad_id):
                        n_new = position + (token == end_id)
                        break
                decoded = processor.tokenizer.decode(ids[:n_new], skip_special_tokens=False)
                out.append({"doctags": decoded, "new_tokens": int(n_new)})
            return out

        return cls(runner, resolved_device, str(dtype).removeprefix("torch."), source, batch_runner, model, processor, _weight_digest(root))

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

    # ------------------------------------------------------------------------------------------------------
    # Adaptation contract (one instruction on transcribed text lines, through the DocTags output)
    # ------------------------------------------------------------------------------------------------------

    @staticmethod
    def _strip(doctags: str) -> str:
        for terminator in _TERMINATORS:
            doctags = doctags.replace(terminator, "")
        return doctags.strip()

    def transcribe(
        self,
        images: Sequence[Image.Image],
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        max_new_tokens: int = DEFAULT_LINE_MAX_NEW_TOKENS,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert many images under `instruction` with greedy decoding, in batches (left-padded, since the tile count
        and hence the prompt length vary with the image); one ``{doctags, text, new_tokens, truncated}`` per image, in
        order, where ``text`` is the plain text the DocTags carry (`doctags_to_text`, whitespace-normalised). With an
        injected runner and no batch runner the images are converted one by one through the runner."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        checked = [_check_inputs(image, instruction, max_new_tokens)[0] for image in images]
        out: list[dict[str, Any]] = []
        for start in range(0, len(checked), batch_size):
            batch = checked[start : start + batch_size]
            if self._batch_runner is not None:
                raw = self._batch_runner(batch, instruction, max_new_tokens)
            else:
                raw = [self._runner(image, instruction, max_new_tokens) for image in batch]
            if not isinstance(raw, list) or len(raw) != len(batch) or any(not isinstance(r, dict) or "doctags" not in r for r in raw):
                raise RuntimeError("batch runner must return one dict with 'doctags' per image")
            for item in raw:
                doctags = self._strip(str(item["doctags"]))
                new_tokens = int(item.get("new_tokens", 0))
                out.append({"doctags": doctags, "text": normalise_text(doctags_to_text(doctags)), "new_tokens": new_tokens, "truncated": new_tokens >= max_new_tokens})
            if progress is not None:
                progress(len(out), len(checked))
        return out

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise RuntimeError("this pipeline has no loaded model (injected runner); use from_pretrained for adapt/save_artifact/load_artifact")
        return self._model, self._processor

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        max_new_tokens: int = DEFAULT_LINE_MAX_NEW_TOKENS,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Convert every validated record under `instruction` and score the text the DocTags carry as a transcript
        with ``metrics.ocr_metrics`` (micro and macro CER / WER, exact match). Works with an injected runner too."""
        from .metrics import ocr_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        items = self.transcribe([r["image"] for r in checked], instruction=instruction, max_new_tokens=max_new_tokens, batch_size=batch_size, progress=progress)
        hypotheses = [item["text"] for item in items]
        metrics = ocr_metrics(hypotheses, checked)
        metrics.update(
            {
                "hypotheses": hypotheses,
                "doctags": [item["doctags"] for item in items],
                "instruction": instruction,
                "truncated": sum(item["truncated"] for item in items),
                "new_tokens": sum(item["new_tokens"] for item in items),
                "max_new_tokens": max_new_tokens,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        epochs: int = 8,
        lr: float = 1e-4,
        batch_size: int = 8,
        seed: int = 0,
        max_new_tokens: int = DEFAULT_LINE_MAX_NEW_TOKENS,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the last `TRAINABLE_LAYERS` text-decoder layers and the final norm on transcribed lines
        with the causal language-model loss over the assistant turn — the DocTags of one `<text>` element that fills
        the image, carrying the transcript (`line_doctags`), then the end-of-utterance token; the image tokens and the
        user turn are masked — the checkpoint's own instruction-tuning objective and output contract. The frozen
        prefix — vision encoder, connector, embeddings and the first decoder layers — is run once per line under no
        gradient and its output hidden states are cached, so each step runs only the trainable tail; the loss equals
        the full model's loss exactly. AdamW (no weight decay), gradient clipping at `GRAD_CLIP`, seeded shuffling, no
        scheduler, no augmentation. Epoch 0 records the frozen model's validation metrics; the epoch with the lowest
        validation CER is kept (the final one without a validation split). On any exception the frozen weights are
        restored."""
        model, processor = self._require_model()  # refuse before importing torch
        import torch

        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not isinstance(lr, int | float) or not 0.0 < float(lr) <= 1e-2:
            raise ValueError("lr must be in (0, 1e-2]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        _check_inputs(Image.new("RGB", (16, 16)), instruction, max_new_tokens)  # the instruction contract
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        names = _trainable_names(model)
        name_set = set(names)
        device = torch.device(self.device)
        language_model = model.model.text_model
        first = _TRAINABLE_FIRST_LAYER
        tokenizer = processor.tokenizer
        end_id = tokenizer.convert_tokens_to_ids(END_OF_UTTERANCE)
        pad_id = tokenizer.pad_token_id
        chat_text = processor.apply_chat_template(build_messages(instruction), add_generation_prompt=True)
        frozen_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        previous_adapter = self.adapter
        cudnn_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # repeatable on one device
        history: list[dict[str, Any]] = []
        started = time.perf_counter()

        def _val() -> dict[str, Any] | None:
            if val_checked is None:
                return None
            result = self.evaluate(val_checked, instruction=instruction, max_new_tokens=max_new_tokens)
            return {"cer": result["cer"], "wer": result["wer"], "cer_macro": result["cer_macro"], "exact_match": result["exact_match"], "n": result["n"]}

        def _encode(batch: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], Any]:
            processor.tokenizer.padding_side = "right"  # the cache is right-padded: causal tokens never see a pad
            encoded = processor(text=[chat_text] * len(batch), images=[[r["image"]] for r in batch], return_tensors="pt", padding=True)
            ids, labels_list = [], []
            for i, record in enumerate(batch):
                prompt_ids = encoded["input_ids"][i][encoded["attention_mask"][i].bool()]
                target = torch.tensor(tokenizer(line_doctags(record["text"]), add_special_tokens=False)["input_ids"] + [end_id], dtype=torch.long)
                ids.append(torch.cat([prompt_ids, target]))
                labels_list.append(torch.cat([torch.full_like(prompt_ids, -100), target]))
            length = max(int(x.shape[0]) for x in ids)
            input_ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
            labels = torch.full((len(batch), length), -100, dtype=torch.long)
            mask = torch.zeros((len(batch), length), dtype=torch.long)
            for i, (x, y) in enumerate(zip(ids, labels_list, strict=True)):
                input_ids[i, : x.shape[0]] = x
                labels[i, : y.shape[0]] = y
                mask[i, : x.shape[0]] = 1
            inputs = {"input_ids": input_ids.to(device), "attention_mask": mask.to(device), "pixel_values": encoded["pixel_values"].to(device)}
            if encoded.get("pixel_attention_mask") is not None:
                inputs["pixel_attention_mask"] = encoded["pixel_attention_mask"].to(device)
            return inputs, labels

        def _tail_loss(hidden: Any, labels: Any) -> Any:
            position_ids = torch.arange(hidden.shape[1], device=device).unsqueeze(0).expand(hidden.shape[0], -1)
            embeddings = language_model.rotary_emb(hidden, position_ids)
            for layer in language_model.layers[first:]:
                hidden = layer(hidden, attention_mask=None, position_ids=position_ids, position_embeddings=embeddings)
            # logits only where a target token is predicted: the same cross-entropy as the full model's, without a
            # batch x length x vocabulary logit tensor
            targets = labels[:, 1:]
            keep = targets != -100
            logits = model.lm_head(language_model.norm(hidden[:, :-1][keep]))
            return torch.nn.functional.cross_entropy(logits.float(), targets[keep])

        try:
            # 1. cache the frozen prefix: the hidden states entering the first trainable layer, per line
            cache: list[tuple[Any, Any]] = []
            for start in range(0, len(train_checked), CACHE_BATCH_SIZE):
                batch = train_checked[start : start + CACHE_BATCH_SIZE]
                inputs, labels = _encode(batch)
                with torch.no_grad():
                    hidden = model.model(**inputs, output_hidden_states=True).hidden_states[first]
                for k in range(len(batch)):
                    n = int(inputs["attention_mask"][k].sum())
                    cache.append((hidden[k, :n].detach().to("cpu"), labels[k, :n]))
                del hidden
            cache_seconds = round(time.perf_counter() - started, 3)
            # 2. train the tail on the cached states
            params = []
            for name, param in model.named_parameters():
                if name in name_set:
                    param.requires_grad_(True)
                    params.append(param)
            n_trainable = sum(p.numel() for p in params)
            entry = {"epoch": 0, "train_loss": None, "val": _val(), "note": "frozen model"}
            history.append(entry)
            if progress is not None:
                progress(entry)
            best_epoch, best_score = 0, (history[0]["val"] or {}).get("cer", float("inf"))
            best_state = frozen_state
            optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
            rng = random.Random(seed)
            torch.manual_seed(seed)
            width = int(cache[0][0].shape[1])
            for epoch in range(1, epochs + 1):
                model.train()
                order = list(range(len(cache)))
                rng.shuffle(order)
                losses = []
                for start in range(0, len(order), batch_size):
                    items = [cache[k] for k in order[start : start + batch_size]]
                    length = max(int(h.shape[0]) for h, _ in items)
                    hidden = torch.zeros((len(items), length, width), dtype=items[0][0].dtype)
                    labels = torch.full((len(items), length), -100, dtype=torch.long)
                    for k, (h, lab) in enumerate(items):
                        hidden[k, : h.shape[0]] = h
                        labels[k, : lab.shape[0]] = lab
                    loss = _tail_loss(hidden.to(device), labels.to(device))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": _val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                if val_checked is None or entry["val"]["cer"] < best_score:
                    best_epoch, best_score = epoch, (entry["val"] or {}).get("cer", float("inf"))
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            model.load_state_dict(best_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
        except BaseException:
            model.load_state_dict(frozen_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            self.adapter = previous_adapter
            raise
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags
        self.adapter = {
            "instruction": instruction,
            "target": LINE_DOCTAGS,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "first_trainable_layer": first,
            "epochs": epochs,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "selection": "lowest validation CER" if val_checked is not None else "final epoch (no validation split)",
            "loss": "causal language-model cross-entropy over the assistant turn (the DocTags line target and the end-of-utterance token); image tokens and the user turn masked; computed on the cached frozen-prefix hidden states",
            "lr": float(lr),
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "n_train": len(train_checked),
            "n_val": len(val_checked) if val_checked is not None else 0,
            "cache_seconds": cache_seconds,
            "history": history,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained tensors as safetensors plus a manifest naming the base, the digests, the instruction and
        the training configuration. Requires a prior `adapt`."""
        model, _processor = self._require_model()  # refuse before importing torch
        import torch
        from safetensors.torch import save_file

        if self.adapter is None:
            raise RuntimeError("nothing to save: call adapt() first")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = list(self.adapter["trainable_names"])
        state = model.state_dict()
        tensors = {name: state[name].detach().cpu().contiguous() for name in names}
        weights = out / ADAPTER_WEIGHTS
        save_file(tensors, str(weights), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_file": WEIGHTS_FILE, "weight_sha256": self.weight_sha256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": names,
            "files": [{"path": ADAPTER_WEIGHTS, "bytes": weights.stat().st_size, "sha256": _sha256(weights)}],
            "torch": torch.__version__,
            "metadata": dict(metadata or {}),
        }
        with open(out / ADAPTER_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Overlay a saved adapter onto this (freshly loaded) pipeline after checking its manifest, digest and exact
        tensor set. Refuses tensors outside the last decoder layers and the final norm."""
        model, _processor = self._require_model()  # refuse before importing safetensors
        from safetensors.torch import load_file

        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        _check_artifact_manifest(manifest, artifact, self.weight_sha256 or "")
        expected = _trainable_names(model)
        if sorted(manifest["tensors"]) != sorted(expected):
            raise ValueError("artifact tensor set does not match its recorded configuration")
        tensors = load_file(str(artifact / ADAPTER_WEIGHTS))
        if sorted(tensors) != sorted(expected):
            raise ValueError("artifact tensor names differ from the manifest")
        state = model.state_dict()
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != tuple(state[name].shape):
                raise ValueError(f"artifact tensor {name} has shape {tuple(tensor.shape)}, base has {tuple(state[name].shape)}")
        model.load_state_dict({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()}, strict=False)
        model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": expected, "history": manifest.get("history", [])}
        return dict(self.adapter)

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> SmolDoclingPipeline:
        """Check the adapter manifest against the base snapshot's recorded weight digest, load the verified base, then
        overlay the adapter (checked again, and the tensor set, before deserialising). A refused manifest never loads
        a model."""
        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        _check_artifact_manifest(manifest, artifact, _weight_digest(root) or "")
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe

# SmolDocling-256M-preview document extraction pipeline

DIMER pipeline for **SmolDocling-256M-preview** (`docling-project/SmolDocling-256M-preview`), IBM Research's 256M-parameter vision–language model (SigLIP encoder + SmolLM-2 decoder in the Idefics3 arrangement, fine-tuned from SmolVLM-256M-Instruct) that converts one document page image into **DocTags** — layout elements with location tokens, reading order, OCR text and OTSL tables — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline accepts one of the seven instructions the upstream README lists, decodes greedily under a caller-owned token budget, and returns the DocTags string with the plain text recovered from it and a `truncated` flag; it does not render PDFs, export documents, or score its own output. On top of inference it carries the **adaptation contract for one instruction** (`Convert this page to docling.`) on transcribed text lines, through the DocTags output: corpus-level character and word error rates over the text the DocTags carry, two non-adapted baselines, a bounded fine-tuning of the last eight SmolLM2 decoder layers on cached prefix hidden states with a DocTags line target, and a verified safetensors adapter that reloads against the pinned base.

## Upstream alignment

- Model: `docling-project/SmolDocling-256M-preview`
- Revision: `ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8`
- Upstream weight license: CDLA-Permissive-2.0 (README front matter; the README body says Apache 2.0 — see `docs/WEIGHTS.md`)
- Upstream task: image-text-to-text — document page image + instruction → DocTags markup
- Repository adaptation: **bounded supervised fine-tuning of one instruction through the DocTags output** — the last eight of the 30 SmolLM2 decoder layers and the final norm (28,321,344 of 256,484,928 parameters) on `{id, image, text}` line records, each trained to emit `<doctag><text><loc_0><loc_0><loc_500><loc_500>` + transcript + `</text></doctag>` (the markup the frozen model already uses for one text element that fills the image) under the causal language-model loss; the SigLIP vision encoder, the connector, the embeddings, the output head and the first 22 decoder layers stay frozen. Trained tensors are exported as a safetensors adapter with a manifest and overlaid on a freshly loaded, re-verified base. The other six instructions are inference-only; the tuned decoder answers them too, which the tutorial shows on one rendered page and the card records.

## Quick start

```python
from PIL import Image
from smoldocling_document_extraction_pipeline import SmolDoclingPipeline, doctags_summary

pipe = SmolDoclingPipeline.from_pretrained()          # stages + verifies weights/smoldocling-256m-preview first
result = pipe.convert(Image.open("page.png"))          # default instruction: "Convert this page to docling."
print(result["doctags"][:500])                         # DocTags markup: <doctag><section_header_level_1><loc_..>…
print(result["text"][:200])                            # plain words, tags and <loc_N> removed
print(result["truncated"], result["new_tokens"])       # True when the max_new_tokens budget was exhausted
print(doctags_summary(result["doctags"]))              # per-element counts, loc tokens, OTSL cells

# other supported instructions and a larger budget
result = pipe.convert(Image.open("table.png"), instruction="Convert table to OTSL.", max_new_tokens=4096)

from smoldocling_document_extraction_pipeline import fetch_sample_dataset
splits = fetch_sample_dataset()                    # 800 digest-pinned Belfort handwritten lines, 600 / 60 / 140
print(pipe.evaluate(splits["test"])["cer"])        # frozen corpus CER of the text the DocTags carry
pipe.adapt(splits["train"], splits["validation"])  # last eight decoder layers, lowest-validation-CER epoch kept
print(pipe.evaluate(splits["test"])["cer"])
pipe.save_artifact("outputs/adapter")
again = SmolDoclingPipeline.from_artifact("outputs/adapter")   # re-verifies the base, checks the manifest, overlays
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed; `tests/test_model_backed.py` runs only where the snapshot is staged). On a fresh clone the manifest is committed but the weights are not: `SmolDoclingPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/smoldocling-256m-preview/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (13 files)
  config.json                # Idefics3ForConditionalGeneration: SigLIP vision + 30-layer llama text, 8192 positions
  preprocessor_config.json   # longest edge 2048, 512-px tiles, image splitting on
  processor_config.json  chat_template.json  generation_config.json
  tokenizer.json  tokenizer_config.json  vocab.json  merges.txt  added_tokens.json  special_tokens_map.json
  model.safetensors          # git-ignored, 513,028,808 bytes
  README.md
```

## Input ceilings and request parameters

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 16384`, `MAX_IMAGE_PIXELS = 4096²` (the ceilings bound decode and resize memory: a 9,000 px wide text line is accepted); `INSTRUCTIONS` — the seven forms from the upstream README (`DEFAULT_INSTRUCTION = "Convert this page to docling."`); `MAX_NEW_TOKENS = 8192` (the README example's budget and the text model's `max_position_embeddings`), `DEFAULT_MAX_NEW_TOKENS = 2048`; `DECODING = "greedy"`. One page image per call; the processor resizes it so its longest edge is 2048 px and splits it into 512-px tiles. Any other instruction is refused (`ValueError`). See `MODEL_CARD.md` for who owns the budget, the licence discrepancy and the measured CPU timings.

## Adaptation contract

- **Records:** `{id, image, text}` — a PIL image (sides within the ceilings) and its transcript (1..512 characters after whitespace runs are collapsed); `validate_dataset` checks the structure, `split_dataset` de-duplicates by decoded pixels and `check_split_disjoint` asserts no image is shared. The default sample (`samples.py`) is the first eight parquet row groups of the Belfort-line test shard (`Teklia/Belfort-line`, MIT; nineteenth-century French council minutes in cursive) read over HTTPS range requests at an immutable Hub revision, each row group refused on any SHA-256 or byte-total mismatch — the same digest-pinned sample and split as the sibling `got-ocr2-pipeline`, `florence2-vision-language-pipeline` and `smolvlm-vision-language-pipeline` rows; `load_byod_dataset` reads a zip or directory of line images plus `transcripts.csv`.
- **Measures (`metrics.py`):** `ocr_metrics` — micro CER and WER (total edits over total reference characters or words) over the text the DocTags carry (`doctags_to_text`), macro rates, exact match, and the hypothesis length; uncapped, so a rate above 1.0 means the model generates text the line does not carry. `empty_baseline` (CER 1.0 by construction) and `constant_baseline` (the medoid training transcript for every line).
- **Fine-tuning:** `adapt(train, val, *, instruction=DEFAULT_INSTRUCTION, epochs=8, lr=1e-4, batch_size=8, seed=0)` caches the hidden states entering decoder layer 22 for every training line (the tiles' visual tokens, the chat-templated user turn and the assistant turn, one frozen forward each), then trains layers 22–29 and the final norm on those states with the causal LM loss over the assistant turn — the DocTags line target `line_doctags(text)` and the end-of-utterance token — AdamW (no weight decay), gradient clipping at 1.0 and seeded shuffling; the loss equals the full model's loss exactly. Epoch 0 records the frozen validation rates; the epoch with the lowest validation CER is kept; on any exception the frozen weights are restored.
- **Artifact:** `save_artifact` writes `adapter.safetensors` (about 113 MB) + `manifest.json` (`org.valcorza.smoldocling-256m-preview.adapter.v1`: base identity and weight digest, tensor names, file size and SHA-256, the instruction and the line target, configuration, history); `from_artifact` re-verifies the base and checks the manifest, digest and exact tensor set before deserialising.
- **Build record (Tesla T4, seed 42 split):** frozen CER 1.439 / WER 2.491 on the 140 held-out lines (the frozen model answers the transcription instruction with almost nothing — 62 of the 140 hypotheses are empty and 100 are three characters or fewer — while 21 lines loop on a digit or a syllable to the 160-token budget (29 truncated), which is how a CER above 1.0 coexists with hypotheses shorter than the references), adapted **0.782** / **0.975** (epoch 8 of 8, 2 lines exact), reload parity 8/8; the report page after adaptation: before adaptation 630 new tokens, page WER 0.192, element counts (expected, observed) `section_header_level_1` 1/1, `text` 3/3, `otsl` 2/2, verdict `sample-sanity`; after adaptation 92 new tokens, page WER 0.709, element counts (expected, observed) `section_header_level_1` 1/0, `text` 3/1, `otsl` 2/0, verdict `sample-sanity`. One seeded split of one 800-line sample; no dispersion estimate. The sibling rows on the same split: GOT-OCR 2.0 0.759, Florence-2 0.797, SmolVLM-500M 0.890.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/smoldocling-document-extraction-pipeline/blob/main/tutorials/smoldocling_document_extraction_colab.ipynb)

`tutorials/smoldocling_document_extraction_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, the model identity, the manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`). Its default `Run all` path stages and verifies the pinned snapshot, fetches the eight pinned Belfort row groups and splits the 800 lines 600 / 60 / 140, converts a synthetic report page through the inference contract, measures the frozen `Convert this page to docling.` instruction's CER and WER on the held-out lines (on the text its DocTags carry) beside the empty and constant baselines, runs `adapt` with validation-CER epoch selection, scores the held-out lines again, writes six line panels and converts the page again with the adapted model, and exports the adapter and reloads it with verified transcript parity. BYOD is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. The earlier `TASK-INFERENCE` notebook's Kaggle CPU run (2026-09-14) is retained as history and is not evidence for the `E2E` blob; the supported-runtime run of the exact release revision is recorded in `docs/release-verification.md` when it exists.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, licence note, the adapter artifacts, the sample corpus and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are CDLA-Permissive-2.0; the Belfort-line sample is MIT and is not redistributed; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.

---
license: cdla-permissive-2.0
model_card_spec: "1.1"
pipeline_tag: image-text-to-text
base_model: docling-project/SmolDocling-256M-preview
date_published: "2025-02-12"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2025-02-12T15:40:33Z, https://huggingface.co/api/models/docling-project/SmolDocling-256M-preview, originally published as ds4sd/SmolDocling-256M-preview); the accompanying paper arXiv:2503.11576 is dated 2025-03-14 and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# SmolDocling-256M-preview (DIMER package v0.1.0) — Document Page → DocTags Extraction (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-docling--project%2FSmolDocling--256M--preview-ffcc4d?style=flat)](https://huggingface.co/docling-project/SmolDocling-256M-preview)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-docling--project%2Fdocling-181717?style=flat&logo=github&logoColor=white)](https://github.com/docling-project/docling)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2503.11576-b31b1b.svg)](https://arxiv.org/abs/2503.11576)
[![License: CDLA-Permissive-2.0](https://img.shields.io/badge/License-CDLA--Permissive--2.0-blue.svg)](https://cdla.dev/permissive-2-0/)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/smoldocling-document-extraction-pipeline/blob/main/tutorials/smoldocling_document_extraction_colab.ipynb) [`smoldocling_document_extraction_colab.ipynb`](https://github.com/kurtvalcorza/smoldocling-document-extraction-pipeline/blob/main/tutorials/smoldocling_document_extraction_colab.ipynb)  
  *Full-page conversion of a report page rendered in code with the pinned `docling-project/SmolDocling-256M-preview` weights: DocTags under a caller-owned instruction and token budget, the element summary and recovered text, and `word_error_rate` / `element_count` against the rendered reference as sanity evidence only — no OCR, layout or table benchmark.*

---

#### Description

`docling-project/SmolDocling-256M-preview` is IBM Research's Docling-team release of SmolDocling, the 256M-parameter vision–language model described in "SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion" (Nassar et al., arXiv:2503.11576), fine-tuned from `HuggingFaceTB/SmolVLM-256M-Instruct` and pinned here to revision `ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8` (the repository was first published as `ds4sd/SmolDocling-256M-preview` and the Hub's `main` resolved to this commit on 2026-09-14). The snapshot `config.json` declares `Idefics3ForConditionalGeneration` (`model_type` idefics3): a SigLIP-style vision encoder (`idefics3_vision`, hidden size 768, 12 heads, 16-px patches on 512-px tiles, `use_base_siglip`) whose features are pixel-shuffled by a factor of 4 into a 30-layer llama-architecture text decoder (hidden size 576, 9 heads, vocabulary 49,280, `max_position_embeddings` 8192) that was trained to emit **DocTags** — a markup in which each page element (`<section_header_level_N>`, `<text>`, `<otsl>` tables with `<ched>`/`<fcel>`/`<nl>` cells, `<picture>`, `<caption>`, `<formula>`, `<code>`, lists, page headers and footers, …) is preceded by four `<loc_N>` tokens on a 0–500 grid, in reading order, with the OCR'd text inside. The processor (`Idefics3ImageProcessor`, `preprocessor_config.json`) resizes the page so its longest edge is 2048 px and splits it into 512-px tiles of 64 visual tokens each plus one global view; the chat template wraps one image and one instruction as a user turn. Upstream reports training on the DoclingMatix mixture plus the SynthCodeNet, SynthFormulaNet and SynthChartNet synthetic sets and labels the checkpoint a *preview* (its README now points to `ibm-granite/granite-docling-258M` as the maintained successor). Nothing is trained or adapted here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `SmolDoclingPipeline.from_pretrained` (verified local loading through `AutoModelForImageTextToText`/`AutoProcessor` with `trust_remote_code=False`, float32 on CPU and the checkpoint's bfloat16 on CUDA), `convert` (input validation, the seven supported instructions, greedy decoding under a caller-owned `max_new_tokens`, terminator stripping, a `truncated` flag), `doctags_summary`, `doctags_to_text`, `word_error_rate`, and the `validate_inputs` and `evaluation_report` stage helpers.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is document page → DocTags conversion: input one page image (`PIL.Image.Image`, any mode, converted to RGB), one of the seven instructions the upstream README's supported-instructions table lists (full-page conversion by default; chart → table, formula → LaTeX, code → text, table → OTSL, text/section-header retrieval, footer detection), and a token budget; output the DocTags string, the plain text recovered from it, the number of tokens generated and whether the budget was exhausted. Envisioned applications are the first stage of document understanding for born-digital or cleanly rendered pages — reports, papers, manuals — where the DocTags are then loaded by `docling_core` into a `DoclingDocument` and exported to Markdown, HTML or JSON, or where the located elements feed downstream extraction; and single-element conversion (a cropped table to OTSL, a formula to LaTeX) when the caller already has the crop. Within DIMER the pipeline is an inference component and a zero-configuration baseline for page conversion, not a certified OCR or layout engine for any document family.

###### Primary Intended Users

Intended users are machine-learning engineers, document-processing developers, and data analysts integrating page conversion into research prototypes, internal enterprise document tooling, or the DIMER workbench. A user is expected to understand that the output is *generated text* — it carries no probability, no correctness signal, and can drift, repeat or invent content while remaining well-formed — that the `<loc_N>` boxes are approximate (a 500-step grid, about 2 px per step on a letter page at 850 px wide), that the instruction must be one of the seven trained forms and a paraphrase is undefined behaviour, that the token budget is theirs to set (a dense page can exceed 2048 tokens and be reported `truncated`), that the model is a preview trained largely on rendered and synthetic documents so scans, photographs, handwriting and non-Latin scripts are distribution shifts, that greedy decoding is reproducible on a fixed device and dtype but GPU bfloat16 and CPU float32 outputs need not match, and that OCR accuracy, layout quality and table structure can only be measured on labelled pages they supply. Users who need PDF rendering, Markdown/HTML export, multi-page documents or batch throughput are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no PDF or multi-page handling (one page image per call), no document export (DocTags → Markdown/HTML/JSON is `docling_core`'s job), no batching or streaming, no sampling or beam search, no instruction outside the seven listed, no chemical, multi-page or improved-chart features the upstream README lists as "coming soon", and no confidence or quality signal on the output.
2. **Input boundary:** `convert` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, instructions not in `INSTRUCTIONS`, and budgets outside `[1, MAX_NEW_TOKENS = 8192]` (`ValueError`). Every page is resized so its longest edge is 2048 px and tiled at 512 px, so glyphs that are only a few pixels tall at that scale are unlikely to be read; the decoder's 8192-position context bounds the longest page that can be converted in one call, and a page needing more output than the caller's budget is returned `truncated`.
3. **Input boundary:** the training mixture is rendered documents (DoclingMatix) and synthetic code, formula and chart renders. Scanned or photographed pages (skew, shading, noise), handwriting, forms with dense rules, multi-column newspapers, non-Latin scripts and right-to-left layouts fall outside what the upstream authors report and what this repository measured; results on them are undefined, not merely degraded. An image that is not a document at all still produces markup (see §Risks and harms).
4. **Decision boundary:** not for autonomous decisions that act on extracted values — invoice or contract processing feeding payments or compliance, clinical-record extraction, regulatory filings — without a human reviewing the recovered text and structure against the page, and a locally measured error rate on the deployment's own labelled pages.

#### Factors

###### Groups

This pipeline is not human-centric by design: it transcribes and lays out a page image and never classifies, identifies or scores people. The training mixture (DoclingMatix — rendered documents assembled by the Docling team and Hugging Face — plus IBM's synthetic code, formula and chart sets, per the snapshot README and the paper) contains no evaluation groups in the demographic sense, and neither the upstream authors nor this repository audited it for anything of the kind. What does vary is the document population: the corpora are predominantly English, rendered rather than scanned, and skewed toward scientific, technical and business layouts, so pages in other languages and scripts, historical typesetting, handwritten or degraded scans, and layouts the mixture under-represents are the groups whose transcription quality is unknown, not known to be equal. Where pages carry personal data — medical records, HR files, identity documents — the pipeline's output makes that data machine-readable; the operator who processes such documents is responsible for a fairness and privacy audit on their own page set, stratified by document family, before relying on the output.

###### Instrumentation

The upstream training pages were produced by rendering documents to images and pairing them with DocTags derived from the source markup or from Docling's own conversion, so the "instrument" is a renderer over born-digital typesetting: crisp glyphs, consistent fonts, no sensor noise, with synthetic augmentation for the code, formula and chart sets. Inference images arrive from whatever produced them — a PDF renderer at some DPI, a scanner, a phone camera, a screenshot — and resolution, skew, JPEG artefacts, contrast and font all change the visual evidence; the longest-edge-2048 resize and 512-px tiling (`preprocessor_config.json`) fix the visual token budget regardless of the source, so a small-type page loses detail while a large blank margin costs tiles. The pipeline validates type and size only; it cannot detect a low-DPI render, a skewed scan or a page cut mid-column. The synthetic tutorial page (Pillow's bundled font, ruled tables, generous margins) is itself a rendering instrument whose glyphs differ from any publisher's — visibly so: on it the model read both tables cell-perfectly but dropped three 6–7-word spans of the paragraphs, repeated one 25-word span, and dropped the colon from the heading.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible (bfloat16, the checkpoint's stored dtype) but was not exercised for this card. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and the Hub offline (`HF_HUB_OFFLINE=1`): `verify_snapshot` on the 13-file, 518 MB snapshot 0.27–0.32 s; load 6.3 s (page cache warm) to 21.8 s (cold); one 850×1100 rendered report page under the default instruction and budget 21.1–21.7 s for 630 generated tokens — cost is dominated by autoregressive decoding, so it scales with the tokens a page needs, not with the caller's pixel count once the 2048-px resize applies. Data environment: the model assumes the image is one cleanly rendered document page in the styles of its training mixture; the synthetic tutorial page satisfies that assumption and is where the measured behaviour holds. Scans, photographs, handwriting, dense or unusual layouts and non-Latin scripts violate it to degrees this repository did not measure, and the pipeline reports no signal when they do — nor when the image is not a document at all.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. Generated DocTags carry no score, no probability and no correctness signal; `new_tokens` and `truncated` describe the generation, and `doctags_summary` counts elements, `<loc_N>` tokens and OTSL cells — descriptions of the output, not metrics. The repository ships two helpers because they are the primitives page-conversion evaluation is built from: `word_error_rate(reference, hypothesis)` (word-level Levenshtein distance over lower-cased, whitespace-tokenised text with tags and location tokens removed, divided by the reference length) and the element-count comparison; character error rate, layout mAP, reading-order metrics and the TEDS table-structure score the upstream paper reports are not implemented, since they need labelled pages with matching conventions that the caller must choose. To evaluate, the caller supplies ground-truth text per page and computes `word_error_rate` on `doctags_to_text`, or loads the DocTags with `docling_core` and applies layout and table metrics against annotations. The public `evaluation_report(result, reference_text=None, expected_counts=None)` stage returns that report in machine-readable form: a `word_error_rate` entry when reference text is supplied and one `element_count` entry per expected tag when counts are supplied, with the verdict `sample-sanity`, or the verdict `not-measurable` naming the labelled set that would be required when neither is supplied. The upstream paper's OCR, layout, table and code/formula numbers are upstream-reported and this pipeline does not reproduce or claim them.

###### Decision thresholds

No score threshold exists: the model generates tokens until it emits a terminator or the budget is exhausted, and nothing is filtered. The decision parameters are the **instruction** — one of seven trained forms, refused otherwise, because the model's behaviour on a paraphrase is undefined — and the **token budget** `max_new_tokens`, default `DEFAULT_MAX_NEW_TOKENS = 2048` (a practical page budget chosen by this repository; the synthetic page needed 630) with ceiling `MAX_NEW_TOKENS = 8192` (the budget the upstream README's transformers example passes and the decoder's `max_position_embeddings`). A budget that is too small is reported, not hidden: `truncated` is true whenever `new_tokens` reaches it, and the caller should raise the budget and rerun. Decoding is greedy (`do_sample=False`) with no temperature, no beams and no repetition penalty — the upstream README example's setting — so a repetition loop runs to the budget rather than being cut short. A deployment owns choosing the budget per document family (dense pages and tables need more) and deciding whether a `truncated` page is retried or rejected.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings, token counts, element counts and one word error rate on one synthetic page, not accuracy. Run-to-run variability comes only from floating-point kernel selection across CPU builds and accelerators; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable, but because decoding is autoregressive a single differing token changes the rest of the sequence, and the CPU float32 output need not match a CUDA bfloat16 output; the synthetic page's own bytes depend on the Pillow build's bundled font. On the synthetic page the model produced exactly the drawn structure (1 section header, 3 text blocks, 2 tables, 68 OTSL cell tokens = the 55 drawn cells plus 13 `<nl>` row terminators) and a word error rate of 0.19 (212 words recovered for 203 rendered: both tables cell-perfect; in the running text three 6–7-word spans dropped and one 25-word span repeated; the heading's colon dropped) — one observation, not an estimate. A caller who needs an accuracy estimate must supply labelled pages and compute it over many pages or bootstrap resamples themselves; a caller who needs a confidence signal per element has none from this model.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README lists the training data as `HuggingFaceM4/DoclingMatix` and IBM's `ds4sd/SynthCodeNet`, `ds4sd/SynthFormulaNet` and `ds4sd/SynthChartNet`; the paper describes DoclingMatix as rendered pages from public document sources with DocTags targets, and the synthetic sets as generated code, formula and chart renders. Public documents can contain personal data (names in papers and reports, addresses in forms), so personal data in the training corpus is not ruled out; it is not enumerated by the upstream authors and was not audited here. This repository distributes code, tests, and documentation; it does not distribute the 513,028,808-byte `model.safetensors`, which is staged locally under `weights/smoldocling-256m-preview/` and git-ignored, and it ships no sample documents — the tutorial page is rendered in code. The upstream README declares two licences (front matter `cdla-permissive-2.0`, body "Apache 2.0"); this repository records the machine-readable front-matter value and flags the discrepancy in `docs/WEIGHTS.md`. The operator must audit the pages they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will transcribe a medical record as readily as a datasheet.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — transcribing clinical records for automated coding, invoices or contracts for automated payment or compliance, identity documents for verification, legal filings for screening — would be admissible only with human review of the recovered text and structure against the original page (the model can invent or drop content without any signal), a locally measured error rate on the deployment's own labelled pages, a documented instruction and budget policy with `truncated` handling, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 13 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False`, and the smoke run loaded with `HF_HUB_OFFLINE=1`. No pickle checkpoint exists at the pinned revision and the upstream `onnx/` export is neither listed nor loaded. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported.
- **Input integrity:** the public `validate_inputs(image, *, instruction, max_new_tokens)` stage applies exactly the checks `convert` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, the observed input, the request and the verdict; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; instructions outside the seven trained forms, non-integer or boolean budgets, and budgets outside `[1, 8192]` are rejected; `convert` raises on a malformed runner result; `evaluation_report` rejects unknown element tags.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; greedy decoding with no sampling; every result carries `model_id`, `model_revision`, the instruction, the budget, `new_tokens`, `truncated`, the device and the dtype.
- **Refusals:** no batching, no download without the explicit flag, no free-form prompts, no sampling, no pickle deserialisation, no attempt to guess whether the input is a document.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Plausible wrong text:** the model's failure mode is fluent drift — merged lines, repeated phrases, invented or dropped words — inside well-formed markup with no signal; on the synthetic page the paragraphs came back with three spans dropped and one repeated while every table cell was right. Downstream consumers that trust the text (search indexes, extraction rules, LLM pipelines) inherit the errors silently.
- **Markup on non-document input:** the model generates DocTags for any image — a photograph, a figure, a blank page — and an operator who feeds one gets a confident "document" of nothing.
- **Truncation and loops:** a dense page can exceed the budget (reported via `truncated`) and a repetition loop runs to the budget; a caller who ignores the flag ships a partial page.
- **Approximate localisation:** `<loc_N>` boxes are on a 500-step grid and can be offset; a caller who crops by them for a second pass can cut content.
- **Automation bias:** clean, well-structured markup invites trust that generated text has not earned.
- **Privacy exposure:** pages containing personal or confidential data are transcribed without any content check and made searchable.
- **Bias amplification:** any document family the rendered, English-centric mixture under-represents (non-Latin scripts, handwriting, historical or degraded scans, unusual layouts) is reproduced as uneven accuracy, undetected because no per-family evaluation exists.
- **Resource use:** a 518 MB model and ~21 s per page on the reference CPU; a page stream saturates a shared host quickly, and the CUDA path was not measured.

###### Use cases

Prohibited even where the model would work: transcribing documents in order to extract personal data for surveillance, profiling, social scoring, or unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access; processing documents the operator has no right to process, or paywalled and licence-restricted publications in breach of their terms; deceptive uses that present generated text or structure as a verified transcript or as evidence; and any use that violates the upstream CDLA-Permissive-2.0 licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the documents processed. Autonomous high-consequence actions triggered by unreviewed extracted content are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `docling-project/SmolDocling-256M-preview`
- Revision: `ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8`
- Snapshot manifest: `weights/smoldocling-256m-preview/dimer-base-manifest.json`, 13 files, `totalBytes` 517896538
- `model.safetensors` SHA-256: `cdcdf5d823c5684029c7d8e52177cf10f9034b3aba6577549cfb1a9ce36ad0a2` (513,028,808 bytes, bfloat16 tensors)
- `config.json` SHA-256: `57af2810c65b9896a8d1d65c67aabdb9296d497aa10a6329f4bb2ddce623586f` (3,903 bytes; `Idefics3ForConditionalGeneration`, text `max_position_embeddings` 8192)
- `preprocessor_config.json` SHA-256: `6cb6e36d6fcb88ca1502c4a26750715dc3e7dedddc9a8f17b27d8d167d1457e7` (longest edge 2048, tiles 512, image splitting on)
- Weight format: SafeTensors; loader `AutoModelForImageTextToText.from_pretrained(<dir>, local_files_only=True, trust_remote_code=False, dtype=float32|bfloat16)` with `AutoProcessor` from the same directory; the chat template is the snapshot's `chat_template.json`. No pickle checkpoint exists at this revision; the upstream `onnx/` directory is not part of the manifest.

## Input/output contract

- `SmolDoclingPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible (bfloat16), else `cpu` (float32).
- `convert(image, *, instruction="Convert this page to docling.", max_new_tokens=2048) -> dict` with keys `doctags` (terminators stripped), `text` (`doctags_to_text`), `instruction`, `image_size`, `new_tokens`, `truncated`, `generation` (`max_new_tokens`, `do_sample` false, `decoding` greedy), `device`, `dtype`, `source`, `model_id`, `model_revision`.
- `doctags_summary(doctags) -> dict` with `counts` per element tag, `n_elements`, `n_loc_tokens`, `n_table_cells`, `wrapped_in_doctag`, `n_chars`; `doctags_to_text(doctags) -> str`; `word_error_rate(reference, hypothesis) -> float`; `build_messages(instruction)`.
- Ceilings and constants: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `INSTRUCTIONS` (seven forms; `DEFAULT_INSTRUCTION` is the first), `MAX_NEW_TOKENS = 8192`, `DEFAULT_MAX_NEW_TOKENS = 2048`, `DECODING = "greedy"`, `INPUT_SCHEMA`.
- `validate_inputs(image, *, instruction, max_new_tokens, names) -> dict`; `evaluation_report(result, reference_text=None, expected_counts=None, *, sample_kind) -> dict`; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32 on CPU, bfloat16 on CUDA; preprocessing resize so the longest edge is 2048 px, 512-px tiles of 64 visual tokens plus one global view (`Idefics3ImageProcessor`, snapshot defaults, nothing overridden); greedy decoding.
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`, two runs: `verify_snapshot` 0.27 / 0.32 s (13 files, 518 MB); load 6.28 s (warm page cache) / 21.8 s (cold); `convert` on a synthetic 850×1100 report page (heading, two six-line paragraphs, an 8×5 and a 5×3 ruled table, one closing line, rendered with Pillow's bundled font) under `Convert this page to docling.` with `max_new_tokens=2048` → 630 new tokens, 1,917 characters of DocTags, `truncated` false, in 21.7 / 21.1 s; `doctags_summary`: `section_header_level_1` 1, `text` 3, `otsl` 2, 24 `<loc_N>` tokens, 68 OTSL cell tokens, wrapped in `<doctag>`; `evaluation_report` against the rendered text and counts: `word_error_rate` 0.192 (212 words recovered vs 203 rendered — tables cell-perfect; running text with three 6–7-word spans dropped and one 25-word span repeated; heading colon dropped), all three `element_count` entries expected = observed, verdict `sample-sanity`. Identical DocTags on both runs.
- Tutorial execution: `tutorials/smoldocling_document_extraction_colab.ipynb` ran top-to-bottom in a fresh local kernel (all 8 code cells, 78.8 s, snapshot staged by the carried `stage_missing_files`, identical DocTags to the smoke run); recorded in `docs/release-verification.md` as pre-flight, not supported-runtime evidence.
- Tests: `pytest -q -o addopts= tests` — offline, no weights required; `ruff check src tests tools` clean.
- Not executed: CUDA/bfloat16 path, instructions other than the full-page default, pages that exceed the default budget, any error-rate measurement against labelled pages, scans or photographs, non-Latin scripts, formulas, code or charts (the synthetic page has none).

## References

- Nassar et al. SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion. 2025. https://arxiv.org/abs/2503.11576
- Auer et al. Docling Technical Report. 2024. https://arxiv.org/abs/2408.09869
- Laurençon et al. Building and better understanding vision-language models: insights and future directions (Idefics3). 2024. https://arxiv.org/abs/2408.12637
- Lysak et al. Optimized Table Tokenization for Table Structure Recognition (OTSL). ICDAR 2023. https://arxiv.org/abs/2305.03393
- Upstream code (Docling, DocTags loading and export): https://github.com/docling-project/docling
- Upstream card: https://huggingface.co/docling-project/SmolDocling-256M-preview
- Transformers `Idefics3` documentation: https://huggingface.co/docs/transformers/model_doc/idefics3
- CDLA-Permissive-2.0 licence text: https://cdla.dev/permissive-2-0/

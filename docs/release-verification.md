# Release verification

`tutorials/smoldocling_document_extraction_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record for the
notebook. The earlier `TASK-INFERENCE` notebook's Kaggle CPU run (2026-09-14, retained below) is history for a
superseded blob, not evidence for this one.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 13-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the isolated-environment bootstrap (one kernel cell, no
  kernel pip install, no restart request, environment reuse by lock digest); `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the Belfort-line
  parquet-conversion revision `c4a74bbd39f2df314752e7e6026649a39d365cbb` is the one other 40-hex revision the
  documents may cite);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `SmolDoclingPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `write_dataset_csv`, the four dataset refusal probes, the ceiling print,
  `validate_inputs` with the unsupported-instruction refusal probe, `convert` for the report page with the
  structural checks, `doctags_summary` and the `evaluation_report` against the rendered words and counts, `empty_baseline`, `constant_baseline`,
  `pipe.evaluate` on the frozen model, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on the
  validation and test splits after adaptation with the two CER assertions, the report page converted again after
  adaptation, the example panels, `pipe.save_artifact`, `SmolDoclingPipeline.from_artifact` and the transcript-parity
  assertion, and the result fields `weight_file` / `weight_format` / `weight_sha256`, the `corpus` block), the seven expected `outputs/` paths, the learner-facing statements (CDLA-Permissive-2.0 weights, greedy decoding,
  generated markup with no score, adaptation of one instruction on transcribed lines through the DocTags output, the
  two non-adapted baselines, CER and WER, rates above 1.0, the empty-string and constant-transcript baselines, the
  DocTags line target, the causal language-model loss, the frozen-prefix cache, lowest validation CER, no dispersion
  estimate, no score exists, the caller-owned request parameters, float32 on every device, the leakage guidance, the
  excluded PDF/multi-page scope, the snapshot note, the troubleshooting block) and the gated-off BYOD default;
  forbidden patterns (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` / repository import on the primary
  path, a mutable `revision='main'`, direct `from transformers import` / `AutoModelForImageTextToText` /
  `AutoProcessor` / `apply_chat_template(` / `model.generate(` / `torch.inference_mode(` / `from
  huggingface_hub import` / `urllib.request` / `pyarrow` / `safetensors` imports / `torch.optim` / `.backward(` /
  `requires_grad` / `pipe._model` / `pipe._processor` / `extractall(` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `huggingface-hub`, `safetensors`, `numpy`,
`pillow` and `pyarrow`, the package with `--no-deps`, runs `ruff check src tests tools`, `tools/build_notebook.py
--check`, and the unit suite (`tests/`, including `test_adaptation.py`, `test_import_boundary.py`,
`test_role_helpers.py`, `test_notebook_parity.py`; injected runner and parquet opener, no weights —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (CUDA; a CPU runtime is not practical for the default path) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Kaggle script kernel (pre-flight only) | Fresh GPU container that clones the candidate branch, installs the pins and runs `tests/test_model_backed.py` plus the package-API recipe probe | Builder pre-flight to catch defects and fix the recipe before spending a notebook run; **not** promotion evidence for the notebook blob |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CUDA runtime (Colab GPU, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/smoldocling-256m-preview/` or the row-group cache `weights/belfort/` (the standalone path writes
   the manifest itself, stages the missing files from the Hub and reads the pinned row groups over range requests, so
   neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `LINE_MAX_NEW_TOKENS = 128`, `EPOCHS = 8`, `LEARNING_RATE = 1e-4`,
   `BATCH_SIZE = 8`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`,
   `numpy==2.5.3`, `pillow==11.3.0`, `pyarrow==25.0.1`, installed by Section 1 into the isolated environment
   (`dimer_isolated_env_<lock digest>/`, managed CPython 3.12.12, `--require-hashes`) with **no** restart: a run that
   needed a restart is recorded with `restarted: true` and is not promotion evidence (RUN10/REL2);
5. verify every default-path stage completes:
   - the isolated environment built from the carried hash lock with no GitHub access, every later cell routed to its worker;
   - the three carried module cells execute (defining `SmolDoclingPipeline`, `verify_snapshot`, `stage_missing_files`,
     `validate_inputs`, `build_messages`, `evaluation_report`, `doctags_to_text`, `doctags_summary`, `line_doctags`, `edit_distance`, `normalise_text`, `ocr_metrics`,
     `empty_baseline`, `constant_baseline`, `medoid_transcript`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`, `load_byod_dataset`,
     `write_dataset_csv`, the instruction constant and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 13 manifest entries fetched from `docling-project/SmolDocling-256M-preview` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (13 files, the 513 MB
     `model.safetensors` re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified
     directory on `cuda:0` in float32 ;
   - Section 4: `fetch_corpus` reading the eight pinned row groups over HTTPS range requests with every SHA-256 and
     byte total matching (800 lines, about 44 MB); the seeded split into 600 / 60 / 140 with `check_split_disjoint`
     reporting no shared image and the three dataset digests printed; `outputs/…_train.csv` and
     `outputs/…_example_line.png` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings surfaced; the report page rendered; the input manifest written to
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the
     unsupported-instruction probe); the page converted with every structural check `True`,
     `outputs/…_page_frozen.json` written and the `evaluation_report` verdict `sample-sanity` with a
     `word_error_rate` and three `element_count` entries (the inference-only card recorded 0.19 with both tables
     cell-perfect — an observation, not an assertion);
   - Section 6: the empty baseline (CER 1.0 exactly), the constant-transcript baseline (≈ 0.94) and the frozen model's
     test rates (≈ 1.439 CER / 2.491 WER in the Tesla T4 build record — the frozen model answers the transcription instruction with almost nothing — 62 of the 140 hypotheses are empty and 100 are three characters or fewer — while 21 lines loop on a digit or a syllable to the 160-token budget (29 truncated), which is how a CER above 1.0 coexists with hypotheses shorter than the references) with four hypotheses printed under their references;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 28,321,344 trainable of 256,484,928 parameters,
     `first_trainable_layer` 22, and an eight-epoch history with the validation CER falling (build record:
     1.600 → 2.254 / 1.904 / 2.289 / 1.815 / 1.129 / 1.068 / 1.024 / 0.871, `best_epoch` 8);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, the hypothesis
     lengths and `outputs/…_evaluation_report.json` written (the verdicts `adapted_vs_frozen_cer` and `adapted_below_empty_baseline` are
     reported and recorded, not asserted — 0.782 against 1.439 in the earlier blob's Kaggle record, WER 2.491 →
     0.975; the adapted model also cleared the constant baseline there);
   - Section 9: six example panels under `outputs/…_examples/`; the report page converted again by the
     adapted model with `outputs/…_page_adapted.json` (build record: before adaptation 630 new tokens, page WER 0.192, element counts (expected, observed) `section_header_level_1` 1/1, `text` 3/3, `otsl` 2/2, verdict `sample-sanity`; after adaptation 92 new tokens, page WER 0.709, element counts (expected, observed) `section_header_level_1` 1/0, `text` 3/1, `otsl` 2/0, verdict `sample-sanity` — a recorded observation,
     not an assertion); `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (37
     tensors, about 113 MB) and `SmolDoclingPipeline.from_artifact` reloading it with 8/8 identical transcripts on
     eight test lines (the cell asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model
     identity and licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus` block,
     the inference-contract records before and after adaptation, the comparison, the artifact digest, the reload
     parity, the runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the row-group cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `smoldocling_document_extraction_colab.ipynb` (`E2E`, current blob) | the PR head under review | — | Colab Tesla T4, fresh runtime, one pass | **pending** — no hosted one-pass `Run all` of the current blob is recorded |
| `smoldocling_document_extraction_colab.ipynb` (`E2E`) — **earlier blob (generator /1, in-kernel pip install), superseded; history only** | `bc77841` / `aaf8fbf0` | 2026-09-21 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-smoldocling-document-extraction` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | **History only (REL14), `restarted: true`** — 11/11 code cells ok only after a manual restart after the install cell, so not clean-runtime evidence under RUN10/REL2, and the blob is not the current one; 36 files, 560 MB staged from the Hub into a clean cache; comparison {cer: {empty: 1, constant: 0.944, frozen: 1.439, adapted: 0.782}, wer: {empty: 1, constant: 0.999, frozen: 2.491, adapted: 0.975}, cer_macro: {empty: 1, constant: 0.952, frozen: 1.392, adapted: 0.882}, exact_match: {empty: 0, constant: 0, frozen: 0, adapted: 0.014}, delta_vs_frozen: {cer: -0.657, wer: -1.516, cer_macro: -0.51, exact_match: 0.014}, hypothesis_length: {ref_chars: 6169, frozen_hyp_chars: 4839, adapted_hyp_chars: 5225, frozen_truncated: 29, adapted_truncated: 2}}; drawing / scene / page check frozen vs adapted {frozen: {word_error_rate: 0.192}, adapted: {word_error_rate: 0.709}}; reload parity {identical_lines: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-smoldocling-document-extraction/v3/evidence/` in the workspace |
| `smoldocling_document_extraction_colab.ipynb` (`TASK-INFERENCE`, superseded) | `c81ca57` / `1c51b0de3d6d` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-smoldocling-document-extraction` v1) | PASSED — 8/8 code cells, 28 files, 518 MB staged, 367.9 s; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/smoldocling_document_extraction_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/smoldocling_document_extraction_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-21 | `7415a38` / `c27ef8ae` (pre-flight: the build-record placeholders still unfilled in the prose, code identical) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-smoldocling-document-extraction` v2) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout | 1708.5 s | **PASSED after a manual restart — history only** (earlier blob, `restarted: true`; not evidence for the current blob) — 11/11 code cells ok (1 restart after install cell); 36 files, 560 MB staged; the metrics the build record quotes |
| 2026-09-21 | `bc77841` / `aaf8fbf0` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-smoldocling-document-extraction` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 1730.3 s | **PASSED after a manual restart — history only** (earlier blob, `restarted: true`; not evidence for the current blob) — 11/11 code cells ok (1 restart after install cell); 36 files, 560 MB staged from the Hub into a clean cache; comparison {cer: {empty: 1, constant: 0.944, frozen: 1.439, adapted: 0.782}, wer: {empty: 1, constant: 0.999, frozen: 2.491, adapted: 0.975}, cer_macro: {empty: 1, constant: 0.952, frozen: 1.392, adapted: 0.882}, exact_match: {empty: 0, constant: 0, frozen: 0, adapted: 0.014}, delta_vs_frozen: {cer: -0.657, wer: -1.516, cer_macro: -0.51, exact_match: 0.014}, hypothesis_length: {ref_chars: 6169, frozen_hyp_chars: 4839, adapted_hyp_chars: 5225, frozen_truncated: 29, adapted_truncated: 2}}; drawing / scene / page check frozen vs adapted {frozen: {word_error_rate: 0.192}, adapted: {word_error_rate: 0.709}}; reload parity {identical_lines: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-smoldocling-document-extraction/v3/evidence/` in the workspace |
| 2026-09-21 | package API at `7415a38` (pre-flight, not the notebook blob) | Kaggle Tesla T4 script kernel (`kurtvalcorza/dimer-probe-smoldocling-e2e` v1; `torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12, `cuda:0`, float32), branch cloned, pins installed, snapshot staged from the Hub | `tests/test_model_backed.py` (7 passed, 19 warnings in 136.42s (0:02:16)) and the recipe probe: the eight pinned row groups read over range requests (800 lines, digest match), empty and constant baselines, frozen transcription instruction on the 140 test lines, `adapt(epochs=8, lr=1e-4, batch_size=8)` with validation-CER selection (the DocTags line target), adapted evaluation, artifact round trip | 1907 s | **PASS (history only; package API of an earlier revision, not a notebook run)** — 7 passed, 19 warnings in 136.42s (0:02:16); the notebook's 8 code cells re-executed through the package API in 1563 s with peak CUDA memory 5.24 GB; the metrics it produced are the ones the notebook run above recorded (same seed, same split, same recipe) |
| 2026-09-14 | `c81ca57` / `1c51b0de3d6d` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-smoldocling-document-extraction` v1) | Default sample path, `Run all` from a fresh interpreter, no repository checkout | 367.9 s | PASSED — 8/8 code cells, 28 files, 518 MB staged; not evidence for the `E2E` blob |

### Local lock-only drives (not clean-runtime evidence)

Builder pre-flight of the current blob `e3d9e0a51040` (generated at `1ba0fa8`): the notebook's own code cells run in
order in an IPython shell standing in for the kernel, in WSL Ubuntu x86_64 with the GPU hidden
(`CUDA_VISIBLE_DEVICES=-1`), so every stage ran on the CPU. The isolated environment was pre-built by the builder
from `tutorials/requirements-colab.lock.txt` with `--require-hashes` (managed CPython 3.12.12) **minus the 22 CUDA
packages, with the `+cpu` builds of torch 2.14.0 / torchvision 0.29.0 / torchaudio 2.11.0**, and marked with the lock
digest, so Section 1 took its reuse path; the full CUDA lock was not installed here (the same lock, byte-identical
apart from comment lines, built and ran on a Colab T4 in `smolvlm-vision-language-pipeline`). Snapshot fetched from
the Hub (13 files, re-hashed) with `HF_HUB_DISABLE_XET=1` (the Xet transfer stalled in WSL). Driver-only changes, all
printed: reduced split sizes, `EPOCHS = 2`, budgets lowered as stated, `RUN_EXPERIMENT = True` with
`EXPERIMENT_EPOCHS = 2`. Not a supported runtime, not hosted, not the default configuration, and not promotion
evidence; the numbers below are not the default path's.

| Date (UTC) | Path exercised | Cells | Wall | Printed results |
|---|---|---|---|---|
| 2026-10-11 | Default budgets, split 24 / 8 / 12, `EPOCHS = 2` (stopped by the builder during Section 9's adapted page re-conversion, which had run over 60 min on the CPU at the 2048-token budget; the frozen conversion took 40 min) | 11/13 ok before the stop | 7040 s for the 11 cells | Section 5 frozen page: 630 new tokens, page WER 0.192, element counts `section_header_level_1` 1/1, `text` 3/3, `otsl` 2/2, verdict `sample-sanity` (identical to the 2026-09-21 T4 record); Section 7 `started_from: pinned base`, 28,321,344 of 256,484,928 trainable, first layer 22; Section 8 verdicts `adapted_vs_frozen_cer: improved`, `adapted_below_empty_baseline: False`, and the notebook carried on (SDL-M3) |
| 2026-10-11 | Split 16 / 4 / 8, `EPOCHS = 2`, `PAGE_MAX_NEW_TOKENS = 64`, `LINE_MAX_NEW_TOKENS = 48`, Section 10 on; then Section 1 re-run, then BYOD by `BYOD_PATH` (20 Belfort lines in a folder) through Sections 4 and 6 | 13/13 ok, then 5/5 re-run and BYOD cells ok | 2381 s (13 cells) | Section 10: epoch-0 validation CER default 0.881 = experiment 0.881; `default_exports_unchanged` adapter / evaluation_report / result all `True`; Section 9 reload parity 8/8, adapter 73 tensors, 113,294,224 bytes; Section 1 re-run `reused: True`, `worker_reused: True`, later cells kept their state; BYOD re-run of Section 4: `restored_pinned_base: 73`, adapter `None`, 0 tensors still differing from the base, split 13 / 3 / 4 with the 20-test-line caution, effective minimum 12, so Section 6 scored the frozen model (SDL-M2, SDL-m1) |

## Current status

**Candidate.** No clean-runtime execution of the current notebook blob is recorded; a hosted one-pass Colab T4 `Run all` of it is **pending**. The 2026-09-21 Kaggle T4 runs above are of earlier blobs and needed a manual restart after the install cell, so they are history only (REL14) and do not satisfy RUN10/REL2; the package-API pre-flight and the superseded TASK-INFERENCE run are history as well. The local lock-only drives are builder pre-flight, not clean-runtime evidence. Static validation (`tools/validate_release_assets.py`), the generator parity check and the offline unit suite pass on the current source, which is necessary but not sufficient. The registry status stays **Candidate** until a reviewer confirms a recorded one-pass run against the blob under review; promotion is not performed by the builder.

Facts a reviewer should weigh: the sample is nineteenth-century French cursive, far outside the checkpoint's rendered-document
training distribution, which is why the frozen `Convert this page to docling.` instruction sits near the empty baseline (the frozen model answers the transcription instruction with almost nothing — 62 of the 140 hypotheses are empty and 100 are three characters or fewer — while 21 lines loop on a digit or a syllable to the 160-token budget (29 truncated), which is how a CER above 1.0 coexists with hypotheses shorter than the references) and why the gain is a
repair of a domain gap, not evidence about other hands or scripts; the rates are uncapped micro
CER/WER over one crowdsourced transcription and the notebook says so; the 60-line validation split selects the epoch;
the SigLIP encoder and the connector are frozen, so what they cannot resolve in a 128-px line tiled at 512 px
stays unread; the decoder that was tuned answers every instruction, and the page re-converted after adaptation is the
only evidence about what happened to page conversion. Greedy decoding is deterministic on a fixed device and dtype, but the
training of eight decoder layers is not bit-reproducible across GPUs, so a Kaggle number a few hundredths off the build
record is the expected spread, not a finding. The sibling rows on the same split — GOT-OCR 2.0 at 0.759, Florence-2 at 0.797 and
SmolVLM-500M at 0.890 CER — are compared in the card.

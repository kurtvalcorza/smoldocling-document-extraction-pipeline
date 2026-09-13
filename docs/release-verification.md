# Release verification

`tutorials/smoldocling_document_extraction_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) is a
**release candidate** until the exact notebook revision has executed top-to-bottom in a clean
supported runtime. Unit tests, JSON validation, code-cell compilation, the generator parity checks
and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under
DIMER Notebook Specification 2.0. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that
  profile, spec `2.0`, a pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on
  the primary path; exactly one cell tagged `embedded_module` equal to
  `src/smoldocling_document_extraction_pipeline/pipeline.py` after the generator's documented rewrites; the
  inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to `tools/build_notebook.py`
  output for its recorded revision; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline
  manifest, which the notebook asserts against the module before fetching), the revision is a 40-hex
  immutable commit, and the same identity string appears in `README.md`, `MODEL_CARD.md`, and
  `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `SmolDoclingPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `convert`, `doctags_summary`,
  `evaluation_report`), the ceiling print (`MIN_IMAGE_SIDE`, `MAX_IMAGE_SIDE`, `MAX_NEW_TOKENS`,
  `DEFAULT_MAX_NEW_TOKENS`, `DECODING`, `INSTRUCTIONS`), the exports, the learner-facing statements (caller-owned
  instruction and token budget, no score in generated markup, the `truncated` flag, no OCR/layout benchmark,
  word error rate as sanity check, capability exclusions) and the gated-off BYOD default listed in the
  validator; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the
  primary path, a mutable `revision='main'`, direct `from transformers import` / `AutoModelForImageTextToText`
  / `AutoProcessor` / `apply_chat_template(` / `model.generate(` / `from huggingface_hub import` use **outside
  the carried module cell**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy` and
`pillow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit
suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`;
injected runner, no weights). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (**no repository checkout is needed — the notebook is standalone**) |
| Local Windows-venv harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `instruction = 'Convert this page to docling.'`,
   `max_new_tokens = 2048`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded
   in `metadata.dimer.generated_from` and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `SmolDoclingPipeline`, `validate_inputs`,
     `evaluation_report`, `doctags_summary`, `doctags_to_text`, `word_error_rate`, `verify_snapshot`,
     `stage_missing_files`) with no import of the repository package;
   - synthetic 850×1100 report page rendered in code with its RGB SHA-256 printed and the ceilings
     (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `MAX_NEW_TOKENS` 8192, `DEFAULT_MAX_NEW_TOKENS` 2048,
     `DECODING` greedy, the seven `INSTRUCTIONS`) surfaced;
   - pinned `docling-project/SmolDocling-256M-preview` acquisition at the immutable revision through the
     carried module: the inline `MANIFEST` is asserted against the module identity and written to
     `weights/smoldocling-256m-preview/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports
     all 13 manifest entries on a clean runtime, `verify_snapshot` returns its summary dict, and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loads from the verified directory;
   - `validate_inputs` writes `outputs/smoldocling_document_extraction_input_manifest.json` (verdict
     `accepted`, one recorded rejection finding from the unsupported-instruction probe);
   - `convert` returning DocTags with `truncated` false on the default budget; record the element counts
     and the word error rate (the card-pass CPU smoke generated 630 tokens and counted exactly 1
     `section_header_level_1`, 3 `text` and 2 `otsl` with 68 OTSL cell tokens, word error rate 0.19; a
     materially different result is a finding to record, not a failure by itself, because no metric
     is asserted — greedy decoding on a different device or dtype can diverge);
   - `evaluation_report` writes `outputs/smoldocling_document_extraction_evaluation_report.json` with verdict
     `sample-sanity`, one `word_error_rate` entry and three `element_count` entries on the synthetic sample
     (`not-measurable` on BYOD), stated as such;
   - `outputs/smoldocling_document_extraction_result.json`, `outputs/smoldocling_document_extraction.dt`,
     `outputs/smoldocling_document_extraction.txt` and `outputs/smoldocling_document_extraction_annotated.png`
     written with `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/smoldocling_document_extraction_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/smoldocling_document_extraction_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Local pre-flight evidence (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | notebook blob `76c69f4712bb` (commit `60c20ea`, generated at `12b2ef0`; `NOTEBOOK_SOURCE.repository_revision` = `12b2ef0…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 8 code cells: pinned install skipped (pre-installed), `stage_missing_files` fetched all 13 manifest entries (518 MB) from the Hub cache at the pinned revision into the scratch `weights/`, `verify_snapshot` PASS (13 files), `convert` → 630 tokens, `truncated` false, 1 `section_header_level_1` / 3 `text` / 2 `otsl`, 68 OTSL cell tokens, 6 located elements, `evaluation_report` `sample-sanity` (word error rate 0.192, all three `element_count` entries expected = observed), 6 outputs written; a first run of the pre-commit working tree (same code cells, earlier prose) gave identical DocTags in 80.7 s | 78.8 s | PASS — pre-flight only; not promotion evidence |

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Default sample path | | pending — no Colab/Kaggle run yet |

## Current status

No clean-runtime execution in a **supported** runtime (Colab or Kaggle) has been recorded yet; the run is
**pending**. What exists: static validation (`tools/validate_release_assets.py`), the generator parity
checks (`--check` OK), the offline unit suite, and one **local fresh-kernel execution** of the generated
notebook (table above) that exercised the standalone carrier, the real `hf_hub_download` staging path
into an empty `weights/` directory, verification, conversion, the evaluation report and every export —
which is necessary but not promotion evidence because the workstation is not a supported runtime. The
registry status remains **Candidate** until a reviewer confirms a recorded supported-runtime run against
the notebook blob under review and an integrator promotes it. Facts a reviewer should weigh: the CUDA
path has not been executed and, because the model is stored in bfloat16 and loaded that way on CUDA,
GPU output can diverge from the CPU float32 output token-for-token; on the synthetic page both tables
came back cell-perfect while the running text dropped three 6–7-word spans and repeated one 25-word span (word error rate 0.19,
212 words recovered against 203 rendered) and the heading lost its colon — the model's failure mode is
plausible-looking text drift, which no field of the output flags; a single page takes ~21 s on the
reference CPU at 630 tokens, so a Colab CPU runtime should expect the conversion cell to be the long
one; and the upstream README declares two licences (front matter CDLA-Permissive-2.0, body Apache 2.0).

"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package.

This template configures an E2E instruction-adaptation workflow: the pinned docling-project/SmolDocling-256M-preview
snapshot is digest-verified and loaded, 800 MIT-licensed Belfort handwritten line images with their transcripts are
fetched as eight digest-pinned parquet row groups over HTTPS range requests, the records are validated and split by
line, a synthetic report page is converted through the inference contract, the frozen model's `Convert this page to
docling.` is scored over the held-out lines on the text its DocTags carry (character and word error rates) beside an
empty-string and a constant-transcript baseline, a bounded fine-tuning of the last eight decoder layers runs on
cached prefix hidden states with a DocTags line target and validation-CER epoch selection, the held-out split is
scored again, six held-out lines and the report page are re-run with the adapted model, and the adapter is exported
and reloaded.

2026-10-05 review fix cycle (SDL-M1..M4, SDL-m1, SDL-m3): generator /2.2 keys ``isolated_runtime`` (nothing installed
into the kernel; no restart), ``infrastructure_labels`` and ``guided``; quality outcomes are reported verdicts;
``SmolDoclingPipeline.adapt`` restarts from the pinned base and Section 4 restores it before a BYOD re-run; BYOD takes a
path or one uploaded zip; the split's shared-hand boundary is disclosed.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "smoldocling_document_extraction_pipeline",
    "repo_name": "smoldocling-document-extraction-pipeline",
    "stem": "smoldocling_document_extraction",
    "notebook_name": "smoldocling_document_extraction_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "SmolDoclingPipeline",
    "weights_key": "smoldocling-256m-preview",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers", "PIL"],
    "title": "SmolDocling-256M-preview — DIMER E2E handwritten-line transcription adaptation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/smoldocling-document-extraction-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/smoldocling-document-extraction-pipeline/blob/main/tutorials/smoldocling_document_extraction_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-docling--project%2FSmolDocling--256M--preview-ffcc4d?style=flat",
            "https://huggingface.co/docling-project/SmolDocling-256M-preview",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-docling--project%2Fdocling-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/docling-project/docling",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2503.11576-b31b1b.svg", "https://arxiv.org/abs/2503.11576"),
    ],
    "capability": "document page image → DocTags markup under one of the seven supported instructions, and bounded supervised fine-tuning of one instruction — `Convert this page to docling.` on transcribed text lines, through the DocTags output — using the pinned SmolDocling-256M-preview weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned `docling-project/SmolDocling-256M-preview` snapshot (a 513 MB `model.safetensors`; no pickle is opened "
        "anywhere), fetches the first eight row groups of the Belfort-line test shard from the Hugging Face Hub at an "
        "immutable revision with HTTPS range requests (about 44 MB; each row group refused on any SHA-256 or byte-total "
        "mismatch), validates the 800 line records and splits them by line into 600 / 60 / 140, converts a synthetic report "
        "page through the inference contract with an input manifest and a rejection probe, scores the frozen model's "
        "`Convert this page to docling.` over the 140 held-out lines on the text its DocTags carry (character and word error "
        "rates) beside an empty-string and a constant-transcript baseline, runs a bounded fine-tuning of the last eight "
        "decoder layers on cached prefix hidden states with a DocTags line target and validation-CER epoch selection, "
        "scores the held-out lines again, re-runs six held-out lines and the report page with the adapted model, exports the "
        "adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify transcript "
        "parity. The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and "
        "no configuration edit (NOTEBOOK_SPEC 2.5 §5). On a Colab Tesla T4 the recorded runs of the default path took between 26.5 and 27.2 minutes "
        "of session wall (1,633.5 s, 1,590.8 s and 1,618.9 s), of which the eight epochs took 1,103.7 s to 1,129.1 s and scoring the 140 frozen lines 156.5 s to 163.1 s (the hosted Colab T4 runs of blobs `e3d9e0a5`, 2026-10-11, `5b9a54dc` and `282a5afd`, 2026-10-25; yours may differ); a CUDA runtime is used "
        "automatically when present, and **a CPU runtime is not practical for the default path** (greedy decoding of some "
        "800 lines plus 600 cached forwards of a 256M-parameter model)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a zip or folder "
        "in the runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one zip in Colab, then choose **Run after** from "
        "that cell. The zip or folder holds line images plus a `transcripts.csv` (`file`, `text`, optional `id`; one row per "
        "image). After the split the training split needs eight images and the validation split one, so the **effective "
        "minimum is 12 distinct images** (split 8 / 2 / 2); with so few, the held-out rates rest on two lines and the cell says "
        "so. Section 4 first puts the model back to the pinned base, so Sections 5 and 6 score the frozen model on your lines. "
        "The records pass through the same validation, image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact "
        "export and reload-parity cells as the Belfort sample. Uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
    ),
    "intro": (
        "SmolDocling-256M-preview is an Idefics3-arrangement vision–language model built for document conversion: a SigLIP "
        "vision encoder turns each 512-px tile of the page (the longest edge resized to 2048 px, aspect ratio preserved, plus "
        "one global view) into 64 visual tokens through a pixel-shuffle connector, and a 30-layer SmolLM2 decoder reads them "
        "with one of seven supported instructions and generates **DocTags** — a markup in which every element "
        "(`<section_header_level_1>`, `<text>`, `<otsl>` table, `<picture>`, …) is preceded by four `<loc_N>` tokens on a "
        "0–500 grid (256,484,928 parameters in all, published under the **CDLA-Permissive-2.0** licence). Decoding is "
        "**greedy** (`do_sample=False`), deterministic on a fixed device and dtype. The output is **generated markup with no "
        "score**: a well-formed document is not evidence that its words are right.\n\n"
        "What this notebook adds to inference is **adaptation of one instruction on transcribed lines, through the DocTags "
        "output**. The instruction stays `Convert this page to docling.`; the lines are nineteenth-century French council "
        "minutes in cursive — the Belfort-line dataset — far outside the rendered-document distribution the preview was "
        "fine-tuned on [1], and on them the frozen model emits a `<text>` element (or a `<picture>`) whose content is empty, "
        "spaced-out letters or a loop: a character error rate of **1.439** on the 140 held-out lines (the Tesla T4 figure of "
        "the recorded runs (the Kaggle T4 run of blob `aaf8fbf0`, 2026-09-21, and the Colab T4 runs of blobs `e3d9e0a5`, 2026-10-11, `5b9a54dc` and `282a5afd`, 2026-10-25, which all printed it)). So the honest question is narrow: does a bounded fine-tuning of the last eight decoder "
        "layers on 600 transcribed lines — trained to emit exactly the DocTags the model already uses for one text element "
        "that fills the image, with the transcript inside it — move the held-out **CER** and **WER** on a line-disjoint test "
        "split past two **non-adapted baselines** and the frozen model, and what does it do to the report page the same "
        "decoder converts? Three sibling rows adapted GOT-OCR 2.0, Florence-2 and SmolVLM-500M on the same split with their own fixed recipes; this is the "
        "fourth model on the same 800 lines, and the only one whose output contract is a document markup. Nothing here is a "
        "claim about your documents or your script: it is one seeded split of one small labelled set.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (a 13-file manifest with the tokenizer, chat "
        "template and processor files) — no pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies "
        "those files before the processor or the model is constructed. The pipeline loads the checkpoint in **float32 on "
        "every device** (the inference-only tutorial used bfloat16 on CUDA): the adapter is trained in float32 and overlays "
        "without a cast, and CPU, Tesla-class and consumer GPUs then run the same arithmetic."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter, and wants to see what a small document-conversion model does with handwriting outside its documented training data, how to measure that honestly with character and word error rates, and how to adapt it to a small transcribed set without fooling themselves — including what the adaptation costs elsewhere. No prior experience with SmolDocling, vision–language models or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. A CUDA GPU runtime (T4) is required in practice: the default path takes about 25 minutes there and hours on CPU.\n\n**Input → Model → Output.**\n\n| | Page conversion | Line transcription (scored) | Bounded fine-tuning |\n|---|---|---|---|\n| Input | one page image and one of seven supported instructions | one handwritten line image and `Convert this page to docling.` | transcribed line images (600 training and 60 validation in the sample) |\n| Model | SigLIP vision encoder, pixel-shuffle connector and a 30-layer decoder, greedy decoding | the same model | the last eight decoder layers and the final norm, trained with the causal language-model loss on a DocTags line target; everything else frozen |\n| Output | DocTags markup (elements with `<loc_N>` positions), the text it carries, a `truncated` flag — no score | the text inside the DocTags, scored by CER and WER against the transcript | a 113 MB safetensors adapter, and held-out CER / WER / exact match |\n\n**How to use this notebook.** Choose a GPU runtime (**Runtime → Change runtime type → T4 GPU**), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer from the recorded runs (the Kaggle T4 run of blob `aaf8fbf0`, 21 September 2026, and the Colab T4 runs of blobs `e3d9e0a5` on 11 October 2026 and `5b9a54dc` on 25 October 2026; none re-run for this revision). Where those runs agree a worked answer quotes the number; where they differ it gives the observed range — the adapted test CER was 0.782 in the first two and 0.781 in the third — and another GPU can move the last digits. Section 10 is a **change-one-thing experiment**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the Belfort lines, validation and the split *(evaluation practice)* → 5 the inference contract on a rendered report page *(core concept: DocTags)* → 6 two baselines and the frozen model *(evaluation practice)* → 7 bounded fine-tuning *(core concept: what is trained)* → 8 held-out evaluation → 9 the lines by eye, the page again, export and reload *(engineering)* → 10 change one thing (optional) → conclude."
            )
        ]
    },
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled line set with its transcripts, validate it and split it by line "
        "without leakage; convert a synthetic report page through the public API and read DocTags correctly (elements, "
        "location tokens, OTSL cells, the `truncated` flag, a `sample-sanity` report against words you rendered yourself); "
        "measure the frozen instruction's corpus CER and WER on the text its DocTags carry beside two non-adapted baselines; "
        "run a bounded fine-tuning with the model's own instruction-tuning loss and a DocTags line target, explicit "
        "hyperparameters and validation-based epoch selection; evaluate on a line-disjoint test split; look at the adapted "
        "transcripts next to the frozen ones and the references, and at what the report page's conversion does after the "
        "shared decoder was tuned; and export a safetensors adapter that reloads against the pinned base with verified "
        "parity."
    ),
    "exclusions": (
        "conversion of PDFs or multi-page documents (one page image per call), export to Markdown/HTML/JSON documents (that "
        "is `docling_core`, not installed here), sampling or beam search, instructions other than the seven the upstream "
        "README lists, fine-tuning of the vision encoder, the connector, the embeddings, the output head or the first 22 "
        "decoder layers, fine-tuning of any instruction but `Convert this page to docling.`, a metric for layout or table "
        "structure (the report page's element counts stay sanity evidence), evaluation on an OCR or document-conversion "
        "benchmark proper (only one seeded 800-line sample is scored here), and any claim that French cursive minutes stand "
        "in for your documents. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with SmolDocling, vision–language models or fine-tuning. The notebook explains DocTags and location tokens, the chat template, greedy decoding, micro and macro CER/WER, the medoid baseline, the frozen-prefix cache and the adapter where they are first used; the Glossary repeats them.",
        "- **Overlap with the training data:** SmolDocling [1] was fine-tuned from SmolVLM-256M-Instruct on document-conversion "
        "sets (SynthCodeNet, SynthFormulaNet, SynthChartNet, DoclingMatix; model card), and SmolVLM-256M was trained on The Cauldron "
        "and Docmatix (model card). None of those cards names the Belfort lines [2] (Zenodo 8041668, public since 2023), and The "
        "Cauldron's subset list (checked 2026-10-25) does not include them, so no overlap with the test lines is documented; it is "
        "inferred to be unlikely, not verified, because the PDF-derived sets are not enumerated line by line. The Cauldron does include "
        "IAM English handwriting lines, so the base has seen some handwriting, though no French cursive is documented.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime with a CUDA GPU (Google Colab or Kaggle GPU). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it. The default path uses CUDA automatically when present. Generation is batched for the corpus stages — the prompt length varies with the image's tile count, so batches are left-padded — and the Kaggle T4 run of an earlier blob (`aaf8fbf0`, 2026-09-21) took under 3 minutes to score 140 lines and about 18 minutes for the eight epochs (caching the prefix hidden states for 600 lines took under 6 minutes), and the Colab T4 runs of blobs `e3d9e0a5` (2026-10-11) and `5b9a54dc` (2026-10-25) took about the same, about 25 minutes of cell time for the whole path; a CPU runtime would take hours. The pinned `torch==2.14.0` install and the 513 MB checkpoint are the large downloads of the run; the row groups are about 44 MB.",
        "- **Knowledge:** basic Python and PIL; what a chat template, a user turn and greedy decoding are; what DocTags elements and location tokens are; what character and word error rate measure and why they are not capped at 1; why a self-rendered page is a plumbing check while a held-out split of one labelled set is a measurement of that set only.",
        "- **Data contract:** records are `{id, image, text}` — `image` a PIL image (or a file decodable by Pillow) with sides within 16..16,384 px and at most 4096² pixels, `text` its transcript (1..512 characters after whitespace runs are collapsed; case and punctuation kept). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting de-duplicates by decoded pixels so no image lands in two splits. BYOD accepts one zip (or directory) of images plus a `transcripts.csv` in the layout named above; after the split the training split needs eight images and the validation split one, so the effective BYOD minimum is **12 distinct images** (`min_byod_records()` computes it). Images are matched by file name: two images with the same name in different folders are refused, and `__MACOSX` entries, hidden files and non-image files are skipped and listed. Every refusal names the `transcripts.csv` line, the file and the rule.",
        "- **Validation is structural, not semantic:** every image is decoded and every transcript checked for length, but nothing checks that a transcript says what its image shows — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads eight row groups of `default/test/0000.parquet` from `https://huggingface.co/datasets/Teklia/Belfort-line/resolve/<revision>/` at the immutable parquet-conversion revision `c4a74bbd…` with HTTPS range requests (the parquet footer plus about 44 MB of row-group bytes out of a 210 MB shard), each row group pinned by SHA-256 and byte total in the carried `samples.py` and refused on any mismatch. Belfort-line is published under the MIT licence (Teklia; Tarride et al. 2023); nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Belfort lines, the transcripts and the split\n\n"
                "`fetch_corpus` returns the eight pinned row groups from the cache under `weights/belfort/` or the Hub at the "
                "pinned parquet-conversion revision — `pyarrow` reads the shard's footer and exactly those row groups over "
                "HTTPS range requests; every cached file is re-hashed and every fetched row group refused on any SHA-256 or "
                "byte-total mismatch — and `read_corpus` turns each row into a record: the line image (128 px tall, 145 to "
                "8,956 px wide) and its crowdsourced transcript with whitespace runs collapsed. `build_sample_dataset` draws a "
                "seeded line-level split (600 / 60 / 140). `validate_dataset` then checks every record against the contract, "
                "`check_split_disjoint` asserts no image (by decoded-pixel digest) is shared, and the training split's summary "
                "table is written to `outputs/{stem}_train.csv`.\n\n"
                "Look for: 800 lines and 33,117 reference characters, three digests, and four refusal probes — a duplicate id, "
                "an empty transcript, an image above the side ceiling, and a dataset too small to use — each rejected before "
                "the model does anything.\n\n"
                "*Evaluation practice.* **What the split preserves, and what it does not.** The 800 lines are consecutive rows "
                "of one council's minutes, and the split is made **by line**: no image is in two splits (the pixel-digest check "
                "proves that), but lines cut from the same page share a hand, so test lines can share pages and writers with "
                "training lines. The held-out rates in Section 8 are therefore a within-collection estimate and are optimistic "
                "for a hand the model has never seen.\n\n"
                "**Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (a zip or a folder holding "
                "`transcripts.csv` and the line images, as a path in this runtime — this works on Colab, Kaggle and Jupyter) "
                "or leave `BYOD_PATH` empty to upload exactly one zip through the Colab dialog; then choose **Run after** from "
                "this cell. This cell first puts the model back to the pinned base, so Sections 5 and 6 score the frozen model "
                "on your lines, not the Belfort-adapted one. The effective minimum is 12 distinct images.\n\n"
                "**Predict before running:** the split draws 600 / 60 / 140 lines at random from 800 consecutive lines of one "
                "volume. Can a test line share a writer with a training line — and would that make the test score higher or "
                "lower than on a new writer?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if pipe.adapter is not None:\n"
                "    # A re-run after Section 7 (BYOD, or a new split): Sections 5 and 6 must score the frozen model, not the adapted one.\n"
                "    print({{'restored_pinned_base': len(pipe.restore_base()), 'note': 'the adapted weights were removed; Sections 5-6 score the frozen model again'}})\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_zip = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_zip.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding transcripts.csv and the line images.')\n"
                "        file_name = byod_zip.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding transcripts.csv and the line images.')\n"
                "        byod_zip = Path('work') / 'byod.zip'\n"
                "        byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_zip.write_bytes(payload)\n"
                "    byod_report = {{}}\n"
                "    records = load_byod_dataset(byod_zip, report=byod_report)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records), 'skipped_files': byod_report['skipped'], 'duplicate_images_dropped': len(records) - sum(len(part) for part in splits.values()), 'effective_minimum': min_byod_records()['total']}}\n"
                "    if len(splits['test']) < 20:\n"
                "        print({{'caution': f\"only {{len(splits['test'])}} held-out test lines: the rates move in large steps and carry no dispersion estimate; add lines before reading them\"}})\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_groups = fetch_corpus(cache_dir='weights/belfort')\n"
                "    corpus = read_corpus(corpus_groups)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} @ {{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'row_groups': len(corpus_groups), 'lines': sum(len(v) for v in corpus_groups.values()), 'bytes': sum(len(r['image']) + len(r['text'].encode('utf-8')) for v in corpus_groups.values() for r in v), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "# The training split must hold MIN_RECORDS; validation and test only need one line each (split_dataset checks that).\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'chars': manifest['text_chars'], 'words': manifest['text_words']['total'], 'width': manifest['image_width'], 'height': manifest['image_height'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "example['image'].save('outputs/{stem}_example_line.png')\n"
                "print({{'example': {{'id': example['id'], 'image': list(example['image'].size), 'text': example['text']}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty transcript': [{{**train_records[0], 'text': '   '}}, *train_records[1:8]],\n"
                "    'image above the side ceiling': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 32))}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the three split sizes and their digests, the example line with its transcript, and the four refusals (each names its rule).\n\n<details><summary>Check your reasoning</summary>Yes. The split keeps *images* apart, not *writers*: lines cut from the same page of the same minutes can fall on both sides, and the model can learn a hand's letter shapes from training lines and meet them again in the test lines. That makes the held-out rate optimistic for a new writer — how much is not measured here (the dataset exposes no page or writer key in the pinned rows). On your own data, split by page, writer or volume when your lines come from few sources.</details>"
            ),
        },
        {
            "md": (
                "## 5. Convert a synthetic report page through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: an 850 × 1100 report page "
                "rendered in code — a heading, two paragraphs, an 8×5 and a 5×3 ruled table and a closing sentence — whose "
                "words and element counts are known because you rendered them; a different image family from the handwritten "
                "lines, and a page the adapted model will convert again in Section 9. `validate_inputs` applies exactly the "
                "checks `convert` applies (a PIL image with sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` and at most "
                "`MAX_IMAGE_PIXELS`, one of the seven `INSTRUCTIONS`, `max_new_tokens` in 1..`MAX_NEW_TOKENS`) and returns an "
                "input manifest; a paraphrased instruction is validated too and its rejection recorded as a finding. The "
                "instruction and the token budget are **caller-owned request parameters**. `convert` returns the DocTags, the "
                "text they carry (`doctags_to_text`), `new_tokens` and a `truncated` flag that is true when the budget was "
                "exhausted — **no score exists** in the markup. `doctags_summary` counts the elements the model claims the page "
                "has, and `evaluation_report` with the rendered words and counts is `sample-sanity`: a `word_error_rate` and "
                "one `element_count` entry per tag against references you drew, plumbing evidence for one page (the "
                "inference-only card recorded 0.19 with both tables cell-perfect); whether the model *reads handwriting* is "
                "what Section 6 measures on 140 lines with a metric.\n\n"
                "*Core concept.* **DocTags** is SmolDocling's output markup: each element (`<text>`, `<section_header_level_1>`, an "
                "`<otsl>` table of cell tokens, …) is preceded by four `<loc_N>` tokens giving its box on a 0–500 grid. The model "
                "generates it greedily, token by token, and attaches no score to any of it.\n\n"
                "**Predict before running:** the page you rendered has one heading, three text blocks and two ruled tables. Will "
                "the frozen model report exactly those element counts — and does a well-formed answer tell you its words are right?"
            ),
            "code": (
                "def synthetic_page(width=850, height=1100):\n"
                "    \"\"\"Heading, two paragraphs, an 8x5 and a 5x3 ruled table, one closing line. Returns page, reference text, expected counts.\"\"\"\n"
                "    page = Image.new('RGB', (width, height), 'white')\n"
                "    d = ImageDraw.Draw(page)\n"
                "    body, head = ImageFont.load_default(size=15), ImageFont.load_default(size=22)\n"
                "    words = 'quarterly revenue by region and product line for the fiscal year with notes on methodology'.split()\n"
                "    reference = ['Annual Report: Regional Results']\n"
                "    d.text((70, 50), reference[0], fill='black', font=head)\n"
                "    y = 95\n"
                "    for _para in range(2):\n"
                "        for line in range(6):\n"
                "            text = ' '.join(words[(line * 3 + k) % len(words)] for k in range(11 - (line % 3)))\n"
                "            d.text((70, y), text, fill=(40, 40, 40), font=body)\n"
                "            reference.append(text)\n"
                "            y += 20\n"
                "        y += 16\n"
                "    for x0, y0, x1, y1, rows, cols, first in ((70, 330, 780, 660, 8, 5, 'Region'), (70, 760, 430, 990, 5, 3, 'Item')):\n"
                "        d.rectangle([x0, y0, x1, y1], outline='black', width=2)\n"
                "        rh, cw = (y1 - y0) / rows, (x1 - x0) / cols\n"
                "        d.line([(x0, y0 + rh), (x1, y0 + rh)], fill='black', width=2)\n"
                "        for r in range(2, rows):\n"
                "            d.line([(x0, y0 + rh * r), (x1, y0 + rh * r)], fill=(120, 120, 120), width=1)\n"
                "        for c in range(1, cols):\n"
                "            d.line([(x0 + cw * c, y0), (x0 + cw * c, y1)], fill=(120, 120, 120), width=1)\n"
                "        for r in range(rows):\n"
                "            for c in range(cols):\n"
                "                token = (first if c == 0 else f'Q{{c}}') if r == 0 else (f'North {{r}}' if c == 0 else f'{{(r * 7 + c * 13) % 97 + 1}},{{(r * 31 + c) % 900 + 100:03d}}')\n"
                "                d.text((x0 + cw * c + 8, y0 + rh * r + rh / 2 - 8), token, fill='black', font=body)\n"
                "                reference.append(token)\n"
                "    tail = 'Table 2 summarises the line items; see the appendix for the full breakdown.'\n"
                "    d.text((70, 1010), tail, fill=(40, 40, 40), font=body)\n"
                "    reference.append(tail)\n"
                "    return page, ' '.join(reference), {{'section_header_level_1': 1, 'text': 3, 'otsl': 2}}\n\n\n"
                "page, page_reference, page_counts = synthetic_page()\n"
                "page_name = 'synthetic_report_page_850x1100'\n"
                "page_sha256 = hashlib.sha256(np.asarray(page).tobytes()).hexdigest()\n"
                "PAGE_MAX_NEW_TOKENS = 2048\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_IMAGE_PIXELS': MAX_IMAGE_PIXELS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DEFAULT_LINE_MAX_NEW_TOKENS': DEFAULT_LINE_MAX_NEW_TOKENS, 'DECODING': DECODING, 'INSTRUCTIONS': list(INSTRUCTIONS), 'LINE_DOCTAGS': LINE_DOCTAGS, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'device': pipe.device, 'dtype': pipe.dtype}}}})\n"
                "input_manifest = validate_inputs(page, instruction=DEFAULT_INSTRUCTION, max_new_tokens=PAGE_MAX_NEW_TOKENS, names=[page_name])\n"
                "try:\n"
                "    validate_inputs(page, instruction='Please convert this page to markdown.')\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'unsupported-instruction-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'page': page_name, 'sha256': page_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def convert_page(pipeline, label):\n"
                "    started = time.perf_counter()\n"
                "    result = pipeline.convert(page, instruction=DEFAULT_INSTRUCTION, max_new_tokens=PAGE_MAX_NEW_TOKENS)\n"
                "    seconds = round(time.perf_counter() - started, 3)\n"
                "    summary = doctags_summary(result['doctags'])\n"
                "    checks = {{\n"
                "        'doctags_is_text': isinstance(result['doctags'], str) and bool(result['doctags']),\n"
                "        'greedy_settings': result['generation']['do_sample'] is False and result['generation']['decoding'] == DECODING,\n"
                "        'budget_honoured': result['new_tokens'] <= PAGE_MAX_NEW_TOKENS,\n"
                "        'identity_reported': result['model_id'] == MODEL_ID and result['model_revision'] == MODEL_REVISION,\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'convert output failed a sanity check: {{checks}}')\n"
                "    report = evaluation_report(result, page_reference, page_counts, sample_kind='synthetic (rendered in this notebook)')\n"
                "    with open(f'outputs/{stem}_page_{{label}}.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'result': result, 'summary': summary, 'report': report}}, handle, indent=2, ensure_ascii=False)\n"
                "    wer = next((m['value'] for m in report['metrics'] if m['id'] == 'word_error_rate'), None)\n"
                "    counts = {{m['tag']: (m['expected'], m['observed']) for m in report['metrics'] if m['id'] == 'element_count'}}\n"
                "    print({{label: {{'seconds': seconds, 'checks': checks, 'new_tokens': result['new_tokens'], 'truncated': result['truncated'], 'n_elements': summary['n_elements'], 'n_table_cells': summary['n_table_cells'], 'word_error_rate': None if wer is None else round(wer, 3), 'element_counts (expected, observed)': counts, 'verdict': report['verdict']}}}})\n"
                "    print(result['text'][:300] + ('…' if len(result['text']) > 300 else ''))\n"
                "    return result, seconds, checks, report\n\n\n"
                "frozen_page, frozen_page_seconds, frozen_page_checks, frozen_page_report = convert_page(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                '**What to notice:** the structural checks, `new_tokens` against the budget, the element counts (expected, observed) and the page WER.\n\n<details><summary>Check your reasoning</summary>In the recorded runs the frozen model used 630 new tokens and got every count right — `section_header_level_1` 1/1, `text` 3/3, `otsl` 2/2 — with a page word error rate of 0.192 and the verdict `sample-sanity`. That is a well-converted page, but the counts only show the structure is plausible: the WER, against words you rendered yourself, is the only check on the text, and one page you drew is plumbing evidence, not a measurement of conversion quality.</details>'
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test lines\n\n"
                "Two non-adapted baselines frame the adaptation, each scored by `ocr_metrics` (carried in `metrics.py`): the "
                "**character error rate** and **word error rate** as micro averages — total Levenshtein edits over total "
                "reference characters or words, the corpus CER/WER of the handwriting-recognition literature — beside the "
                "macro (per-line mean) rates and the exact-match rate. Neither rate is capped: a hypothesis longer than its "
                "reference pushes the rate **above 1.0**, the signal that the model is generating text the line does not carry. "
                "The **empty-string** baseline predicts nothing and scores CER 1.0 exactly (every reference character is a "
                "deletion) — the floor any recogniser must beat to do better than silence. The **constant-transcript** baseline "
                "predicts one training transcript — the medoid, the line closest on average to the others — for every test "
                "line: what corpus statistics buy without reading the image. The **frozen model** is scored by `pipe.evaluate`, "
                "which converts every line under `DEFAULT_INSTRUCTION` in left-padded batches of `EVAL_BATCH_SIZE` under a "
                "`LINE_MAX_NEW_TOKENS` budget, strips the DocTags with `doctags_to_text` and scores the text that remains as "
                "the hypothesis. The recorded runs put the frozen model **near the empty baseline**: they measured 1.439 "
                "(hypotheses 0.78 times the reference length — the frozen model answers the transcription instruction with almost nothing — 62 of the 140 hypotheses are empty and 100 are three characters or fewer — while 21 lines loop on a digit or a syllable to the 160-token budget (29 truncated), which is how a CER above 1.0 coexists with hypotheses shorter than the references); read four of them under the "
                "references, with the DocTags they came from.\n\n"
                "*Evaluation practice.* **Predict before running:** an empty answer scores CER exactly 1.0. Will the frozen "
                "model — a document converter tuned on rendered pages, whose base saw some English handwriting — score below 1.0, near it, or above it? And what "
                "would a score *above* 1.0 mean?"
            ),
            "code": (
                "METRICS = ('cer', 'wer', 'cer_macro', 'exact_match')\n"
                "LINE_MAX_NEW_TOKENS = 160  # @param {{type:\"integer\"}}\n\n"
                "baseline_empty = empty_baseline(test_records)\n"
                "baseline_constant = constant_baseline(train_records, test_records)\n"
                "print({{'empty_baseline': {{k: round(baseline_empty[k], 3) for k in METRICS}}, 'n': baseline_empty['n'], 'note': baseline_empty['baseline']}})\n"
                "print({{'constant_baseline': {{k: round(baseline_constant[k], 3) for k in METRICS}}, 'note': baseline_constant['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, instruction=DEFAULT_INSTRUCTION, max_new_tokens=LINE_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'ref_chars': frozen_test['ref_chars'], 'hyp_chars': frozen_test['hyp_chars'], 'truncated': frozen_test['truncated'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for record, hypothesis, doctags in zip(test_records[:4], frozen_test['hypotheses'][:4], frozen_test['doctags'][:4], strict=True):\n"
                "    print({{'id': record['id'], 'reference': record['text'], 'frozen': hypothesis[:120], 'doctags': doctags[:90]}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the three systems' CER and WER, `hyp_chars` against `ref_chars`, the `truncated` count and the four example lines.\n\n<details><summary>Check your reasoning</summary>In the recorded runs, above it: the empty baseline scored CER 1.0 (by construction), the constant-transcript (medoid) baseline 0.944 and the frozen model **1.439** (WER 2.491): worse than saying nothing. Its hypotheses were shorter than the references in total (0.78 times the length) — 62 of the 140 were empty — but 21 lines looped on a digit or a syllable to the 160-token budget, and every extra character is an insertion. An uncapped rate above 1.0 is the signal that a model is generating, not reading.</details>"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the last decoder layers\n\n"
                "`pipe.adapt` trains only the last eight of the 30 SmolLM2 decoder layers and the final norm — 28,321,344 of "
                "256,484,928 parameters — while the SigLIP vision encoder, the connector, the embeddings, the output head and "
                "the first 22 decoder layers stay frozen. Each training line is the chat-templated user turn (the tiles' visual "
                "tokens and the instruction) followed by the assistant turn: the **DocTags line target** — "
                "`<doctag><text><loc_0><loc_0><loc_500><loc_500>` + transcript + `</text></doctag>`, exactly the markup the "
                "frozen model already emits for one text element that fills the image — and the end-of-utterance token; the "
                "loss is the **causal language-model cross-entropy** over the assistant turn with the user turn masked out — "
                "the checkpoint's own instruction-tuning objective and output contract. Because everything before layer 22 is "
                "frozen, its output for every training line is computed once under no gradient and cached (the "
                "**frozen-prefix cache**), and each step runs only the eight trainable layers, the norm and the output head on "
                "those cached states — the loss equals the full model's loss exactly, at a fraction of the cost. AdamW without "
                "weight decay at a fixed learning rate, gradient clipping at 1.0, seeded shuffling, no scheduler, no "
                "augmentation. Epoch 0 records the frozen model's validation rates; every epoch is scored on the 60 validation "
                "lines, and the epoch with the **lowest validation CER** is kept.\n\n"
                "In the recorded runs the validation CER went from 1.600 to 0.871 (epoch 8) "
                "while the loss dropped from about 1.36 to 0.04: the adapted model's edits still add up to 78 % of the reference characters, and 2 of 140 lines are exact. The three sibling rows "
                "on the same split report their own numbers in their own repositories (see the model card); they were not re-run "
                "or paired with this run.\n\n"
                "*Core concept.* Every call to `pipe.adapt` starts from the **pinned base**: tensors an earlier call changed are "
                "restored first, so epoch 0 is always the frozen model. Re-running this cell with a changed field is a fresh run, "
                "not continued training — but it replaces the default results that Sections 8 and 9 export; to compare a change "
                "side by side and leave the default exports untouched, use Section 10.\n\n"
                "**Predict before running:** the training loss will fall every epoch. Will the validation CER still be falling "
                "at epoch 8, or will it have bottomed out earlier?"
            ),
            "code": (
                "EPOCHS = 8  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "settings = {{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE}}\n"
                "if settings != {{'epochs': 8, 'lr': 1e-4, 'batch_size': 8}}:\n"
                "    print({{'note': 'changed settings: this run starts again from the pinned base and replaces the default results of Sections 8-9; Section 10 compares a change side by side instead', 'settings': settings}})\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, instruction=DEFAULT_INSTRUCTION, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, max_new_tokens=LINE_MAX_NEW_TOKENS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'instruction': adapt_result['instruction'], 'target': adapt_result['target'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'first_trainable_layer': adapt_result['first_trainable_layer'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'cache_seconds': adapt_result['cache_seconds'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '**What to notice:** epoch 0 (the frozen model, before any update), the loss column, the validation CER column and `best_epoch`.\n\n<details><summary>Check your reasoning</summary>In the recorded runs, still falling: the validation CER went from 1.600 at epoch 0 (the frozen model) to 0.871 at epoch 8, and epoch 8 — the last — was kept; it was still moving between the last epochs (1.024 → 0.871), so eight epochs are a floor for this recipe, not a converged point. The training loss fell from about 1.36 to 0.04, far faster than the validation CER: a falling loss is not evidence that the model reads better.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test lines were never used for training or epoch selection, and no image appears in two splits. The "
                "adapted model is scored exactly as the frozen model was in Section 6 and the four systems are put side by "
                "side. Read it in this order: **CER** first (the measure the epoch was selected on — the recorded runs measured "
                "1.439 → **0.781–0.782** in the table printed below, past both baselines), then **WER** (2.491 → 0.975), "
                "then the hypothesis length (from 0.78 times the reference length to 0.85), "
                "then the exact-match rate (2 of the 140 lines read perfectly). The cell reports two **verdicts** — whether the "
                "adapted CER `improved`, showed `no gain` or got `worse` against the frozen one, and whether it is below the "
                "empty baseline's 1.0 — and records them in the evaluation report; a fine-tune that does not help is a finding, "
                "not an error, and Section 9 still exports, reloads and writes the result. The split is by line (Section 4): test "
                "lines can share pages and writers with training lines, so the rates are a within-collection estimate, optimistic "
                "for a new hand. One hundred and forty lines from one seeded "
                "split give **no dispersion estimate**; the deltas are sample-sanity evidence that the adaptation contract "
                "works, not a benchmark, and a result on one French council's minutes says nothing about other hands, other "
                "languages or other scripts until you measure them.\n\n"
                "**Predict before running:** will the adapted model beat the empty baseline (CER 1.0)? Will it read whole lines "
                "exactly?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, instruction=DEFAULT_INSTRUCTION, max_new_tokens=LINE_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE)\n"
                "adapted_val = pipe.evaluate(val_records, instruction=DEFAULT_INSTRUCTION, max_new_tokens=LINE_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE)\n"
                "comparison = {{metric: {{'empty': round(baseline_empty[metric], 3), 'constant': round(baseline_constant[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['hypothesis_length'] = {{'ref_chars': adapted_test['ref_chars'], 'frozen_hyp_chars': frozen_test['hyp_chars'], 'adapted_hyp_chars': adapted_test['hyp_chars'], 'frozen_truncated': frozen_test['truncated'], 'adapted_truncated': adapted_test['truncated']}}\n"
                "# Reported verdicts, not assertions: a non-improving fine-tune is a result to record, and export and reload still run.\n"
                "delta_cer = adapted_test['cer'] - frozen_test['cer']\n"
                "adaptation_verdict = 'improved' if delta_cer < 0 else ('no gain' if delta_cer == 0 else 'worse')\n"
                "comparison['verdicts'] = {{'adapted_vs_frozen_cer': adaptation_verdict, 'adapted_below_empty_baseline': bool(adapted_test['cer'] < baseline_empty['cer'])}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'instruction': DEFAULT_INSTRUCTION,\n"
                "    'target': LINE_DOCTAGS,\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'empty': {{k: v for k, v in baseline_empty.items() if k != 'rows'}}, 'constant': {{k: v for k, v in baseline_constant.items() if k != 'rows'}}}},\n"
                "    'frozen_test': {{k: v for k, v in frozen_test.items() if k != 'rows'}},\n"
                "    'validation_metrics': {{k: v for k, v in adapted_val.items() if k != 'rows'}},\n"
                "    'test_metrics': {{k: v for k, v in adapted_test.items() if k != 'rows'}},\n"
                "    'per_line': [{{**frozen_row, 'frozen_hypothesis': frozen_hyp, 'frozen_doctags': frozen_dt, 'adapted_cer': adapted_row['cer'], 'adapted_hypothesis': adapted_hyp, 'adapted_doctags': adapted_dt}} for frozen_row, frozen_hyp, frozen_dt, adapted_row, adapted_hyp, adapted_dt in zip(frozen_test['rows'], frozen_test['hypotheses'], frozen_test['doctags'], adapted_test['rows'], adapted_test['hypotheses'], adapted_test['doctags'], strict=True)],\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'verdicts': comparison['verdicts'], 'adapted_beats_both_baselines': adapted_test['cer'] < min(baseline_empty['cer'], baseline_constant['cer'])}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the four-way table, the `verdicts` row and the hypothesis-length row.\n\n<details><summary>Check your reasoning</summary>In the recorded runs, yes, but it rarely read a line exactly. The verdict was *improved* in each: test CER 1.439 → 0.781–0.782 across the recorded runs (read your value in the table printed above) and WER 2.491 → 0.975, below both baselines (1.0 and 0.944), with the hypotheses at 0.85 times the reference length; only 2 of the 140 lines were exact. The model learned to emit transcripts of about the right length in the right DocTags element, but a CER of about 0.78 still means the edits needed to correct it add up to 78 % of the reference characters — a start, far from a finished recogniser, and measured on lines whose writers it may already have seen.</details>'
            ),
        },
        {
            "md": (
                "## 9. Look at the lines, convert the page again, export the adapter and reload it\n\n"
                "Six held-out lines are written as panels (`outputs/{stem}_examples/`: the line image with the reference, the "
                "frozen answer and the adapted answer beneath it) so the numbers can be checked by eye: look whether the adapted rows "
                "carry the cursive the frozen rows left empty or spaced out. The report page from Section 5 is then converted "
                "again by the adapted model — the decoder that was tuned answers every instruction, so this is a small look at "
                "what the adaptation did *outside* its corpus: the recorded runs measured before adaptation 630 new tokens, page WER 0.192, element counts (expected, observed) `section_header_level_1` 1/1, `text` 3/3, `otsl` 2/2, verdict `sample-sanity`; after adaptation 92 new tokens, page WER 0.709, element counts (expected, observed) `section_header_level_1` 1/0, `text` 3/1, `otsl` 2/0, verdict `sample-sanity` — one page of "
                "evidence, not a measurement.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the eight decoder layers and the norm, about 113 MB in "
                "float32 — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "instruction and the line target, the training configuration and the epoch history (OUT8). "
                "`SmolDoclingPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest, its digest "
                "and its exact tensor set **before** deserialising, refuses any tensor outside the last eight decoder layers "
                "and the norm, and overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory "
                "model (VER2). The cell asserts identical transcripts on eight test lines (VER4) — a contract check, so it stays "
                "a hard check: a reloaded artifact that reads differently would be a broken export.\n\n"
                "**Predict before running:** only the decoder's last layers were tuned, on lines. Will the report page from "
                "Section 5 convert as well as before?"
            ),
            "code": (
                "import shutil\n\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "caption_font = ImageFont.load_default(size=18)\n"
                "for record, frozen_hyp, adapted_hyp in zip(test_records[:6], frozen_test['hypotheses'][:6], adapted_test['hypotheses'][:6], strict=True):\n"
                "    line = record['image']\n"
                "    width = min(1400, line.width)\n"
                "    line = line.resize((width, max(1, round(line.height * width / record['image'].width))))\n"
                "    sheet = Image.new('RGB', (max(width, 1400), line.height + 96), (255, 255, 255))\n"
                "    sheet.paste(line, (0, 0))\n"
                "    marker = ImageDraw.Draw(sheet)\n"
                "    for i, (tag, text) in enumerate((('REF', record['text']), ('FROZEN', frozen_hyp), ('ADAPTED', adapted_hyp))):\n"
                "        marker.text((8, line.height + 6 + i * 28), f'{{tag}}: {{text[:140]}}', fill=(20, 20, 20) if tag != 'FROZEN' else (150, 40, 40), font=caption_font)\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'rows': ['reference', 'frozen answer', 'adapted answer']}})\n\n"
                "adapted_page, adapted_page_seconds, adapted_page_checks, adapted_page_report = convert_page(pipe, 'adapted')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'instruction': artifact_manifest['adapter']['instruction'], 'best_epoch': artifact_manifest['adapter']['best_epoch']}})\n\n"
                "reloaded = SmolDoclingPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [item['text'] for item in pipe.transcribe([r['image'] for r in test_records[:8]], instruction=DEFAULT_INSTRUCTION, max_new_tokens=LINE_MAX_NEW_TOKENS)]\n"
                "after = [item['text'] for item in reloaded.transcribe([r['image'] for r in test_records[:8]], instruction=DEFAULT_INSTRUCTION, max_new_tokens=LINE_MAX_NEW_TOKENS)]\n"
                "parity = {{'identical_lines': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_lines'] == parity['of']\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'file': CORPUS_FILE, 'license': CORPUS_LICENSE, 'language': CORPUS_LANGUAGE, 'row_groups': sorted(ROW_GROUP_PINS), 'shard_bytes': CORPUS_BYTES}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'page': {{'name': page_name, 'sha256': page_sha256, 'reference_words': len(page_reference.split()), 'expected_counts': page_counts}}, 'frozen': {{'result': frozen_page, 'seconds': frozen_page_seconds, 'checks': frozen_page_checks, 'report': frozen_page_report}}, 'adapted': {{'result': adapted_page, 'seconds': adapted_page_seconds, 'checks': adapted_page_checks, 'report': adapted_page_report}}, 'output_files': ['outputs/{stem}_page_frozen.json', 'outputs/{stem}_page_adapted.json']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': 'outputs/{stem}_examples',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'pillow': PIL.__version__, 'device': pipe.device, 'source': pipe.source, 'dtype': pipe.dtype}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "**What to notice:** the panels in `outputs/{stem}_examples/`, the adapted page's tokens, counts and WER, and `identical_lines` against `of`.\n\n<details><summary>Check your reasoning</summary>In the recorded runs, no: the adapted model answered the same page with 92 new tokens instead of 630, the heading and both tables vanished (element counts 1/0, 3/1, 2/0) and the page WER rose from 0.192 to 0.709: the decoder that was tuned answers every instruction, and it learned to emit one text line. Reload parity held — 8 of 8 identical transcripts — because the adapter carries the trained tensors byte for byte and greedy decoding is deterministic on one device.</details>"
            ),
        },
        {
            "md": (
                "## 10. Change one thing: the learning rate (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, change **one** field — by default the learning rate is "
                "halved to `5e-5` — and run this cell (it needs the variables of Sections 4–9, so run it after them). The "
                "experiment loads its **own** pipeline from the verified snapshot, so it starts from the frozen model and never "
                "touches the default `pipe`; it writes only to `outputs/{stem}_experiment/`, prints the default and the changed "
                "run side by side, and checks that the default exports (adapter, evaluation report, result) are byte-identical "
                "afterwards. It loads a second copy of the model and trains again (the default run's eight epochs took 1,103.7 s to 1,129.1 s on a T4 in the recorded runs).\n\n"
                "**Predict:** with half the learning rate and the same eight epochs, will the held-out CER end higher, lower or "
                "about the same as the default run's? What should epoch 0 of the experiment show?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "EXPERIMENT_EPOCHS = 8  # @param {{type:\"integer\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': Path('outputs/{stem}_evaluation_report.json'), 'result': Path('outputs/{stem}_result.json')}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own pipeline from the verified snapshot: the experiment starts from the frozen model and the default pipe is untouched.\n"
                "    experiment_pipe = SmolDoclingPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n"
                "    experiment_result = experiment_pipe.adapt(train_records, val_records, instruction=DEFAULT_INSTRUCTION, epochs=EXPERIMENT_EPOCHS, lr=EXPERIMENT_LEARNING_RATE, batch_size=BATCH_SIZE, max_new_tokens=LINE_MAX_NEW_TOKENS, progress=report)\n"
                "    experiment_test = experiment_pipe.evaluate(test_records, instruction=DEFAULT_INSTRUCTION, max_new_tokens=LINE_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE)\n"
                "    side_by_side = {{\n"
                "        'settings': {{'default': {{'lr': adapt_result['lr'], 'epochs': adapt_result['epochs']}}, 'experiment': {{'lr': EXPERIMENT_LEARNING_RATE, 'epochs': EXPERIMENT_EPOCHS}}}},\n"
                "        'epoch0_validation_cer (frozen model)': {{'default': round(adapt_result['history'][0]['val']['cer'], 3), 'experiment': round(experiment_result['history'][0]['val']['cer'], 3)}},\n"
                "        'best_epoch': {{'default': adapt_result['best_epoch'], 'experiment': experiment_result['best_epoch']}},\n"
                "        'test': {{metric: {{'frozen': round(frozen_test[metric], 3), 'default': round(adapted_test[metric], 3), 'experiment': round(experiment_test[metric], 3)}} for metric in METRICS}},\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    with open(experiment_dir / 'experiment_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'history': experiment_result['history'], 'test_metrics': {{k: v for k, v in experiment_test.items() if k != 'rows'}}}}, handle, indent=2, ensure_ascii=False)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': str(experiment_dir)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare the `test` columns and the two `best_epoch` values.\n\n"
                "<details><summary>Check your reasoning</summary>Epoch 0 of the experiment must equal epoch 0 of the default run "
                "(validation CER 1.600 in the recorded runs): both are the frozen model, which is what the restart-from-base rule "
                "guarantees. The held-out result is yours to explain. The default run's "
                "validation CER was still falling at its last epoch, so halving the step size most likely leaves the model "
                "further from convergence after the same eight epochs — a higher CER — unless the default rate was "
                "overshooting. With 60 validation and 140 test lines, a difference of a few hundredths is within what a "
                "different seed could produce; say which it was on your run.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A document-conversion model trained on rendered pages, asked to convert a line of nineteenth-century cursive French, "
        "emits the markup it knows and little of the text: the frozen model scores a character error rate of 1.439 "
        "on the Belfort lines. A bounded fine-tuning of the last eight decoder layers on 600 transcribed lines — with the "
        "transcript inside the model's own `<text>` element — moves it to 0.781–0.782 CER and 0.975 WER across the "
        "recorded hosted T4 runs (2 of the held-out lines exact), with a 113 MB adapter that reloads line-for-line. That "
        "is the claim: the adaptation contract works end to end on one instruction of a document-conversion model with a "
        "real labelled set, through the model's own output contract, and the numbers it produces are read as micro and macro "
        "rates, against two non-adapted baselines and the frozen model, with the hypothesis length beside them rather than "
        "in isolation. The three sibling rows on the same split (GOT-OCR 2.0, Florence-2, SmolVLM-500M; their numbers are in their own "
        "repositories and this model card) ran separate fixed recipes, one seeded 140-line split, no paired test and no dispersion estimate, so differences of a few hundredths of CER between them are not a ranking; SmolDocling is the smallest of the four (256M) and started from a frozen position that read nothing; the validation CER was still falling at the last epoch (1.024 → 0.871), so eight epochs are a floor for this recipe, not a converged point; 7 of 140 lines end under 0.5 CER and 19 above 1.0; and the cost is visible on the report page — after adaptation the same instruction returns 92 tokens instead of 630, the section header and both tables vanish (element counts 1/0, 3/1, 2/0) and the page WER moves 0.192 → 0.709: the tuned decoder learned lines and page structure paid for it, the drift the card names.\n\n"
        "The test split is 140 lines from one seeded draw of one 800-line sample, the validation split that picks the epoch is "
        "60, and both rates are corpus edit distances over one crowdsourced transcription of the text the DocTags carry — not "
        "a benchmark, not a measure of layout, reading order or table structure. So a result here says the contract works on "
        "one council's minutes, not that the adapted model handles other hands, other languages, other scripts or your "
        "scans. The decoder that was tuned answers every instruction: the report page re-converted in Section 9 is one page "
        "of evidence about what the tuning did to document conversion (before adaptation 630 new tokens, page WER 0.192, element counts (expected, observed) `section_header_level_1` 1/1, `text` 3/3, `otsl` 2/2, verdict `sample-sanity`; after adaptation 92 new tokens, page WER 0.709, element counts (expected, observed) `section_header_level_1` 1/0, `text` 3/1, `otsl` 2/0, verdict `sample-sanity`), not a measurement, and a "
        "deployment that needs page conversion after adapting must measure it. The decoder was adapted, not the vision "
        "encoder: what the tiles cannot resolve stays unread.\n\n"
        "Three things to carry to real data. **Baselines first:** the empty and constant-transcript rates on *your* transcripts, "
        "and the frozen model's hypothesis length, are the numbers to read before any adapted one. **Rates above 1.0:** an "
        "uncapped CER tells you the model is generating, not reading; a capped one would hide it. **Leakage:** keep every image "
        "in one split (the contract de-duplicates by decoded pixels) and split by page, writer or volume when your lines come "
        "from few sources — lines cut from the same page share a hand.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled line set, validate the "
        "demonstrated dataset contract without leakage, execute the inference contract for one page and a bounded fine-tuning "
        "of one instruction with the model's own objective and output contract, evaluate against two non-adapted baselines "
        "and the frozen model on a line-disjoint split, and emit the shown machine-readable artifacts — without the repository "
        "being reachable. It does **not** establish benchmark superiority, transcription quality on any other hand, language "
        "or document family, page-conversion quality after adaptation, or production fitness.\n\n"
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 halves the learning "
        "rate in its own pipeline and prints it beside the default run — change `EXPERIMENT_LEARNING_RATE` or "
        "`EXPERIMENT_EPOCHS` there (one at a time) and run that cell again; every run starts from the frozen model and writes "
        "only to `outputs/{stem}_experiment/`. Changing `EPOCHS`, `LEARNING_RATE` or `BATCH_SIZE` in Section 7 and choosing "
        "**Run after** also starts from the pinned base, but replaces the default results and exports. Lowering "
        "`LINE_MAX_NEW_TOKENS` in Section 6 to `64` and choosing **Run after** from Section 6 shows how the truncation count "
        "changes. BYOD: `USE_BYOD` and `BYOD_PATH` in Section 4, then **Run after** from Section 4, and read the two "
        "baselines before the adapted number.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"This notebook needs a Linux x86_64 runtime\"** — you are on Windows, macOS or an ARM "
        "machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n"
        "- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a "
        "complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering "
        "`files.pythonhosted.org` or `pypi.org`.\n"
        "- **\"The isolated environment's Python process exited\"** — usually out of memory (Section 9 holds the adapted and "
        "the reloaded model, Section 10 a third). Restart the session and choose **Run all**; leave Section 10 off on a small "
        "GPU.\n"
        "- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the "
        "cells after it keep working. After a session restart, run from the top.\n"
        "- **Where the environment lives** — Section 1 builds it in a folder named `dimer_isolated_env_<lock digest>` in the working "
        "directory. Set the environment variable `DIMER_ISOLATED_ENV` before Section 1 to build it somewhere else.\n"
        "- **No GPU / it is very slow** — the default path needs a CUDA runtime in practice (26.5 to 27.2 minutes of session wall on a T4 in the recorded runs, hours "
        "on CPU). Choose **Runtime → Change runtime type → T4 GPU** and run all again.\n"
        "- **`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3** — a staged file is "
        "incomplete or altered: delete it from the snapshot folder Section 3 prints as `weights_dir` and run Section 3 again "
        "(it needs `huggingface.co`).\n"
        "- **Section 4 cannot fetch a row group, or one fails its digest** — `fetch_corpus` names it; the Hub may be briefly "
        "unreachable. Run Section 4 again (cached row groups are re-hashed); delete `weights/belfort/` if a cached file is "
        "corrupt.\n"
        "- **CUDA out of memory in Section 7 or 10** — set `BATCH_SIZE = 4` in Section 7 and choose **Run after** from "
        "Section 7 (the numbers may differ from the recorded runs).\n"
        "- **BYOD: \"BYOD_PATH … does not exist\"** — the path is relative to the working directory printed in the "
        "message.\n"
        "- **BYOD: \"the upload dialog exists only in Google Colab\"** — on Kaggle or Jupyter, put the zip in the runtime and "
        "set `BYOD_PATH`.\n"
        "- **BYOD: \"Upload exactly one .zip file\"** — the dialog was cancelled or several files were chosen; run Section 4 "
        "again.\n"
        "- **BYOD: \"transcripts.csv line N (file …): …\"** — that row names an image that is not in the dataset or cannot be "
        "decoded; fix that row.\n"
        "- **BYOD: \"two images share the file name …\"** — rows name images by file name; rename one of the two.\n"
        "- **BYOD: \"… image file(s) have no transcripts.csv row\"** — every image needs a row (non-image files and "
        "`__MACOSX` entries are skipped).\n"
        "- **BYOD: \"… leave the training split with … supply at least 12 distinct images\"** — the set is too small for "
        "the split.\n\n"
        "## Glossary\n\n"
        "- **DocTags** — SmolDocling's output markup: one tag per element (`<text>`, `<section_header_level_1>`, `<otsl>`, "
        "`<picture>`, …), each with four location tokens.\n"
        "- **Location tokens (`<loc_N>`)** — an element's box on a 0–500 grid over the page.\n"
        "- **OTSL** — the table markup inside `<otsl>`: one token per cell (`<fcel>`, `<ecel>`, …) and `<nl>` per row.\n"
        "- **Instruction / chat template** — the user turn the model is given (`Convert this page to docling.`) and the "
        "tokenizer's recipe for wrapping it with the image tokens.\n"
        "- **Greedy decoding** — always taking the most likely next token; deterministic on one device, no score attached.\n"
        "- **CER / WER** — character and word error rates: Levenshtein edits over reference characters or words. "
        "**Micro** sums over the corpus; **macro** averages per line. Neither is capped at 1.\n"
        "- **Exact match** — the share of lines read with no error.\n"
        "- **Empty / constant (medoid) baseline** — predict nothing; predict for every line the training transcript closest "
        "on average to the others.\n"
        "- **Causal language-model loss** — cross-entropy of each next token of the target; here only the assistant turn "
        "(the DocTags line) is scored.\n"
        "- **Frozen-prefix cache** — the hidden states after the frozen layers, computed once per line, so training runs "
        "only the tuned tail.\n"
        "- **Epoch / learning rate / validation selection** — one pass over the training lines; the update step size; "
        "choosing the epoch on lines that are neither trained on nor used for the final score.\n"
        "- **Held-out test split / leakage** — lines never used for training or selection; the same image, page or writer "
        "on both sides of a split.\n"
        "- **Adapter / reload parity** — the trained tensors only (safetensors) overlaid on the pinned base; the reloaded "
        "pipeline reads the same lines identically.\n"
        "- **Drift** — what tuning for one task does to the model's other behaviour (here, page conversion).\n"
        "- **BYOD** — bring your own data: your transcribed lines through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The task was ___ on ___ held-out lines.\n"
        "- The empty baseline scored CER ___, the constant baseline ___ and the frozen model ___.\n"
        "- After fine-tuning, CER moved from ___ to ___ and WER from ___ to ___; the verdict was ___; ___ lines were exact.\n"
        "- The report page went from ___ to ___ page WER after tuning, because ___.\n"
        "- One reason not to trust this number on my own documents yet: ___ (for example shared writers across the split, "
        "the size of the test split, or drift).\n\n"
        "## References\n\n"
        "[1] A. Nassar *et al.*, \"SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion,\" arXiv:2503.11576, 2025. [Online]. Available: https://arxiv.org/abs/2503.11576\n\n"
        "[2] S. Tarride *et al.*, \"Handwritten text recognition from crowdsourced annotations,\" in *Proc. 7th Int. Workshop Historical Document Imaging and Processing (HIP)*, 2023, doi: 10.1145/3604951.3605517. Dataset (Teklia, MIT): https://huggingface.co/datasets/Teklia/Belfort-line\n\n"
        "[3] Docling Project, \"docling,\" GitHub repository. [Online]. Available: https://github.com/docling-project/docling\n\n"
        "[4] Upstream model checkpoint. [Online]. Available: https://huggingface.co/{MODEL_ID}\n\n"
        "[5] K. Valcorza, \"smoldocling-document-extraction-pipeline,\" README, weight provenance and model card. [Online]. Available: https://github.com/kurtvalcorza/smoldocling-document-extraction-pipeline/blob/main/README.md, https://github.com/kurtvalcorza/smoldocling-document-extraction-pipeline/blob/main/docs/WEIGHTS.md; model card: https://github.com/kurtvalcorza/smoldocling-document-extraction-pipeline/blob/main/MODEL_CARD.md\n\n"
        "[6] K. Valcorza, sibling rows on the same split. [Online]. Available: https://github.com/kurtvalcorza/got-ocr2-pipeline, https://github.com/kurtvalcorza/florence2-vision-language-pipeline, https://github.com/kurtvalcorza/smolvlm-vision-language-pipeline\n\n"
        "[7] DIMER Notebook Specification 2.5 and Model Card Specification 1.1, fleet specs in the ml-worker repository."
    ),
}

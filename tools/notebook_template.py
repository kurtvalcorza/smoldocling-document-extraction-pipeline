"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "smoldocling_document_extraction_pipeline",
    "repo_name": "smoldocling-document-extraction-pipeline",
    "stem": "smoldocling_document_extraction",
    "notebook_name": "smoldocling_document_extraction_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "SmolDoclingPipeline",
    "weights_key": "smoldocling-256m-preview",
    "runtime_imports": ["torch", "transformers"],
    "title": "SmolDocling-256M-preview — DIMER document page → DocTags extraction tutorial (standalone)",
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
    "capability": "document page image → DocTags markup (layout elements with location tokens, reading order, OCR text and OTSL tables) under one of the seven supported instructions, using the pinned `docling-project/SmolDocling-256M-preview` weights",
    "intro": (
        "At inference the 256M-parameter vision–language model (a SigLIP vision encoder feeding a SmolLM-2 decoder in the "
        "Idefics3 arrangement) reads one page image — resized so its longest edge is 2048 px and split into 512-px tiles of 64 "
        "visual tokens each plus one global view — together with one instruction wrapped in the snapshot's chat template, and "
        "generates **DocTags**: a markup in which every element (`<section_header_level_1>`, `<text>`, `<otsl>` table, "
        "`<picture>`, `<caption>`, …) is preceded by four `<loc_N>` tokens on a 0–500 grid and OTSL tables carry their cells as "
        "`<ched>`/`<fcel>`/`<nl>` tokens. Decoding is greedy (`do_sample=False`) up to a caller-owned `max_new_tokens` budget. "
        "**No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting happens in this "
        "notebook — the upstream checkpoint supplies the weights, processor and chat template, and the carried module adds "
        "snapshot verification, the input contract (a supported instruction, image side ceilings, the token budget), a fixed "
        "output contract, and the `doctags_summary`, `doctags_to_text`, `word_error_rate`, `validate_inputs` and "
        "`evaluation_report` helpers. The default sample is a report page rendered in code from known text, so its words and "
        "its element counts serve as references; the resulting word error rate and count comparison are demonstration "
        "(plumbing) evidence for one page, not a document-conversion benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, render a synthetic report page with known reference text and element counts "
        "(or upload your own page image) and validate it into an input manifest, choose a supported instruction and a token "
        "budget, run the supported task, read DocTags correctly (elements, location tokens, OTSL cells, the `truncated` flag), "
        "recover the plain text with `doctags_to_text`, exercise an optional BYOD path, produce an evaluation report that is "
        "`sample-sanity` with `word_error_rate` and `element_count` measures only when references exist and `not-measurable` "
        "otherwise, and export the DocTags, the recovered text, an annotated page and provenance."
    ),
    "exclusions": (
        "conversion of PDFs or multi-page documents (one page image per call; rendering a PDF to images is the caller's step), "
        "export to Markdown/HTML/JSON documents (that is `docling_core`'s `DoclingDocument.load_from_doctags`, not installed "
        "here), batch or streaming generation, sampling or beam search, instructions other than the seven the upstream README "
        "lists, OCR accuracy or layout/table-structure evaluation on a labelled page set (which this repository does not "
        "ship), and any training. The model is a **preview** release trained on rendered documents; scans, photographs, "
        "handwriting and non-Latin scripts are outside what this notebook measures, and its output can be truncated, repeated "
        "or hallucinated without any signal in the markup itself."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (bfloat16). CPU is adequate but slow: the repository's model card records 6.3 s to load and 21.7 s for one 850×1100 page (630 generated tokens) in the Windows venv (Intel Core Ultra 9 275HX); a denser page or a larger `max_new_tokens` budget scales roughly with the tokens generated. The pinned `torch==2.14.0` install and the 513 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a vision–language model's generated tokens are; what word error rate measures; that markup well-formedness is not correctness.",
        "- **Data:** the default sample is a deterministic 850×1100 report page rendered in code with Pillow's bundled font — a heading, two paragraphs, an 8×5 and a 5×3 ruled table and a closing sentence — so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar) of a **single document page**, any colour mode, sides between 16 and 4096 px. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Render the synthetic report page or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own references: a page with a heading, two six-line "
                "paragraphs of short words, an 8×5 and a 5×3 ruled table of short tokens and a closing sentence is rendered "
                "with Pillow's bundled font at 850×1100 — the same page the repository's smoke run used. The words drawn on it, in "
                "reading order, are the reference text for the `word_error_rate` sanity check later, and the drawn structure "
                "(1 section header, 3 text blocks, 2 tables) is the expected element count. Neither is a labelled dataset, so "
                "nothing here is an OCR or layout benchmark. The image digest is printed for the record. BYOD is optional and "
                "disabled by default; when enabled, upload one page image — no reference exists for it, so the evaluation "
                "report will be `not-measurable`.\n\n"
                "The instruction and the token budget are **caller-owned request parameters**: `instruction` must be one of "
                "the seven forms in `INSTRUCTIONS` (the pinned README's supported-instructions table; the default converts the "
                "whole page), and `max_new_tokens` bounds the generation (`DEFAULT_MAX_NEW_TOKENS = 2048` is a practical "
                "page budget, `MAX_NEW_TOKENS = 8192` is the ceiling the README example uses). Nothing is validated in this "
                "cell — the next section hands the image and the request to the pipeline's own validation stage, which is the "
                "only checker. Look for a dictionary naming the sample kind, the page size and digest, the request, and the "
                "number of reference words."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "instruction = 'Convert this page to docling.'  # @param ['Convert this page to docling.', 'Convert table to OTSL.', \"Find all 'text' elements on the page, retrieve all section headers.\", 'Detect footer elements on the page.']\n"
                "max_new_tokens = 2048  # @param {{type:\"integer\"}}\n\n\n"
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
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    reference_text, expected_counts = None, None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic page: no randomness, so no seed is needed and the digest is stable per Pillow build.\n"
                "    image, reference_text, expected_counts = synthetic_page()\n"
                "    image_name = 'synthetic_report_page_850x1100.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'instruction': instruction, 'max_new_tokens': max_new_tokens, 'reference_words': None if reference_text is None else len(reference_text.split()), 'expected_counts': expected_counts}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `convert` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, an instruction that is one of the seven "
                "`INSTRUCTIONS`, and `max_new_tokens` in `[1, MAX_NEW_TOKENS]` — and returns an **input manifest** naming the "
                "schema (including the preprocessing the processor applies and the decoding rule), the input's observed mode "
                "and size, the request, and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To "
                "show what rejection looks like, the cell also validates a paraphrased instruction the model was not trained on "
                "and records the pipeline's own error message as a finding. Inside the pipeline the image is converted to RGB and "
                "tiled by the processor; nothing else is dropped or altered. The pipeline cannot tell whether the image is a "
                "document page: that contract is the caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING, 'INSTRUCTIONS': list(INSTRUCTIONS)}}}})\n"
                "input_manifest = validate_inputs(image, instruction=instruction, max_new_tokens=max_new_tokens, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, instruction='Please convert this page to markdown.')\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'unsupported-instruction-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Convert the page and read DocTags correctly\n\n"
                "`convert` returns a dict with `doctags` — the generated markup with the terminator tokens removed — `text` "
                "(the plain words `doctags_to_text` recovers: tags, `<loc_N>` and OTSL cell tokens stripped, whitespace "
                "collapsed), the `instruction`, `image_size`, `new_tokens`, a `truncated` flag that is true when the budget "
                "was exhausted, the generation settings and the model identity. **No score exists**: generated markup carries no "
                "probability and no correctness signal, and well-formed tags are not evidence that the text or the layout is "
                "right. `doctags_summary` counts the opening tags per element type, the `<loc_N>` tokens (four per located "
                "element), the OTSL cell tokens and whether the string is wrapped in `<doctag>…</doctag>`. Greedy decoding is "
                "deterministic on a fixed device and dtype; CUDA kernel selection and bfloat16 on GPU can change a token and "
                "therefore the rest of the sequence, so GPU and CPU outputs need not match. As recorded in the model card, the "
                "repository's CPU smoke on this same page generated 630 tokens in 21.7 s and counted exactly 1 "
                "`section_header_level_1`, 3 `text` and 2 `otsl` elements with 68 OTSL cell tokens — both tables cell-perfect, "
                "the paragraphs with three short spans dropped and one span repeated, and the heading without its colon; that is one observation "
                "on one rendered page, not a calibration point."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.time()\n"
                "result = pipe.convert(image, instruction=instruction, max_new_tokens=max_new_tokens)\n"
                "elapsed = time.time() - t0\n"
                "summary = doctags_summary(result['doctags'])\n"
                "print({{'seconds': round(elapsed, 1), 'new_tokens': result['new_tokens'], 'truncated': result['truncated'], 'device': pipe.device, 'dtype': pipe.dtype, 'n_elements': summary['n_elements'], 'n_table_cells': summary['n_table_cells'], 'wrapped_in_doctag': summary['wrapped_in_doctag']}})\n"
                "print({{tag: n for tag, n in summary['counts'].items() if n}})\n"
                "print(result['doctags'][:1200] + ('…' if len(result['doctags']) > 1200 else ''))\n"
                "print('--- recovered text ---')\n"
                "print(result['text'][:600] + ('…' if len(result['text']) > 600 else ''))\n"
                "if result['truncated']:\n"
                "    print('The token budget was exhausted: the DocTags are incomplete. Raise max_new_tokens (ceiling MAX_NEW_TOKENS) and rerun.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No conversion "
                "metric is reported by default: OCR accuracy, layout mAP or table TEDS need labelled pages, and this repository "
                "ships none. The repository's metric helpers are `word_error_rate` (lower-cased, whitespace-tokenised "
                "Levenshtein distance over words, punctuation kept) and the element-count comparison; when reference text is "
                "supplied the report carries one `word_error_rate` entry, and when expected counts are supplied one "
                "`element_count` entry per tag comparing expected and observed, with the verdict `sample-sanity`. On the "
                "synthetic path those references are words and structure **you rendered yourself**, so a low error rate proves "
                "only that the input contract, forward pass, decoding and text recovery round-trip. On BYOD no reference exists, "
                "the verdict is `not-measurable`, and the report states what would make the task measurable. The report is "
                "written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, reference_text, expected_counts, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'doctags_summary')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    if metric['id'] == 'word_error_rate':\n"
                "        print(f\"word_error_rate {{metric['value']:.3f}}  ({{metric['normalisation']}})\")\n"
                "    else:\n"
                "        print(f\"element_count {{metric['tag']:24}} expected {{metric['expected']:>3}}  observed {{metric['observed']:>3}}\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference text or expected counts exist for this input, so nothing is measured; inspect the annotated PNG and the recovered text instead.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full result (DocTags, recovered text, the request, `new_tokens`, "
                "`truncated`), the element summary, the evaluation report, the input manifest, the sample identity, digest and "
                "references, the notebook's source (repository, revision, embedded module digest, generator), the model "
                "identifier, the immutable model revision, the model licence, and the runtime identity (Python, `torch`, "
                "`transformers`, device). The DocTags are also written verbatim to `outputs/{stem}.dt` (the file form "
                "`docling_core` loads) and the recovered text to `outputs/{stem}.txt`, and an annotated PNG draws each located "
                "element's `<loc_N>` box (0–500 grid scaled to the page; headers in red, tables in green, everything else in "
                "blue) for visual inspection — a supplement to, not a replacement for, the machine-readable files. No "
                "credentials are recorded."
            ),
            "code": (
                "import re\n\n"
                "LOC_ELEMENT = re.compile(r'<([a-z_0-9]+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)>')\n"
                "COLOURS = {{'section_header_level_1': (200, 30, 30), 'section_header_level_2': (200, 30, 30), 'otsl': (0, 160, 0)}}\n"
                "annotated = image.convert('RGB').copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "width, height = annotated.size\n"
                "located = []\n"
                "for match in LOC_ELEMENT.finditer(result['doctags']):\n"
                "    tag, x0, y0, x1, y1 = match.group(1), *(int(v) for v in match.groups()[1:])\n"
                "    box = [x0 / 500 * width, y0 / 500 * height, x1 / 500 * width, y1 / 500 * height]\n"
                "    located.append({{'tag': tag, 'box': [round(v, 1) for v in box]}})\n"
                "    draw.rectangle(box, outline=COLOURS.get(tag, (40, 90, 220)), width=2)\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "with open('outputs/{stem}.dt', 'w', encoding='utf-8') as handle:\n"
                "    handle.write(result['doctags'])\n"
                "with open('outputs/{stem}.txt', 'w', encoding='utf-8') as handle:\n"
                "    handle.write(result['text'])\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'doctags_summary': summary,\n"
                "    'located_elements': located,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'reference_text': reference_text, 'expected_counts': expected_counts}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': pipe.dtype,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print({{'located_elements': len(located)}})\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The DocTags are the structure and text the model *claims* the page has; nothing in the markup scores that claim, "
        "and a well-formed document with plausible tables can still carry wrong words, merged lines, repeated phrases or "
        "invented elements. On the synthetic page the `word_error_rate` and `element_count` entries in the evaluation report "
        "compare the output with words and structure you rendered yourself and the verdict is `sample-sanity`, which proves "
        "only that the input contract, forward pass, decoding and text recovery work (the repository's smoke run measured a "
        "word error rate of 0.19 on this page, with both tables cell-perfect and the errors in the running text); they say "
        "nothing about scans, photographs, dense multi-column layouts, formulas, code, non-Latin scripts or long pages, and a "
        "BYOD result is a single-page observation with the verdict `not-measurable`. **The model generates markup for any "
        "image** and stops only at a terminator or the token budget: check `truncated`, expect repetition loops on inputs "
        "unlike its training renders, and treat the `<loc_N>` boxes as approximate (a 0–500 grid, about 2 px per step on this "
        "page). The pipeline provides no PDF rendering, no document export, no batch generation, no evaluation on labelled "
        "pages and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** switch `instruction` to `Convert table to OTSL.` and crop the page to the first table to see a "
        "bare `<otsl>` answer; lower `max_new_tokens` to 200 and watch `truncated` turn true; enable `USE_BYOD` with a page whose "
        "text you know, pass that text as `reference_text` to `evaluation_report` and see the verdict switch to `sample-sanity`; "
        "then install `docling_core` and load `outputs/{stem}.dt` with `DoclingDocument.load_from_doctags` to export Markdown.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/smoldocling-document-extraction-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/smoldocling-document-extraction-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/smoldocling-document-extraction-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code (Docling): https://github.com/docling-project/docling\n"
        "- SmolDocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion (Nassar et al., 2025): https://arxiv.org/abs/2503.11576\n"
        "- Docling Technical Report (Auer et al., 2024): https://arxiv.org/abs/2408.09869"
    ),
}

# Hallu Detect

An evidence-grounded verification starter for generated text. It splits a response into sentence-level claims, retrieves relevant passages from supplied documents, and classifies each claim/evidence pair as **Supported**, **Contradicted**, or **Unknown** with a configured NLI sequence-classification checkpoint.

> This repository is a project starter, not a trained Hallu Detect 2.0 release. No model checkpoint or curated training dataset is included. Sentence splitting is a transparent baseline; it does not guarantee semantic atomicity. See `methodology.md` for scope and limitations.

## Quick start

Requires Python 3.10 or newer. Install the package and the training, API, and test tools:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[train,api,dev]"
```

The optional visual dashboard has its own extra:

```powershell
python -m pip install -e ".[dashboard]"
```

Create `input.json`:

```json
{
  "response": "Water boils at 100 degrees Celsius at sea level. The Moon is made of cheese.",
  "documents": [
    "At standard atmospheric pressure, water boils at 100 degrees Celsius.",
    "The Moon is Earth's only natural satellite and is made mostly of rock."
  ]
}
```

Run verification with a fine-tuned sequence-classification checkpoint:

```powershell
$env:HALLU_MODEL = ".\checkpoints\hallu-detect"
hallu-verify .\input.json
```

The checkpoint must expose meaningful class names (`entailment`, `contradiction`, `neutral`) in its model config, or use `HALLU_LABEL_MAP` to specify them, for example:

```powershell
$env:HALLU_LABEL_MAP = '{"supported":2,"contradicted":0,"unknown":1}'
```

The mapping values are classifier output indices. Do not rely on this example unless it matches the checkpoint's label order.

## Optional dashboard

```powershell
$env:HALLU_MODEL = ".\checkpoints\hallu-detect"
streamlit run .\src\hallu_detect\dashboard.py
```

It highlights each verdict and displays retrieved passages. The starter processes text with the configured model; it has no authentication or production privacy controls.

## Public NLI baseline: prepare, train, evaluate

The optional baseline uses the public Multi-Genre Natural Language Inference (MultiNLI) corpus. It maps entailment to Supported, contradiction to Contradicted, and neutral to Unknown. This is a general NLI benchmark, **not** a hallucination-specific dataset, so its metrics must not be presented as proof of production hallucination-detection performance.

```powershell
hallu-prepare-mnli --output-dir .\data\generated\mnli --train-size 10000 --validation-size 1000 --test-size 1000 --seed 42
hallu-train --train-file .\data\generated\mnli\train.jsonl --validation-file .\data\generated\mnli\validation.jsonl --base-model microsoft/deberta-v3-xsmall --output-dir .\checkpoints\hallu-detect --epochs 1 --batch-size 8
hallu-evaluate --model .\checkpoints\hallu-detect --data-file .\data\generated\mnli\test_mismatched.jsonl --output .\outputs\mnli-mismatched-metrics.json
```

The preparer saves a manifest with source, label mapping, sample sizes, and seed. Training and evaluation data are JSONL with `premise`, `hypothesis`, and `label` fields. Evaluation reports per-class precision, recall, F1, macro averages, and binary not-supported metrics (Contradicted or Unknown). Keep the mismatched split held out until final evaluation. For project-specific claims, replace this benchmark with curated, licensed, domain-relevant data and separately evaluate retrieval.

## Optional API

```powershell
$env:HALLU_MODEL = ".\checkpoints\hallu-detect"
uvicorn hallu_detect.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Send `POST /verify` with the same JSON shape as `input.json`. The API returns a claim-level result and overall supported-claim ratio. Do not expose the development server to untrusted networks without adding authentication, request limits, and operational safeguards.

## Repository guide

- `src/hallu_detect/`: claim splitting, BM25 evidence retrieval, NLI inference, CLI, optional API/dashboard, training, and evaluation.
- `tests/`: dependency-light tests for deterministic core behavior.
- `problem_statement.md`, `background.md`, `academic_notes.md`, `methodology.md`, `references.md`: project proposal and implementation report tabs.

See `methodology.md` for the two-week plan, assumptions, and evaluation protocol.

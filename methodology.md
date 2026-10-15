# Methodology

## System pipeline

1. **Input:** receive a generated response and caller-supplied documents. Preserve document identifiers when provided.
2. **Claim segmentation:** split text into sentence-level units in the starter. This is deliberately a baseline, not a guarantee of atomic factual claims. Evaluate or replace it with a semantic claim decomposition component before claiming atomicity.
3. **Chunking and retrieval:** split evidence into overlapping word chunks and rank by BM25 for each claim. Keep the top passages and their source identifiers.
4. **NLI classification:** pass the retrieved premise and claim hypothesis to a configured three-class sequence classifier. Resolve label indices from model metadata or an explicitly verified mapping; fail on ambiguous labels.
5. **Aggregation:** return a claim verdict, class scores, evidence passages, and an overall faithfulness ratio (supported claims / extracted claims). If no evidence is retrieved, return Unknown without presenting an unsupported classifier result.
6. **Review:** present the claim result with its evidence. Human review remains important for ambiguous, high-impact, or out-of-domain claims.

## Data format

Training, validation, and evaluation files use JSON Lines. Each line is an NLI pair:

```json
{"premise":"Water boils at 100 degrees Celsius at standard atmospheric pressure.","hypothesis":"At sea level, water boils at 100 degrees Celsius.","label":"supported"}
{"premise":"The source says the meeting is on Tuesday.","hypothesis":"The meeting is on Wednesday.","label":"contradicted"}
{"premise":"The source discusses the meeting location only.","hypothesis":"The meeting starts at noon.","label":"unknown"}
```

Use labels `supported`, `contradicted`, or `unknown`. These examples illustrate the schema only; they are not a training dataset. Obtain and document data permissions, provenance, domain coverage, and annotation protocol before training.

## Fine-tuning

The `hallu-prepare-mnli` entry point downloads the public MultiNLI benchmark and writes seeded, size-limited train, matched validation, and mismatched test JSONL files. It maps entailment to Supported, neutral to Unknown, and contradiction to Contradicted; unlabeled examples are filtered. The generated manifest records the dataset source, seed, mapping, and counts. MultiNLI trains sentence-pair NLI, not end-to-end hallucination detection; results on it are an initial baseline only.

The training entry point fine-tunes a Hugging Face sequence classifier (for example, a DeBERTa-v3 or RoBERTa checkpoint) and saves a checkpoint with explicit output-label metadata. The starter label indices are Contradicted=0, Unknown=1, Supported=2. Training hyperparameters are command-line configurable for epochs and batch size; record the chosen base model, learning rate, seed, hardware, and full data configuration in the final report.

Example:

```powershell
hallu-prepare-mnli --output-dir .\data\generated\mnli --train-size 10000 --validation-size 1000 --test-size 1000 --seed 42
hallu-train --train-file .\data\generated\mnli\train.jsonl --validation-file .\data\generated\mnli\validation.jsonl --base-model microsoft/deberta-v3-xsmall --output-dir .\checkpoints\hallu-detect --epochs 1 --batch-size 8
hallu-evaluate --model .\checkpoints\hallu-detect --data-file .\data\generated\mnli\test_mismatched.jsonl --output .\outputs\mnli-mismatched-metrics.json
```

Do not choose the final model using test-set results. Compare against a baseline and report class-wise results.

## Evaluation protocol

Run `hallu-evaluate` on a held-out JSONL set containing the same premise/hypothesis/label fields. Report:

- Precision, recall, F1, and support for each of the three NLI classes.
- Macro averages (to expose minority-class failures) and overall accuracy.
- Binary hallucination-detection precision, recall, and F1. The implementation defines a positive as Contradicted **or** Unknown (not supported); document this definition wherever results are presented.
- Dataset composition, split method, and confidence intervals or repeated-run variability if time and data allow.

Separately inspect evidence retrieval quality and representative errors. A strong NLI score on preselected premises does not establish end-to-end verifier quality if retrieval misses relevant evidence.

## Two-week delivery plan

| Days | Work | Exit artifact |
|---|---|---|
| 1–2 | Confirm scope, checkpoint provenance, dataset licenses, label definitions, compute, and acceptance criteria. | Frozen experiment/data plan |
| 3–4 | Curate and validate data; define source-level train/validation/test splits and annotation checks. | Versioned JSONL splits and data card |
| 5–7 | Establish pretrained baseline; fine-tune candidate NLI model; track parameters and checkpoints. | Reproducible trained checkpoint |
| 8–9 | Integrate retrieval, inference, API/CLI; test missing-evidence and label-mapping behavior. | End-to-end runnable pipeline |
| 10–11 | Run held-out evaluation, error analysis, and retrieval checks; revise only using training/validation data. | Metrics and error analysis |
| 12–13 | Complete proposal tabs, model/data documentation, limitations, and reproducibility instructions. | Populated reports and runbook |
| 14 | Clean environment run, verify artifacts, package deliverables, and identify remaining deployment work. | Release candidate and handoff |

This schedule assumes suitable data and compute are available immediately. Training a reliable model and curating high-quality labels cannot be guaranteed within two weeks without those inputs.

## Current implementation boundaries

The starter implements sentence splitting, BM25 retrieval, configurable three-class Hugging Face inference, a CLI, an optional FastAPI endpoint, fine-tuning, and held-out metric reporting. It does not include curated data, a trained checkpoint, semantic atomic-claim decomposition, a dashboard, production authentication, or deployment automation. These must be completed or explicitly excluded from the final acceptance criteria.
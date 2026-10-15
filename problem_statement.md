# Problem statement

## Project

**Hallu Detect: evidence-grounded verification for generated language-model responses**

## Problem

Large language models can produce fluent statements that are unsupported by their source material or contradicted by it. A user who receives only the generated response cannot easily distinguish well-grounded claims from invented or uncertain ones. Manual verification is slow and difficult to apply consistently across long outputs.

## Aim

Design and evaluate a model-agnostic verification layer that accepts a generated response and a set of supplied source documents, identifies claim-level verification outcomes, and makes the evidence behind each outcome visible.

## Objectives

1. Segment a response into claim-sized units and retain their text for review.
2. Retrieve relevant passages from user-provided source documents for each claim.
3. Apply a fine-tuned three-way natural language inference (NLI) classifier to each premise-claim pair.
4. Return Supported, Contradicted, or Unknown verdicts, confidence values, and evidence.
5. Report claim-level and aggregate evaluation metrics, including precision, recall, and F1 for detecting claims that are not supported.
6. Document model, dataset, limitations, and reproducible evaluation choices.

## Research questions

- How accurately does the selected NLI model distinguish supported, contradicted, and unknown claims in the chosen evaluation data?
- How does evidence retrieval affect claim-level verification compared with using a fixed premise?
- What trade-offs arise between evidence recall, latency, and output interpretability?

## Scope and exclusions

The initial implementation operates on text and source documents supplied by the caller. It is not a fact-checking service over the open web, does not establish truth beyond those sources, and does not guarantee that a sentence-level split is a semantically atomic claim. The optional API is a development interface, not a hardened enterprise deployment.

## Deliverables

- Reproducible Python code for retrieval, inference, training, and evaluation.
- A trained three-class checkpoint and held-out evaluation report, once suitable data and compute are supplied.
- A documented methodology, limitations, and references.
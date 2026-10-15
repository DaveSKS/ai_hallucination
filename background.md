# Background

## Motivation

Text generation is probabilistic: a plausible answer is not necessarily entailed by the evidence available to a user. Verification should therefore be explicit about its evidence scope. This project treats the provided documents as the reference corpus and assesses whether each generated claim is supported by that corpus.

## Natural language inference

Natural language inference (NLI) classifies a premise-hypothesis pair by the relationship between the premise and hypothesis. In this project, the retrieved source passage(s) form the premise and a response claim is the hypothesis:

| NLI relation | Project verdict | Interpretation |
|---|---|---|
| Entailment | Supported | The supplied evidence supports the claim. |
| Contradiction | Contradicted | The supplied evidence conflicts with the claim. |
| Neutral | Unknown | The evidence does not establish either support or contradiction. |

These are model predictions, not guarantees of truth. In particular, retrieval failure can turn a supportable claim into Unknown.

## Retrieval and grounding

The starter uses BM25 lexical ranking over chunks of the documents provided in the request. This has no external service or embedding-model requirement and offers a transparent baseline. It can miss paraphrases and semantically related passages; a dense retriever or hybrid retriever can be evaluated as a later extension. Evidence provenance is returned with each claim so a reviewer can inspect the selected text.

## Model-agnostic design

The inference adapter accepts a Hugging Face sequence-classification checkpoint and maps its three output indices to canonical labels. The mapping must come from the checkpoint's label metadata or be explicitly configured after checking its label order. The repository does not assume that a model named “Hallu Detect 2.0” is publicly available or that its class indices follow any particular order.

## Users and use cases

The intended users are developers, enterprise teams, and system architects who need an explainable verification component in LLM applications. Results should be surfaced as decision support: high-impact claims still require appropriate human review and domain-specific validation.
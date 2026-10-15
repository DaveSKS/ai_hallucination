# Academic notes

## Core concepts

- **Hallucination (operational definition):** a response claim that is contradicted by, or not supported by, the reference evidence available to this verifier. “Unknown” is not proof that a claim is false.
- **Claim:** a response text unit assessed independently against evidence. The current baseline uses sentence-level units; coordinated or multi-fact sentences may need semantic decomposition.
- **Premise:** retrieved source text supplied to the NLI model.
- **Faithfulness ratio:** supported claims divided by all extracted claims. This is an aggregate descriptive score, not a calibrated probability that the complete response is true.
- **Retrieval recall:** whether useful supporting or contradicting evidence is among the retrieved passages. The NLI model cannot assess evidence it never receives.

## Design choices to record

For every experiment, record the dataset version and license, split strategy, source-domain distribution, label definitions, annotator agreement if available, pretrained checkpoint, fine-tuning parameters, random seed, hardware, and metric averaging method. Keep a held-out test set untouched until model and threshold decisions are complete.

## Risks to validity

1. **Dataset leakage:** overlapping source passages or paraphrases across splits can inflate scores. Split by document/source where feasible.
2. **Class imbalance:** accuracy can conceal poor performance on Contradicted or Unknown. Report per-class precision, recall, F1, support, macro-F1, and a clearly defined binary metric.
3. **Annotation ambiguity:** evidence may be incomplete, contradictory, or require external knowledge. Define annotation rules and adjudicate uncertain examples.
4. **Retrieval confounding:** an NLI error may be caused by a poor retrieved premise rather than the classifier. Record retrieved passages and evaluate retrieval separately where possible.
5. **Domain shift:** a model fine-tuned on one domain may not generalize to another.
6. **Confidence interpretation:** softmax scores need not be calibrated. Do not present them as verified probabilities without calibration experiments.

## Open decisions for the final study

- Which licensed, task-appropriate training and evaluation datasets will be used?
- Is “Hallu Detect 2.0” a supplied checkpoint, a model to be trained, or a project label? Confirm its provenance, license, label order, and intended use.
- What hardware and maximum training budget are available?
- Which deployment domain, latency target, and privacy constraints apply?

Do not report model performance until these decisions and the data splits are fixed and the evaluation script has been run on the held-out set.
# Problem Statement

Large language models (LLMs) frequently generate fluent, confident, yet factually incorrect claims, posing a critical reliability challenge for real-world deployment. While existing hallucination detection methods attempt to verify LLM outputs, most rely on single-component approaches: either an LLM-as-judge, similarity-based matching, or retrieval-augmented prompting with a capped verification accuracy of only 70–80% (Huo, 2023). Moreover, these systems typically lack a clear **three-class output** (Supported, Contradicted, Unknown), often misclassifying "no evidence found" as "false" and failing to distinguish between unsupported claims and unverifiable information.

The core problem this project addresses is the absence of a **reproducible, evidence-grounded verification pipeline** that can:
1. Extract individual factual claims from LLM-generated responses,
2. Retrieve external evidence for each claim,
3. Classify each claim as **Supported**, **Contradicted**, or **Unknown** using a dedicated NLI-based model,
4. Produce a transparent, actionable reliability report for end users.

Additionally, this project investigates a secondary research question: **Which LLM generator produces the most verifiable and least hallucinated responses when their answers are passed through a unified external-evidence verifier?**

### Scope and Constraints
- Focuses on open-domain question answering and short-to-medium length responses.
- Uses publicly available evidence sources (e.g., web search or curated corpora) for retrieval.
- Compares three LLM generators (e.g., ChatGPT, Gemini, Claude) against a baseline (TF-IDF + Logistic Regression) and a fine-tuned NLI model.
- Does not finetune a large generative model; focuses on verification and evaluation.

### Significance
This work provides a scientifically defensible framework for auditing LLM outputs, enabling users to distinguish between "verified facts," "contradicted claims," and "unverifiable information." The AI-vs-AI comparison component offers actionable insights into which generators produce outputs that are easier to fact-check, which directly informs both users and model developers.

> *Note: This section should be refined with specific metrics, datasets, and target LLMs once the methodology is finalized.*
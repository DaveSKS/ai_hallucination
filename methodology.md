# Methodology

## 1. System Overview
The proposed system implements an evidence-based hallucination detection pipeline. Given an LLM-generated response, the system:
1. Extracts atomic factual claims,
2. Retrieves supporting evidence for each claim,
3. Classifies each claim using a fine-tuned NLI model into: **Supported**, **Contradicted**, or **Unknown**,
4. Aggregates results into a reliability report with per-claim evidence and an overall score.

**Architecture Flow:**  
User → LLM Generator → Claim Extraction → Evidence Retrieval → NLI Verifier → 3-Class Output + Report

## 2. Data Collection and Generation
- **LLM Generators:** Three LLMs (e.g., ChatGPT, Gemini, Claude) will be queried using standardized prompts across predefined question categories.
- **Question Categories:**
  - Simple facts (1-hop)
  - Multi-step facts (2+ hops)
  - Difficult/obscure facts
  - Recent/current events
  - Long-form explanations
- **Claims:** For each response, atomic factual claims will be extracted using rule-based heuristics and/or a lightweight LLM-assisted extractor.

## 3. Evidence Retrieval
- **Source:** Web search API or a curated corpus (e.g., Wikipedia, search results).
- **Retrieval Strategy:** For each claim, the claim or question is used to retrieve the top-k relevant passages.
- **Handling Missing Evidence:** If retrieval fails or returns irrelevant results, the claim is labeled **Unknown** rather than forced into a binary classification.

## 4. Claim Verification (NLI Model)
- **Baseline Model:** TF-IDF + Logistic Regression (similarity-based)
- **Main Model:** Pre-trained NLI model (e.g., DeBERTa, RoBERTa) fine-tuned on claim–evidence pairs.
- **Output Classes:** 
  - `Supported`: Evidence entails the claim.
  - `Contradicted`: Evidence contradicts the claim.
  - `Unknown`: Evidence is missing, irrelevant, or insufficient to determine truth.
- **Fine-tuning Dataset:** A domain-agnostic claim–evidence dataset (e.g., FEVER, SciFact, or a custom-built set of LLM outputs and verified claims) will be used to fine-tune the NLI model.

## 5. AI-vs-AI Comparison Experiment
- All three LLM generators answer the same set of questions.
- Each response is passed through the same verification pipeline.
- Metrics per generator:
  - % Supported / % Contradicted / % Unknown
  - Overall reliability score
  - Average claims per response
- This experiment tests whether certain generators produce outputs that are more easily verifiable or less prone to hallucination.

## 6. Evaluation Metrics
- **Binary Verification Metrics** (Supported vs. Contradicted):
  - Accuracy, Precision, Recall, F1-score
- **Three-Class Metrics:**
  - Macro-F1, Confusion Matrix
- **Reliability Analysis:**
  - Average % Unknown (should be >0; "no evidence" should not be treated as false)
  - Contradiction rate across generators
- **AI-vs-AI Metrics:**
  - Verifiability ratio, Contradiction rate, Claim diversity

## 7. Implementation Steps
1. Set up environment & install dependencies (Python, scikit-learn, transformers, pandas, etc.)
2. Implement LLM query interface (API or local model)
3. Build claim extraction module
4. Integrate evidence retrieval (search API or local corpus)
5. Train/load NLI model & fine-tune on claim–evidence data
6. Build verification pipeline & reliability report generator
7. Run AI-vs-AI experiments & collect metrics
8. Evaluate, document results, and write final report

> *Note: This methodology should be refined with specific hyperparameters, dataset splits, and evaluation thresholds as the project progresses.*
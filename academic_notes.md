# Academic Framing: Hallucination Detection (Research Notes)

## Paper #1: Retrieving supporting evidence for LLMs generated answers
- **Year / Venue:** 2023 (arXiv:2306.13781)
- **Type:** Retrieval + LLM verification
- **Core question:** Can an LLM reliably verify its own generated answer when provided with retrieved evidence? What is the upper bound of this verification accuracy?
- **Method used:** 
  1. Query a corpus using the combination of the question + generated answer.
  2. Present the LLM with the question + generated answer + retrieved evidence.
  3. Prompt the LLM to indicate whether the generated answer can be supported by the retrieved evidence.
  4. Tested three retrieval approaches: standard BM25 → QA stack → full QA stack with LLM reader.
- **Dataset(s):** MS MARCO V1 test collection (question + passage pairs).
- **Key finding:** LLMs can verify their generated answers when supporting material is provided, but accuracy caps at 70–80%. This approach cannot be fully relied upon to detect hallucinations.
- **Limitations / gaps:** 
  1. Relies on a single LLM as the verifier rather than a dedicated NLI classifier.
  2. Accuracy ceiling (~70–80%) is too low for production use.
  3. Does not handle an “Unknown” classification.
  4. Verification quality depends heavily on retrieval quality.
- **Connection to our project:** Directly validates your core idea: verify LLM claims against external evidence. You can improve it by using a dedicated NLI model (fine-tuned) instead of LLM prompting, and by extending the output to a 3-class system (Supported / Contradicted / Unknown).
- **Quote / snippet to cite:**  We find that an LLM is capable of verifying its generated answer if appropriate supporting material is provided. However, with an accuracy of 70-80%, this approach cannot be fully relied upon to detect hallucinations.
- **Link:** https://arxiv.org/abs/2306.13781

## Paper #2: Natural Language Inference Fine-tuning for Scientific Hallucination Detection
- **Year / Venue:** 2025 (ACL Workshop on Scholarly Document Processing, 2025.sdp-1.33)
- **Type:** NLI fine-tuning
- **Core question:** How can we fine-tune an NLI model to detect hallucinations in scientific text, especially in RAG systems where evidence is provided but claims may still contradict or be unverifiable?
- **Method used:** 
  1. Frames hallucination detection as an NLI task: given a claim and supporting evidence, classify whether the claim is entailed, contradicted, or neutral/unverifiable relative to the evidence.
  2. Fine-tunes a transformer-based NLI model on claim–evidence pairs.
- **Dataset(s):** Scientific document dataset (used for the SciHal 2025 shared task).
- **Key finding:** Fine-tuning an NLI model on claim–evidence pairs improves hallucination detection compared to similarity-based or prompt-based methods. NLI framing turns verification into a reasoning task.
- **Limitations / gaps:** 
  1. Focused on scientific texts, which may limit generalization to open-domain conversational claims.
  2. Likely uses a limited output label set; does not emphasize handling an “Unknown” class.
  3. Relies on curated scientific evidence, which may be harder to scale to open-domain questions.
- **Connection to our project:** Justifies NLI-based verification step. We can adapt this approach to our claim–evidence pairs. Our 3-class system (Supported / Contradicted / Unknown) is a direct improvement over typical NLI setups that only use entailment/contradiction.
- **Quote / snippet to cite:**  "Even in the case of Retrieval-Augmented Generation (RAG) systems, where relevant context is first retrieved and passed in the input, the generated facts can contradict or not be verifiable by the provided references."
- **Link:** https://aclanthology.org/2025.sdp-1.33/

## Paper #3: A Survey on Hallucination in Large Language Models
- **Year / Venue:** 2024 (Revised v2, arXiv:2311.05232)

- **Type:** Survey
- **Core question:** How is hallucination defined, detected, and mitigated across current research? (Implies comprehensive review of definitions, methods, and evaluation practices.)
- **Method used:** Comprehensive review of recent advances in understanding, detecting, and mitigating hallucinations in LLMs. Distinguishes hallucination from related dimensions like factuality, faithfulness, and factual consistency.
- **Dataset(s):** Surveys existing datasets, benchmarks, and evaluation protocols used in hallucination research.
- **Key finding:** Hallucination is an umbrella term; operational meaning varies across studies. No single agreed-upon benchmark. Detection methods are highly heterogeneous and fragmented.
- **Limitations / gaps:** 
  1. Definitions of hallucination are inconsistent across papers.
  2. Evaluation is fragmented; lacks a unified benchmark.
  3. Few works use a 3-class “Supported / Contradicted / Unknown” output.
  4. Many rely on LLM-as-judge rather than NLI + retrieval pipelines.
- **Connection to our project:** Gives us a strong academic foundation for our Background section. We will use it to: 
  1. Cite for formal hallucination definitions.
  2.  Argue that existing work is fragmented and lacks a clear 3-class verification pipeline.
  3.  Position your project as a structured, evidence-based verification approach.
- **Quote / snippet to cite:** "Despite the strong capabilities of Large Language Models (LLMs) across a wide range of natural language processing tasks, their outputs can still be unreliable. Hallucination is widely used as an umbrella label for several related failures, but its operational meaning varies across tasks and studies. No single agreed-upon benchmark exists."
- **Link:**  https://arxiv.org/abs/2311.05232

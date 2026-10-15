"""Claim-to-evidence verification orchestration."""

from dataclasses import asdict, dataclass
from typing import Protocol

from .claims import extract_claims
from .nli import NliPrediction
from .retrieval import BM25Retriever, EvidenceChunk


class Classifier(Protocol):
    def predict(self, premise: str, hypothesis: str) -> NliPrediction: ...


@dataclass(frozen=True)
class ClaimResult:
    claim: str
    verdict: str
    confidence: float
    probabilities: dict[str, float]
    evidence: list[dict[str, str | float]]


class HallucinationVerifier:
    def __init__(self, classifier: Classifier, *, top_k: int = 3) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._classifier = classifier
        self._top_k = top_k

    def verify(self, response: str, documents: list[str | dict[str, str]]) -> dict:
        claims = extract_claims(response)
        retriever = BM25Retriever(documents)
        results: list[ClaimResult] = []

        for claim in claims:
            evidence = retriever.search(claim, top_k=self._top_k)
            if not evidence:
                prediction = NliPrediction(
                    label="unknown",
                    confidence=1.0,
                    probabilities={"supported": 0.0, "contradicted": 0.0, "unknown": 1.0},
                )
            else:
                prediction = self._classifier.predict(
                    premise="\n".join(chunk.text for chunk in evidence),
                    hypothesis=claim,
                )
            results.append(
                ClaimResult(
                    claim=claim,
                    verdict=prediction.label,
                    confidence=prediction.confidence,
                    probabilities=prediction.probabilities,
                    evidence=[
                        {"text": chunk.text, "source": chunk.source, "score": chunk.score}
                        for chunk in evidence
                    ],
                )
            )

        supported_count = sum(result.verdict == "supported" for result in results)
        total = len(results)
        return {
            "overall_faithfulness": supported_count / total if total else 0.0,
            "claim_count": total,
            "supported_count": supported_count,
            "contradicted_count": sum(result.verdict == "contradicted" for result in results),
            "unknown_count": sum(result.verdict == "unknown" for result in results),
            "claims": [asdict(result) for result in results],
        }

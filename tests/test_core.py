import unittest

from hallu_detect.claims import extract_claims
from hallu_detect.nli import _map_from_model_config
from hallu_detect.pipeline import HallucinationVerifier
from hallu_detect.retrieval import BM25Retriever


class FakeClassifier:
    def predict(self, premise: str, hypothesis: str):
        from hallu_detect.nli import NliPrediction

        return NliPrediction(
            label="supported",
            confidence=0.9,
            probabilities={"supported": 0.9, "contradicted": 0.05, "unknown": 0.05},
        )


class CoreTests(unittest.TestCase):
    def test_extracts_sentence_level_claims_and_bullets(self):
        self.assertEqual(
            extract_claims("Facts:\n- Earth orbits the Sun. The Moon orbits Earth."),
            ["Earth orbits the Sun.", "The Moon orbits Earth."],
        )

    def test_retriever_prefers_relevant_document(self):
        retriever = BM25Retriever([
            "The capital of France is Paris.",
            "Penguins are flightless birds."
        ])
        matches = retriever.search("What is the capital of France?")
        self.assertEqual(matches[0].text, "The capital of France is Paris.")

    def test_verifier_marks_claim_unknown_without_retrieved_evidence(self):
        result = HallucinationVerifier(FakeClassifier()).verify(
            "A fact with no matching evidence.", ["Unrelated source material."]
        )
        self.assertEqual(result["claims"][0]["verdict"], "unknown")
        self.assertEqual(result["overall_faithfulness"], 0.0)

    def test_verifier_reports_supported_ratio(self):
        result = HallucinationVerifier(FakeClassifier()).verify(
            "Paris is in France.", ["Paris is the capital of France."]
        )
        self.assertEqual(result["overall_faithfulness"], 1.0)

    def test_checkpoint_canonical_labels_map_to_expected_verdicts(self):
        self.assertEqual(
            _map_from_model_config({0: "contradicted", 1: "unknown", 2: "supported"}),
            {"contradicted": 0, "unknown": 1, "supported": 2},
        )


if __name__ == "__main__":
    unittest.main()

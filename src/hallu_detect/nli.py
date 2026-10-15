"""Hugging Face sequence-classification adapter for three-way NLI."""

import json
import os
import re
from dataclasses import dataclass


class NliConfigurationError(ValueError):
    """Raised when a model's output labels cannot be mapped safely."""


@dataclass(frozen=True)
class NliPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


_CANONICAL_LABELS = ("supported", "contradicted", "unknown")
_LABEL_MATCHES = {
    "supported": ("entail", "support"),
    "contradicted": ("contrad",),
    "unknown": ("neutral", "unknown"),
}


def load_label_map(value: str | None = None) -> dict[str, int] | None:
    """Parse an explicit JSON mapping of canonical labels to classifier output indices."""
    raw = value if value is not None else os.getenv("HALLU_LABEL_MAP")
    if not raw:
        return None
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError as error:
        raise NliConfigurationError("HALLU_LABEL_MAP must be a JSON object") from error
    if (
        not isinstance(mapping, dict)
        or set(mapping) != set(_CANONICAL_LABELS)
        or any(not isinstance(index, int) or isinstance(index, bool) for index in mapping.values())
        or len(set(mapping.values())) != 3
    ):
        raise NliConfigurationError(
            "label map must map supported, contradicted, and unknown to distinct integer indices"
        )
    return mapping


def _map_from_model_config(id2label: dict[int, str]) -> dict[str, int]:
    normalized = {
        index: re.sub(r"[^a-z]", "", str(label).casefold())
        for index, label in id2label.items()
    }
    mapping: dict[str, int] = {}
    for canonical, markers in _LABEL_MATCHES.items():
        matches = [
            index for index, label in normalized.items()
            if any(marker in label for marker in markers)
        ]
        if len(matches) != 1:
            raise NliConfigurationError(
                "model labels do not identify entailment, contradiction, and neutral; "
                "set HALLU_LABEL_MAP to the checkpoint's verified output indices"
            )
        mapping[canonical] = matches[0]
    if len(set(mapping.values())) != 3:
        raise NliConfigurationError("model label names do not map to three distinct classes")
    return mapping


class NliClassifier:
    """Loads a local or Hub sequence-classification checkpoint for inference."""

    def __init__(self, model_name_or_path: str, label_map: dict[str, int] | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "NLI inference dependencies are missing; install the project requirements"
            ) from error

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        config_labels = {
            int(index): label for index, label in self._model.config.id2label.items()
        }
        self._label_map = label_map or _map_from_model_config(config_labels)
        if len(config_labels) != 3 or any(
            index not in config_labels for index in self._label_map.values()
        ):
            raise NliConfigurationError(
                "checkpoint must have exactly three output classes and a valid label mapping"
            )
        self._model.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def predict(self, premise: str, hypothesis: str) -> NliPrediction:
        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits[0]
            probabilities_by_index = self._torch.softmax(logits, dim=-1).cpu().tolist()

        probabilities = {
            label: float(probabilities_by_index[index])
            for label, index in self._label_map.items()
        }
        predicted_label = max(probabilities, key=probabilities.get)
        return NliPrediction(
            label=predicted_label,
            confidence=probabilities[predicted_label],
            probabilities=probabilities,
        )

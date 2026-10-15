"""Report NLI classification and binary not-supported detection metrics."""

import argparse
import json

from .nli import NliClassifier, load_label_map


LABELS = ("supported", "contradicted", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Fine-tuned three-class checkpoint")
    parser.add_argument("--data-file", required=True, help="Labeled JSONL test file")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    try:
        from sklearn.metrics import classification_report, precision_recall_fscore_support
    except ImportError as error:
        raise RuntimeError("install the training extras with: python -m pip install -e '.[train]'") from error

    classifier = NliClassifier(args.model, label_map=load_label_map())
    gold: list[str] = []
    predicted: list[str] = []
    with open(args.data_file, encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(key in row for key in ("premise", "hypothesis", "label")):
                raise ValueError(f"line {line_number} must contain premise, hypothesis, and label")
            if row["label"] not in LABELS:
                raise ValueError(f"line {line_number} has unsupported label: {row['label']}")
            gold.append(row["label"])
            predicted.append(classifier.predict(row["premise"], row["hypothesis"]).label)

    if not gold:
        raise ValueError("evaluation dataset contains no labeled examples")
    report = classification_report(
        gold,
        predicted,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    binary_gold = [label != "supported" for label in gold]
    binary_predicted = [label != "supported" for label in predicted]
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_gold,
        binary_predicted,
        average="binary",
        zero_division=0,
    )
    result = {
        "examples": len(gold),
        "classes": report,
        "hallucination_detection": {
            "positive_definition": "contradicted or unknown (not supported)",
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }
    serialized = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(serialized + "\n")
    else:
        print(serialized)


if __name__ == "__main__":
    main()

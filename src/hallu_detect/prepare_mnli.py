"""Prepare reproducible JSONL subsets of the public MultiNLI benchmark."""

import argparse
import json
from pathlib import Path


MNLI_TO_PROJECT_LABEL = {
    0: "supported",
    1: "unknown",
    2: "contradicted",
}


def _write_split(dataset, path: Path, limit: int, seed: int) -> int:
    filtered = dataset.filter(lambda row: row["label"] in MNLI_TO_PROJECT_LABEL)
    selected = filtered.shuffle(seed=seed).select(range(min(limit, len(filtered))))
    with path.open("w", encoding="utf-8") as output:
        for row in selected:
            record = {
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": MNLI_TO_PROJECT_LABEL[row["label"]],
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/generated/mnli")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_size, args.validation_size, args.test_size) < 1:
        parser.error("all split sizes must be at least 1")

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "install the data preparation extra with: python -m pip install -e '.[train]'"
        ) from error

    dataset = load_dataset("nyu-mll/multi_nli")
    required_splits = {"train", "validation_matched", "validation_mismatched"}
    if not required_splits.issubset(dataset):
        raise ValueError(f"MultiNLI is missing expected splits: {sorted(required_splits)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": _write_split(
            dataset["train"], output_dir / "train.jsonl", args.train_size, args.seed
        ),
        "validation_matched": _write_split(
            dataset["validation_matched"],
            output_dir / "validation.jsonl",
            args.validation_size,
            args.seed,
        ),
        "test_mismatched": _write_split(
            dataset["validation_mismatched"],
            output_dir / "test_mismatched.jsonl",
            args.test_size,
            args.seed,
        ),
    }
    manifest = {
        "dataset": "nyu-mll/multi_nli",
        "source": "https://huggingface.co/datasets/nyu-mll/multi_nli",
        "seed": args.seed,
        "requested_sizes": {
            "train": args.train_size,
            "validation_matched": args.validation_size,
            "test_mismatched": args.test_size,
        },
        "actual_sizes": counts,
        "label_mapping": MNLI_TO_PROJECT_LABEL,
        "note": (
            "General NLI benchmark baseline; this is not a hallucination-specific "
            "evaluation dataset."
        ),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

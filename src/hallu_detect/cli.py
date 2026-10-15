"""Command-line interface for verifying a response against supplied documents."""

import argparse
import json
import os
import sys

from .nli import NliClassifier, load_label_map
from .pipeline import HallucinationVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify generated claims against source documents.")
    parser.add_argument("input", help="JSON file containing response and documents")
    parser.add_argument("--model", default=os.getenv("HALLU_MODEL"))
    parser.add_argument("--output", help="Optional path for the JSON result")
    args = parser.parse_args()

    if not args.model:
        parser.error("provide --model or set HALLU_MODEL to a fine-tuned NLI checkpoint")
    with open(args.input, encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
        parser.error("input JSON must contain a string field named 'response'")
    if not isinstance(payload.get("documents"), list):
        parser.error("input JSON must contain a list field named 'documents'")

    verifier = HallucinationVerifier(
        NliClassifier(args.model, label_map=load_label_map())
    )
    result = json.dumps(
        verifier.verify(payload["response"], payload["documents"]),
        indent=2,
        ensure_ascii=False,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(result + "\n")
    else:
        sys.stdout.write(result + "\n")


if __name__ == "__main__":
    main()

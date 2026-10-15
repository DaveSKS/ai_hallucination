"""Fine-tune a three-class sequence classifier on labeled premise/hypothesis pairs."""

import argparse


LABEL_TO_ID = {"contradicted": 0, "unknown": 1, "supported": 2}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, help="Training JSONL file")
    parser.add_argument("--validation-file", required=True, help="Validation JSONL file")
    parser.add_argument("--base-model", required=True, help="Pretrained Hugging Face model")
    parser.add_argument("--output-dir", required=True, help="Directory for the fine-tuned checkpoint")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    try:
        import numpy as np
        from datasets import load_dataset
        from sklearn.metrics import accuracy_score, f1_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError("install the training extras with: python -m pip install -e '.[train]'") from error

    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.validation_file},
    )
    required_columns = {"premise", "hypothesis", "label"}
    for split in ("train", "validation"):
        if not required_columns.issubset(dataset[split].column_names):
            raise ValueError(f"{split} JSONL rows must contain premise, hypothesis, and label")

    def encode_labels(batch: dict) -> dict:
        try:
            batch["labels"] = [LABEL_TO_ID[label] for label in batch["label"]]
        except KeyError as error:
            raise ValueError(f"unsupported training label: {error.args[0]}") from error
        return batch

    dataset = dataset.map(encode_labels, batched=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=512)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=[column for column in dataset["train"].column_names if column != "labels"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    def compute_metrics(evaluation) -> dict[str, float]:
        predictions = np.argmax(evaluation.predictions, axis=-1)
        return {
            "accuracy": accuracy_score(evaluation.label_ids, predictions),
            "macro_f1": f1_score(evaluation.label_ids, predictions, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

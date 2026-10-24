
"""Fine-tune DistilBERT for legal clause classification."""

import json
import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from config import DATA_DIR, ARTIFACTS_DIR, RANDOM_SEED

MODEL_ID = "distilbert-base-uncased"
MAX_LENGTH = 128

# Start small for CPU training. Increase later for stronger evaluation.
MAX_TRAIN_SAMPLES = int(os.getenv("LEGALTEX_MAX_TRAIN_SAMPLES", "1200"))


def train():
    train_path = DATA_DIR / "train.csv"
    eval_path = DATA_DIR / "evaluation.csv"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            "Dataset files not found. Check data/train.csv and data/evaluation.csv."
        )

    train_df = pd.read_csv(train_path).dropna(subset=["text", "label"])
    eval_df = pd.read_csv(eval_path).dropna(subset=["text", "label"])

    for df in (train_df, eval_df):
        df["text"] = df["text"].astype(str).str.strip()

    train_df = train_df[train_df["text"] != ""].copy()
    eval_df = eval_df[eval_df["text"] != ""].copy()

    labels = sorted(set(train_df["label"]) & set(eval_df["label"]))

    if len(labels) < 2:
        raise ValueError("At least two shared clause categories are required.")

    train_df = train_df[train_df["label"].isin(labels)].copy()
    eval_df = eval_df[eval_df["label"].isin(labels)].copy()

    # Keep examples from every category while limiting initial CPU workload.
    samples_per_label = max(1, MAX_TRAIN_SAMPLES // len(labels))
    train_df = (
        train_df.groupby("label", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), samples_per_label),
                random_state=RANDOM_SEED,
            ),
            include_groups=True,
        )
        .reset_index(drop=True)
    )

    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    train_df["label"] = train_df["label"].map(label2id).astype(int)
    eval_df["label"] = eval_df["label"].map(label2id).astype(int)

    print(f"Training examples: {len(train_df)}")
    print(f"Evaluation examples: {len(eval_df)}")
    print(f"Categories: {len(labels)}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = Dataset.from_pandas(
        train_df[["text", "label"]], preserve_index=False
    ).map(tokenize, batched=True)

    eval_ds = Dataset.from_pandas(
        eval_df[["text", "label"]], preserve_index=False
    ).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = ARTIFACTS_DIR / "distilbert_model"

    args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR / "distilbert_checkpoints"),
        num_train_epochs=1,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to="none",
        seed=RANDOM_SEED,
        fp16=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()

    predictions = trainer.predict(eval_ds)
    predicted_ids = np.argmax(predictions.predictions, axis=1)
    true_ids = np.asarray(eval_df["label"].tolist())

    metrics = {
        "model": MODEL_ID,
        "model_type": "DistilBERT fine-tuned for sequence classification",
        "train_size": int(len(train_df)),
        "test_size": int(len(eval_df)),
        "num_labels": len(labels),
        "max_length": MAX_LENGTH,
        "epochs": 1,
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "macro_f1": float(
            f1_score(true_ids, predicted_ids, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(true_ids, predicted_ids, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            true_ids,
            predicted_ids,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "labels": labels,
        "note": "Results are from one fine-tuning epoch. The baseline uses a different model.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics_path = ARTIFACTS_DIR / "distilbert_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nDistilBERT evaluation:")
    print(json.dumps(
        {key: value for key, value in metrics.items()
         if key != "classification_report"},
        indent=2,
    ))
    print(f"\nSaved model to: {output_dir}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    train()

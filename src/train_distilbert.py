"""Optional, resource-intensive DistilBERT fine-tuning. Run only after baseline works."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from config import DATA_DIR, ARTIFACTS_DIR, RANDOM_SEED

MODEL_ID = "distilbert-base-uncased"

def train():
    train_df = pd.read_csv(DATA_DIR / "train.csv").dropna(subset=["text", "label"])
    eval_df = pd.read_csv(DATA_DIR / "evaluation.csv").dropna(subset=["text", "label"])
    labels = sorted(set(train_df["label"]) & set(eval_df["label"]))
    train_df, eval_df = train_df[train_df.label.isin(labels)].copy(), eval_df[eval_df.label.isin(labels)].copy()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    train_df["label"] = train_df["label"].map(label2id)
    eval_df["label"] = eval_df["label"].map(label2id)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)
    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    eval_ds = Dataset.from_pandas(eval_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    output_dir = ARTIFACTS_DIR / "distilbert_model"
    args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR / "distilbert_checkpoints"),
        num_train_epochs=1,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=RANDOM_SEED,
        fp16=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
    trainer.train()
    pred = trainer.predict(eval_ds)
    predicted_ids = np.argmax(pred.predictions, axis=1)
    true_ids = np.array(eval_df["label"].tolist())
    metrics = {
        "model": MODEL_ID, "train_size": len(train_df), "test_size": len(eval_df),
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "macro_f1": float(f1_score(true_ids, predicted_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_ids, predicted_ids, average="weighted", zero_division=0)),
        "classification_report": classification_report(true_ids, predicted_ids, output_dict=True, zero_division=0),
        "labels": labels,
        "note": "Measured after a single fine-tuning epoch; compare carefully with the baseline split."
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (ARTIFACTS_DIR / "distilbert_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report"}, indent=2))

if __name__ == "__main__":
    train()

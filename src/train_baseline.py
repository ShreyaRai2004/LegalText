"""Train TF-IDF + Logistic Regression and save measured metrics."""
from pathlib import Path
import json
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)
from config import DATA_DIR, ARTIFACTS_DIR, RANDOM_SEED

def train():
    train_path, eval_path = DATA_DIR / "train.csv", DATA_DIR / "evaluation.csv"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError("Dataset files missing. Run: python -m src.data_loader")
    train_df = pd.read_csv(train_path).dropna(subset=["text", "label"])
    eval_df = pd.read_csv(eval_path).dropna(subset=["text", "label"])
    if train_df["label"].nunique() < 2:
        raise ValueError("Need at least two categories to train a classifier.")
    # Keep only categories represented in both splits to ensure a meaningful evaluation.
    common = sorted(set(train_df["label"]) & set(eval_df["label"]))
    train_df = train_df[train_df["label"].isin(common)]
    eval_df = eval_df[eval_df["label"].isin(common)]
    if len(common) < 2 or eval_df.empty:
        raise ValueError("Not enough overlapping labels between training and evaluation splits.")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000,
                                  sublinear_tf=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=RANDOM_SEED, solver="liblinear"))
    ])
    start = time.perf_counter()
    pipeline.fit(train_df["text"].astype(str), train_df["label"].astype(str))
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = pipeline.predict(eval_df["text"].astype(str))
    inference_seconds = time.perf_counter() - start
    labels = sorted(set(eval_df["label"].astype(str)))
    report = classification_report(eval_df["label"].astype(str), predictions,
                                   labels=labels, output_dict=True, zero_division=0)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS_DIR / "baseline_pipeline.joblib")
    metrics = {
        "dataset": "LexGLUE LEDGAR (selected subset)",
        "train_size": int(len(train_df)),
        "test_size": int(len(eval_df)),
        "labels": labels,
        "accuracy": float(accuracy_score(eval_df["label"].astype(str), predictions)),
        "macro_f1": float(f1_score(eval_df["label"].astype(str), predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(eval_df["label"].astype(str), predictions, average="weighted", zero_division=0)),
        "training_seconds": float(training_seconds),
        "inference_seconds_total": float(inference_seconds),
        "classification_report": report,
    }
    (ARTIFACTS_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = confusion_matrix(eval_df["label"].astype(str), predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.65), max(6, len(labels)*0.55)))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=90, colorbar=False)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"Saved pipeline and metrics in {ARTIFACTS_DIR}")

if __name__ == "__main__":
    train()

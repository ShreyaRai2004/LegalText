"""Load LEDGAR from the LexGLUE benchmark and create a manageable CSV subset."""
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from config import DATA_DIR, MAX_LABELS, MAX_TRAIN_SAMPLES, MAX_EVAL_SAMPLES, RANDOM_SEED

DATASET_ID = "coastalcph/lex_glue"
DATASET_CONFIG = "ledgar"

def _as_frame(split, label_names):
    frame = split.to_pandas()
    # LexGLUE LEDGAR uses text and label fields; support common schema variants.
    text_col = "text" if "text" in frame.columns else next((c for c in frame.columns if "text" in c.lower()), None)
    label_col = "label" if "label" in frame.columns else next((c for c in frame.columns if "label" in c.lower()), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not identify text/label columns. Found: {list(frame.columns)}")
    frame = frame[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label_id"})
    frame = frame.dropna(subset=["text", "label_id"])
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].str.len() > 0]
    frame["label_id"] = frame["label_id"].astype(int)
    frame["label"] = frame["label_id"].map(lambda x: label_names[x] if 0 <= x < len(label_names) else str(x))
    return frame[["text", "label"]]

def prepare_dataset(output_dir: Path = DATA_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {DATASET_ID}/{DATASET_CONFIG} from Hugging Face Datasets...")
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG)
    train = _as_frame(dataset["train"], dataset["train"].features["label"].names)
    # Select top labels from training data, then apply the same label set to every split.
    top_labels = train["label"].value_counts().head(MAX_LABELS).index.tolist()
    train = train[train["label"].isin(top_labels)]
    # Keep a stratified sample when possible, preserving representation for selected classes.
    if len(train) > MAX_TRAIN_SAMPLES:
        train = train.groupby("label", group_keys=False).apply(
            lambda x: x.sample(max(1, round(MAX_TRAIN_SAMPLES * len(x) / len(train))),
                               random_state=RANDOM_SEED)
        )
        if len(train) > MAX_TRAIN_SAMPLES:
            train = train.sample(MAX_TRAIN_SAMPLES, random_state=RANDOM_SEED)
    train = train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    eval_split = "validation" if "validation" in dataset else "test"
    evaluation = _as_frame(dataset[eval_split], dataset[eval_split].features["label"].names)
    evaluation = evaluation[evaluation["label"].isin(top_labels)]
    if len(evaluation) > MAX_EVAL_SAMPLES:
        evaluation = evaluation.groupby("label", group_keys=False).apply(
            lambda x: x.sample(max(1, round(MAX_EVAL_SAMPLES * len(x) / len(evaluation))),
                               random_state=RANDOM_SEED)
        )
        if len(evaluation) > MAX_EVAL_SAMPLES:
            evaluation = evaluation.sample(MAX_EVAL_SAMPLES, random_state=RANDOM_SEED)
    evaluation = evaluation.reset_index(drop=True)

    train_path, eval_path = output_dir / "train.csv", output_dir / "evaluation.csv"
    train.to_csv(train_path, index=False)
    evaluation.to_csv(eval_path, index=False)
    metadata = {
        "dataset": f"{DATASET_ID}/{DATASET_CONFIG}",
        "source": "Hugging Face Datasets / LexGLUE LEDGAR",
        "train_records": len(train),
        "evaluation_records": len(evaluation),
        "labels": sorted(train["label"].unique().tolist()),
        "train_split": "train",
        "evaluation_split": eval_split,
        "max_labels_configured": MAX_LABELS,
    }
    import json
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(train)} training records to {train_path}")
    print(f"Saved {len(evaluation)} evaluation records to {eval_path}")
    print("Selected labels:", ", ".join(metadata["labels"]))
    return metadata

if __name__ == "__main__":
    prepare_dataset()

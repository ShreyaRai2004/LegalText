
"""Load LegalText models and classify legal clause text."""

from pathlib import Path
import time

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocessing import validate_text


def load_baseline(path: Path):
    """Load the TF-IDF + Logistic Regression baseline."""
    if not path.exists():
        raise FileNotFoundError(f"Baseline model not found: {path}")
    return joblib.load(path)



def load_distilbert(path: Path | None = None):
    """Load the trained DistilBERT model locally or from Hugging Face."""

    model_id = "Sret12/legaltext-distilbert"

    # Use local files when available; otherwise download from Hugging Face.
    if path is not None and (path / "config.json").exists():
        model_source = str(path)
    else:
        model_source = model_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)

    model.to(device)
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
    }



def predict_text(model, text: str):
    """Return the predicted category, confidence score, and latency."""
    valid_text = validate_text(text)
    start = time.perf_counter()

    # Primary model: fine-tuned DistilBERT
    if isinstance(model, dict) and "tokenizer" in model:
        tokenizer = model["tokenizer"]
        transformer = model["model"]
        device = model["device"]

        inputs = tokenizer(
            valid_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = transformer(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())
        label = transformer.config.id2label.get(
            predicted_id,
            transformer.config.id2label.get(str(predicted_id), str(predicted_id)),
        )
        score = float(probabilities[predicted_id].item())

    # Comparison model: TF-IDF + Logistic Regression
    else:
        label = str(model.predict([valid_text])[0])
        score = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([valid_text])[0]
            score = float(max(probabilities))

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "label": label,
        "score": score,
        "elapsed_ms": elapsed_ms,
    }

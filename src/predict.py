"""Load the baseline model and perform inference."""
from pathlib import Path
import time
import joblib
from src.preprocessing import validate_text

def load_baseline(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Trained baseline not found: {path}. Train it first.")
    return joblib.load(path)

def predict_text(model, text: str):
    valid_text = validate_text(text)
    start = time.perf_counter()
    label = str(model.predict([valid_text])[0])
    score = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([valid_text])[0]
        score = float(max(probabilities))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {"label": label, "score": score, "elapsed_ms": elapsed_ms}

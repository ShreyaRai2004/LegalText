from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MAX_LABELS = int(os.getenv("LEGALTEX_MAX_LABELS", "15"))
MAX_TRAIN_SAMPLES = int(os.getenv("LEGALTEX_MAX_TRAIN_SAMPLES", "12000"))
MAX_EVAL_SAMPLES = int(os.getenv("LEGALTEX_MAX_EVAL_SAMPLES", "2500"))
RANDOM_SEED = 42

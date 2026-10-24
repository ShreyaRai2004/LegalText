# LegalText — AI-Powered Legal Document Classification

LegalText is an educational NLP application that predicts dataset-defined categories for legal contract provisions. It combines a TF-IDF + Logistic Regression baseline with an optional DistilBERT fine-tuning workflow, measured evaluation, Streamlit UI, and local SQLite prediction history.

> **Important:** LegalText is not a legal advice system, contract-validity checker, or substitute for a qualified lawyer. Do not paste confidential real-world documents into the app.

## Features

- Public labelled legal text dataset: LexGLUE LEDGAR, prepared as a manageable subset.
- TF-IDF + Logistic Regression baseline.
- Optional DistilBERT fine-tuning (resource intensive).
- Accuracy, macro-F1, weighted-F1, per-class report, and confusion matrix.
- Streamlit pages for classification, performance, history, and project information.
- SQLite history stored locally, with a limited text preview.
- Basic unit tests.

## Workflow

1. Download and inspect LEDGAR using `src/data_loader.py`.
2. Train the baseline using `src/train_baseline.py`.
3. The fitted vectorizer and classifier are saved together as a Joblib pipeline.
4. Streamlit loads the saved pipeline to predict categories for new text.
5. Prediction metadata is saved in a local SQLite database.
6. Measured evaluation outputs are loaded from JSON; no metrics are hardcoded.

## Requirements

- Windows 10/11, PowerShell, Python 3.11 recommended.
- Internet connection for the first dataset download and optional model downloads.
- Several GB of free disk space is recommended for dependencies and cached data.
- The baseline can run on CPU. DistilBERT fine-tuning can be slow and memory-intensive on older laptops; it is optional.

## Windows PowerShell setup

Open PowerShell in the `LegalText` project directory.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If Python 3.11 is not installed, install it first or use an installed compatible Python version and create the environment with `python -m venv .venv`.

## Prepare the dataset

```powershell
python -m src.data_loader
```

The script downloads the `lex_glue` dataset with the `ledgar` configuration, chooses the most frequent training labels (15 by default), creates a capped subset, and saves `data/train.csv`, `data/evaluation.csv`, and `data/dataset_metadata.json`.

Default caps can be adjusted before running the script:

```powershell
$env:LEGALTEX_MAX_LABELS="10"
$env:LEGALTEX_MAX_TRAIN_SAMPLES="6000"
$env:LEGALTEX_MAX_EVAL_SAMPLES="1500"
python -m src.data_loader
```

The selected labels are determined from the actual training data; they are not invented in code. Review the dataset's upstream documentation and applicable license/terms before use. First download requires an internet connection and can take time.

## Train the baseline

```powershell
python -m src.train_baseline
```

This trains TF-IDF + Logistic Regression and writes:
- `artifacts/baseline_pipeline.joblib`
- `artifacts/baseline_metrics.json`
- `artifacts/confusion_matrix.png`

The script prints measured metrics. Do not claim results before this command completes successfully.

## Run the application

```powershell
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open that address in your browser. The classification page remains unavailable until the baseline model has been trained.

## Optional: DistilBERT

Only try this after the baseline works. On an older laptop, prefer a cloud notebook with sufficient RAM/GPU, or skip this optional phase and clearly document that the transformer was not fine-tuned.

Install compatible PyTorch and Transformers versions using the official installation guidance for your platform, then install:

```powershell
pip install transformers
```

Then run:

```powershell
python -m src.train_distilbert
```

This script fine-tunes `distilbert-base-uncased` for one epoch using the prepared subset. It downloads pretrained weights and can require several GB of disk space and substantial memory. If it fails due to memory or package compatibility, the Logistic Regression application remains the primary working implementation. Do not describe the transformer as trained unless the script finishes and saves the model.

## Tests

```powershell
pytest -q
```

## How evaluation works

The baseline reports accuracy, macro-F1, weighted-F1, and per-class precision/recall/F1. Macro-F1 gives every class equal weight; weighted-F1 weights classes by their number of examples. The confusion matrix shows category-level errors. The prepared evaluation split is used as a held-out evaluation set; do not repeatedly tune based on it.

## Privacy and limitations

- Prediction history is stored in `data/legaltext_history.db` on the same computer.
- The app stores a text preview limited to 300 characters, not the full input.
- Do not enter confidential or personally identifiable legal information.
- The class taxonomy is inherited from the selected dataset and may not fit every real-world document.
- Predictions and model scores can be wrong; probability-like scores may not be calibrated.
- This prototype does not interpret legal rights, determine enforceability, or provide legal advice.

## Screenshots

Add real screenshots after the application has been run successfully.

## Suggested future improvements

- Add a validation split for model selection and probability calibration.
- Add error analysis and confidence calibration.
- Add a privacy option to disable text-preview storage.
- Compare more baselines and evaluate on an independently sourced dataset.
- Add a documented model card and dataset card.

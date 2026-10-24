
from pathlib import Path
import json

import streamlit as st

from src.predict import load_distilbert, load_baseline, predict_text


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DISTILBERT_DIR = ARTIFACTS_DIR / "distilbert_model"
BASELINE_PATH = ARTIFACTS_DIR / "baseline_pipeline.joblib"


st.set_page_config(
    page_title="LegalText | Legal Clause Classifier",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ LegalText")
st.subheader("AI-Powered Legal Clause Classification")
st.write(
    "Classify legal agreement clauses into one of 10 categories "
    "using a fine-tuned DistilBERT NLP model."
)


@st.cache_resource
def get_distilbert():
    return load_distilbert()


@st.cache_resource
def get_baseline():
    return load_baseline(BASELINE_PATH)


def read_metrics(filename):
    path = ARTIFACTS_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


metrics = read_metrics("distilbert_metrics.json")

st.markdown("### Model Performance")

if metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    col2.metric("Macro F1-score", f"{metrics.get('macro_f1', 0):.2%}")
    col3.metric("Weighted F1-score", f"{metrics.get('weighted_f1', 0):.2%}")
else:
    st.info("Evaluation metrics are not available yet.")

st.divider()

st.markdown("### Classify a Legal Clause")
legal_text = st.text_area(
    "Paste a clause from a legal agreement:",
    height=180,
    placeholder=(
        "Example: This agreement shall be governed by "
        "the laws of the State of California."
    ),
)

if st.button("Classify Clause", type="primary"):
    if not legal_text.strip():
        st.warning("Please enter a legal clause first.")
    else:
        try:
            with st.spinner("Analyzing clause with DistilBERT..."):
                model = get_distilbert()
                result = predict_text(model, legal_text)

            st.success("Classification complete")
            st.markdown("### Predicted Category")
            st.subheader(result["label"])

            if result["score"] is not None:
                st.progress(result["score"])
                st.write(f"Model confidence: **{result['score']:.2%}**")

            st.caption(
                f"Inference time: {result['elapsed_ms']:.0f} ms"
            )

            with st.expander("Compare with Logistic Regression baseline"):
                try:
                    baseline = get_baseline()
                    baseline_result = predict_text(baseline, legal_text)
                    st.write(
                        f"Baseline prediction: **{baseline_result['label']}**"
                    )
                    if baseline_result["score"] is not None:
                        st.write(
                            "Baseline confidence: "
                            f"**{baseline_result['score']:.2%}**"
                        )
                except Exception as error:
                    st.warning(f"Baseline comparison unavailable: {error}")

        except Exception as error:
            st.error(f"Could not classify this clause: {error}")

st.divider()

with st.expander("About this project"):
    st.write(
        """
        **Primary model:** Fine-tuned DistilBERT  
        **Framework:** PyTorch and Hugging Face Transformers  
        **Baseline:** TF-IDF with Logistic Regression  
        **Dataset:** LEDGAR from the LexGLUE benchmark  
        **Task:** Classify legal clauses into 10 predefined categories.
        """
    )

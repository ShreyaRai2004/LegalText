
from pathlib import Path
import json
import streamlit as st
from src.predict import load_baseline, predict_text

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(
    page_title="LegalText | Legal NLP",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
.hero {
    padding: 1.5rem 1.7rem;
    border-radius: 18px;
    background: linear-gradient(120deg,#eaf2ff,#f8fbff);
    border: 1px solid #d7e5fb;
}
.hero h1 {color: #163a70; margin-bottom: .35rem;}
.small-note {color: #53657d; font-size: .92rem;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚖️ LegalText")
    st.caption("AI-powered legal text classification")
    page = st.radio(
        "Navigate",
        ["Classify text", "Model performance", "About"]
    )
    st.divider()
    st.caption("Experimental classification aid — not legal advice.")

if page == "Classify text":
    st.markdown("""
    <div class="hero">
        <h1>LegalText</h1>
        <p>Classify legal contract provisions using natural
        language processing and machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    model_path = ARTIFACTS / "baseline_pipeline.joblib"

    if not model_path.exists():
        st.warning(
            "Baseline model not found. Run "
            "`python -m src.train_baseline` first."
        )

    text = st.text_area(
        "Paste a legal clause or paragraph",
        height=220,
        placeholder="Paste a non-confidential legal clause here..."
    )

    st.caption(
        "For privacy, do not enter confidential, personal, "
        "or sensitive legal documents."
    )

    if st.button(
        "Analyze text",
        type="primary",
        disabled=not model_path.exists(),
        use_container_width=True
    ):
        if not text.strip():
            st.error("Please enter some text before analyzing.")
        else:
            try:
                model = load_baseline(model_path)
                result = predict_text(model, text)

                st.success("Classification complete!")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted category",
                        result["label"]
                    )

                with col2:
                    if result["score"] is not None:
                        st.metric(
                            "Model score",
                            f'{result["score"]:.1%}'
                        )

                    st.metric(
                        "Inference time",
                        f'{result["elapsed_ms"]:.1f} ms'
                    )

                st.info(
                    "The score is the model's estimated class "
                    "probability, not a guarantee of correctness."
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

elif page == "Model performance":
    st.title("Model performance")
    metrics_path = ARTIFACTS / "baseline_metrics.json"

    if metrics_path.exists():
        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )

        st.caption(
            f"Dataset: {metrics.get('dataset', 'Unknown')} · "
            f"Evaluation records: {metrics.get('test_size', 'Unknown')}"
        )

        cols = st.columns(4)
        keys = ["accuracy", "macro_f1", "weighted_f1", "training_seconds"]
        labels = ["Accuracy", "Macro-F1", "Weighted-F1", "Training time (s)"]

        for col, key, label in zip(cols, keys, labels):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                display = f"{value:.1f}" if key == "training_seconds" else f"{value:.3f}"
            else:
                display = "—"
            col.metric(label, display)

        st.write("### Per-class report")
        report = metrics.get("classification_report", {})

        if report:
            import pandas as pd
            rows = [
                {"Category": category, **scores}
                for category, scores in report.items()
                if isinstance(scores, dict) and "f1-score" in scores
            ]
            if rows:
                st.dataframe(
                    pd.DataFrame(rows).set_index("Category"),
                    use_container_width=True
                )

        cm_path = ARTIFACTS / "confusion_matrix.png"
        if cm_path.exists():
            st.image(
                str(cm_path),
                caption="Confusion matrix for the evaluation dataset"
            )
    else:
        st.info("Evaluation results not found. Train the baseline model first.")

    st.write("### Model concepts")
    st.markdown("""
    - **TF-IDF:** Converts text into weighted numerical features.
    - **Logistic Regression:** Learns patterns associated with each category.
    - **Macro-F1:** Gives equal weight to each category's F1-score.
    - **Weighted-F1:** Weights each category by its number of examples.
    - **Confusion matrix:** Shows correct and incorrect category predictions.
    """)

else:
    st.title("About LegalText")

    st.markdown("""
    **Problem statement:** Legal documents contain many provisions
    that can be time-consuming to organize manually. LegalText
    demonstrates supervised NLP for classifying legal text into
    predefined categories.

    **Workflow:** Public labelled dataset → data preparation →
    TF-IDF → Logistic Regression → evaluation → saved model →
    Streamlit classification app.

    **Dataset:** LEDGAR, provided through the LexGLUE benchmark.
    The preparation script selects a manageable subset and a limited
    number of categories.

    **Responsible use:** This is an educational prototype. Predictions
    can be incorrect, dataset labels have limitations, and results may
    not generalize to other legal documents. It does not provide legal
    advice or determine legal validity.

    **Privacy:** Text entered for classification is not deliberately
    saved to a local prediction-history database by this app.
    Avoid entering confidential documents.
    """)

    st.write("### Project components")
    st.markdown("""
    - `src/data_loader.py`: Downloads and prepares the dataset.
    - `src/train_baseline.py`: Trains the baseline classifier.
    - `src/train_distilbert.py`: Optional transformer training script.
    - `src/predict.py`: Loads the saved model and predicts categories.
    """)

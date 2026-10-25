# LegalText — AI-Powered Legal Clause Classification

LegalText is an NLP-based application that classifies legal contract clauses into 10 predefined categories using a fine-tuned DistilBERT model. It combines transformer-based text classification with a traditional machine learning baseline to compare model performance and demonstrate practical NLP techniques for legal text.

## Key Features

- **Legal Clause Classification:** Predicts the category of a legal contract clause.
- **DistilBERT NLP Model:** Uses a fine-tuned transformer to understand contextual information in legal text.
- **Model Comparison:** Includes a TF-IDF + Logistic Regression baseline.
- **Performance Evaluation:** Measures accuracy, macro F1-score, and weighted F1-score, with per-class metrics and a confusion matrix.
- **Prediction Confidence:** Displays the model's confidence score alongside the predicted category.
- **Inference Time:** Shows the time taken to generate a prediction.
- **Interactive Web App:** Provides a Streamlit interface for entering and classifying legal text.

## Tech Stack

- **Language:** Python
- **NLP & Deep Learning:** PyTorch, Hugging Face Transformers, DistilBERT
- **Machine Learning:** Scikit-learn, TF-IDF, Logistic Regression
- **Application:** Streamlit
- **Model Hosting:** Hugging Face Hub
- **Data Format & Artifacts:** JSON, Joblib

## Dataset

The project uses the **LEDGAR dataset from the LexGLUE benchmark**, which contains legal provisions labelled by category.

A selected subset of 10 categories is used to build and evaluate the classification models, keeping the project manageable while demonstrating legal text classification.

## Model Performance

The models were evaluated on a held-out evaluation dataset.

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| TF-IDF + Logistic Regression | 98.53% | 98.19% |
| Fine-tuned DistilBERT | 98.33% | 97.85% |

**Understanding the metrics:**

- **Accuracy:** The proportion of evaluated clauses classified correctly.
- **Macro F1-score:** The average F1-score across categories, giving each category equal weight.
- **Weighted F1-score:** The average F1-score weighted by the number of examples in each category.

The baseline achieved slightly higher scores on the recorded evaluation. This comparison demonstrates why traditional machine learning models remain useful benchmarks when evaluating transformer-based approaches.

## How It Works

1. **Data Preparation:** Loads and prepares labelled legal provisions from LEDGAR.
2. **Text Representation:** Converts text into numerical TF-IDF features for the baseline model.
3. **Baseline Training:** Trains Logistic Regression using the extracted text features.
4. **Transformer Fine-Tuning:** Fine-tunes DistilBERT for the selected legal clause categories.
5. **Model Evaluation:** Calculates classification metrics and generates a confusion matrix.
6. **Prediction:** Processes user-provided text and predicts its category.
7. **Deployment:** Provides predictions through a Streamlit application, loading the fine-tuned model from Hugging Face when needed.

## Project Links

- **Live Demo:** https://legaltext-w2vrappppdc2txrcrshbzmnb.streamlit.app/
- **GitHub Repository:** https://github.com/ShreyaRai2004/LegalText
- **Fine-Tuned DistilBERT Model:** https://huggingface.co/Sret12/legaltext-distilbert

## Limitations

- Classification is limited to the 10 categories used in the project.
- Performance may differ on legal documents outside the training dataset.
- Confidence scores are not guarantees of correctness.
- The application classifies text; it does not determine contract validity or provide legal advice.


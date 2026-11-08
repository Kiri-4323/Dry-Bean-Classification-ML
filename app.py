
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# --------------------------------------------------
# 2. Feature names
# --------------------------------------------------

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]

# --------------------------------------------------
# 3. Load saved models and preprocessing objects
# --------------------------------------------------

@st.cache_resource
def load_saved_objects():

    models = {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            MODEL_DIR / "decision_tree.pkl"
        ),
        "KNN": joblib.load(
            MODEL_DIR / "knn.pkl"
        ),
        "Gaussian Naive Bayes": joblib.load(
            MODEL_DIR / "gaussian_naive_bayes.pkl"
        ),
        "SVM": joblib.load(
            MODEL_DIR / "svm.pkl"
        )
    }

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

    return models, scaler, label_encoder


models, scaler, label_encoder = load_saved_objects()

# --------------------------------------------------
# 4. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("Dry Bean Classification System")

st.write(
    "This application evaluates five trained machine learning "
    "classifiers using an uploaded Dry Bean test dataset."
)

st.info(
    "Upload a CSV file containing the 16 input features "
    "and the actual Class column."
)

# --------------------------------------------------
# 5. Model selection
# --------------------------------------------------

selected_model_name = st.selectbox(
    "Select a Classification Model",
    list(models.keys())
)

# --------------------------------------------------
# 6. Upload test dataset
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.write("Dataset Shape:", test_df.shape)
        st.dataframe(test_df.head())

        # Check required columns
        required_columns = FEATURES + ["Class"]

        missing_columns = [
            column for column in required_columns
            if column not in test_df.columns
        ]

        if missing_columns:

            st.error(
                "The uploaded CSV is missing these columns: "
                + ", ".join(missing_columns)
            )

        else:

            # Separate input features and actual target
            X_test = test_df[FEATURES].copy()
            y_actual = test_df["Class"].astype(str)

            # Convert class labels into encoded values
            try:
                y_test = label_encoder.transform(y_actual)

                # Select the trained model
                selected_model = models[selected_model_name]

                # Apply scaling only to scale-sensitive models
                if selected_model_name in [
                    "Logistic Regression",
                    "KNN",
                    "SVM"
                ]:
                    X_input = scaler.transform(X_test)
                else:
                    X_input = X_test

                # Generate predictions and probabilities
                y_pred = selected_model.predict(X_input)
                y_prob = selected_model.predict_proba(X_input)

                # --------------------------------------------------
                # 7. Evaluation metrics
                # --------------------------------------------------

                accuracy = accuracy_score(y_test, y_pred)

                auc = roc_auc_score(
                    y_test,
                    y_prob,
                    multi_class="ovr",
                    average="weighted",
                    labels=np.arange(len(label_encoder.classes_))
                )

                precision = precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )

                mcc = matthews_corrcoef(y_test, y_pred)

                # --------------------------------------------------
                # 8. Display evaluation metrics
                # --------------------------------------------------

                st.subheader("Model Evaluation Results")
                st.write("Selected Model:", selected_model_name)

                col1, col2, col3 = st.columns(3)

                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("AUC", f"{auc:.4f}")
                col3.metric("Precision", f"{precision:.4f}")

                col4, col5, col6 = st.columns(3)

                col4.metric("Recall", f"{recall:.4f}")
                col5.metric("F1-Score", f"{f1:.4f}")
                col6.metric("MCC", f"{mcc:.4f}")

                # --------------------------------------------------
                # 9. Confusion matrix
                # --------------------------------------------------

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=np.arange(len(label_encoder.classes_))
                )

                fig, ax = plt.subplots(figsize=(9, 6))

                ax.imshow(cm, interpolation="nearest", cmap="Blues")
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted Class")
                ax.set_ylabel("Actual Class")

                ax.set_xticks(
                    np.arange(len(label_encoder.classes_))
                )
                ax.set_yticks(
                    np.arange(len(label_encoder.classes_))
                )

                ax.set_xticklabels(
                    label_encoder.classes_,
                    rotation=45,
                    ha="right"
                )
                ax.set_yticklabels(label_encoder.classes_)

                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            j, i, str(cm[i, j]),
                            ha="center",
                            va="center"
                        )

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # --------------------------------------------------
                # 10. Classification report
                # --------------------------------------------------

                st.subheader("Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    labels=np.arange(len(label_encoder.classes_)),
                    target_names=label_encoder.classes_,
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(report).transpose()

                st.dataframe(report_df.round(4))

                # --------------------------------------------------
                # 11. Prediction results
                # --------------------------------------------------

                st.subheader("Prediction Results")

                results_df = test_df.copy()

                results_df["Predicted Class"] = (
                    label_encoder.inverse_transform(y_pred)
                )

                results_df["Prediction Status"] = np.where(
                    results_df["Class"].astype(str)
                    == results_df["Predicted Class"],
                    "Correct",
                    "Incorrect"
                )

                st.dataframe(results_df.head(20))

            except ValueError as error:
                st.error(
                    "Unable to evaluate the uploaded dataset. "
                    "Please check that the Class column contains "
                    "valid Dry Bean class labels and that the "
                    "dataset contains all seven classes. "
                    f"Details: {error}"
                )

    except Exception as error:
        st.error(f"Error reading the uploaded CSV: {error}")

else:
    st.warning("Please upload a test CSV file to evaluate the model.")

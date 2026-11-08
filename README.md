# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to classify dry beans into seven different varieties using machine learning classification algorithms based on their physical characteristics.

Five classification algorithms are implemented and compared to identify the model that provides the best classification performance.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Number of instances:** 13,611  
**Number of input features:** 16  
**Target variable:** Class  
**Number of classes:** 7

The dataset contains measurements such as Area, Perimeter, Major Axis Length, Minor Axis Length, Eccentricity, Compactness, and other shape-related characteristics.

The seven bean classes are:
- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

## 3. Data Preprocessing

The following preprocessing steps were performed:

- Checked the dataset structure and data types.
- Checked for missing values.
- Explored the distribution of the target classes.
- Separated input features and target variable.
- Applied Label Encoding to the target variable.
- Split the dataset into 80% training data and 20% testing data using stratified sampling.
- Applied StandardScaler to the training and testing features for scale-sensitive algorithms.

## 4. Machine Learning Algorithms

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 5. Model Evaluation

The models were evaluated using the following performance metrics:

- Accuracy
- Area Under the Curve (AUC)
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8917 | 0.9330 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Gaussian Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| SVM | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

## 6. Observations

**Logistic Regression:** Achieved strong classification performance with an accuracy of 92.14% and an AUC of 0.9934.

**Decision Tree:** Achieved an accuracy of 89.17%. Its performance was lower than the other models except Gaussian Naive Bayes.

**KNN:** Achieved an accuracy of 91.66%, demonstrating good classification performance.

**Gaussian Naive Bayes:** Achieved an accuracy of 76.39%. It had the lowest accuracy among the five classifiers.

**SVM:** Achieved the highest accuracy of 92.21% and the highest AUC of 0.9937, along with the best Precision, Recall, F1-Score, and MCC values.

## 7. Overall Best Model

Based on the evaluation results, **Support Vector Machine (SVM)** performed best among the five classification algorithms.

It achieved:
- Accuracy: 92.21%
- AUC: 0.9937
- Precision: 0.9224
- Recall: 0.9221
- F1-Score: 0.9222
- MCC: 0.9059

Therefore, SVM was selected as the overall best-performing classifier for this dataset.

## 8. Streamlit Application

A Streamlit application was developed to evaluate the trained classification models.

The application provides:
- CSV test dataset upload.
- Classification model selection.
- Evaluation metrics.
- Confusion matrix.
- Classification report.
- Prediction results.

The application loads the previously saved models and preprocessing objects. It does not retrain the models when a test CSV is uploaded.

**Live Application:** https://xycsd7uk7m876pcfy968yk.streamlit.app/

## 9. GitHub Repository

**GitHub Repository:** https://github.com/Kiri-4323/Dry-Bean-Classification-ML/tree/main

## 10. Project Structure

```text
ML Assignment/
│
├── app.py
├── test_data.csv
├── requirements.txt
├── README.md
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── gaussian_naive_bayes.pkl
│   ├── svm.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
└── notebook/
    └── ML_Assignment_2.ipynb
```

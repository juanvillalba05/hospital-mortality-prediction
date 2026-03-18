# Hospital Mortality Prediction

Supervised machine learning classifier to predict in-hospital mortality
of critically ill patients using physiological, demographic, and disease
severity features.

## Overview

Accurate prediction of hospital mortality enables anticipatory clinical
decisions that can reduce painful and prolonged end-of-life processes.
This project builds and evaluates several classification models on a
dataset of 9,105 ICU patient records, selecting the best-performing
model through rigorous cross-validation and hyperparameter tuning.

## Pipeline
```
Raw data → Imputation + Encoding + Scaling → Model comparison → Hyperparameter tuning → Evaluation
```

## Key steps

- **Dataset:** 9,105 ICU patient records with physiological measurements,
  demographic data, disease severity indicators, and mortality labels (`hospdead`)
- **Preprocessing pipeline:** Median imputation for numerical variables,
  mode imputation for categorical variables, StandardScaler, OneHotEncoder —
  all integrated in a `ColumnTransformer` + `Pipeline`
- **Models compared:** Logistic Regression, Random Forest, Gaussian Naive Bayes,
  Linear Discriminant Analysis, Quadratic Discriminant Analysis
- **Model selection:** Stratified 5-fold cross-validation with ROC-AUC as
  the primary metric (chosen given class imbalance in the target variable)
- **Best model:** L2-regularized Logistic Regression (`C=0.1`)
- **Hyperparameter tuning:** GridSearchCV over regularization strength and
  class weighting strategies

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC (CV) | 0.9397 |
| ROC-AUC (test) | 0.9466 |
| Accuracy | 0.9023 |
| Precision | 0.8568 |
| Recall | 0.7479 |
| F1-score | 0.7986 |

Results are consistent between cross-validation and the held-out test set,
indicating the model generalizes well to unseen patients.

## Tech stack

- Python
- scikit-learn (LogisticRegression, RandomForest, GaussianNB, LDA, QDA,
  Pipeline, ColumnTransformer, GridSearchCV, StratifiedKFold)
- NumPy, pandas
- Matplotlib, Seaborn
- Jupyter Notebook

## How to run
```bash
git clone https://github.com/tu-usuario/hospital-mortality-prediction
cd hospital-mortality-prediction
pip install -r requirements.txt
jupyter notebook hospital-mortality-prediction.ipynb
```

## Context

Final exam project — Supervised Machine Learning course,
Master's in Artificial Intelligence.

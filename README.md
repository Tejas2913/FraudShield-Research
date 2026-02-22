
# FraudShield AI – Research & Model Training

This repository contains the research notebooks, experimentation pipeline, and model training workflow used to build the scam detection engine.

## Problem Statement
Digital scams and phishing attempts are rapidly increasing. This system aims to classify messages as Scam or Not Scam using NLP and machine learning.

## Dataset
- Text-based scam dataset
- Binary classification problem
- Cleaned and preprocessed before training

## Preprocessing Steps
- Lowercasing
- Stopword removal
- Punctuation cleaning
- Tokenization
- TF-IDF vectorization

## Models Experimented
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Naive Bayes

## Final Selected Model
Logistic Regression (best F1-score and ROC-AUC performance)

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Experiment Notebooks
```
01_experiments.ipynb
model_training.ipynb
```
## Model Export
Final model saved as:
model.pkl

Deployment Flow:
```
Research Repo → Export model.pkl → Backend loads model → API exposes prediction endpoint → Frontend consumes API
```

================================================================================

# System Architecture Overview

```
Frontend (React + Vercel)
        ↓
Backend API (FastAPI + Render)
        ↓
ML Model (Scikit-learn)
        ↓
Database (SQLite / PostgreSQL)
```

================================================================================

# Future Improvements

- PostgreSQL production database
- Refresh token support
- Role-based access control
- Model explainability (SHAP/LIME)
- Deep learning-based NLP (LSTM/Transformer)
- Monitoring and logging integration
- Continuous retraining pipeline

================================================================================

# License

MIT License

# Loan Prediction System — Motz Financial Services

## Intent
Predicts whether a loan application will be **Approved (Y)** or **Rejected (N)**
using applicant demographics, income, loan details and credit history.
Built for the DIAL3110 Fundamentals of AI group assignment.

## Dataset
`data/loan_data_set.csv` — 614 applications, 13 columns.
Target: `Loan_Status` (Y/N). Class balance is imbalanced (~69% Y / 31% N).

## Models compared
- Logistic Regression (interpretable baseline)
- Random Forest (bagged trees)
- Gradient Boosting (boosted trees)

## Pipeline
1. Drop `Loan_ID` (identifier, no predictive value).
2. Impute missing values: median (numeric), mode (categorical), mode for `Credit_History`.
3. Flag/report outliers via IQR (kept — genuine high-income applicants).
4. Feature engineering: `TotalIncome`, `LoanAmount_log`, `EMI`, `Balance_Income`.
5. One-hot encode categoricals; scale numerics for LR.
6. Stratified 80/20 split (random_state=42).
7. 5-fold CV grid search on training data only.
8. Final evaluation on held-out test set: accuracy, precision, recall, F1, ROC-AUC.

## Run
```bash
pip install -r requirements.txt
python train_pipeline.py      # trains, tunes, saves models + figures
streamlit run app.py          # launches dashboard on localhost:8501
# `models/` — trained artifacts

Populated by `python train_pipeline.py`. Not committed (regenerate locally).

Expected files after a successful run:

| File | Contents |
|---|---|
| `LogisticRegression.joblib` | Best LR pipeline (preprocessor + estimator) |
| `RandomForest.joblib`       | Best RF pipeline |
| `GradientBoosting.joblib`   | Best GB pipeline |
| `best_model.joblib`         | Model with highest CV F1 — used by `app.py` |
| `model_comparison.csv`      | Accuracy/Precision/Recall/F1/ROC-AUC per model |

Load in code:
```python
import joblib
model = joblib.load("models/best_model.joblib")
model.predict(X_new)          # expects the same columns the pipeline was fit on
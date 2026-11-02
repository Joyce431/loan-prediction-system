"""End-to-end training script.

Run:  python train_pipeline.py
Outputs: models/*.joblib, figures/*.png, results summary printed.
"""
import os, json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.data_cleaning import load_data, inspect, descriptive_stats, outlier_report, clean
from src.preprocessing import engineer_features
from src.model_training import get_models, tune_model
from src.evaluate import metrics_dict, save_confusion_matrix, save_roc_curves

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ---------- 1. Load & inspect ----------
df_raw = load_data("data/loan_data_set.csv")
print("=== Inspection ===")
for k, v in inspect(df_raw).items():
    print(f"{k}: {v}")

print("\n=== Descriptive stats ===")
print(descriptive_stats(df_raw).round(2))
print("\n=== Outlier report ===")
print(outlier_report(df_raw))

# ---------- 2. Clean & engineer ----------
df = clean(df_raw)
df = engineer_features(df)
print("\nAfter cleaning/engineering:", df.shape)
print("Missing:", df.isna().sum().sum())

# Save the cleaned dataset for the report
df.to_csv("data/loan_clean.csv", index=False)

# ---------- 3. Train/test split ----------
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"].map({"N": 0, "Y": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print("Train class balance:", y_train.value_counts(normalize=True).round(3).to_dict())

# ---------- 4. Figures for report (Task 8a) ----------
sns.set_theme(style="whitegrid")

# Fig 1: Target distribution
plt.figure(figsize=(4, 3.5))
sns.countplot(x=y.map({0: "N", 1: "Y"}))
plt.title("Loan Status Distribution"); plt.xlabel("Status"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig("figures/fig1_target.png", dpi=150); plt.close()

# Fig 2: Income distribution (raw vs log)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
sns.histplot(df["ApplicantIncome"], bins=40, ax=ax[0]); ax[0].set_title("ApplicantIncome (raw)")
sns.histplot(df["LoanAmount_log"], bins=40, ax=ax[1]); ax[1].set_title("LoanAmount (log1p)")
plt.tight_layout(); plt.savefig("figures/fig2_income_loan.png", dpi=150); plt.close()

# Fig 3: Boxplots for outliers
fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
for i, c in enumerate(["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]):
    sns.boxplot(y=df[c], ax=ax[i]); ax[i].set_title(c)
plt.tight_layout(); plt.savefig("figures/fig3_outliers.png", dpi=150); plt.close()

# Fig 4: Credit history vs status
plt.figure(figsize=(4.5, 3.5))
sns.countplot(data=df, x="Credit_History", hue="Loan_Status")
plt.title("Credit History vs Loan Status")
plt.tight_layout(); plt.savefig("figures/fig4_credit.png", dpi=150); plt.close()

# Fig 5: Correlation heatmap
num = df.select_dtypes(include=np.number)
plt.figure(figsize=(6, 4.5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=False,
            annot_kws={"size": 7})
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig("figures/fig5_corr.png", dpi=150); plt.close()

# ---------- 5. Train + tune each model ----------
models = get_models()
tuned = {}
results = {}
roc_data = {}

for name, model in models.items():
    print(f"\n=== Tuning {name} ===")
    gs = tune_model(name, model, X_train, y_train)
    print("Best params:", gs.best_params_)
    print("Best CV F1:", round(gs.best_score_, 4))
    tuned[name] = gs.best_estimator_

    # Evaluate on held-out test set
    y_pred = gs.predict(X_test)
    y_proba = gs.predict_proba(X_test)[:, 1]
    results[name] = metrics_dict(y_test, y_pred, y_proba)
    roc_data[name] = (y_test, y_proba)
    save_confusion_matrix(y_test, y_pred, name,
                          f"figures/cm_{name}.png")
    joblib.dump(gs.best_estimator_, f"models/{name}.joblib")
    print(name, results[name])

# ---------- 6. Comparison + ROC ----------
results_df = pd.DataFrame(results).T
print("\n=== Model comparison ===")
print(results_df)
results_df.to_csv("models/model_comparison.csv")
save_roc_curves(roc_data, "figures/fig_roc.png")

# Pick best by F1
best_name = results_df["F1"].idxmax()
print(f"\nBest model (F1): {best_name}")
joblib.dump(tuned[best_name], "models/best_model.joblib")

# Feature importance for tree models
for name in ["RandomForest", "GradientBoosting"]:
    est = tuned[name].named_steps["clf"]
    if hasattr(est, "feature_importances_"):
        pre = tuned[name].named_steps["pre"]
        feat_names = pre.get_feature_names_out()
        imp = pd.Series(est.feature_importances_, index=feat_names).sort_values(ascending=False).head(15)
        plt.figure(figsize=(6, 4))
        imp[::-1].plot(kind="barh")
        plt.title(f"Top features — {name}")
        plt.tight_layout(); plt.savefig(f"figures/featimp_{name}.png", dpi=150); plt.close()

print("\nDone. Artifacts in models/ and figures/.")
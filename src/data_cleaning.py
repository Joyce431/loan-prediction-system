"""Task 1 — Data gathering, inspection and cleaning.

Functions
---------
load_data(path)          -> pd.DataFrame
inspect(df)              -> dict          (shape, missing, duplicates, dtypes)
descriptive_stats(df)    -> pd.DataFrame  (count, mean, median, std, min, quartiles, max)
outlier_report(df, cols) -> pd.DataFrame  (IQR bounds + counts)
outlier_mask(df, col)    -> pd.Series     (True where row is an IQR outlier)
clean(df)                -> pd.DataFrame  (imputation + type coercion, documented)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PATH = "data/loan_data_set.csv"


# --------------------------------------------------------------------------- #
# Load / inspect
# --------------------------------------------------------------------------- #
def load_data(path: str = DEFAULT_PATH) -> pd.DataFrame:
    """Load the raw CSV exactly as supplied (no cleaning)."""
    return pd.read_csv(path)


def inspect(df: pd.DataFrame) -> dict:
    """Return a summary dict you can paste into the report."""
    return {
        "shape": df.shape,
        "n_missing_total": int(df.isna().sum().sum()),
        "missing": df.isna().sum().to_dict(),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "duplicates_rows": int(df.duplicated().sum()),
        "duplicates_loan_id": int(df["Loan_ID"].duplicated().sum())
        if "Loan_ID" in df.columns else None,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "target_counts": df["Loan_Status"].value_counts().to_dict()
        if "Loan_Status" in df.columns else None,
    }


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / median / min / quartiles / max for numeric columns.

    Returns one row per numeric column. `count` = number of non-null values.
    """
    num = df.select_dtypes(include=np.number)
    stats = num.describe().T  # count, mean, std, min, 25, 50, 75, max
    stats["median"] = num.median()
    return stats[["count", "mean", "median", "std",
                  "min", "25%", "50%", "75%", "max"]]


def outlier_report(
    df: pd.DataFrame,
    cols: tuple[str, ...] = ("ApplicantIncome", "CoapplicantIncome",
                             "LoanAmount", "Loan_Amount_Term"),
) -> pd.DataFrame:
    """IQR outlier summary — one row per column."""
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float)
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < low) | (s > high)).sum())
        rows.append({
            "column": c,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "lower_bound": round(low, 2),
            "upper_bound": round(high, 2),
            "n_outliers": n_out,
            "pct_outliers": round(100 * n_out / len(s), 2),
        })
    return pd.DataFrame(rows)


def outlier_mask(df: pd.DataFrame, col: str) -> pd.Series:
    """Boolean mask: True where value lies outside [Q1-1.5·IQR, Q3+1.5·IQR]."""
    s = df[col].astype(float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented cleaning decisions.

    Decisions
    ---------
    1. Drop `Loan_ID` — identifier, no predictive value.
    2. `Dependents` '3+' -> 3; force numeric.
    3. Categorical imputation: mode for Gender, Married, Dependents, Self_Employed.
    4. Numeric imputation: median for LoanAmount, mode for Loan_Amount_Term.
    5. `Credit_History` missing -> mode (strongest predictor; dropping would
       remove ~8% of the dataset). Cast to int.
    6. Outliers are KEPT (genuine high-income applicants, not errors).
       LoanAmount is log-transformed later in preprocessing.
    """
    df = df.copy()

    # 1. Identifier
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # 2. Dependents
    if "Dependents" in df.columns:
        df["Dependents"] = (
            df["Dependents"].astype(str).str.replace("3+", "3", regex=False)
        )
        df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")

    # 3. Categorical modes (computed AFTER dtype coercion so Dependents is valid)
    for c in ("Gender", "Married", "Dependents", "Self_Employed"):
        if c in df.columns and df[c].isna().any():
            mode_val = df[c].mode(dropna=True)
            if not mode_val.empty:
                df[c] = df[c].fillna(mode_val.iloc[0])
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].fillna(0).astype(int)

    # 4. Numeric imputation
    if "LoanAmount" in df.columns:
        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    if "Loan_Amount_Term" in df.columns and df["Loan_Amount_Term"].isna().any():
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
            df["Loan_Amount_Term"].mode().iloc[0]
        )

    # 5. Credit history
    if "Credit_History" in df.columns:
        if df["Credit_History"].isna().any():
            df["Credit_History"] = df["Credit_History"].fillna(
                df["Credit_History"].mode().iloc[0]
            )
        df["Credit_History"] = df["Credit_History"].astype(int)

    return df
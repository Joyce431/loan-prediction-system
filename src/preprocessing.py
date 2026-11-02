"""Feature extraction & preprocessing (Task 2)."""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature extraction (Task 2a).

    - TotalIncome     : ApplicantIncome + CoapplicantIncome
    - LoanAmount_log  : log1p(LoanAmount) — reduces right skew
    - EMI             : LoanAmount / Loan_Amount_Term
    - Balance_Income  : TotalIncome - (EMI * 1000)  (monthly surplus proxy)
    """
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
    df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"].replace(0, np.nan)
    df["Balance_Income"] = df["TotalIncome"] - (df["EMI"] * 1000)
    return df


CAT_COLS = ["Gender", "Married", "Dependents", "Education",
            "Self_Employed", "Property_Area"]
NUM_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History",
            "TotalIncome", "LoanAmount_log", "EMI", "Balance_Income"]


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """ColumnTransformer for numeric + categorical features."""
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), NUM_COLS),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CAT_COLS),
        ],
        remainder="drop",
    )
    return pre
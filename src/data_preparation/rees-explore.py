import pandas as pd
import numpy as np
import os
from datetime import datetime

BASE = "../../data/raw/"  # change if your path is different

# Common filename: E Commerce Dataset.csv or churn_dataset.csv — check folder and update
rees46_file = os.path.join(
    BASE, "rees46-churn/rees46_customer_model.csv"
)  # ← UPDATE THIS
if not os.path.exists(rees46_file):
    print("File not found:", rees46_file)
    # rees46_file = os.path.join(BASE, "rees46/churn_dataset.csv")  # try this too

rees46 = pd.read_csv(rees46_file)
print(f"Shape: {rees46.shape}")
print(f"Columns ({len(rees46.columns)}):\n{sorted(rees46.columns.tolist())}")
print("\nData types:\n", rees46.dtypes)
print(
    "\nMissing values (%):\n",
    (rees46.isnull().sum() / len(rees46) * 100).round(2).sort_values(ascending=False),
)
print("\nFirst 3 rows:\n", rees46.head(3))
if "churn" in rees46.columns:
    print(
        f"\nChurn rate: {rees46['churn'].mean():.2%} ({rees46['churn'].sum()} churned)"
    )
else:
    print("\nWARNING: No 'churn' column found!")

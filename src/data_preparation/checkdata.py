import pandas as pd

features = pd.read_csv("../../data/raw/synthetic_events.csv")

print("Final shape:", features.shape)
print("Columns:", features.columns.tolist())
# print("\nChurn distribution:\n", features["churn"].value_counts())
print("\nNull check:\n", features.isnull().sum()[features.isnull().sum() > 0])
print("\nSample:\n", features.sample(15).to_string())
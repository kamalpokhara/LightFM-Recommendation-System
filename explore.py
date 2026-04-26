import pandas as pd
df = pd.read_csv("products.csv")

print(df["category"].value_counts())
print(df["category"].unique())
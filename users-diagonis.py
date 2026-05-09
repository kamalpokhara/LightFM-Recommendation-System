import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("users_export.csv")

df_category = df["category"].unique().tolist()

print("Unique categories: ", df_category)
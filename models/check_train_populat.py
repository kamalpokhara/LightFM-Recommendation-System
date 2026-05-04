# check_popular_ids.py
import joblib
import pandas as pd

popular_data = joblib.load("popular_products.pkl")
popular_ids  = [int(p) for p in popular_data]

products = pd.read_csv("../products.csv")
products.columns = products.columns.str.lower()

print("Popular IDs:", popular_ids[:5])
print("Products ID sample:", products["id"].head(5).tolist())

# Check all popular IDs exist in products
missing = [pid for pid in popular_ids if pid not in products["id"].values]
print(f"\nMissing from products: {missing}")
print(f"All popular IDs valid: {len(missing) == 0}")
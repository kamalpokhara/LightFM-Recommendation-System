# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime

import pandas as pd

path = "../data/raw/retailrocket/events.csv"  

events = pd.read_csv(path)

print("Shape:", events.shape)
print("Columns:", events.columns.tolist())
print("\nFirst 5 rows:\n", events.head().to_string())
print("\nEvent distribution:\n", events["event"].value_counts(dropna=False))
print("\nUnique users:", events["visitorid"].nunique())
print("Unique products:", events["itemid"].nunique())

events["ts"] = pd.to_datetime(events["timestamp"], unit="ms")
print("\nDate range:", events["ts"].min(), "—", events["ts"].max())

# BASE = "../data/raw/retailrocket/" 

# print("=== RETAILROCKET EVENTS.CSV ===")
# events = pd.read_csv(
#     os.path.join(BASE, "events.csv"),
#     dtype={"visitorid": "int32", "itemid": "int32", "transactionid": "float32"},
# )

# print(f"Shape: {events.shape}")
# print(f"Columns: {events.columns.tolist()}")
# print("\nData types:\n", events.dtypes)
# print("\nMissing values (%):\n", (events.isnull().sum() / len(events) * 100).round(3))

# print("\nEvent types distribution:")
# print(events["event"].value_counts(normalize=True).round(4) * 100)

# # Convert timestamp (critical)
# events["interaction_timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")
# print("\nDate range:")
# print(events["interaction_timestamp"].min())
# print(events["interaction_timestamp"].max())
# print(
#     f"Total days: {(events['interaction_timestamp'].max() - events['interaction_timestamp'].min()).days}"
# )

# print("\nUnique users (visitorid):", events["visitorid"].nunique())
# print("Unique items (itemid):", events["itemid"].nunique())
# print("Events per user (mean):", events.groupby("visitorid").size().mean().round(2))

# print("\nFirst 5 rows:\n", events.head())

# # Quick churn preview (using your definition)
# last_activity = events.groupby("visitorid")["interaction_timestamp"].max()
# today = events["interaction_timestamp"].max()  # use dataset end as "today" for now
# days_inactive = (today - last_activity).dt.days
# churn_rate_preview = (days_inactive >= 30).mean()
# print(
#     f"\nPreview churn rate ( >=30 days inactive at dataset end): {churn_rate_preview:.2%}"
# )

# print("\n" + "=" * 80)

# print("=== ITEM PROPERTIES (Quick Look - This file is large) ===")
# # Load only first 100k rows for speed, or full if your machine can handle
# props = pd.read_csv(os.path.join(BASE, "item_properties_part1.csv"), nrows=100000)
# print(f"Shape (sample): {props.shape}")
# print(f"Columns: {props.columns.tolist()}")
# print("\nSample properties:\n", props["property"].value_counts().head(10))

# print("\nUnique items in properties:", props["itemid"].nunique())

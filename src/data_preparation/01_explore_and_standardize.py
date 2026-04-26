import pandas as pd
from datetime import datetime


events_path = "../../data/raw/retailrocket/events.csv"

df = pd.read_csv(events_path)

print("Original shape:", df.shape)
print("Original columns:", df.columns.tolist())

# Standardize column names (this is what we will use everywhere)
df = df.rename(
    columns={
        "visitorid": "user_id",
        "itemid": "product_id",
        "event": "interaction_type",
        "timestamp": "timestamp_ms",
    }
)

# Step 2: Convert timestamp to proper datetime
df["interaction_timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")

# Step 3: Add interaction weight (important for LightFM)
weight_map = {"view": 1, "addtocart": 2, "transaction": 3}
df["interaction_weight"] = df["interaction_type"].map(weight_map)

print("\n=== AFTER STANDARDIZATION ===")
print("Shape:", df.shape)
print("Columns now:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head().to_string(index=False))
print(
    "\nInteraction weight distribution:\n",
    df["interaction_weight"].value_counts().sort_index(),
)

# Save standardized version (we will use this going forward)
# df.to_parquet("../../data/processed/retailrocket_standardized.parquet", index=False)
# print(
#     "\nSaved standardized data to ../../data/processed/retailrocket_standardized.parquet"
# )


import pandas as pd
import numpy as np

events   = pd.read_parquet("data/processed/recsys_events.parquet")
products = pd.read_parquet("data/processed/recsys_products.parquet")
users    = pd.read_parquet("data/processed/recsys_users.parquet")

# ── HOW MANY ITEMS SHOULD WE KEEP? ───────────────────────────────────────────
# Target: each user should have seen at least 5% of catalog
# mean interactions per user = 8
# 8 / 0.05 = 160 items maximum

# Keep top N most interacted items
TOP_N_ITEMS = 200

item_counts = events.groupby("product_id").size().sort_values(ascending=False)
top_items   = item_counts.head(TOP_N_ITEMS).index.tolist()

print(f"Keeping top {TOP_N_ITEMS} most interacted items")
print(f"These items cover: {item_counts.head(TOP_N_ITEMS).sum()} "
      f"of {len(events)} events "
      f"({item_counts.head(TOP_N_ITEMS).sum()/len(events):.1%})")

# Filter events to top items only
events_filtered = events[events["product_id"].isin(top_items)].copy()
products_filtered = products[products["product_id"].isin(top_items)].copy()

print(f"\nEvents after item filter: {len(events_filtered)}")
print(f"Products after filter:    {len(products_filtered)}")

# Dedup
deduped = events_filtered.drop_duplicates(
    subset=["user_id", "product_id"], keep="last"
)

user_counts = deduped.groupby("user_id").size()
print(f"\nAfter dedup:")
print(f"  Mean interactions per user: {user_counts.mean():.2f}")
print(f"  Coverage per user: {user_counts.mean()/TOP_N_ITEMS:.2%}")
print(f"  Users remaining: {deduped['user_id'].nunique()}")

# Save filtered versions
events_filtered.to_parquet("data/processed/recsys_events_filtered.parquet",
                            index=False)
products_filtered.to_parquet("data/processed/recsys_products_filtered.parquet",
                              index=False)
print("\nSaved filtered files")
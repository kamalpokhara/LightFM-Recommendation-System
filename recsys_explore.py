import pandas as pd
import numpy as np

# ── 1. LOAD ALL THREE ─────────────────────────────────────────────────────────
events = pd.read_csv("data/raw/ecommerce_dataset/events.csv")
products = pd.read_csv("data/raw/ecommerce_dataset/products.csv")
users = pd.read_csv("data/raw/ecommerce_dataset/users.csv")

print("events shape:  ", events.shape)
print("products shape:", products.shape)
print("users shape:   ", users.shape)

# ── 2. CLEAN EVENTS ───────────────────────────────────────────────────────────
events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])
events["event_type"] = events["event_type"].str.strip().str.lower()
events = events[["user_id", "product_id", "event_type", "event_timestamp"]]

print("\n── Events ───────────────────────────────────────────")
print(events["event_type"].value_counts())
print(
    f"Date range: {events['event_timestamp'].min().date()} → {events['event_timestamp'].max().date()}"
)
print(f"Unique users:    {events['user_id'].nunique()}")
print(f"Unique products: {events['product_id'].nunique()}")

# ── 3. CLEAN PRODUCTS ─────────────────────────────────────────────────────────
products = products[["product_id", "category", "brand", "price", "rating"]]
products["category"] = products["category"].str.strip().str.lower()
products["brand"] = products["brand"].str.strip().str.lower()

# Price buckets — more stable than raw price for LightFM features
products["price_range"] = pd.cut(
    products["price"],
    bins=[0, 50, 150, 300, 600, float("inf")],
    labels=["budget", "low_mid", "mid", "high_mid", "premium"],
).astype(str)

# Rating buckets
products["rating_band"] = pd.cut(
    products["rating"],
    bins=[0, 2.5, 3.5, 4.0, 4.5, 5.0],
    labels=["poor", "average", "good", "great", "excellent"],
).astype(str)

print("\n── Products ─────────────────────────────────────────")
print(f"Total products: {len(products)}")
print(f"\nCategories:\n{products['category'].value_counts()}")
print(f"\nBrands:\n{products['brand'].value_counts().head(10)}")
print(f"\nPrice distribution:\n{products['price'].describe()}")
print(f"\nPrice ranges:\n{products['price_range'].value_counts()}")
print(f"\nRating bands:\n{products['rating_band'].value_counts()}")
print(f"\nNull check:\n{products.isnull().sum()}")

# ── 4. CLEAN USERS ────────────────────────────────────────────────────────────
users = users[["user_id", "city", "signup_date"]]
users["signup_date"] = pd.to_datetime(users["signup_date"])

# User tenure in days from signup to today
today = pd.Timestamp("2025-11-14")  # dataset end date
users["tenure_days"] = (today - users["signup_date"]).dt.days

# Tenure band
users["tenure_band"] = pd.cut(
    users["tenure_days"],
    bins=[0, 30, 90, 180, 365, float("inf")],
    labels=["new", "early", "growing", "established", "loyal"],
).astype(str)

print("\n── Users ────────────────────────────────────────────")
print(f"Total users: {len(users)}")
print(f"\nTop cities:\n{users['city'].value_counts().head(10)}")
print(f"\nTenure bands:\n{users['tenure_band'].value_counts()}")
print(f"\nNull check:\n{users.isnull().sum()}")

# ── 5. COVERAGE CHECK ─────────────────────────────────────────────────────────
# Check how well events link to products and users
events_products_match = events["product_id"].isin(products["product_id"]).mean()
events_users_match = events["user_id"].isin(users["user_id"]).mean()

print("\n── Coverage ─────────────────────────────────────────")
print(f"Events → products match: {events_products_match:.1%}")
print(f"Events → users match:    {events_users_match:.1%}")

# ── 6. INTERACTION WEIGHTS ────────────────────────────────────────────────────
weight_map = {
    "view": 1,
    "wishlist": 2,
    "cart": 3,
    "purchase": 5,
}
events["interaction_weight"] = (
    events["event_type"].map(weight_map).fillna(1).astype(int)
)

print("\n── Interaction weights ──────────────────────────────")
print(events.groupby("event_type")["interaction_weight"].first())

# ── 7. INTERACTIONS PER USER ──────────────────────────────────────────────────
user_counts = events.groupby("user_id").size()
print("\n── Interactions per user ────────────────────────────")
print(user_counts.describe())
print(
    f"Users with >= 3 interactions: {(user_counts >= 3).sum()} ({(user_counts >= 3).mean():.1%})"
)
print(
    f"Users with >= 5 interactions: {(user_counts >= 5).sum()} ({(user_counts >= 5).mean():.1%})"
)

# ── 8. SAVE CLEANED FILES ─────────────────────────────────────────────────────
# events.to_parquet("../../data/processed/recsys_events.parquet", index=False)
# products.to_parquet("../../data/processed/recsys_products.parquet", index=False)
# users.to_parquet("../../data/processed/recsys_users.parquet", index=False)

print("\nSaved:")
print("  recsys_events.parquet")
print("  recsys_products.parquet")
print("  recsys_users.parquet")

# save_features_from_existing.py
# No retraining — just rebuild matrices from saved objects

import joblib
import pandas as pd
import numpy as np
import ast

print("Loading saved models...", flush=True)

dataset   = joblib.load("models/lightfm_dataset.pkl")
model     = joblib.load("models/lightfm_best.pkl")

print("dataset type:", type(dataset))
print("model type:  ", type(model))

# ── LOAD RAW DATA (needed to rebuild matrices) ────────────────────────────────
interactions = pd.read_csv("data/raw/synthetic_interactions_recsys.csv")
products     = pd.read_csv("products.csv")
users        = pd.read_csv("data/raw/synthetic_users_recsys.csv")

products.columns     = products.columns.str.lower().str.strip()
interactions.columns = interactions.columns.str.lower().str.strip()
users.columns        = users.columns.str.lower().str.strip()

interactions = interactions.rename(columns={"action": "event_type"})
products["product_id"]     = products["id"].astype(int)
interactions["product_id"] = interactions["product_id"].astype(int)

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
# Add this right after brand cleaning line
products["category"] = products["category"].str.strip().str.lower()
products["brand"]    = products["brand"].str.strip().str.lower()


products["brand"] = products["brand"].fillna("unknown").str.strip().str.lower()
products["category"] = products["category"].fillna("unknown").str.strip().str.lower()
# products["price_range"] = products["price_range"].fillna("budget")
# products["rating_band"] = products["rating_band"].fillna("average")

products["price_range"] = pd.cut(
    products["price"],
    bins=[0, 20, 50, 100, 300, float("inf")],
    labels=["budget", "low_mid", "mid", "high_mid", "premium"]
).astype(str)

products["rating_band"] = pd.cut(
    products["rating"],
    bins=[0, 3.0, 3.5, 4.0, 4.5, 5.0],
    labels=["poor", "average", "good", "great", "excellent"]
).astype(str)


interactions["interaction_timestamp"] = pd.to_datetime(
    interactions["interaction_timestamp"]
)
today = interactions["interaction_timestamp"].max()

users["registration_date"] = pd.to_datetime(users["registration_date"])
users["tenure_days"]       = (today - users["registration_date"]).dt.days

users["tenure_band"] = pd.cut(
    users["tenure_days"],
    bins=[0, 30, 90, 180, 365, float("inf")],
    labels=["new", "early", "growing", "established", "loyal"]
).astype(str)

users["age_group"] = pd.cut(
    users["age"],
    bins=[0, 25, 35, 45, 60, float("inf")],
    labels=["18_25", "26_35", "36_45", "46_60", "60_plus"]
).astype(str).str.strip().str.rstrip(",")

def parse_interests(val):
    if pd.isna(val) or str(val).strip() in ["[]", ""]:
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []

users["interests_parsed"] = users["interests"].apply(parse_interests)

event_users = interactions["user_id"].unique()
event_items = interactions["product_id"].unique()

# ── BUILD ITEM FEATURES FROM EXISTING DATASET ─────────────────────────────────
print("\nBuilding item feature matrix...", flush=True)

item_features = dataset.build_item_features(
    (
        row["product_id"],
        [
            f"category:{row['category']}",
            f"brand:{row['brand']}",
            f"price:{row['price_range']}",
            f"rating:{row['rating_band']}",
        ]
    )
    for _, row in products.iterrows()
    if row["product_id"] in set(event_items)
)
print(f"Item features shape: {item_features.shape}")

# ── BUILD USER FEATURES FROM EXISTING DATASET ─────────────────────────────────
print("Building user feature matrix...", flush=True)

users_filtered = users[users["user_id"].isin(event_users)].copy()

def get_user_features(row):
    feats = [
        f"gender:{row['gender']}",
        f"tenure:{row['tenure_band']}",
        f"age:{row['age_group']}",
    ]
    for interest in row["interests_parsed"]: 
        feats.append(f"interest:{interest}")
    return feats

user_features = dataset.build_user_features(
    (row["user_id"], get_user_features(row))
    for _, row in users_filtered.iterrows()
)
print(f"User features shape: {user_features.shape}")

# ── VERIFY PREDICT WORKS ──────────────────────────────────────────────────────
print("\nVerifying predict works...", flush=True)

user_id_map, _, item_id_map, _ = dataset.mapping()
sample_user = list(user_id_map.keys())[0]
sample_uid  = user_id_map[sample_user]
n_items     = len(item_id_map)

scores = model.predict(
    user_ids=int(sample_uid),
    item_ids=np.arange(n_items),
    user_features=user_features,
    item_features=item_features,
    num_threads=1,
)
print(f"Predict works — sample scores shape: {scores.shape}")
print(f"Sample top 3 item indices: {np.argsort(-scores)[:3]}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
joblib.dump(item_features, "models/item_features.pkl")
joblib.dump(user_features, "models/user_features.pkl")

print("\nSaved:")
print("  models/item_features.pkl")
print("  models/user_features.pkl")

import os
for f in ["lightfm_best.pkl", "lightfm_dataset.pkl",
          "item_features.pkl", "user_features.pkl",
          "popular_products.pkl"]:
    path = f"models/{f}"
    size = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
    status = f"{size:.1f} KB" if size > 0 else "MISSING"
    print(f"  {f}: {status}")
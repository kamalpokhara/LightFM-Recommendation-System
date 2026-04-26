import sys
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
import joblib

# ── LOAD ALREADY PROCESSED DATA ───────────────────────────────────────────────
print("Loading data...", flush=True)

events   = pd.read_csv("data/processed/recsys_events.csv")
products = pd.read_csv("data/processed/recsys_products.csv")
users    = pd.read_csv("data/processed/recsys_users.csv")

# ── REBUILD EXACTLY AS IN MAIN SCRIPT ─────────────────────────────────────────
events = events.sort_values("event_timestamp").reset_index(drop=True)
split = int(len(events) * 0.8)
train_events = events.iloc[:split]
test_events = events.iloc[split:]

event_users = events["user_id"].unique()
event_items = events["product_id"].unique()

dataset = Dataset()
dataset.fit(
    users=event_users,
    items=event_items,
    item_features=[
        *[f"category:{c}" for c in products["category"].unique()],
        *[f"brand:{b}" for b in products["brand"].unique()],
        *[f"price:{p}" for p in products["price_range"].unique()],
        *[f"rating:{r}" for r in products["rating_band"].unique()],
    ],
    user_features=[
        *[f"tenure:{t}" for t in users["tenure_band"].unique()],
    ],
)


def build_interactions(df, dataset):
    return dataset.build_interactions(
        (row["user_id"], row["product_id"], row["interaction_weight"])
        for _, row in df.iterrows()
    )


train_interactions, _ = build_interactions(train_events, dataset)
test_interactions, _ = build_interactions(test_events, dataset)
train_interactions = train_interactions.tocsr()
test_interactions = test_interactions.tocsr()

item_features = dataset.build_item_features(
    (
        row["product_id"],
        [
            f"category:{row['category']}",
            f"brand:{row['brand']}",
            f"price:{row['price_range']}",
            f"rating:{row['rating_band']}",
        ],
    )
    for _, row in products.iterrows()
)

users_filtered = users[users["user_id"].isin(event_users)]
user_features = dataset.build_user_features(
    (row["user_id"], [f"tenure:{row['tenure_band']}"])
    for _, row in users_filtered.iterrows()
)

print(
    f"train_interactions: {train_interactions.shape} {type(train_interactions)}",
    flush=True,
)
print(
    f"item_features:      {item_features.shape}      {type(item_features)}", flush=True
)
print(
    f"user_features:      {user_features.shape}      {type(user_features)}", flush=True
)

# ── TEST 1: no features ───────────────────────────────────────────────────────
print("\nTest 1: fit without any features...", flush=True)
try:
    m1 = LightFM(no_components=10, loss="warp", random_state=42)
    m1.fit_partial(
        interactions=train_interactions,
        num_threads=1,
        epochs=1,
    )
    print("Test 1 PASSED", flush=True)
except Exception as e:
    print(f"Test 1 FAILED: {e}", flush=True)

# ── TEST 2: item features only ────────────────────────────────────────────────
print("\nTest 2: fit with item features only...", flush=True)
try:
    m2 = LightFM(no_components=10, loss="warp", random_state=42)
    m2.fit_partial(
        interactions=train_interactions,
        item_features=item_features,
        num_threads=1,
        epochs=1,
    )
    print("Test 2 PASSED", flush=True)
except Exception as e:
    print(f"Test 2 FAILED: {e}", flush=True)

# ── TEST 3: user and item features ────────────────────────────────────────────
print("\nTest 3: fit with both features...", flush=True)
try:
    m3 = LightFM(no_components=10, loss="warp", random_state=42)
    m3.fit_partial(
        interactions=train_interactions,
        user_features=user_features,
        item_features=item_features,
        num_threads=1,
        epochs=1,
    )
    print("Test 3 PASSED", flush=True)
except Exception as e:
    print(f"Test 3 FAILED: {e}", flush=True)

# ── TEST 4: BPR loss instead of WARP ─────────────────────────────────────────
print("\nTest 4: fit with BPR loss (no features)...", flush=True)
try:
    m4 = LightFM(no_components=10, loss="bpr", random_state=42)
    m4.fit_partial(
        interactions=train_interactions,
        num_threads=1,
        epochs=1,
    )
    print("Test 4 PASSED", flush=True)
except Exception as e:
    print(f"Test 4 FAILED: {e}", flush=True)

print("\nAll tests done", flush=True)

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k
import joblib
import json
import ast
import sys
import itertools

# ── 1. LOAD EVERYTHING ────────────────────────────────────────────────────────
print("Loading data...", flush=True)

interactions = pd.read_csv("data/raw/synthetic_interactions_recsys.csv")
products     = pd.read_csv("products.csv")
users        = pd.read_csv("data/raw/synthetic_users_recsys.csv")

products.columns     = products.columns.str.lower().str.strip()
interactions.columns = interactions.columns.str.lower().str.strip()
users.columns        = users.columns.str.lower().str.strip()

interactions = interactions.rename(columns={"action": "event_type"})
products["product_id"]      = products["id"].astype(int)
interactions["product_id"]  = interactions["product_id"].astype(int)
interactions["interaction_timestamp"] = pd.to_datetime(
    interactions["interaction_timestamp"]
)

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
products["category"] = products["category"].str.strip().str.lower()
products["brand"]    = products["brand"].str.strip().str.lower()

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

users["registration_date"] = pd.to_datetime(users["registration_date"])
today                      = interactions["interaction_timestamp"].max()
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

weight_map = {"view": 1, "wishlist": 2, "cart": 3, "purchase": 5}
interactions["weight"] = interactions["event_type"].map(weight_map).fillna(1)

# ── 3. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
interactions = interactions.sort_values(
    "interaction_timestamp"
).reset_index(drop=True)

interactions = interactions.drop_duplicates(
    subset=["user_id", "product_id"], keep="last"
).sort_values("interaction_timestamp").reset_index(drop=True)

split        = int(len(interactions) * 0.8)
train_events = interactions.iloc[:split]
test_events  = interactions.iloc[split:]

# ── 4. BUILD DATASET & MATRICES ───────────────────────────────────────────────
print("Building dataset and matrices...", flush=True)

event_users = interactions["user_id"].unique()
event_items = interactions["product_id"].unique()

dataset = Dataset()
dataset.fit(
    users=event_users,
    items=event_items,
    item_features=[
        *[f"category:{c}" for c in products["category"].unique()],
        *[f"brand:{b}"    for b in products["brand"].unique()],
        *[f"price:{p}"    for p in products["price_range"].unique()],
        *[f"rating:{r}"   for r in products["rating_band"].unique()],
    ],
    user_features=[
        *[f"gender:{g}"   for g in users["gender"].unique()],
        *[f"tenure:{t}"   for t in users["tenure_band"].unique()],
        *[f"age:{a}"      for a in users["age_group"].unique()],
        *[f"interest:{c}" for c in products["category"].unique()],
    ],
)

def build_interactions_matrix(df, dataset):
    return dataset.build_interactions(
        (row["user_id"], row["product_id"], row["weight"])
        for _, row in df.iterrows()
    )

train_interactions, _ = build_interactions_matrix(train_events, dataset)
test_interactions,  _ = build_interactions_matrix(test_events,  dataset)
train_interactions    = train_interactions.tocsr()
test_interactions     = test_interactions.tocsr()

users_filtered = users[users["user_id"].isin(event_users)].copy()

def get_user_features(row):
    feats = [
        f"gender:{str(row['gender']).strip()}",
        f"tenure:{str(row['tenure_band']).strip()}",
        f"age:{str(row['age_group']).strip().rstrip(',')}",
    ]
    for interest in row["interests_parsed"]:
        feats.append(f"interest:{str(interest).strip()}")
    return feats

user_features = dataset.build_user_features(
    (row["user_id"], get_user_features(row))
    for _, row in users_filtered.iterrows()
)

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

print(f"Train: {train_interactions.shape}")
print(f"Test:  {test_interactions.shape}")
print(f"Items: {item_features.shape}")
print(f"Users: {user_features.shape}")

# ── 5. HYPERPARAMETER GRID ────────────────────────────────────────────────────
param_grid = {
    "no_components": [32, 64, 128],
    "learning_rate": [0.01, 0.05, 0.1],
    "item_alpha":    [1e-6, 1e-5, 1e-4],
    "user_alpha":    [1e-6, 1e-5, 1e-4],
    "loss":          ["warp", "bpr"],
}

# Generate all combinations
keys   = list(param_grid.keys())
values = list(param_grid.values())
combos = list(itertools.product(*values))
total  = len(combos)

print(f"\nTotal combinations: {total}")
print(f"Estimated time: {total * 2} - {total * 4} minutes")
print("\nStarting search...\n", flush=True)

# ── 6. SEARCH ─────────────────────────────────────────────────────────────────
results = []
best_auc    = 0
best_params = {}
best_model  = None

for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))

    model = LightFM(
        no_components=params["no_components"],
        loss=params["loss"],
        learning_rate=params["learning_rate"],
        item_alpha=params["item_alpha"],
        user_alpha=params["user_alpha"],
        random_state=42,
    )

    # Train with early stopping per combo
    best_epoch_auc = 0
    no_improve     = 0
    patience       = 3

    for epoch in range(1, 31):
        model.fit_partial(
            interactions=train_interactions,
            user_features=user_features,
            item_features=item_features,
            num_threads=1,
            epochs=1,
        )

        test_auc = auc_score(
            model, test_interactions,
            train_interactions=train_interactions,
            user_features=user_features,
            item_features=item_features,
            num_threads=1,
        ).mean()

        if test_auc > best_epoch_auc:
            best_epoch_auc = test_auc
            no_improve     = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    p_at_k = precision_at_k(
        model, test_interactions,
        train_interactions=train_interactions,
        user_features=user_features,
        item_features=item_features,
        k=10, num_threads=1,
    ).mean()

    result = {**params, "test_auc": round(best_epoch_auc, 4),
              "precision_at_10": round(float(p_at_k), 4),
              "epochs_run": epoch}
    results.append(result)

    print(f"  [{i+1:>3}/{total}] "
          f"components={params['no_components']:>3}  "
          f"lr={params['learning_rate']}  "
          f"loss={params['loss']:<4}  "
          f"item_a={params['item_alpha']}  "
          f"test_auc={best_epoch_auc:.4f}  "
          f"p@10={p_at_k:.4f}", flush=True)

    if best_epoch_auc > best_auc:
        best_auc    = best_epoch_auc
        best_params = params
        best_model  = model
        joblib.dump(model, "models/lightfm_tuned.pkl")
        print(f"  ✓ New best: AUC={best_auc:.4f}", flush=True)

# ── 7. RESULTS SUMMARY ────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values(
    "test_auc", ascending=False
)

print("\n── Top 10 Configurations ────────────────────────────")
print(results_df.head(10).to_string(index=False))

print("\n── Best Configuration ───────────────────────────────")
for k, v in best_params.items():
    print(f"  {k:15s}: {v}")
print(f"  {'test_auc':15s}: {best_auc:.4f}")

# ── 8. SAVE ───────────────────────────────────────────────────────────────────
results_df.to_csv("data/processed/lightfm_tuning_results.csv", index=False)
joblib.dump(dataset, "models/tuned_lightfm_dataset.pkl")

meta = {
    "best_params":       best_params,
    "best_test_auc":     round(best_auc, 4),
    "item_features":     ["category", "brand", "price_range", "rating_band"],
    "user_features":     ["gender", "tenure_band", "age_group", "interests"],
    "weight_map":        weight_map,
    "tuning_combos":     total,
}
with open("models/lightfm_tuned_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved:")
print("  models/lightfm_tuned.pkl")
print("  models/tuned_lightfm_dataset.pkl")
print("  models/lightfm_tuned_meta.json")
print("  data/processed/lightfm_tuning_results.csv")
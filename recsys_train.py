import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
import joblib
import json
import ast
import sys
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


interactions = pd.read_csv("data/raw/synthetic_interactions_recsys.csv")
products = pd.read_csv("products.csv")
users = pd.read_csv("data/raw/synthetic_users_recsys.csv")

#1 clean column names
print(" data loading ")
products.columns = products.columns.str.lower().str.strip()
interactions.columns = interactions.columns.str.lower().str.strip()
users.columns = users.columns.str.lower().str.strip()

interactions = interactions.rename(columns={"action": "event_type"})
# products["product_id"] = products["id"].astype(str) #org
products["product_id"] = products["id"].astype(int) #to fix mismatch
interactions["product_id"] = interactions["product_id"].astype(int) #this too

interactions["interaction_timestamp"] = pd.to_datetime(interactions["interaction_timestamp"])

# FIX: Handle NaNs before they become features
products['brand'] = products['brand'].fillna('unknown')
products['category'] = products['category'].fillna('other')

print("interactions: ",interactions.shape)
print("products: ",products.shape)
print("users: ",users.shape)

#2 products fearute engineering
print("Product eature engineering", flush = True)

products["category"] = products["category"].str.strip().str.lower()
products["brand"] = products["brand"].str.strip().str.lower()
#categorizing price using pandas cut fun
products["price_range"] = pd.cut(
    products["price"],
    bins=[0,20,50,100,300, float("inf")], #float("inf") means infinity
    labels= ["budget", "low_mid", "mid", "high_mid", "premium"]
).astype(str)

products["rating_band"] = pd.cut(
    products["rating"],
    bins=[0, 2.9, 3.3, 4.0, 4.5, 5.0], #float("inf") means infinity
    labels= ["poor", "average", "good", "great", "excellent"]
).astype(str)

print("price ranges: ", products["price_range"].value_counts())
print("rating bands: ", products["rating_band"].value_counts())

#3 user feature engineering
print("user eature engineering", flush = True)

users["registration_date"] = pd.to_datetime(users["registration_date"])
today = interactions["interaction_timestamp"].max()
users["tenure_days"] = (today - users["registration_date"]).dt.days

users["tenure_band"]= pd.cut(
    users["tenure_days"],
    bins =[0, 15, 30, 90, 270, float("inf")],
    labels = ["new", "early", "growing", "established", "loyal"]
).astype(str)

users["age_group"] = pd.cut(
    users["age"],
    bins= [0, 25, 35, 45, 60, float("inf")],
    labels=["18_25", "26_35", "36_45", "46_60", "60+"]
).astype(str)

#convert text from intrested column to list
def parse_intrests(val):
    if pd.isna(val) or str(val).strip() in ["[]", ""]:
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []
    
users["interests_parsed"] = users ["interests"].apply(parse_intrests)

print("Tenure bands: ", users["tenure_band"].value_counts())
print("Age Groups: ", users["age_group"].value_counts())

#interaion  weights
weight_map= { "view":1, "wishlist":2, "cart":3 , "purchase":5}
interactions["weight"] = interactions["event_type"].map(weight_map).fillna(1)

#4 train test split
print("train test split", flush = True)
  #sorting ascending 
interactions = interactions.sort_values("interaction_timestamp").reset_index(drop=True)

interactions = interactions.drop_duplicates(
    subset = ["user_id", "product_id"],
    keep = "last"
).sort_values("interaction_timestamp").reset_index(drop=True)

split = int(len(interactions)*0.8)
train_events = interactions.iloc[:split]
test_events = interactions.iloc[split:]

train_pairs = set(zip(train_events["user_id"], train_events["product_id"]))
test_pairs = set(zip(test_events["user_id"], test_events["product_id"]))
overlap = train_pairs & test_pairs

print(f"overlap: {len(overlap)}")
print(f"Train events: {len(train_events)}")
print(f"Test events: {len(test_events)}")

print(f"Train range: {train_events['interaction_timestamp'].min().date()} -> \
{train_events['interaction_timestamp'].max().date()}")
print(f"Test range: {test_events['interaction_timestamp'].min().date()} -> \
{test_events['interaction_timestamp'].max().date()}")
      
#5 building lightfm dataset -sekelton
print("\nBuilding LightFM Dataset", flush = True)
event_users = interactions["user_id"].unique()
event_items = interactions["product_id"].unique()

# the exact values being registered
print("Age groups in users_df:", users["age_group"].unique().tolist())
print("Tenure bands:", users["tenure_band"].unique().tolist())
print("Genders:", users["gender"].unique().tolist())

# using lighfm's Dataset() class 
dataset = Dataset()
#mapping users and items to internal ids and building the dataset 
# basically we are making skeleton here, later we will fill the interactions and features in this skeleton
dataset.fit(
    users = event_users,
    items = event_items,
    item_features=[
        *[f"category:{c}" for c in products["category"].unique()],
        *[f"brand:{b}" for b in products["brand"].unique()],
        *[f"price:{p}" for p in products["price_range"].unique()],
        *[f"rating:{r}" for r in products["rating_band"].unique()],  
    ],
    user_features=[
        *[f"gender:{g}" for g in users["gender"].unique()],
        *[f"tenure:{t}" for t in users["tenure_band"].unique()],
        *[f"age:{a}" for a in users["age_group"].unique()],
        *[f"interest:{c}" for c in products["category"].unique()],
    ],
    )

n_users, n_items = dataset.interactions_shape()
print(f"Registered users: {n_users}, items {n_items}")

#6 generating interaction matrix
print("\nbuilding interaction matrix", flush = True)

def build_interactions(df, dataset):
    '''returns two matricess: interactions and weights '''
    return dataset.build_interactions(
        (row["user_id"], row["product_id"], row["weight"])
        for _ , row in df.iterrows()
    )

train_interactions, _ = build_interactions(train_events, dataset)
test_interactions, _ = build_interactions(test_events, dataset)

train_interactions = train_interactions.tocsr()
test_interactions = test_interactions.tocsr()

print(f"Train matrix: {train_interactions.shape}")
print(f"Test matrix: {test_interactions.shape}")

#7 build item fearture matrix
print("\nbuilding item feature matrix", flush = True)
item_features = dataset.build_item_features(
    #generatior expression to create item features in the format expected by lightfm
    #tuple structure (id, [features]) -> expected by lightfm
    ( 
        row["product_id"],
        [ 
            #make it similar to dataset.fit. thi section is sensitive 
            #ensure the strings in .fit() and .build_..._features()
            f"category:{row['category']}",
            f"brand:{row['brand']}",
            f"price:{row['price_range']}",
            f"rating:{row['rating_band']}",
        ]
    ) for _ , row in products.iterrows() if row["product_id"] in set(event_items)
)
print(f"Item feature matrix: {item_features.shape}")

#8 build user feature matrix
print("\nbuilding user feature matrix", flush = True)
users_filtered = users[users["user_id"].isin(event_users)].copy()

# Add these debug lines right before build_user_features
print("users_filtered age groups:", users_filtered["age_group"].unique().tolist())
print("users_filtered columns:", users_filtered.columns.tolist())

first_row = users_filtered.iloc[0]

print("age_group in users_filtered:", "age_group" in users_filtered.columns)
print("age groups:", users_filtered["age_group"].unique().tolist())
def get_user_features(row):
    feats = [
        f"gender:{row['gender']}",
        f"tenure:{row['tenure_band']}",
        f"age:{row['age_group']}",
    ]
    for interest in row["interests_parsed"]: 
        feats.append(f"interest:{interest}")
    return feats

print("Sample user features for first user:")
print("get user features first row",get_user_features(first_row))

user_features = dataset.build_user_features(
    (row["user_id"], get_user_features(row))
    for _ , row in users_filtered.iterrows()
)
print(f"user feature matrix: {user_features.shape}")

#9 train

print("\n Trainnig Lightfm ", flush = True)

model = LightFM(
    no_components = 64,
    loss = "warp",
    learning_rate = 0.05,
    item_alpha = 1e-5,
    user_alpha = 1e-5,
    random_state = 42,
)
EPOCHS = 15 # old 30
PATIENCE = 5
train_aucs = []
test_aucs = []
best_test_auc = 0
best_epoch = 0
no_improve = 0

for epoch in range(1, EPOCHS +1):
    try:
        model.fit_partial(
            interactions =  train_interactions,
            user_features = user_features,
            item_features = item_features,
            epochs  = 1,
        )
    except Exception as e:
        print(f"Crash fit epoch {epoch}: {e}")
        traceback.print_exc()
        sys.exit(1) 
    
    train_auc  = auc_score(
        model, train_interactions,
        user_features =  user_features,
        item_features =  item_features,
        num_threads =  1,
    ).mean()

    test_auc = auc_score(
        model, test_interactions,
        train_interactions =  train_interactions,
        user_features =  user_features,
        item_features =  item_features,
        num_threads =  1,
    ).mean()

    train_aucs.append(train_auc)
    test_aucs.append(test_auc)

    print(f"Epoch {epoch: >2} train={train_auc:.4f} test={test_auc:.4f}", flush = True)

    if test_auc > best_test_auc :
        best_test_auc = test_auc
        best_epoch = epoch
        no_improve = 0
        joblib.dump(model, "models/lightfm_best.pkl")
        print(f"Best model saved at epoch {epoch} with test AUC {test_auc:.4f}", flush = True)
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly Stopping at epoch {epoch} - best epoch {best_epoch} test AUC = {best_test_auc:.4f}")

            break
model = joblib.load("models/lightfm_best.pkl")

#10 evaluate

print(f"\n Final metrics " )
print(f"best epoch : {best_epoch}")
print(f"Besti train AUC : {train_aucs[best_epoch-1]:.4f}")
print(f"best test AUC : {best_test_auc:.4f}")

p_at_k = precision_at_k(
    model, test_interactions,
    train_interactions = train_interactions,
    user_features = user_features,
    item_features = item_features,

    k=10, num_threads=1,
).mean()

r_at_k  = recall_at_k(
    model, test_interactions,
    train_interactions = train_interactions,
    user_features = user_features,
    item_features = item_features,

    k=10, num_threads=1,
).mean()

print(f" Precision@10: {p_at_k:.4f}")
print(f" Recall@10: {r_at_k:.4f}")

#11 sample reccomendations
print("\n Sample reccomendations", flush = True)

user_id_map, _ , item_id_map, _ = dataset.mapping() 
index_to_item = {v: k for k , v in item_id_map.items()}

popular_df = pd.read_csv("data/processed/popular_products.csv")

def get_popular_products(n = 10):
    top_ids = popular_df.head(n)["product_id"].astype(int).tolist()
    result = products[products["product_id"].isin(top_ids)].copy()
    result["order"] = result["product_id"].map(
        {pid: idx for idx, pid in enumerate(top_ids)}
    )
    result = result.sort_values("order").drop(columns=["order"])
    print(f"found {len(result)} popular products")
    return result[["product_id", "title", "category", "brand", "price", "rating"]]

#------------------------------------------------------------------------
    #debug for get recommendations
# DEBUG 
# print("\nDebug -----------------------------------------------------")
# # Check popular_df
# print("popular_df product_id sample:", popular_df["product_id"].head(5).tolist())
# print("products product_id sample:  ", products["product_id"].head(5).tolist())
# print("popular_df dtypes:", popular_df.dtypes)
# print("products dtypes:  ", products.dtypes)

# # Check mapping
# user_id_map, _, item_id_map, _ = dataset.mapping()
# index_to_item = {v: k for k, v in item_id_map.items()}
# print("\nitem_id_map sample:", list(item_id_map.items())[:5])
# print("index_to_item sample:", list(index_to_item.items())[:5])

# # Check train_interactions shape
# print("\ntrain_interactions shape:", train_interactions.shape)
# print("train_interactions type:", type(train_interactions))
# n_items   = train_interactions.shape[1]
# all_items = np.arange(n_items)
# print("n_items:", n_items)
# print("all_items type:", type(all_items))
# print("all_items shape:", all_items.shape)
# print("all_items[:5]:", all_items[:5])
#------------------------------------------------------------------------

def get_recommendations(user_id, n = 10):
    if user_id not in user_id_map:
        print(f" {user_id} not in training data - returning popular data")
        return get_popular_products(n)

    user_idx =  user_id_map[user_id]
    n_items = train_interactions.shape[1]
    all_items = np.arange(n_items)

    scores = model.predict(
        user_ids =  int(user_idx),
        item_ids = all_items,
        user_features =user_features,
        item_features = item_features,
    )
    # Exclude already interacted items
    known = set(
        train_events[train_events["user_id"] == user_id]["product_id"]
    )
    top_indices = np.argsort(-scores)
    top_items = [
        index_to_item[i] for i in top_indices
        if index_to_item[i] not in known
    ][:n]

    print(f"  Top item indices: {top_items[:3]}")  # debug

    result = products[products["product_id"].isin(top_items)].copy()
    result["order"] = result["product_id"].map(
        {pid: idx for idx, pid in enumerate(top_items)}
    )
    result = result.sort_values("order").drop(columns=["order"])
    return result[["title", "category", "brand", "price", "rating"]]    

    #popular products
print("\n Our 10 popular products ")
print(get_popular_products(10).to_string())

    #persionalized fo 3 users
sample_users = train_events["user_id"].drop_duplicates().sample(3, random_state= 42).tolist()

for uid in sample_users:
    user_info = users[users["user_id"] == uid].iloc[0] 
    print(f"\n Recommended for {uid}: ")
    print(f"\n  Gender: {user_info['gender']}, Age: {user_info['age']}, Interests:  {user_info['interests']} ")
    print(get_recommendations(uid, n=10).to_string())

#12 CURVE 
actual_epochs = len(train_aucs)
plt.figure(figsize=(10, 5))
plt.plot(range(1, actual_epochs+1), train_aucs,
         label="Train AUC", marker="o")
plt.plot(range(1, actual_epochs+1), test_aucs,
         label="Test AUC",  marker="o")
plt.axvline(x=best_epoch, color="red", linestyle="--",
            label=f"Best epoch {best_epoch}")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("LightFM Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/figures/lightfm_learning_curve.png", dpi=150)
print("\nPlot saved: lightfm_learning_curve.png")

#12 SAVE
joblib.dump(dataset, "models/lightfm_dataset.pkl")
meta = {
    "best_epoch":      int(best_epoch),
    "no_components":   64,
    "loss":            "warp",
    "learning_rate":   0.05,
    "item_alpha":      1e-5,
    "user_alpha":      1e-5,
    "best_test_auc":   round(float(best_test_auc), 4),
    "train_auc":       round(float(train_aucs[best_epoch-1]), 4),
    "precision_at_10": float(p_at_k.item() if hasattr(p_at_k, 'item') else p_at_k),
    "recall_at_10":    float(r_at_k.item() if hasattr(r_at_k, 'item') else r_at_k),
    "item_features":   ["category", "brand", "price_range", "rating_band"],
    "user_features":   ["gender", "tenure_band", "age_group", "interests"],
    "weight_map":      {"view": 1, "wishlist": 2, "cart": 3, "purchase": 5},
}
with open("models/lightfm_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved:")
print("  models/lightfm_best.pkl")
print("  models/lightfm_dataset.pkl")
print("  models/lightfm_meta.json")
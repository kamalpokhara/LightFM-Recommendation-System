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
products     = pd.read_csv("products.csv")
users        = pd.read_csv("data/raw/synthetic_users_recsys.csv")

#1 clean column names
print(" data loadeding ")
products.columns = products.columns.str.lower().str.strip()
interactions.columns = interactions.columns.str.lower().str.strip()
users.columns = users.columns.str.lower().str.strip()

interactions = interactions.rename(columns={"action": "event_type"})
products["product_id"] = products["id"].astype(str)
interactions["interaction_timestamp"] = pd.to_datetime(interactions["interaction_timestamp"])

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
    labels=["18-25", "26-35", "36-45", "46-60", "60+"]
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
event_users = interactions["user_id"].unique()
event_items = interactions["product_id"].unique()

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

#7 generating item fearture matrix
print("\nbuilding item feature matrix", flush = True)
item_features = dataset.build_item_features(
    #generatior expression to create item features in the format expected by lightfm
    #tuple structure (id, [features]) -> expected by lightfm
    ( 
        row["product_id"],
        [ 
            f"category: {row['category']}",
            f"brand:{row['brand']}",
            f"price: {row['price_range']}",
            f"rating: {row['rating_band']}",
        ]
    ) for _ , row in products.iterrows() if row["product_id"] in set(event_items)
)
print(f"Item feature matrix: {item_features.shape}")

#8 generating user feature matrix
print("\nbuilding user feature matrix", flush = True)
users_filtered = users[users["user_id"].isin(event_users)].copy()

def get_user_features(row):
    feats = [
        f"gender:{row['gender']}",
        f"tenure:{row['tenure_band']}",
        f"age:{row['age_group']},"
    ]
    for interest in row["interests_parsed"]: 
        feats.append(f"interest:{interest}")
    return feats

user_features = dataset.build_user_features(
    (row["user_id"], get_user_features(row))
    for _ , row in users_filtered.iterrows()
)
print(f"user feature matrix: {user_features.shape}")

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

# clean column names
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

# products fearute engineering
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

# user feature engineering
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

def parse_intrests(val):
    if pd.isna(val) or str(val).strip() in ["[]", ""]:
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []
users["interests_parsed"] = users ["intrested"].apply(parse_intrests)
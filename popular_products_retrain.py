import pandas as pd
import numpy as np
import joblib
import json
 
# 1 load products

print("Loading data", flush=True)

interactions = pd.read_csv("interactions_export.csv")
products = pd.read_csv("products_export.csv")

print("first 5 products ")
print(products.head(20)[[
    "id", "title", "category", "brand","price", "rating",
    ]])


products.columns = products.columns.str.lower().str.strip()
interactions.columns = interactions.columns.str.lower().str.strip()

interactions = interactions.rename(columns={"action": "event_type"})
# products["product_id"] = products["id"].astype(str)
# fix product_id to integer
products["product_id"] = products["id"].astype(int)
interactions["product_id"] = interactions["product_id"].astype(int)

print(f"Interactions: {interactions.shape}")
print(f"Products:     {products.shape}")

# 2 interaction weights
weight_map = {
    "view":1,
    "wishlist":2,
    "cart":3,
    "purchase":5,
}
interactions["weight"] = interactions["event_type"].map(weight_map).fillna(1)

# 3 coompute popular
print("Computing popular products", flush=True)

popular = (
    interactions.groupby("product_id").agg(
        total_score=("weight", "sum"),
        view_count=("event_type", lambda x: (x == "view").sum()),
        cart_count=("event_type", lambda x: (x == "cart").sum()),
        wishlist_count=("event_type", lambda x: (x == "wishlist").sum()),
        purchase_count=("event_type", lambda x: (x == "purchase").sum()),
        unique_users=("user_id", "nunique"),
    
    ).reset_index().sort_values("total_score", ascending=False).reset_index(drop=True)
)
popular["product_id"] = popular["product_id"].astype(str).str.strip()
products["product_id"] = products["product_id"].astype(str).str.strip()
popular = popular.merge(
    products[["product_id", "title", "category",
        "brand", "price", "rating"]],
    on="product_id",
    how="left"
)

print(f"top 20 popular prodcuts: ")
print(popular.head(20)[[
    "product_id", "title", "category", "brand","price", 
    "rating", "total_score", "view_count", "purchase_count", 
    "unique_users"
    ]])

# 4 save
popular.to_csv("popular_products.csv", index=False)
top20_ids= popular.head(20)["product_id"].tolist()
joblib.dump(top20_ids, "models/popular_products.pkl")

meta = {
    "top_n": 20,
    "weight_map": weight_map,
    "total_products": len(popular),
    "scoring": "weighted sum: view=1, wishlist=2, cart=3, purchase=5",
}
with open("models/popular_products_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved:")
print("  popular_products.csv")
print("  models/popular_products.pkl")
print("  models/popular_products_meta.json")

print("Sample IDs in Interactions:", popular["product_id"].head(5).tolist())
print("Sample IDs in Products:", products["product_id"].head(5).tolist())
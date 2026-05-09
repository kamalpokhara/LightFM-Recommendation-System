import pandas as pd

interactions = pd.read_csv("interactions_export.csv")
users = pd.read_csv("users_export.csv")
products = pd.read_csv("products_export.csv")

print("=== INTERACTIONS ===")
print(f"Total rows: {len(interactions)}")
print(f"Unique users: {interactions['user_id'].nunique()}")
print(f"Unique products: {interactions['product_id'].nunique()}")
print(f"Event types:\n{interactions['event_type'].value_counts()}")

print("\n=== AFTER DEDUP (user+product) ===")
deduped = interactions.drop_duplicates(subset=["user_id", "product_id"], keep="last")
print(f"Rows after dedup: {len(deduped)}")
print(f"Train size: {int(len(deduped)*0.8)}")
print(f"Test size: {len(deduped) - int(len(deduped)*0.8)}")

print("\n=== USERS ===")
print(f"Total users: {len(users)}")
print(f"Users with interests: {users[users['interests'] != '[]'].shape[0]}")
print(f"Age nulls: {users['age'].isna().sum()}")
print(f"Gender nulls: {users['gender'].isna().sum()}")

print("\n=== OVERLAP CHECK ===")
interaction_users = set(interactions['user_id'].unique())
user_file_users = set(users['user_id'].unique())
print(f"Users in interactions: {len(interaction_users)}")
print(f"Users in users file: {len(user_file_users)}")
print(f"In interactions but NOT in users file: {interaction_users - user_file_users}")
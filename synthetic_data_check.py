import pandas as pd

syn_users = pd.read_csv("data/raw/synthetic_users_recsys.csv")
syn_interactions = pd.read_csv("data/raw/synthetic_interactions_recsys.csv")

print("=== SYNTHETIC USERS ===")
print(syn_users.columns.tolist())
print(syn_users.head(3))
print(f"Total: {len(syn_users)}")

print("\n=== SYNTHETIC INTERACTIONS ===")
print(syn_interactions.columns.tolist())
print(syn_interactions.head(3))
print(f"Total: {len(syn_interactions)}")
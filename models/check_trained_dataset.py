# inspect_model.py
import joblib
import numpy as np

model = joblib.load("lightfm_best.pkl")
dataset = joblib.load("lightfm_dataset.pkl")

print("═══ MODEL ═══════════════════════════════════════")
print(f"Type: {type(model)}")
print(f"no_components:  {model.no_components}")
print(f"loss:           {model.loss}")
print(f"item_embeddings shape: {model.item_embeddings.shape}")
print(f"user_embeddings shape: {model.user_embeddings.shape}")

print("\n═══ DATASET MAPPING ═════════════════════════════")
user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

print(f"Total users:         {len(user_id_map)}")
print(f"Total items:         {len(item_id_map)}")
print(f"Total user features: {len(user_feature_map)}")
print(f"Total item features: {len(item_feature_map)}")

print("\n── Sample user_ids ──────────────────────────────")
print(list(user_id_map.keys())[:5])

print("\n── Sample item_ids ──────────────────────────────")
print(list(item_id_map.keys())[:5])

print("\n── ALL user feature names ───────────────────────")
print(sorted(user_feature_map.keys()))

print("\n── ALL item feature names ───────────────────────")
print(sorted(item_feature_map.keys(), key=lambda x: str(x))) #error originated form here

print("\n═══ CAN WE PREDICT WITHOUT FEATURES? ════════════")
# Test predict without user/item features
try:
    scores = model.predict(
        user_ids=0,
        item_ids=np.arange(model.item_embeddings.shape[0]),
        num_threads=1,
    )
    print(f"Predict WITHOUT features works — scores shape: {scores.shape}")
    print(f"Top 3 item indices: {np.argsort(-scores)[:3]}")
except Exception as e:
    print(f"Predict WITHOUT features failed: {e}")
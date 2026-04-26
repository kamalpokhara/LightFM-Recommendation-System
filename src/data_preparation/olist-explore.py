import pandas as pd
import numpy as np
import os
from datetime import datetime


BASE = "../data/raw/"  # change if your path is different

print("=== OLIST DATASET (Multiple Files) ===")
olist_path = os.path.join(BASE, "olist-rec/")

# Load the files we care about for rec-sys + products
products = pd.read_csv(os.path.join(olist_path, "olist_products_dataset.csv"))
order_items = pd.read_csv(os.path.join(olist_path, "olist_order_items_dataset.csv"))
customers = pd.read_csv(os.path.join(olist_path, "olist_customers_dataset.csv"))
orders = pd.read_csv(
    os.path.join(olist_path, "olist_orders_dataset.csv")
)  # for timestamps

print(f"Products shape: {products.shape}")
print(f"Order Items shape: {order_items.shape}")
print(f"Customers shape: {customers.shape}")
print(f"Orders shape: {orders.shape}")

print("\nProducts columns:", sorted(products.columns.tolist()))
print("Order Items columns:", sorted(order_items.columns.tolist()))
print("Customers columns:", sorted(customers.columns.tolist()))
print("Orders columns:", sorted(orders.columns.tolist()))

print(
    "\nMissing % in Products:\n",
    (products.isnull().sum() / len(products) * 100).round(2),
)
print(
    "\nMissing % in Order Items:\n",
    (order_items.isnull().sum() / len(order_items) * 100).round(2),
)

# Quick interaction stats for LightFM
print(f"\nUnique customers (from orders): {orders['customer_id'].nunique()}")
print(f"Unique products: {order_items['product_id'].nunique()}")
print(f"Total purchases: {len(order_items)}")
print(f"Sparsity estimate: very high (typical for e-comm)")

# Date range (important for later temporal features)
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
print(
    f"Date range: {orders['order_purchase_timestamp'].min()} to {orders['order_purchase_timestamp'].max()}"
)

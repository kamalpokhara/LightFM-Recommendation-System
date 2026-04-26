import requests
import pandas as pd

# Fetch all products from DummyJSON
url = "https://dummyjson.com/products?limit=194"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    products = data.get("products", [])
    
    # Convert to DataFrame
    df = pd.DataFrame(products)
    
    # Keep only the required columns
    selected_columns = ["id", "title", "description", "category", 
                        "price", "rating", "stock", "brand", "images"]
    df = df[selected_columns]
    
    # Optional: store only the first image link instead of the full list
    df["image_link"] = df["images"].apply(lambda imgs: imgs[0] if isinstance(imgs, list) and imgs else None)
    df = df.drop(columns=["images"])
    
    # Save to CSV
    df.to_csv("products.csv", index=False)
    print("✅ Products saved to products.csv")
else:
    print("❌ Failed to fetch data. Status code:", response.status_code)

import numpy as np
from scipy.sparse import csr_matrix, eye
from lightfm import LightFM

def check_lightfm():
    try:
        # 1. Test Matrix Creation (Scipy compatibility)
        interactions = csr_matrix([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
        item_features = eye(3).tocsr()
        
        # 2. Test Model Initialization & Training
        # Use 'warp' as it's the most computationally intensive loss
        model = LightFM(loss='warp', no_components=10)
        
        print("Training model...")
        model.fit(interactions, item_features=item_features, epochs=5)
        
        # 3. Test Prediction Logic
        scores = model.predict(0, np.arange(3), item_features=item_features)
        
        if len(scores) == 3 and not np.isnan(scores).any():
            print("✅ LightFM is working properly!")
            print(f"Sample scores: {scores}")
        else:
            print("❌ Model produced invalid scores (NaN or wrong shape).")
            
    except Exception as e:
        print(f"❌ Error detected: {e}")

if __name__ == "__main__":
    check_lightfm()

    
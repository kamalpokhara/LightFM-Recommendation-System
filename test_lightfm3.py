# import os
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.sparse import csr_matrix , coo_matrix
from lightfm import LightFM

# The simplest possible data
# data = coo_matrix([[1]], dtype=np.float32)
# model = LightFM(loss='logistic') # Logistic is simpler/faster than WARP

# print("Starting Fit...")
# if model.fit(data, epochs=1): 
#     print("Fit Finished Successfully!")
# else :
#     print("Fit Failed!")

import numpy as np
from scipy.sparse import coo_matrix
from lightfm import LightFM

print("Testing LightFM with Sparse Data...", flush=True)

# 1. Make the data SPARSE (mostly zeros)
# 50 users, 20 items, but only 10 total interactions
rows = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
cols = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
values = np.ones(10).astype(np.float32)
data = coo_matrix((values, (rows, cols)), shape=(50, 20))

print(f"Matrix shape: {data.shape} | Non-zeros: {data.nnz}", flush=True)

# 2. Try 'bpr' first (it's faster and less likely to hang than WARP on small data)
model = LightFM(no_components=5, loss="bpr", random_state=42)

print("Calling fit with BPR...", flush=True)
model.fit(data, epochs=1, num_threads=1)
print("BPR PASSED", flush=True)

# 3. Now try WARP
print("Calling fit with WARP...", flush=True)
model_warp = LightFM(no_components=5, loss="warp", random_state=42)
print("This may take a moment due to WARP's nature on small datasets...", flush=True)
model_warp.fit(data, epochs=1, num_threads=1)
print("WARP PASSED — LightFM is working 100%", flush=True)
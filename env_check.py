# env_check.py
import sys
import platform

print("── System ───────────────────────────────────────────")
print(f"Python:   {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Machine:  {platform.machine()}")
print(f"CPU:      {platform.processor()}")

print("\n── Package Versions ─────────────────────────────────")
import numpy; print(f"numpy:   {numpy.__version__}")
import scipy; print(f"scipy:   {scipy.__version__}")

print("\n── Scipy Sparse Check ───────────────────────────────")
from scipy.sparse import csr_matrix
import numpy as np
m = csr_matrix(np.ones((5,5), dtype=np.float32))
print(f"sparse matrix ok: {m.shape}")

print("\n── LightFM Import Check ─────────────────────────────")
try:
    import lightfm
    print(f"lightfm version: {lightfm.__version__}")
except Exception as e:
    print(f"lightfm import failed: {e}")

print("\n── LightFM Fast Check ───────────────────────────────")
try:
    import lightfm._lightfm_fast as fast
    print(f"_lightfm_fast loaded ok: {fast}")
except Exception as e:
    print(f"_lightfm_fast failed: {e}")

print("\n── Cython Check ─────────────────────────────────────")
try:
    import Cython
    print(f"Cython: {Cython.__version__}")
except:
    print("Cython: not installed")

print("\n── Microsoft Visual C++ Check ───────────────────────")
import ctypes
try:
    ctypes.CDLL("vcruntime140.dll")
    print("vcruntime140.dll: FOUND")
except:
    print("vcruntime140.dll: NOT FOUND ← possible cause")

try:
    ctypes.CDLL("msvcp140.dll")
    print("msvcp140.dll:     FOUND")
except:
    print("msvcp140.dll:     NOT FOUND ← possible cause")

print("\n── LightFM .pyd file check ──────────────────────────")
import os
import lightfm
lightfm_dir = os.path.dirname(lightfm.__file__)
print(f"LightFM installed at: {lightfm_dir}")
files = os.listdir(lightfm_dir)
print(f"Files: {files}")

print("\nDone.", flush=True)
"""Verify: does quick20_clean.py's 'HF guard' disable the embedder?

quick20_eval.py explicitly warns against this:
    "Do NOT pre-fill _EMBEDDER_CACHE with None here, otherwise
     event penalty will be completely disabled."

quick20_clean.py:190-195 does exactly that.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.container_manager import _EMBEDDER_CACHE, _get_embedder

print("--- BEFORE the guard (cache empty) ---")
print("  cache:", dict(_EMBEDDER_CACHE))
e = _get_embedder("all-MiniLM-L6-v2")
print("  embedder:", "OK" if e is not None else "None")

print()
print("--- Simulating quick20_clean.py:190-195 on a FRESH cache ---")
_EMBEDDER_CACHE.clear()
for _name in ("all-MiniLM-L6-v2",):
    _EMBEDDER_CACHE.setdefault(_name, None)
print("  cache after guard:", dict(_EMBEDDER_CACHE))
e2 = _get_embedder("all-MiniLM-L6-v2")
print("  embedder:", "OK" if e2 is not None else "None  <-- DISABLED")
print()
print("===> embedder disabled by the guard:",
      "CONFIRMED" if e2 is None else "NOT REPRODUCED")

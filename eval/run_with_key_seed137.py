"""Run quick20_clean with DEEPSEEK_API_KEY sourced from eval/api_key.txt
and seed=137.  Embedder stays local (sentence-transformers, no DeepSeek).

Usage:
    python eval/run_with_key_seed137.py
"""
import os
import sys

KEY_PATH = r"c:\Users\zsf\Desktop\lyh\StenRAG\eval\api_key.txt"
try:
    with open(KEY_PATH, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
    os.environ["DEEPSEEK_API_KEY"] = api_key
    print(f"[env] DEEPSEEK_API_KEY set ({len(api_key)} chars)")
except Exception as e:
    print(f"[env] could not read key: {e}")
    sys.exit(1)

# Forward CLI args to quick20_clean.py.
sys.argv = [
    "quick20_clean.py",
    "--n", "20",
    "--seed", "137",
    "--tag", "ev_sig_v1_seed137",
    "--use-llm",
]

PROJECT_ROOT = r"c:\Users\zsf\Desktop\lyh\StenRAG"
sys.path.insert(0, PROJECT_ROOT)
import runpy
runpy.run_path(PROJECT_ROOT + r"\eval\quick20_clean.py", run_name="__main__")

"""Run quick20_clean with a fresh DEEPSEEK_API_KEY sourced from a local file.

Usage:
    python eval/run_with_key.py
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
    "--seed", "7",
    "--tag", "online_hyperedge_test",
    "--use-llm",
]

# Import as a module so its __main__ guard fires.
PROJECT_ROOT = r"c:\Users\zsf\Desktop\lyh\StenRAG"
sys.path.insert(0, PROJECT_ROOT)
import runpy
runpy.run_path(PROJECT_ROOT + r"\eval\quick20_clean.py", run_name="__main__")

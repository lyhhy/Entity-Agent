"""Runner for the 20-question clean eval (any seed).

Usage:
    python eval/_run_seed137.py <tag> <seed> [extra quick20_clean args...]

Reads the API key from eval/api_key.txt, runs eval/quick20_clean.py with
the LLM + offline hyperedges enabled, and tees the FULL stdout to
eval/results/<tag>.out so the complete per-question Q/G/A output is
preserved (not just the tail).
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KEY_FILE = REPO / "eval" / "api_key.txt"

key = KEY_FILE.read_text(encoding="utf-8").strip()
if not key:
    print("[FATAL] no key in eval/api_key.txt")
    raise SystemExit(1)

env = dict(os.environ)
env["DEEPSEEK_API_KEY"] = key
env.setdefault("TRANSFORMERS_OFFLINE", "1")
env.setdefault("HF_HUB_OFFLINE", "1")
env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
env["PYTHONIOENCODING"] = "utf-8"
env["TOKENIZERS_PARALLELISM"] = "false"

tag = sys.argv[1] if len(sys.argv) > 1 else "run"
seed = sys.argv[2] if len(sys.argv) > 2 else "137"
extra = sys.argv[3:]

out = REPO / "eval" / "results" / f"clean_results_{tag}.json"
log = REPO / "eval" / "results" / f"{tag}.out"

cmd = [
    sys.executable, "eval/quick20_clean.py",
    "--n", "20",
    "--seed", str(seed),
    "--use-llm",
    "--offline-hyperedges", "eval/offline_hyperedges/hyperedges.parquet",
    "--out", str(out),
    "--tag", str(tag),
] + extra

print("[run] " + " ".join(cmd), flush=True)
with open(log, "w", encoding="utf-8") as fh:
    proc = subprocess.run(cmd, cwd=str(REPO), env=env,
                          stdout=fh, stderr=subprocess.STDOUT)
print(f"[run] exit={proc.returncode}")
print(f"[run] full output → {log}")
print(f"[run] records    → {out}")
sys.exit(proc.returncode)

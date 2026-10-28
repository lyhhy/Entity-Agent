"""Strip progress-bar noise from a *.out capture and write a clean copy.

Usage:
    python eval/_clean_out.py <in.out> [out.txt]
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

src = REPO / sys.argv[1]
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "eval" / "results" / (
    src.stem + "_clean.txt")

d = src.read_text(encoding="utf-8", errors="replace")
# tqdm progress bars
d = re.sub(r"Batches:.*?it/s\]", "", d, flags=re.S)
d = re.sub(r"\r", "\n", d)
# the "   10|" line-number gutter that _safe_print inherits from the tee
d = re.sub(r"^\s*\d+\|", "", d, flags=re.M)
d = re.sub(r"\n{3,}", "\n\n", d)
d = d.strip()

dst.write_text(d, encoding="utf-8")
print(f"[clean] {src.name} -> {dst}  ({len(d)} chars)")

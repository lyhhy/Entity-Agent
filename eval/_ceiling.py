"""Retrieval ceiling: how much of the remaining error is the extractor's
fault vs. the retrieval stage's fault."""
from __future__ import annotations

import glob
import json
import os

files = sorted(glob.glob("eval/results/clean_results_ho*.json"))
tot_n = tot_em = tot_f1 = tot_ceiling = 0
print(f"{'set':<12} {'n':>4} {'EM':>7} {'F1':>7} {'ceil':>7}")
for p in files:
    d = json.load(open(p, encoding="utf-8"))
    rs = d["records"]
    n = len(rs)
    em = sum(r["em"] for r in rs) / n
    f1 = sum(r["f1"] for r in rs) / n
    ceiling = sum(1 for r in rs if any(e.get("is_gold") for e in r["evidence"])) / n
    name = os.path.basename(p).replace("clean_results_", "").replace(".json", "")
    print(f"{name:<12} {n:>4} {em:>7.3f} {f1:>7.3f} {ceiling:>7.3f}")
    tot_n += n
    tot_em += em * n
    tot_f1 += f1 * n
    tot_ceiling += ceiling * n

print(f"{'ALL':<12} {tot_n:>4} {tot_em/tot_n:>7.3f} {tot_f1/tot_n:>7.3f} "
      f"{tot_ceiling/tot_n:>7.3f}")
print()
print(f"  headroom from retrieval (gold in evidence but not answered): "
      f"{tot_ceiling/tot_n - tot_em/tot_n:+.3f} EM")
print(f"  headroom from retrieval (gold not in evidence at all)      : "
      f"{1 - tot_ceiling/tot_n:+.3f} EM")

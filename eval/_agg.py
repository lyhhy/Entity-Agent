"""Aggregate failure analysis across multiple clean-eval runs.

Usage:
    python eval/_agg.py <records.json> [<records.json> ...]
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "at",
        "is", "was", "were", "are", "by", "with", "as", "that", "this", "it"}


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def toks(s):
    return {t for t in norm(s).split() if t not in STOP and len(t) > 1}


all_runs = []
for p in sys.argv[1:]:
    d = json.load(open(REPO / p, encoding="utf-8"))
    name = Path(p).stem.replace("clean_results_", "").replace("_seed", "_s")
    recs = d["records"]
    all_runs.append((name, d, recs))

print("=" * 88)
print("SUMMARY")
print("=" * 88)
print(f"{'run':<28} {'EM':>7} {'F1':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
tot = {"em": 0.0, "f1": 0.0, "r1": 0.0, "r5": 0.0, "r10": 0.0}
N = 0
for name, d, recs in all_runs:
    n = len(recs)
    N += n
    for k in tot:
        tot[k] += sum(r[k.lower() if k != "r1" else "r1"] for r in recs)
    print(f"  {name:<26} {d['em']:>7.3f} {d['f1']:>7.3f} "
          f"{d['r1']:>7.3f} {d['r5']:>7.3f} {d['r10']:>7.3f}")
print(f"  {'-'*26} " + " ".join(f"{'-'*7:>7}" for _ in range(5)))
print(f"  {'MACRO AVG':<26} " +
      " ".join(f"{tot[k]/len(all_runs):>7.3f}" for k in
               ("em", "f1", "r1", "r5", "r10")))
print(f"  {'MICRO AVG (n=%d)' % N:<26} " +
      " ".join(f"{tot[k]/N:>7.3f}" for k in ("em", "f1", "r1", "r5", "r10")))

print()
print("=" * 88)
print("FAILURE TAXONOMY (over all questions)")
print("=" * 88)
buckets = {
    "OK": [],
    "PARTIAL (F1>0, EM=0)": [],
    "UNANSWERABLE (said unanswerable)": [],
    "WRONG SPAN (retrieved ok)": [],
    "WRONG SPAN (gold not retrieved)": [],
}
em_with_r1_0 = 0
for name, d, recs in all_runs:
    for r in recs:
        tag = (name, r)
        ans = (r.get("answer") or "").strip()
        if r["em"]:
            buckets["OK"].append(tag)
            if not r["r1"]:
                em_with_r1_0 += 1
            continue
        if ans.lower().startswith("unanswerable"):
            buckets["UNANSWERABLE (said unanswerable)"].append(tag)
            continue
        if r["f1"] > 0:
            buckets["PARTIAL (F1>0, EM=0)"].append(tag)
            continue
        if r["r10"]:
            buckets["WRONG SPAN (retrieved ok)"].append(tag)
        else:
            buckets["WRONG SPAN (gold not retrieved)"].append(tag)

for k, v in buckets.items():
    print(f"  {k:<36} {len(v):>3}/{N}  ({len(v)/N*100:4.1f}%)")

print()
print(f"  !! EM hit but R@1=0:  {em_with_r1_0}/{len(buckets['OK'])} of the "
      f"correct answers  -> R@1 does NOT track answer correctness")

for k in ("PARTIAL (F1>0, EM=0)", "UNANSWERABLE (said unanswerable)",
          "WRONG SPAN (retrieved ok)", "WRONG SPAN (gold not retrieved)"):
    if not buckets[k]:
        continue
    print()
    print("─" * 88)
    print(f"─── {k}  ({len(buckets[k])})")
    print("─" * 88)
    for name, r in buckets[k]:
        print(f"  [{name}] {r['qid'][:8]} R@1={r['r1']} R@5={r['r5']} "
              f"R@10={r['r10']} F1={r['f1']:.3f}")
        print(f"    Q: {r['question']}")
        print(f"    G: {r['gold']}")
        print(f"    A: {r['answer']}")

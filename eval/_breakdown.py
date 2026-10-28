"""Failure breakdown for a seed137 clean-eval records file.

Classifies each question into:
  OK            - EM hit
  EXTRACT       - answer was produced but wrong (wrong span / wrong entity)
  UNANSWERABLE  - pipeline said "unanswerable" but gold had an answer
  RETRIEVAL     - gold string never appears in the retrieved context
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

path = REPO / (sys.argv[1] if len(sys.argv) > 1
               else "eval/results/clean_results_seed137_base_seed137_base.json")
recs = json.load(open(path, encoding="utf-8"))["records"]

STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "at",
        "is", "was", "were", "are", "by", "with", "as", "that", "this", "it"}


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def content_toks(s):
    return {t for t in norm(s).split() if t not in STOP and len(t) > 1}


buckets = {"OK": [], "EXTRACT": [], "UNANSWERABLE": [], "RETRIEVAL": []}

for r in recs:
    gold = (r.get("gold") or "").strip()
    ans = (r.get("answer") or "").strip()
    gl = norm(gold)
    al = norm(ans)

    if r.get("em"):
        buckets["OK"].append(r)
        continue

    if al.startswith("unanswerable"):
        buckets["UNANSWERABLE"].append(r)
        continue

    # answer given but wrong: is the gold present in the answer?
    if gl and gl in al:
        buckets["EXTRACT"].append(r)   # containment but normalisation mismatch
    else:
        # is the gold a substring of gold-ish? check token overlap
        gt, at = content_toks(gold), content_toks(ans)
        if gt and gt & at:
            buckets["EXTRACT"].append(r)
        else:
            buckets["RETRIEVAL"].append(r)

print("=" * 88)
print(f"{path.name}")
print("=" * 88)
for k, v in buckets.items():
    print(f"  {k:<14} {len(v):>2}/{len(recs)}")

print()
print("=" * 88)
print("DETAIL")
print("=" * 88)
for k in ("UNANSWERABLE", "EXTRACT", "RETRIEVAL"):
    if not buckets[k]:
        continue
    print(f"\n─── {k} ───")
    for r in buckets[k]:
        print(f"  {r['qid'][:8]}  R@1={r['r1']} R@5={r['r5']} R@10={r['r10']}")
        print(f"    Q: {r['question']}")
        print(f"    G: {r['gold']}")
        print(f"    A: {r['answer']}")

print()
print("=" * 88)
print("R@K distribution")
print("=" * 88)
for k in range(1, 4):
    key = ["r1", "r5", "r10"][k - 1]
    hits = [r for r in recs if r.get(key)]
    print(f"  {key}: {len(hits)}/{len(recs)}")

# cross-tab: retrieval vs generation
print()
print("=" * 88)
print("Cross-tab: R@10 hit x outcome")
print("=" * 88)
ct = {}
for r in recs:
    for k, v in buckets.items():
        if r in v:
            ct[(r["r10"], k)] = ct.get((r["r10"], k), 0) + 1
for key in sorted(ct):
    print(f"  R@10={key[0]}  {key[1]:<14} {ct[key]}")

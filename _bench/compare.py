"""Compare two bench runs question-by-question (not just aggregate)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "_bench"

a = json.loads((BENCH / f"bench_{sys.argv[1]}.json").read_text(encoding="utf-8"))
b = json.loads((BENCH / f"bench_{sys.argv[2]}.json").read_text(encoding="utf-8"))

print(f"A = {sys.argv[1]}   B = {sys.argv[2]}")
print(f"{'metric':<12}{'A':>10}{'B':>10}{'delta':>10}")
print("-" * 44)
for k in ("r1", "r5", "r10", "pool_avg", "final_avg", "ms_avg"):
    va, vb = a[k], b[k]
    d = vb - va
    flag = "" if abs(d) < 1e-9 else ("  <-- CHANGED" if k in ("r1","r5","r10") else "  (timing)")
    print(f"{k:<12}{va:>10.3f}{vb:>10.3f}{d:>+10.3f}{flag}")

print("-" * 44)
qa = {q["qid"]: q for q in a["per_q"]}
qb = {q["qid"]: q for q in b["per_q"]}
assert set(qa) == set(qb), "different question sets!"
diffs = []
for qid in qa:
    x, y = qa[qid], qb[qid]
    for k in ("r1", "r5", "r10", "pool", "n_final", "a_cids", "b_cids", "n_he"):
        if x[k] != y[k]:
            diffs.append((qid, k, x[k], y[k]))

if diffs:
    print(f"\n{len(diffs)} PER-QUESTION DIFFERENCES:")
    for qid, k, x, y in diffs:
        print(f"  {qid}  {k}: {x} -> {y}")
    sys.exit(1)
else:
    print(f"\nIDENTICAL on all {len(qa)} questions "
          f"(r1/r5/r10/pool/n_final/a_cids/b_cids/n_he)")

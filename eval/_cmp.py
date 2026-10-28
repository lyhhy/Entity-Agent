"""Compare two clean-eval records files question by question.

Usage:
    python eval/_cmp.py <fileA> <fileB>
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(p):
    return json.load(open(REPO / p, encoding="utf-8"))["records"]


def summ(recs):
    n = len(recs)
    return {
        "EM": sum(r["em"] for r in recs) / n,
        "F1": sum(r["f1"] for r in recs) / n,
        "R@1": sum(r["r1"] for r in recs) / n,
        "R@5": sum(r["r5"] for r in recs) / n,
        "R@10": sum(r["r10"] for r in recs) / n,
    }


a = load(sys.argv[1])
b = load(sys.argv[2])
A = {r["qid"]: r for r in a}
B = {r["qid"]: r for r in b}

print(f"A = {sys.argv[1]}")
print(f"B = {sys.argv[2]}")
print()
print(f"{'':6} {'EM':>7} {'F1':>7} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
sa, sb = summ(a), summ(b)
for k in sa:
    print(f"  A    {sa[k]:>7.3f}")
for k in sb:
    print(f"  B    {sb[k]:>7.3f}")
print()
print("per-question diff (only changed rows)")
print("=" * 78)
nch = 0
for q in A:
    x, y = A[q], B[q]
    ch = []
    for k in ("em", "f1", "r1", "r5", "r10"):
        if x.get(k) != y.get(k):
            ch.append(f"{k}:{x.get(k)}->{y.get(k)}")
    if ch:
        nch += 1
        print(f"  {q[:8]}  " + "  ".join(ch))
        print(f"    Q: {x['question']}")
        print(f"    G : {x['gold']}")
        print(f"    A : {x['answer']}")
        print(f"    B : {y['answer']}")
print()
print(f"changed questions: {nch}/{len(A)}")

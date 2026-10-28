"""Diagnose why --attr-soft-match had zero effect in the seed137 run."""
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def load(path):
    p = REPO / path
    if not p.exists():
        print(f"[MISS] {path}")
        return None
    return json.load(open(p, encoding="utf-8"))


base = load("eval/results/clean_results_seed137_base_seed137_base.json")
soft = load("eval/results/clean_results_seed137_soft_seed137_soft.json")

if not base or not soft:
    sys.exit(1)

print("=" * 78)
print("A. top-level summary")
print("=" * 78)
for name, d in (("base", base), ("soft", soft)):
    keys = {k: v for k, v in d.items() if not isinstance(v, (list, dict))}
    print(f"  {name}: {keys}")

print()
print("=" * 78)
print("B. is attr_soft_match_mode actually ON in the soft run?")
print("=" * 78)
rec = soft["records"][0]
dbg = rec.get("debug", {}) or {}
print(f"  debug keys: {sorted(dbg.keys())}")

# Does ef_scores exist and does attr_match vary?
ef = dbg.get("ef_scores") or []
print(f"  ef_scores entries: {len(ef)}")
if ef:
    ams = [d.get("attr_match", None) for d in ef]
    ats = [d.get("attr_term", None) for d in ef]
    print(f"  attr_match values: {ams[:10]}")
    print(f"  attr_term  values: {ats[:10]}")
    nz = [a for a in ams if a]
    print(f"  non-zero attr_match: {len(nz)}/{len(ams)}")
else:
    print("  [!] ef_scores MISSING from debug")

print()
print("=" * 78)
print("C. per-question diff (base vs soft)")
print("=" * 78)
b_by = {r["qid"]: r for r in base["records"]}
s_by = {r["qid"]: r for r in soft["records"]}
diffs = 0
for qid in b_by:
    b, s = b_by[qid], s_by[qid]
    changed = []
    for k in ("em", "f1", "r1", "r5", "r10"):
        if b.get(k) != s.get(k):
            changed.append(f"{k}: {b.get(k)} -> {s.get(k)}")
    if changed:
        diffs += 1
        print(f"  {qid[:10]}  " + "; ".join(changed))
print(f"  questions with any metric change: {diffs}/{len(b_by)}")

print()
print("=" * 78)
print("D. final-sid ordering identical?")
print("=" * 78)
same = 0
for qid in b_by:
    bf = (b_by[qid].get("debug") or {}).get("step10_final_sids")
    sf = (s_by[qid].get("debug") or {}).get("step10_final_sids")
    if bf == sf:
        same += 1
print(f"  identical final sids: {same}/{len(b_by)}")

print()
print("=" * 78)
print("E. attr_match in the SOFT run (all questions)")
print("=" * 78)
all_am = []
for qid, r in s_by.items():
    ef = (r.get("debug") or {}).get("ef_scores") or []
    for d in ef:
        all_am.append(d.get("attr_match", 0.0) or 0.0)
if all_am:
    nz = [a for a in all_am if a > 1e-9]
    print(f"  total scored: {len(all_am)}   non-zero: {len(nz)}")
    if nz:
        print(f"  min={min(nz):.4f} max={max(nz):.4f} mean={sum(nz)/len(nz):.4f}")
    else:
        print("  [!] ALL attr_match == 0 -> soft match never fired")
else:
    print("  [!] no ef_scores at all")

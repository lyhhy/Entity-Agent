"""Test: from diag_sparse.json, use step6_pool_sids (before filter) as the pool.
Check if miss_gold sentences are in the pool BEFORE sparse filter."""
import json, os

_PKG = 'c:/Users/zsf/Desktop/lyh/StenRAG'
results_file = os.path.join(_PKG, "eval", "results", "smoke20_diag_sparse.json")
d = json.load(open(results_file, encoding="utf-8"))
records = d.get("records", [])

cut_qids = {'5ae4a2d0', '5a7f669a', '5ae641f6', '5adc0d33', '5a8208ee', '5adea62e'}

print("Analysis: miss_gold sentences in pool (before sparse filter)")
print("=" * 80)

total_gold = 0
total_in_l1 = 0
total_in_l2 = 0
total_in_pool_before = 0
total_in_pool_after = 0
total_lost_at_filter = 0

for rec in records:
    qid = rec["qid"]
    qid_short = qid[:8]
    gold = set(rec.get("diag", {}).get("gold_sentence_ids", []))
    l1 = set(rec.get("diag", {}).get("step4_l1_sids", []))
    l2 = set(rec.get("diag", {}).get("step4_l2_sids", []))
    pool_before = set(rec.get("debug", {}).get("step6_pool_sids", []))
    # The hard attribute filter was removed; on new result files there is
    # no "after filter" pool.  Fall back to the unfiltered pool.
    pool_after = set(rec.get("debug", {}).get("step6_pool_sids_after_filter", pool_before))
    
    if not gold:
        continue
    
    in_l1 = gold & l1
    in_l2 = gold & l2
    in_before = gold & pool_before
    in_after = gold & pool_after
    lost = gold - pool_before  # never made it to pool
    cut = in_before - in_after  # gold in pool but cut by filter
    
    total_gold += len(gold)
    total_in_l1 += len(in_l1)
    total_in_l2 += len(in_l2)
    total_in_pool_before += len(in_before)
    total_in_pool_after += len(in_after)
    total_lost_at_filter += len(cut)
    
    is_cut = qid_short in cut_qids
    mark = ">>> CUT" if is_cut else "    "
    
    if lost or cut or is_cut:
        print(f"{mark} {qid[:12]}")
        print(f"       gold: {gold}")
        print(f"       in_L1: {in_l1}")
        print(f"       in_L2: {in_l2}")
        print(f"       in_pool_before: {in_before}")
        print(f"       in_pool_after: {in_after}")
        print(f"       LOST (never in pool): {lost}")
        print(f"       CUT (by sparse filter): {cut}")
        print()

print("=" * 80)
print(f"TOTAL gold sentences: {total_gold}")
print(f"  In L1:              {total_in_l1}/{total_gold}")
print(f"  In L2:              {total_in_l2}/{total_gold}")
print(f"  In pool (before):   {total_in_pool_before}/{total_gold}")
print(f"  In pool (after):    {total_in_pool_after}/{total_gold}")
print(f"  LOST (never pool):   {total_gold - total_in_pool_before}")
print(f"  CUT (by filter):    {total_lost_at_filter}")

print("\n\n=== HYPEREDGE L2 CONTAINERS TEST ===")
print("We need to check: are miss_gold sentences in containers that would be")
print("recalled by the hyperedge L2 mechanism?")
print()
print("The L2 mechanism: from L1 sentences -> their containers -> expand to L2 containers")
print("The miss_gold sentences are NOT in L1 or L2. This means:")
print("  - They are in containers that are NOT connected via shared entities to L1 containers")
print("  - Even if we expand L2 to include ALL sentences from L2 containers,")
print("    the miss_gold sentences would still be MISS because they're in DIFFERENT containers")
print()

# Check: for miss_gold sentences, are they in containers that share ANY entity with pool sentences?
# We can't check containers directly, but we can infer from the data
# If a gold sentence is not in pool_before, it's either:
# 1. In a container not reached by L1->L2 expansion
# 2. The container exists but sparse filter removed it (but it's not in pool_before, so it's #1)

# For the 6 cut questions: gold was IN pool_before but got filtered out
# For the other miss: gold was NEVER in pool_before
print("\n=== BREAKDOWN ===")
print("Questions where gold was NEVER in pool (seed entity coverage gap):")
for rec in records:
    qid = rec["qid"]
    qid_short = qid[:8]
    gold = set(rec.get("diag", {}).get("gold_sentence_ids", []))
    l1 = set(rec.get("diag", {}).get("step4_l1_sids", []))
    l2 = set(rec.get("diag", {}).get("step4_l2_sids", []))
    pool_before = set(rec.get("debug", {}).get("step6_pool_sids", []))
    lost = gold - l1 - l2
    if lost:
        in_pool = gold & pool_before
        print(f"  {qid[:12]} gold_not_in_pool={lost} gold_in_pool={in_pool}")

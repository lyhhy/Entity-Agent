import json
d = json.load(open('eval/results/smoke20_diag_sparse.json', encoding='utf-8'))

# 6题被sparse砍掉的gold
cut_qids = ['5ae4a2d0', '5a7f669a', '5ae641f6', '5adc0d33', '5a8208ee', '5adea62e']

for r in d['records']:
    qid_short = r['qid'][:8]
    if qid_short not in cut_qids:
        continue
    
    gold = set(r['diag'].get('gold_sentence_ids', []))
    pool_before = set(r['debug'].get('step6_pool_sids', []))
    # The hard attribute filter was removed; on new result files there is
    # no "after filter" pool.  Fall back to the unfiltered pool.
    pool_after = set(r['debug'].get('step6_pool_sids_after_filter', pool_before))
    
    removed = pool_before - pool_after
    gold_removed = gold & removed
    
    gold_mounts = r['diag'].get('gold_mounts', 'N/A')
    gold_required = r['diag'].get('gold_required', 'N/A')
    
    print(f"=== {r['qid'][:12]} r10={r['r10']} ===")
    print(f"  gold:       {gold}")
    print(f"  pool_before: {pool_before}")
    print(f"  gold_in_before: {gold & pool_before}")
    print(f"  pool_after:  {pool_after}")
    print(f"  removed:     {removed}")
    print(f"  gold_removed: {gold_removed}")
    print(f"  gold_mounts: {gold_mounts}")
    print(f"  gold_required: {gold_required}")
    print()

# 另外看看step3 hyperedges
print("\n=== step3 hyperedges for cut questions ===")
for r in d['records']:
    qid_short = r['qid'][:8]
    if qid_short not in cut_qids:
        continue
    hyperedges = r['diag'].get('layer_step3_hyperedges', [])
    print(f"{r['qid'][:12]}: {hyperedges[:5]}")

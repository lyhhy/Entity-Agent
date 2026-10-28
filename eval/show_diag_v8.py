import json
d = json.load(open('eval/diag/v8_diag_seed137.json', encoding='utf-8'))
print("=== MISSING cases (gold not in pool) ===")
for c in d['cases']:
    if c['case'] == 'MISSING':
        print(f"Q{c['i']:>2}  n_A={c['n_A']:>2}  n_L2={c['n_L2']:>2}  n_pool={c['n_pool']:>2}  q={c['question']}")
print()
print("=== IN_POOL cases (gold in pool, sorting problem) ===")
for c in d['cases']:
    if c['case'].startswith('IN_POOL'):
        print(f"Q{c['i']:>2}  rank={c['gold_rank']}  case={c['case']}  q={c['question']}")

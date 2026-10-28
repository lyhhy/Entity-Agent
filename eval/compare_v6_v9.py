import json
d = json.load(open('eval/results/clean_results_seed7_v6baseline.json', encoding='utf-8'))
print("=== v6 baseline seed7 ===")
em_count = 0
for r in d['records']:
    em_count += int(r['em'])
    print(f"Q{r['qid'][:10]} EM={int(r['em'])} F1={r['f1']:.2f} R@1={r['r1']} R@5={r['r5']} A={r['answer'][:50]!r}")
print(f"\nTotal EM: {em_count}/20")

print("\n\n=== v9 seed7 ===")
d2 = json.load(open('eval/results/clean_results_seed7_v9.json', encoding='utf-8'))
em_count2 = 0
for r in d2['records']:
    em_count2 += int(r['em'])
    print(f"Q{r['qid'][:10]} EM={int(r['em'])} F1={r['f1']:.2f} R@1={r['r1']} R@5={r['r5']} A={r['answer'][:50]!r}")
print(f"\nTotal EM: {em_count2}/20")

# Compare per-Q
print("\n\n=== Per-Q diff (v9 - v6) ===")
v6 = {r['qid']: r for r in d['records']}
v9 = {r['qid']: r for r in d2['records']}
for qid in v6:
    a = v6[qid]
    b = v9[qid]
    em_diff = int(b['em']) - int(a['em'])
    f1_diff = b['f1'] - a['f1']
    r1_diff = b['r1'] - a['r1']
    r5_diff = b['r5'] - a['r5']
    if em_diff != 0 or abs(f1_diff) > 0.05 or r1_diff != 0 or r5_diff != 0:
        print(f"Q{qid[:10]}  EM {int(a['em'])}→{int(b['em'])} ({em_diff:+d})  F1 {a['f1']:.2f}→{b['f1']:.2f} ({f1_diff:+.2f})  R@1 {a['r1']}→{b['r1']}  R@5 {a['r5']}→{b['r5']}")

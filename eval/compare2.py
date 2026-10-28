import json, sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open('eval/results/clean_results_seed7_v6baseline.json', encoding='utf-8'))
print("=== v6 baseline ===")
for r in d['records']:
    try:
        q = r['question'][:60]
        g = r['gold'][:30]
        a = r['answer'][:40]
        print(f"{r['qid'][:10]} EM={int(r['em'])} F1={r['f1']:.2f} R@1={r['r1']} R@5={r['r5']}")
        print(f"  G: {g}")
        print(f"  A: {a}")
    except Exception:
        pass

print()
d2 = json.load(open('eval/results/clean_results_seed7_v9.json', encoding='utf-8'))
print("=== v9 ===")
for r in d2['records']:
    try:
        q = r['question'][:60]
        g = r['gold'][:30]
        a = r['answer'][:40]
        print(f"{r['qid'][:10]} EM={int(r['em'])} F1={r['f1']:.2f} R@1={r['r1']} R@5={r['r5']}")
        print(f"  G: {g}")
        print(f"  A: {a}")
    except Exception:
        pass

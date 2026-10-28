import json, sys
sys.stdout.reconfigure(encoding='utf-8')
for tag, path in [
    ('seed=7',   r'eval\results\quick20_seed7_results.json'),
    ('seed=137', r'eval\results\quick20_results.json'),
]:
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    print('=== {}  (n={}) ==='.format(tag, d['n']))
    print('  EM={:.3f}  F1={:.3f}  R@1={:.3f}  R@5={:.3f}  R@10={:.3f}'.format(
        d['em'], d['f1'], d['r1'], d['r5'], d['r10']))
    print('  Paths: {}'.format(d['paths']))
    em_hits = sum(1 for r in d['records'] if r['em'])
    print('  EM hits: {}/{}'.format(em_hits, d['n']))
    print('  Per-question:')
    for i, r in enumerate(d['records'], 1):
        mark = 'OK' if r['em'] else '--'
        print('    Q{:>2} {} EM={} F1={:.3f} R@1={} R@5={} R@10={} | pred={!r}'.format(
            i, r['qid'][:8], mark, r['f1'], r['r1'], r['r5'], r['r10'], r['answer'][:40]))
    print()

"""
诊断每个错题:
- r1/r5/r10=0 表示 gold 句子 **没在召回前K** (召回失败)
- r1/r5/r10>0 但 EM=miss 表示 gold 句子被召回了, **重排/LLM 抽取失败**
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

for tag, path in [
    ('seed=7',   r'eval\results\quick20_seed7_results.json'),
    ('seed=137', r'eval\results\quick20_results.json'),
]:
    d = json.load(open(path, encoding='utf-8'))
    print('=' * 70)
    print('=== {}  EM={:.3f}  F1={:.3f} ==='.format(tag, d['em'], d['f1']))
    print('=' * 70)

    recall_miss = 0     # r10 == 0, gold sentence NOT retrieved
    rerank_miss = 0     # r10 > 0 but EM=0, gold retrieved but not picked
    exact_miss = 0      # EM=miss with F1>0 (close but not exact)

    for r in d['records']:
        if r['em']:
            continue
        r10 = r['r10']
        f1 = r['f1']
        if r10 == 0:
            recall_miss += 1
            label = '[RECALL-MISS]'
        else:
            if f1 == 0:
                rerank_miss += 1
                label = '[RERANK-MISS]'
            else:
                exact_miss += 1
                label = '[CLOSE-MISS]'
        # Check if answer is a number
        gold = r['gold']
        pred = r['answer']
        is_num = any(c.isdigit() for c in gold)
        print('  Q{:>2} {} {} r1={} r5={} r10={} F1={:.2f} num={} | gold={!r} pred={!r}'.format(
            d['records'].index(r)+1, r['qid'][:8], label, r['r1'], r['r5'], r['r10'], f1, is_num, gold[:35], pred[:35]))

    print()
    print('  Summary:')
    print('    Recall-miss (r10=0): {}'.format(recall_miss))
    print('    Rerank-miss (r10>0, F1=0): {}'.format(rerank_miss))
    print('    Close-miss (r10>0, F1>0): {}'.format(exact_miss))
    print()

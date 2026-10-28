"""
Aggregated diagnostic: print cascade state for every recall-miss question.
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open(r'eval\results\quick20_seed7_diag3.json', encoding='utf-8'))
print('Overall  EM={:.3f}  F1={:.3f}  R@10={:.3f}'.format(d['em'], d['f1'], d['r10']))
print()

verdicts = {}
for i, r in enumerate(d['records'], 1):
    diag = r.get('diag', {})
    cascade = diag.get('cascade') or {}
    if r['em']:
        v = 'OK'
    elif r['r10'] == 0 and not cascade:
        v = 'DIAG_FAIL'
    elif not (cascade.get('gold_in_step4_l1') or cascade.get('gold_in_step4_l2')):
        v = 'RETRIEVAL-MISS'
    elif cascade.get('gold_in_step4_l2') and not cascade.get('gold_in_step10_top10'):
        v = 'RERANK-MISS'
    elif cascade.get('gold_in_step10_top10'):
        v = 'EXTRACT-MISS'
    else:
        v = 'OTHER'
    verdicts.setdefault(v, []).append((i, r))

print('=== Cascade verdict breakdown ===')
for v, items in sorted(verdicts.items()):
    print('  {:<14}  x{}'.format(v, len(items)))
print()

print('=== All miss-questions ===')
for v, items in sorted(verdicts.items()):
    if v == 'OK':
        continue
    print()
    print('--- {} ---'.format(v))
    for i, r in items:
        d_ = r['diag']
        cas = d_.get('cascade', {})
        title_hit = cas.get('gold_title_in_any_step4_l2_container')
        print('  Q{:>2} {} EM={} F1={:.2f} ent={} | l1={} l2={} top10={} title_in_container={} | entities={!r} titles={!r}'.format(
            i, r['qid'][:8], 'OK' if r['em'] else '--', r['f1'],
            d_.get('n_entities', 0),
            cas.get('gold_in_step4_l1'),
            cas.get('gold_in_step4_l2'),
            cas.get('gold_in_step10_top10'),
            title_hit,
            r['debug'].get('entities', []),
            d_.get('gold_titles', []),
        ))
        # Show step4_l2_cids for retrieval-miss
        if v == 'RETRIEVAL-MISS':
            l2c = d_.get('step4_l2_cids', [])
            print('       step4_l2_cids: {} (top 10 of {})'.format(l2c[:10], len(l2c)))
            print('       step10_final_sids: {}'.format(d_.get('step10_final_sids')))
            print('       gold_sids: {}'.format(d_.get('gold_sentence_ids')))

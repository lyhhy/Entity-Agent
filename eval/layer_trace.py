"""
Layer-by-layer trace: for each r10=1 / cascade-l2=False question,
show WHERE the gold sentence appears. Also dump each layer's sid
list and the gold_sid.
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open(r'eval\results\quick20_seed7_diag4.json', encoding='utf-8'))

# Q1, Q3, Q8, Q10, Q12, Q14, Q20 from the user's question
TARGETS = {
    '5a79c6a5': 'Q1', '5adf043b': 'Q3', '5abaad58': 'Q8',
    '5a714058': 'Q10', '5abce301': 'Q12', '5ade9923': 'Q14',
    '5a75642c': 'Q20',
}

for rec in d['records']:
    qid = rec['qid']
    label = TARGETS.get(qid[:8])
    if label is None:
        continue
    diag = rec['diag']
    cas = diag.get('cascade', {})
    if cas.get('gold_in_step4_l2'):
        print('SKIP {} - gold WAS in step4_l2'.format(label))
        continue
    print('=' * 70)
    print('{}  qid={}  r10={}  EM={}  F1={:.2f}'.format(
        label, qid[:8], rec['r10'], 'OK' if rec['em'] else '--', rec['f1']))
    print('  Q:', rec['question'][:80])
    print('  G:', rec['gold'])
    print('  A:', rec['answer'][:50])
    print('  entities:', rec['debug'].get('entities', []))
    print('  gold_titles:', diag.get('gold_titles', []))
    print('  gold_sids:', diag.get('gold_sentence_ids', []))
    print()
    print('  --- WHERE DOES GOLD FIRST APPEAR ---')
    print('  gold_first_appear_layer:', diag.get('gold_first_appear_layer'))
    print()
    print('  --- LAYER DUMPS ---')
    print('  step2_a_cids ({}):'.format(len(diag.get('layer_step2_a_cids',[]))),
          diag.get('layer_step2_a_cids',[])[:6])
    print('  step4_l1_sids ({}):'.format(len(diag.get('step4_l1_sids',[]))),
          diag.get('step4_l1_sids',[])[:8])
    print('  step4_b_cids ({}):'.format(len(diag.get('layer_step4_b_cids',[]))),
          diag.get('layer_step4_b_cids',[])[:8])
    print('  step2_5_bridge_titles:'.format(diag.get('layer_step2_5_bridge_titles',[])))
    print('  step2_5_bridge_cids ({}):'.format(len(diag.get('layer_step2_5_bridge_cids',[]))),
          diag.get('layer_step2_5_bridge_cids',[])[:8])
    print('  step3_hyperedges:'.format(diag.get('layer_step3_hyperedges',[])))
    print('  step4_l2_cids ({}):'.format(len(diag.get('step4_l2_cids',[]))),
          diag.get('step4_l2_cids',[])[:8])
    print('  step4_l2_sids ({}):'.format(len(diag.get('step4_l2_sids',[]))),
          diag.get('step4_l2_sids',[])[:8])
    print('  step5_traced ({}):'.format(len(diag.get('layer_step5_traced',[]))),
          diag.get('layer_step5_traced',[])[:8])
    print('  step6_pool_size:', diag.get('layer_step6_pool_size'))
    print('  step7_topn ({}):'.format(len(diag.get('layer_step7_topn',[]))),
          diag.get('layer_step7_topn',[])[:8])
    print('  step8_supplement ({}):'.format(len(diag.get('layer_step8_supplement',[]))),
          diag.get('layer_step8_supplement',[])[:8])
    print('  step9_merged ({}):'.format(len(diag.get('layer_step9_merged',[]))),
          diag.get('layer_step9_merged',[])[:8])
    print('  step10_final_sids ({}):'.format(len(diag.get('step10_final_sids',[]))),
          diag.get('step10_final_sids',[])[:8])
    print()

# Aggregate: where does gold first appear across all r10=1 questions?
print('=' * 70)
print('=== AGG: gold first-appear layer across r10=1 + cascade-l2=False ===')
first = {}
for rec in d['records']:
    diag = rec.get('diag', {})
    cas = diag.get('cascade', {})
    if rec['r10'] == 1 and not cas.get('gold_in_step4_l2'):
        layer = diag.get('gold_first_appear_layer') or 'NONE'
        first.setdefault(layer, []).append(rec['qid'][:8])
for layer, qids in sorted(first.items(), key=lambda x: -len(x[1])):
    print('  {:<20} x{}  {}'.format(layer, len(qids), qids))

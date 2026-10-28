"""
Deep dump for r10=1 but cascade l2=False questions:
  Q1, Q3, Q8, Q10, Q12, Q14, Q20

For each: pick the gold sentence ID, then walk EVERY layer of the
pipeline to see WHERE the gold sentence appears.

Layers to inspect:
  - step4_l2_cids_filtered       → Step4 L2 容器
  - step4_l2_sids_filtered       → Step4 L2 句子
  - step5_traced_sids            → Step5 trace-back
  - step6_pool_size              → Step6 pool 大小
  - step7_top_n_sids             → Step7 top-n
  - step8_supplement_sids        → Step8 supplement
  - step9_merged_sids            → Step9 merged
  - step10_final_sids            → Step10 final

We need a debug.expansion hook but since none exists we re-read
pipeline result through getattr (with fallbacks) plus the builder.
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open(r'eval\results\quick20_seed7_diag3.json', encoding='utf-8'))

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
    if diag.get('cascade', {}).get('gold_in_step4_l2'):
        continue  # skip if it's not what we said
    print('=' * 70)
    print('{}  qid={}  r10={}  entities={!r}'.format(
        label, qid[:8], rec['r10'], rec['debug'].get('entities', [])))
    print('  Q :', rec['question'][:80])
    print('  G :', rec['gold'])
    print('  A :', rec['answer'][:50])
    # Output the cascade
    cas = diag.get('cascade', {})
    print('  cascade:', cas)
    print('  gold_sids:', diag.get('gold_sentence_ids', []))
    print('  step10_final_sids:', diag.get('step10_final_sids', []))
    # Show which final sids are gold (cross-check)
    gold_set = set(diag.get('gold_sentence_ids', []))
    print('  gold ∩ step10 =', sorted(gold_set & set(diag.get('step10_final_sids', []))))
    print()

# Now we need to inspect the pool layer by layer for one specific qid.
# Rerun with full pipeline inspection (we can't from JSON alone).
print()
print('=== Run a runtime probe ===')

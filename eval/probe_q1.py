"""
Quick reload: lookup the L2 cids / sentences for a known Q.
We re-run a single question to print L0 vs L1 vs L2 layers.
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'c:\Users\zsf\Desktop\lyh\StenRAG')

import logging
logging.basicConfig(level=logging.WARNING)

from eval.quick20_eval import build_graph_from_hotpot
from src.graph_builder import _load_spacy_shared

nlp = _load_spacy_shared('en_core_web_lg')
data = json.load(open(r'eval\data\hotpot_dev_distractor_v1.json', encoding='utf-8'))

# Find Q1 (5a79c6a5)
target_qid = '5a79c6a5'
qdata = next(q for q in data if q['_id'].startswith(target_qid))
builder, sentences, titles, para_sents, bridge_titles = build_graph_from_hotpot(qdata, nlp)

# Inspect builder.sentence_registry
cm = builder.container_manager
print('Total containers:', len(cm.get_all_containers()))
print('Total sentences:', len(builder.sentence_registry._nodes))

# For Q1, the 4 L2 cids are: c_0012, c_0008, c_0010, c_0011
for cid in ['c_0012', 'c_0008', 'c_0010', 'c_0011']:
    c = cm.get_container(cid)
    if c is None:
        print(cid, '= NOT FOUND')
        continue
    print(cid, '->', getattr(c, 'name', '?'),
          'sids:', getattr(c, 'sentence_ids', None))

print()
# Now check s_0019 and s_0007 (gold sids for Q1)
for sid in ['s_0019', 's_0007']:
    n = builder.sentence_registry.get_node(sid)
    if n:
        print(sid, 'text:', (n.text or '')[:100])
        print('    containers:', getattr(n, 'containers', None))
    else:
        print(sid, '= NOT FOUND')

# Step2 A cids for Q1
print()
# Verify: count how many c_xxxx containers contain gold s_0019
print('Sentences containing "george cayley" or "Sir George Cayley":')
for sid, n in list(builder.sentence_registry._nodes.items())[:80]:
    txt = (n.text or '').lower()
    if 'cayley' in txt:
        print(' ', sid, ':', n.text[:100])

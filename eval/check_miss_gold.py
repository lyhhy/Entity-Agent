import json, pickle, sys
_PKG = 'c:/Users/zsf/Desktop/lyh/StenRAG'
sys.path.insert(0, _PKG)

d = json.load(open(f'{_PKG}/eval/results/smoke20_diag_sparse.json', encoding='utf-8'))
with open(f'{_PKG}/data/graphs/hotpot_dev_graph.pkl', 'rb') as f:
    saved = pickle.load(f)
cm = saved['container_manager']
registry = saved['sentence_registry']

miss_entries = [
    ('5a886589', 's_0027'),
    ('5a7d2aa0', 's_0029'),
    ('5ae0a481', 's_0005'),
    ('5adc0d33', 's_0022'),
    ('5adc0d33', 's_0017'),
    ('5ae4a2d0', 's_0038'),
    ('5a7f669a', 's_0021'),
    ('5adea62e', 's_0026'),
    ('5ae142c4', 's_0024'),
    ('5ae142c4', 's_0026'),
    ('5ac49a1c', 's_0018'),
    ('5ab41f6b', 's_0047'),
    ('5ab41f6b', 's_0036'),
    ('5ac23f52', 's_0007'),
    ('5a8208ee', 's_0023'),
]

qid_l1 = {}
for r in d['records']:
    qid_short = r['qid'][:8]
    l1 = r['diag'].get('step4_l1_sids', [])
    qid_l1[qid_short] = l1

for qid_short, miss_sid in miss_entries:
    node = registry.get_node(miss_sid)
    if not node:
        print(f'{miss_sid} NOT IN REGISTRY')
        continue
    
    sent_cid = getattr(node, 'container_id', None)
    c = cm.get_container(sent_cid)
    c_entities = list(c.entities) if c and hasattr(c, 'entities') else []
    gold_entities = set(getattr(node, 'entities', []) or [])
    
    l1_sids = qid_l1.get(qid_short, [])
    l1_entities = set()
    for sid in l1_sids:
        n = registry.get_node(sid)
        if n:
            l1_entities.update(getattr(n, 'entities', []) or [])
    
    intersection = gold_entities & l1_entities
    has_int = '有交集' if intersection else '!!!无交集!!!'
    
    txt = getattr(node, 'text', '?') or '?'
    print(f'QID={qid_short} miss={miss_sid}')
    print(f'  gold文本: {txt[:70]}')
    print(f'  gold实体: {gold_entities}')
    print(f'  容器实体: {c_entities}')
    print(f'  L1实体  : {l1_entities}')
    print(f'  交集    : {intersection}')
    print(f'  结论    : {has_int}')
    print()

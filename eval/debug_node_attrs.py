"""Debug: check sentence node attributes in rebuilt graph."""
import json, random, sys, os

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.dirname(_ROOT)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import spacy
from src.graph_builder import GraphBuilder, ComplementedSentence

print("Loading spaCy...")
nlp = spacy.load("en_core_web_lg")

with open(os.path.join(_PKG, "eval", "data", "hotpot_dev_distractor_v1.json"), encoding="utf-8") as f:
    all_data = json.load(f)
rng = random.Random(137)
idx = rng.sample(range(len(all_data)), 20)
idx.sort()
sample = [all_data[i] for i in idx]
hotpot = {r["_id"]: r for r in sample}

# Build first question's graph
d = json.load(open(os.path.join(_PKG, "eval", "results", "smoke20_diag_sparse.json"), encoding="utf-8"))
rec = d["records"][0]
qid = rec["qid"]
qdata = hotpot.get(qid)

context = qdata.get("context", [])
sentences = []
for title, para in context:
    for sent in para:
        sentences.append(ComplementedSentence(
            original_text=sent,
            complemented_text=sent,
            resolved_subject=title,
            entities=[]))

builder = GraphBuilder(shared_nlp=nlp)
for s in sentences:
    builder.add_sentence(s)

# Inspect first few nodes
print("All nodes and their attributes:")
print("=" * 80)
for i, (sid, node) in enumerate(builder.sentence_registry._nodes.items()):
    if i >= 5:
        break
    attrs = [a for a in dir(node) if not a.startswith('_')]
    containers_attr = getattr(node, 'containers', '???')
    mount_attr = getattr(node, 'attribute_mounts', '???')
    print(f"\n{sid}:")
    print(f"  class: {node.__class__.__name__}")
    print(f"  containers: {containers_attr}")
    print(f"  attribute_mounts: {mount_attr}")
    print(f"  text: {(getattr(node, 'text', '') or '')[:60]}")
    print(f"  entities: {getattr(node, 'entities', [])}")

# Check container_manager
print("\n\nContainer Manager:")
print("=" * 80)
cm = builder.container_manager
for i, (cid, c) in enumerate(cm._containers.items()):
    if i >= 5:
        break
    print(f"\n{cid}:")
    print(f"  class: {c.__class__.__name__}")
    print(f"  entities: {getattr(c, 'entities', [])}")
    print(f"  sentence_ids: {getattr(c, 'sentence_ids', [])}")
    print(f"  n_sentences: {getattr(c, 'n_sentences', '?')}")

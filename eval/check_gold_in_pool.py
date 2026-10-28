"""Check which gold sentences made it into the raw pool (before sparse filter)."""
import json, sys, os, random

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_ROOT)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from src.graph_builder import GraphBuilder, ComplementedSentence
from src.stengraph_pipeline import StenGraphPipeline
from src.dispatcher import Dispatcher
import spacy

nlp = spacy.load("en_core_web_lg")

# Load hotpot
with open(os.path.join(_PKG, "eval", "data", "hotpot_dev_distractor_v1.json"), encoding="utf-8") as f:
    all_data = json.load(f)

rng = random.Random(137)
idx = sorted(rng.sample(range(len(all_data)), 20))
sample = [all_data[i] for i in idx]
hotpot = {r["_id"]: r for r in sample}

# Load diag sparse result to get gold_sids
d = json.load(open(os.path.join(_PKG, "eval", "results", "smoke20_diag_sparse.json"), encoding="utf-8"))

# Load pipeline data (built graph)
from src.container_manager import ContainerManager
from src.hyper_edge_registry import HyperEdgeRegistry
from src.sentence_registry import SentenceRegistry

import pickle, os
graph_path = os.path.join(_PKG, "data", "graphs", "hotpot_dev_graph.pkl")
with open(graph_path, "rb") as f:
    saved = pickle.load(f)

cm = saved["container_manager"]
he_registry = saved["hyper_edge_registry"]
registry = saved["sentence_registry"]

pipeline = StenGraphPipeline(
    container_manager=cm,
    hyper_edge_registry=he_registry,
    sentence_registry=registry,
    embedder=None,
    top_a=6, top_b=8, top_k_final=8,
    w_bm25=0.5, w_dense=0.5,
    faiss_top_k=20, faiss_threshold=0.0,
    shared_nlp=nlp,
    strict_entity_match=True,
)
# No hard attribute filter any more (mapper/filter removed).
pipeline.attr_soft_match_mode = False

dispatcher = Dispatcher(
    pipeline=pipeline,
    analyzer=None,
    entity_extractor=None,
    top_k=5,
)

print(f"{'QID':<20} {'gold_in_A':>10} {'gold_in_B':>10} {'gold_in_pool':>11}  verdict")
print("-" * 80)

for rec in d["records"]:
    qid = rec["qid"]
    question = rec["question"]
    gold_sids = set(rec["diag"].get("gold_sentence_ids", []))
    gold_norm = rec["diag"].get("gold_norm", "")

    if not gold_sids:
        print(f"{qid[:12]:<20}  NO_GOLD_SIDS")
        continue

    # Run pipeline without sparse filter
    result = pipeline.retrieve(question, top_k=5)
    
    # Step 2 A containers
    a_cids = set(result.step2_a_containers or [])
    # Step 4 B containers
    b_cids = set()
    if hasattr(result, 'step4_b_containers') and result.step4_b_containers:
        b_cids = set(result.step4_b_containers)
    if hasattr(result, 'step4_l2_cids') and result.step4_l2_cids:
        b_cids |= set(result.step4_l2_cids)
    
    # Pool: all sentences in A + all sentences in B
    pool_a_sids = set()
    for cid in a_cids:
        c = cm.get_container(cid)
        if c:
            pool_a_sids.update(getattr(c, 'sentence_ids', []) or [])
    
    pool_b_sids = set()
    for cid in b_cids:
        if cid in a_cids:
            continue
        c = cm.get_container(cid)
        if c:
            pool_b_sids.update(getattr(c, 'sentence_ids', []) or [])
    
    pool_sids = pool_a_sids | pool_b_sids
    
    gin_a = len(gold_sids & pool_a_sids)
    gin_b = len(gold_sids & pool_b_sids)
    gin_pool = len(gold_sids & pool_sids)
    
    if gin_pool == 0:
        mark = "!!! MISS !!!"
    elif gin_a > 0:
        mark = "in A"
    else:
        mark = "in B only"
    
    print(f"{qid[:12]:<20} {gin_a:>10} {gin_b:>10} {gin_pool:>11}  {mark}")

print()
print("Summary: gold sentences NOT in raw pool = MISS count above")

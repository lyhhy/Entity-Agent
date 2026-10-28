"""
Diagnostic for question 5adf043b:
  Q: Which American cable news and talk radio host was the former GOP representative
  G: Charles Joseph  (gold — actually wrong, correct is Joe Scarborough)
  A: Joe Scarborough  (model answered correctly — F1=0 due to gold mismatch)

Shows: entities, hyperedges, containers, attribute triggers, full retrieval path.
"""
import json, sys, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import spacy
import numpy as np
from eval.quick20_eval import build_graph_from_hotpot
from src.hyper_edge import HyperEdgeRegistry
from src.stengraph_pipeline import StenGraphPipeline
from src.query_analyzer import make_query_analyzer

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("diag")

DATA_PATH = ROOT / "eval/data/hotpot_dev_distractor_v1.json"

# ── Load question ────────────────────────────────────────────────
data = json.load(open(DATA_PATH, encoding="utf-8"))
target_qid = "5adf043b5542993a75d263d4"
qdata = next(q for q in data if q["_id"] == target_qid)
question = qdata["question"]
log.info("QID: %s", target_qid)
log.info("Q : %s", question)
log.info("G : %s", qdata.get("answer"))
log.info("")

# ── Build graph (EXACTLY like quick20_clean.py) ──────────────────
nlp = spacy.load("en_core_web_lg")
builder, sentences, titles, para_sents, bridge_titles = build_graph_from_hotpot(qdata, nlp)
cm = builder.container_manager

# Build container index like quick20_clean.py does
for c in cm._containers.values():
    cm.update_container_index(c.id)
builder.compute_tfidf_index()

# ── Build hyperedges (EXACTLY like quick20_clean.py) ──────────────
log.info("=== Building hyperedges ===")
he = HyperEdgeRegistry(container_manager=cm)
for _p_title, _p_cids in (bridge_titles or {}).items():
    _p_containers = [
        cm._containers.get(_cid)
        for _cid in _p_cids
        if cm._containers.get(_cid) is not None
    ]
    if _p_containers:
        he.build_hyperedges_for_chunk(
            chunk_id=str(_p_title),
            containers=_p_containers,
            time_word="",
            paragraph_title=str(_p_title),
        )

log.info("  total hyperedges: %d", len(he))
log.info("")

# ── Build pipeline ───────────────────────────────────────────────
analyzer = make_query_analyzer(client=None)
pipeline = StenGraphPipeline(
    graph=builder,
    container_manager=cm,
    hyperedge_registry=he,
    sentence_registry=builder.sentence_registry,
    analyzer=analyzer,
    nlp=nlp,
)

# ── Step 1: entity extraction ────────────────────────────────────
log.info("=== Step 1: Query Analysis ===")
analysis = analyzer(question)
log.info("  entity_count : %d", analysis.entity_count)
log.info("  entity_names : %s", list(analysis.entity_names))
log.info("  is_temporal : %s", analysis.is_temporal)
log.info("  attribute_words: %s", analysis.attribute_words)
log.info("  via_llm     : %s", analysis.via_llm)
log.info("")

entities = pipeline._step1_query_analysis(question)
log.info("  _step1_query_analysis result: %s", entities)
log.info("")

# ── Step 1.5: query relations ────────────────────────────────────
log.info("=== Step 1.5: Query Relations ===")
q_relations = pipeline._step1_5_query_relations(question)
log.info("  %s", q_relations)
log.info("")

# ── Step 2: containers A ─────────────────────────────────────────
log.info("=== Step 2: Containers A ===")
a_containers = pipeline._step2_recall_a_containers(question, entities)
log.info("  count: %d", len(a_containers))
for cid in a_containers[:5]:
    c = cm.get_container(cid)
    if c:
        log.info("  [%s] entity=%r  attrs=%s", cid, c.entity, list(c.attributes.keys())[:5])
        hes = he.get_hyperedges_for_container(cid)
        log.info("       hyperedges=%s", [(h.id, h.event_name[:40]) for h in hes])
        for h in hes:
            trigs = h.triggers_for(cid)
            log.info("         %s triggers=%s", h.id, sorted(trigs) if trigs else "(all attrs)")
log.info("")

# ── Step 3: hyperedges for A ─────────────────────────────────────
log.info("=== Step 3: Hyperedges touching A ===")
step3_hids = pipeline._step3_hyperedges_for_containers(a_containers[:5])
log.info("  hyperedge IDs: %s", step3_hids)
for hid in step3_hids:
    he_obj = he.get_hyperedge(hid)
    if he_obj:
        log.info("  %s  event=%r", he_obj.id, he_obj.event_name)
        log.info("    containers: %s", he_obj.container_ids)
        log.info("    triggers: %s", {k: sorted(v) for k, v in he_obj.attribute_triggers.items()})
        log.info("    source_chunk: %s", he_obj.source_chunk_id)
        # Show container details for containers in this hyperedge
        for cid in he_obj.container_ids[:3]:
            c = cm.get_container(cid)
            if c:
                log.info("    container [%s] entity=%r attrs=%s", cid, c.entity, list(c.attributes.keys())[:5])
log.info("")

# ── Full retrieval ───────────────────────────────────────────────
log.info("=== Full Retrieval ===")
result = pipeline.retrieve(question, top_k=8)
log.info("  final_text    : %s", result.final_text)
log.info("  step3_hyperedges: %s", result.step3_hyperedges)
log.info("  step4_b_containers: %d", len(result.step4_b_containers))
log.info("  step7_top_n_sids: %s", result.step7_top_n_sids)
log.info("  step9_merged_sids: %s", result.step9_merged_sids)
log.info("")
log.info("  === Step7 sentences ===")
for sid in result.step7_top_n_sids[:8]:
    node = builder.sentence_registry.get_node(sid)
    if node:
        log.info("  [%s] %s", sid, node.text[:150])
log.info("")
log.info("  === Gold supporting facts (ground truth) ===")
ctx_dict = {t: list(sents) for t, sents in qdata.get("context", [])}
for t, idx in qdata.get("supporting_facts", []):
    sents = ctx_dict.get(t, [])
    if 0 <= idx < len(sents):
        log.info("  [%s][%d] %s", t, idx, sents[idx][:200])

# ── Direct FAISS lookup ────────────────────────────────────────────
log.info("")
log.info("=== Direct FAISS index lookup ===")
idx = getattr(pipeline, "container_index", None)
if idx is not None:
    for name in ["Joe Scarborough", "Charles Joseph", "Morning Joe", "MSNBC"]:
        try:
            q_vec = nlp(name).vector
            nrm = np.linalg.norm(q_vec)
            q_vec_norm = q_vec / nrm if nrm > 0 else q_vec
            dists, ids = idx.search(np.array([q_vec_norm]).astype("float32"), 5)
            top5 = [(int(ids[0][i]), round(float(dists[0][i]), 3)) for i in range(5) if ids[0][i] >= 0]
            log.info("  '%s' -> %s", name, top5)
            for cid, dist in top5:
                c = cm.get_container(cid)
                if c:
                    hes = he.get_hyperedges_for_container(cid)
                    log.info("    cid=%d entity=%r hyperedges=%s", cid, c.entity, [(h.id, h.event_name[:30]) for h in hes])
        except Exception as ex:
            log.info("  '%s' -> ERROR: %s", name, ex)
else:
    log.info("  No container_index found")

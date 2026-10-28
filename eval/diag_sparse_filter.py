"""
Attribute soft-match diagnostic: mount quality and soft-match contribution.
Runs on eval/results/smoke20_sparse_collapse.json (n=20).
"""
import json, sys, os
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, _ROOT)

RESULTS = Path("eval/results/smoke20_sparse_collapse.json")
out: list = []  # accumulates print lines

def log(msg: str):
    out.append(msg)
    print(msg)


def analyse():
    d = json.load(open(RESULTS, encoding="utf-8"))
    records = d.get("records", [])
    log(f"\n{'='*70}")
    log(f"Attribute Soft-Match Diagnostic  |  n={len(records)}")
    log(f"{'='*70}\n")

    # ── 1. Mount quality ─────────────────────────────────────────────
    # For each gold sentence we need its attribute_mounts.
    # We don't have the raw graph, so we read from the pipeline debug
    # (step6_pool_sids + sentence text from diag's step10_final_sids).
    # attribute_mounts are NOT serialised in the result JSON; we read
    # them by looking up each gold sentence node in the graph directly.
    # Re-load the graph once.
    from src.stengraph_pipeline import StenGraphPipeline
    from src.graph_loader import load_graph
    from src.container_manager import ContainerManager
    from src.hyperedge_registry import HyperEdgeRegistry

    graph_path = "graphs/hotpotqa_graph.pkl"
    if not os.path.exists(graph_path):
        log(f"[WARN] Graph not found at {graph_path}, skipping mount quality check")
        graph_path = "graphs/hotpotqa_full.pkl"
    if not os.path.exists(graph_path):
        log(f"[WARN] Graph not found, skipping mount quality check")
        graph_path = None

    pipeline: StenGraphPipeline = None
    if graph_path:
        try:
            graph = load_graph(graph_path)
            cm = ContainerManager(graph)
            he = HyperEdgeRegistry(graph)
            pipeline = StenGraphPipeline(graph, cm, he)
            log("[INFO] Graph loaded OK\n")
        except Exception as e:
            log(f"[WARN] Could not load graph: {e}\n")
            pipeline = None

    log(f"{'─'*70}")
    log(f"TEST 1 · Mount Quality  (gold sentence → attribute_mounts)")
    log(f"{'─'*70}")

    mount_quality = []  # (qid, gold_sid, mounts, verdict)
    for rec in records:
        qid = rec["qid"]
        gold_sids = rec.get("diag", {}).get("gold_sentence_ids", [])
        # Get the first gold sid
        if not gold_sids:
            log(f"\n  [Q] {qid}  →  NO gold sentence found in graph")
            continue
        gold_sid = gold_sids[0]
        gold_text = rec.get("gold", "")[:50]
        mounts = []
        mountable_attrs = set()
        if pipeline:
            node = pipeline.registry.get_node(gold_sid)
            if node:
                for m in getattr(node, "attribute_mounts", []) or []:
                    mounts.append(m.attribute_name)
                entity_type = getattr(node, "entity_type", "PERSON")
                mountable_attrs = set(pipeline.registry.get_all_attribute_names(entity_type))
            else:
                log(f"\n  [Q] {qid}  sid={gold_sid}  not found in registry")
                continue

        em = rec.get("em", False)
        mounts_str = ", ".join(sorted(mounts)) if mounts else "(none)"
        status = "✓" if mounts else "✗ NO MOUNT"
        log(f"\n  [{status}] {qid}  sid={gold_sid}")
        log(f"        gold:  {gold_text!r}")
        log(f"        mounts: {mounts_str}")
        if pipeline:
            entity_type = getattr(pipeline.registry.get_node(gold_sid), "entity_type", "?") if pipeline else "?"
            log(f"        entity_type={entity_type}  mountable={sorted(mountable_attrs)}")
        mount_quality.append((qid, gold_sid, set(mounts), mounts, em, rec["question"][:55]))

    no_mount = [x for x in mount_quality if not x[2]]
    log(f"\n  >> Gold sentences with NO mount: {len(no_mount)}/{len(mount_quality)}")
    log(f"     {['  '.join([q,q2]) for q,q2,_,_,_,_ in no_mount[:5]]}")

    # ── 2. Soft-match contribution ────────────────────────────────────
    log(f"\n{'─'*70}")
    log(f"TEST 2 · Soft-Match Contribution  (attr_term in final score)")
    log(f"{'─'*70}")

    scored_count = 0
    gold_scored_count = 0
    detail: list = []

    for rec in records:
        qid = rec["qid"]
        dbg = rec.get("debug", {})
        ef = dbg.get("ef_scores") or []
        gold_sids = set(rec.get("diag", {}).get("gold_sentence_ids", []))

        if not ef:
            log(f"\n  [?] {qid}  no ef_scores in debug (need re-run with new pipeline)")
            continue

        scored_count += 1
        by_sid = {d.get("sentence_id"): d for d in ef}
        gold_ef = [by_sid[s] for s in gold_sids if s in by_sid]
        if gold_ef:
            gold_scored_count += 1

        top = sorted(ef, key=lambda d: -d.get("final_score", 0))[:3]
        log(f"\n  [{qid}]  pool={len(ef)}  gold_in_ranked={len(gold_ef)}")
        for d in top:
            sid = d.get("sentence_id", "?")
            mark = " ← GOLD" if sid in gold_sids else ""
            log(f"     {sid[:18]:<20} attr_match={d.get('attr_match', 0):.4f} "
                f"attr_term={d.get('attr_term', 0):.4f} "
                f"final={d.get('final_score', 0):.4f}{mark}")
        for d in gold_ef:
            log(f"     GOLD {d.get('sentence_id','?')[:18]:<20} "
                f"attr_match={d.get('attr_match', 0):.4f} "
                f"attr_term={d.get('attr_term', 0):.4f} "
                f"final={d.get('final_score', 0):.4f}")

        detail.append({
            "qid": qid,
            "n_scored": len(ef),
            "gold_scored": len(gold_ef),
            "gold_attr_terms": [d.get("attr_term", 0) for d in gold_ef],
            "top_attr_terms": [d.get("attr_term", 0) for d in top],
        })

    log(f"\n  >> Questions with EF breakdown: {scored_count}/{len(records)}")
    log(f"  >> Questions where a gold sentence was scored: {gold_scored_count}")

    # Summary table
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")
    log(f"  Mount quality:   {len(no_mount)}/{len(mount_quality)} gold sentences have NO mount")
    log(f"  Soft-match:      {gold_scored_count}/{scored_count} questions have a gold sentence scored")
    log(f"  Filter:          removed (mapper/filter deleted; pool is unfiltered)")

    return out

if __name__ == "__main__":
    analyse()

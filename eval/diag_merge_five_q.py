"""
Diagnose 5 questions where hyperedge merge fired:
  Q5  (5ab42b24)  merged  10→9   size=10-19
  Q10 (5a7350d5)  merged   8→7   size=10-19
  Q12 (5a8e5f1f)  fallback 3 members
  Q18 (5ae1d1a5)  fallback 2 members
  Q19 (5ab8494d)  fallback 3 members

For each question runs BOTH merge=ON and merge=OFF, then prints:
  (1) Hyperedges before merge (OFF) / after merge (ON)
  (2) Step-7 reranked candidate sentences with S_h(E,Q) scores
  (3) Whether gold sentences are in top-10
  (4) Whether gold sids appear in step7 / step10
"""
from __future__ import annotations
import json, os, random, sys, io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEEPSEEK_API_KEY", (ROOT / "eval" / "api_key.txt").read_text().strip())

import spacy
import eval.quick20_eval as q20
from src.hyper_edge import HyperEdgeRegistry, merge_paragraph_hyperedges_into

QIDS = {
    "Q5":  "5ab42b24",
    "Q10": "5a7350d5",
    "Q12": "5a8e5f1f",
    "Q18": "5ae1d1a5",
    "Q19": "5ab8494d",
}
GOLD_SENTS_KEY = "supporting_facts"   # for gold sid extraction


def _gold_sids(qdata):
    """Return set of gold sentence indices from supporting_facts."""
    sids = set()
    for sf in qdata.get("supporting_facts", []):
        for idx in sf.get("sentences", []):
            sids.add(idx)
    return sids


def _build_sparse_triggers(builder, paragraph_cids):
    triggers = {cid: set() for cid in paragraph_cids}
    for sid, node in builder.sentence_registry._nodes.items():
        for mount in getattr(node, "attribute_mounts", []) or []:
            attr = mount.attribute_name
            if not attr or attr == "other":
                continue
            for cid in set(node.containers) & set(paragraph_cids):
                triggers[cid].add(attr)
    return {cid: sorted(s) for cid, s in triggers.items() if s}


def run_one(qdata, nlp, use_merge: bool):
    builder, sentences, titles, para_sents, bridge_titles = \
        q20.build_graph_from_hotpot(qdata, nlp)
    cm = builder.container_manager
    for c in cm._containers.values():
        cm.update_container_index(c.id)
    builder.compute_tfidf_index()

    he = HyperEdgeRegistry(container_manager=cm)
    for p_title, p_cids in (bridge_titles or {}).items():
        p_containers = [
            cm._containers.get(c) for c in p_cids
            if cm._containers.get(c) is not None
        ]
        if not p_containers:
            continue
        triggers = _build_sparse_triggers(builder, p_cids)
        he.build_hyperedges_for_chunk(
            chunk_id=str(p_title), containers=p_containers,
            time_word="", paragraph_title=str(p_title),
            attribute_triggers=triggers,
        )

    # ── merge hook ──
    merge_stats = {}
    if use_merge:
        merge_stats = merge_paragraph_hyperedges_into(he)

    from src.dispatcher import build_dispatcher
    disp = build_dispatcher(
        builder, client=None, nlp=nlp,
        hyperedge_registry=he,
        attr_soft_match=True,
    )
    # suppress pipeline INFO noise
    res = disp.run(qdata["question"], bridge_titles=bridge_titles)
    return builder, cm, he, res, merge_stats


def fmt_sid(sid, builder, gold_sids, rank=None):
    n = builder.sentence_registry.get_node(sid)
    if n is None:
        return f"  ?unknown sid={sid}"
    text = n.text[:120].replace("\n", " ")
    star = "★" if sid in gold_sids else " "
    rank_str = f"#{rank:02d}" if rank is not None else "    "
    return f"  {star}{rank_str}  {sid}: {text!r}"


def print_hyperedges(cm, he, builder, label):
    print(f"\n  --- Hyperedges [{label}] (total={len(he)}) ---")
    all_he = sorted(he.get_all_hyperedges(), key=lambda e: e.id)
    for e in all_he:
        # gather entities
        ents = []
        for cid in e.container_ids or []:
            c = cm.get_container(cid)
            if c:
                ents.append(c.entity or "")
        ent_str = "; ".join(ents[:6])
        merged_str = f"  merged_from={e.merged_from}" if e.merged_from else ""
        print(
            f"    {e.id}  title={e.event_name!r:40s}  "
            f"containers={len(e.container_ids)}  "
            f"time={e.time_range}  "
            f"ents=[{ent_str[:80]}]  "
            f"{merged_str}"
        )


def print_step7_with_scores(cm, he, res, builder, gold_sids):
    pr = res.pipeline_result
    s7 = pr.step7_top_n_sids or []
    # Try to pull S_h scores from rerank debug
    rerank_debug = res.debug if hasattr(res, "debug") else {}
    cand_scores = rerank_debug.get("candidate_scores", {})  # sid -> float

    print(f"\n  --- Step-7 top sentences [{len(s7)} total] ---")
    for rank, sid in enumerate(s7[:12], 1):
        score_str = ""
        if cand_scores and sid in cand_scores:
            score_str = f"  S={cand_scores[sid]:.4f}"
        n = builder.sentence_registry.get_node(sid)
        if n is None:
            print(f"    #{rank:02d}  {sid}: <deleted>")
            continue
        star = "★" if sid in gold_sids else " "
        text = n.text[:110].replace("\n", " ")
        print(f"  {star}#{rank:02d}  {sid}: {text!r}{score_str}")


def main():
    # Force UTF-8 stdout
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    rng = random.Random(20)
    with open(ROOT / "eval" / "data" / "hotpot_dev_distractor_v1.json", encoding="utf-8") as fh:
        data = json.load(fh)
    indices = rng.sample(range(len(data)), 20)
    indices.sort()
    sample = [data[i] for i in indices]

    nlp = spacy.load("en_core_web_lg")

    for qname, qid_prefix in QIDS.items():
        qdata = next(q for q in sample if q["_id"].startswith(qid_prefix))
        gold_sids = _gold_sids(qdata)
        print(f"\n{'='*80}")
        print(f"# {qname}: {qdata['_id']}")
        print(f"# Q: {qdata['question']}")
        print(f"# A: {qdata['answer']}")
        print(f"# Gold sids: {sorted(gold_sids)}")

        for use_merge in (False, True):
            label = "MERGE-OFF" if not use_merge else "MERGE-ON"
            print(f"\n{'─'*60}")
            print(f"  >>> {label} <<<")
            try:
                builder, cm, he, res, mstats = run_one(qdata, nlp, use_merge)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                import traceback; traceback.print_exc()
                continue

            if mstats:
                print(f"  merge_stats: {mstats}")

            print_hyperedges(cm, he, builder, label)
            print_step7_with_scores(cm, he, res, builder, gold_sids)

            pr = res.pipeline_result
            s7 = pr.step7_top_n_sids or []
            s10 = pr.step10_final_sids or []
            gold_in_s7 = gold_sids & set(s7)
            gold_in_s10 = gold_sids & set(s10)
            print(
                f"\n  Gold in step7: {len(gold_in_s7)}/{len(gold_sids)} → {sorted(gold_in_s7)}"
            )
            print(
                f"  Gold in step10: {len(gold_in_s10)}/{len(gold_sids)} → {sorted(gold_in_s10)}"
            )
            print(
                f"  Top-10 S7 gold coverage: {len(gold_in_s7 & set(s7[:10]))}/{len(gold_sids)}"
            )


if __name__ == "__main__":
    main()

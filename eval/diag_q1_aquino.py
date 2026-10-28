"""
Q1 (Aquino population) diagnostic dump
======================================

Runs ONLY Q1 = 5ab9d2e2 with both legacy and sparse mode, and dumps:
  (1) Attribute mounts of "102,984" and "24,950" sentences
  (2) he_allowed per L1/L2 container, gold-mount presence
  (3) step4 → step7 → step10 sentence trace, side-by-side
"""
from __future__ import annotations
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import spacy
import eval.quick20_eval as q20
from src.hyper_edge import HyperEdgeRegistry

QID = "5ab9d2e2"

def build_sparse_triggers(builder, paragraph_cids):
    triggers = {cid: set() for cid in paragraph_cids}
    for sid, node in builder.sentence_registry._nodes.items():
        for mount in getattr(node, "attribute_mounts", []) or []:
            attr = mount.attribute_name
            if not attr or attr == "other":
                continue
            for cid in set(node.containers) & set(paragraph_cids):
                triggers[cid].add(attr)
    return {cid: sorted(s) for cid, s in triggers.items() if s}


def find_paragraph_cids_with_title(cm, title, paragraph_to_containers):
    for k, cids in paragraph_to_containers.items():
        if title in str(k):
            return cids, k
    return None, None


def run_one(qdata, nlp, sparse: bool, use_llm=False):
    builder, sentences, titles, para_sents, bridge_titles = \
        q20.build_graph_from_hotpot(qdata, nlp)
    cm = builder.container_manager
    for c in cm._containers.values():
        cm.update_container_index(c.id)
    builder.compute_tfidf_index()

    he = HyperEdgeRegistry(container_manager=cm)
    for p_title, p_cids in (bridge_titles or {}).items():
        p_containers = [cm._containers.get(c) for c in p_cids if cm._containers.get(c)]
        if not p_containers:
            continue
        if sparse:
            triggers = build_sparse_triggers(builder, p_cids)
            he.build_hyperedges_for_chunk(
                chunk_id=str(p_title), containers=p_containers,
                time_word="", paragraph_title=str(p_title),
                attribute_triggers=triggers,
            )
        else:
            he.build_hyperedges_for_chunk(
                chunk_id=str(p_title), containers=p_containers,
                time_word="", paragraph_title=str(p_title),
            )

    from src.dispatcher import build_dispatcher
    client = None
    if use_llm:
        from src.llm_client import DeepSeekClient
        client = DeepSeekClient()
    disp = build_dispatcher(builder, client=client, nlp=nlp,
                            hyperedge_registry=he,
                            attr_soft_match=sparse)
    res = disp.run(qdata["question"], bridge_titles=bridge_titles)
    return builder, cm, he, res, client


def attr_mounts_for_node(node, entity):
    out = []
    for m in getattr(node, "attribute_mounts", []) or []:
        if entity and m.entity != entity:
            continue
        out.append((m.entity, m.attribute_name, m.weight))
    return out


def main():
    import io, sys
    # Force UTF-8 stdout for non-ASCII entity names
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("DEEPSEEK_API_KEY",
        Path("eval/api_key.txt").read_text().strip())
    rng = random.Random(7)
    with open("eval/data/hotpot_dev_distractor_v1.json", encoding="utf-8") as fh:
        data = json.load(fh)
    indices = rng.sample(range(len(data)), 20)
    sample = [data[i] for i in indices]
    qdata = next(q for q in sample if q["_id"].startswith(QID))
    print(f"# Q1: {qdata['question']}")
    print(f"# Gold: {qdata['answer']}\n")
    nlp = spacy.load("en_core_web_lg")

    # === Run both with LLM ===
    for sparse in (False, True):
        print(f"\n{'='*80}\n## MODE = {'SPARSE+LLM' if sparse else 'LEGACY+LLM'}\n{'='*80}")
        builder, cm, he, res, client = run_one(qdata, nlp, sparse=sparse, use_llm=True)

        # (1) attribute mounts for "102,984" and "24,950" sentences
        print("\n### (1) Attribute mounts of population-bearing sentences")
        for sid, node in builder.sentence_registry._nodes.items():
            if "102,984" in node.text or "24,950" in node.text:
                print(f"  sid={sid}: {node.text[:160]!r}")
                for m in getattr(node, "attribute_mounts", []) or []:
                    print(f"    mount entity={m.entity!r} attr={m.attribute_name!r} w={m.weight}")

        # (2) he_allowed per L1/L2 container, gold-mount presence
        print("\n### (2) Hyperedge triggers per container (he_allowed)")
        all_he = []
        for he_id, he_obj in he._edges.items():
            for cid, attrs in getattr(he_obj, "attribute_triggers", {}).items():
                all_he.append((he_id, cid, attrs))
        for he_id, cid, attrs in all_he:
            c = cm.get_container(cid)
            if c is None:
                continue
            ent_disp = repr(c.entity)[:40]
            attrs_s = sorted(attrs)
            print(f"  he={he_id[:8]} cid={cid[:8]}  entity={ent_disp:40}  triggers={attrs_s}")
            # Check each sentence in this container for gold mount
            for sid in c.sentence_ids:
                n = builder.sentence_registry.get_node(sid)
                if n is None:
                    continue
                if "102,984" in n.text or "24,950" in n.text or "Pompano" in n.text:
                    mounts = {m.attribute_name for m in getattr(n, "attribute_mounts", []) or []
                              if m.entity == c.entity}
                    inter = mounts & set(attrs)
                    txt = n.text[:120]
                    print(f"    sid={sid} mounts={mounts} intersect_he={inter}  text={txt!r}")

        # (3) step trace
        print("\n### (3) Step trace (L1/L2/S7/S10)")
        pr = res.pipeline_result
        # a) L1 containers (from A-path) - just print A-related containers
        print("  A-entity containers (touching 'Aquino', 'Pompano'):")
        for cid, c in cm._containers.items():
            if "aquino" in c.entity.lower() or "pompano" in c.entity.lower():
                print(f"    cid={cid[:8]} entity={c.entity!r:50} sentences={len(c.sentence_ids)}")
                for sid in c.sentence_ids:
                    n = builder.sentence_registry.get_node(sid)
                    if n is None: continue
                    if "102,984" in n.text or "24,950" in n.text or "Pompano" in n.text:
                        print(f"      sid={sid} text={n.text[:120]!r}")
        print(f"  step4_b_containers: {len(pr.step4_b_containers)}")
        print(f"  step4_l2_cids_filtered: {len(pr.step4_l2_cids_filtered)}")
        print(f"  step4_l2_sids_filtered: {len(pr.step4_l2_sids_filtered)}")
        print(f"  step6_pool_size: {pr.step6_pool_size}")
        print(f"  step7_top_n_sids ({len(pr.step7_top_n_sids)}):")
        for sid in pr.step7_top_n_sids:
            n = builder.sentence_registry.get_node(sid)
            if n is None: continue
            mark = "★" if "102,984" in n.text else (" " if "24,950" not in n.text else "△")
            print(f"    {mark} {sid}: {n.text[:160]!r}")
        print(f"  step10_final_sids ({len(pr.step10_final_sids)}):")
        for sid in pr.step10_final_sids:
            n = builder.sentence_registry.get_node(sid)
            if n is None: continue
            mark = "★" if "102,984" in n.text else (" " if "24,950" not in n.text else "△")
            print(f"    {mark} {sid}: {n.text[:160]!r}")
        # Step9.5 expansion detail
        if hasattr(pr, 'step9_5_merged_sids') and pr.step9_5_merged_sids:
            print(f"  step9_5_merged_sids ({len(pr.step9_5_merged_sids)}):")
            for sid in pr.step9_5_merged_sids:
                n = builder.sentence_registry.get_node(sid)
                if n is None: continue
                mark = "★" if "102,984" in n.text else (" " if "24,950" not in n.text else "△")
                print(f"    {mark} {sid}: {n.text[:160]!r}")
        added = pr.debug.get("step9_5_added_sids", [])
        if added:
            print(f"  step9_5_added_sids ({len(added)}): {added}")
            for sid in added:
                n = builder.sentence_registry.get_node(sid)
                if n is None: continue
                mark = "★" if "102,984" in n.text else (" " if "24,950" not in n.text else "△")
                print(f"    {mark} {sid}: {n.text[:160]!r}")
        print(f"  final_answer: {pr.final_sids[:5] if hasattr(pr, 'final_sids') else pr.step10_final_sids[:5]}")
        # Print final_text (what LLM actually sees)
        print(f"  final_text (what LLM sees):")
        for line in (pr.final_text or "").splitlines()[:15]:
            print(f"    {line[:200]}")


if __name__ == "__main__":
    main()

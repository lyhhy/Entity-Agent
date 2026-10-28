"""Quick smoke for diag_merge_five_q structure (only Q5, no LLM)."""
from __future__ import annotations
import json, os, sys, io, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import spacy
import eval.quick20_eval as q20
from src.hyper_edge import HyperEdgeRegistry, merge_paragraph_hyperedges_into


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


def run_one(qdata, nlp, use_merge):
    builder, sentences, titles, para_sents, bridge_titles = \
        q20.build_graph_from_hotpot(qdata, nlp)
    cm = builder.container_manager
    for c in cm._containers.values():
        cm.update_container_index(c.id)
    builder.compute_tfidf_index()

    he = HyperEdgeRegistry(container_manager=cm)
    for p_title, p_cids in (bridge_titles or {}).items():
        p_containers = [cm._containers.get(c) for c in p_cids
                        if cm._containers.get(c) is not None]
        if not p_containers:
            continue
        # Mirror run_with_key_seed20.py: no --sparse-attribute → no triggers
        he.build_hyperedges_for_chunk(
            chunk_id=str(p_title), containers=p_containers,
            time_word="", paragraph_title=str(p_title),
        )

    if use_merge:
        merge_paragraph_hyperedges_into(he)

    from src.dispatcher import build_dispatcher
    disp = build_dispatcher(builder, client=None, nlp=nlp,
                            hyperedge_registry=he, attr_soft_match=False)
    res = disp.run(qdata["question"], bridge_titles=bridge_titles)
    return builder, cm, he, res


def main():
    os.environ.setdefault("DEEPSEEK_API_KEY",
        (ROOT / "eval" / "api_key.txt").read_text().strip())
    rng = random.Random(20)
    with open(ROOT / "eval" / "data" / "hotpot_dev_distractor_v1.json",
              encoding="utf-8") as fh:
        data = json.load(fh)
    indices = rng.sample(range(len(data)), 20)
    sample = [data[i] for i in indices]
    qdata = next(q for q in sample if q["_id"].startswith("5ab42b24"))
    print(f"Q: {qdata['question']}\nGold: {qdata['answer']}")

    nlp = spacy.load("en_core_web_lg")

    for use_merge in (False, True):
        label = "MERGE-OFF" if not use_merge else "MERGE-ON"
        print(f"\n{'='*60}\n {label}\n{'='*60}")
        builder, cm, he, res = run_one(qdata, nlp, use_merge)

        # Hyperedge summary
        print(f"\n# Hyperedges ({len(he)} total):")
        for e in sorted(he.get_all_hyperedges(), key=lambda x: x.id):
            ents = [cm.get_container(cid).entity for cid in e.container_ids
                    if cm.get_container(cid)]
            print(f"  {e.id} title={e.event_name!r:30s} containers={len(e.container_ids)} "
                  f"time={e.time_range} merged_from={e.merged_from}")

        pr = res.pipeline_result
        print(f"\n# step7_top_n_sids: {len(pr.step7_top_n_sids or [])}")
        for i, sid in enumerate((pr.step7_top_n_sids or [])[:8], 1):
            n = builder.sentence_registry.get_node(sid)
            txt = (n.text[:80] if n else "<missing>").replace("\n", " ")
            print(f"  #{i:02d} {sid}: {txt!r}")

        # EF scores
        ef = pr.debug.get("ef_scores", [])
        print(f"\n# ef_scores (top-8 by S_h_E):")
        ef_sorted = sorted(ef, key=lambda d: -d.get("S_h_E", 0))[:8]
        for d in ef_sorted:
            sid = d.get("sentence_id")
            n = builder.sentence_registry.get_node(sid)
            txt = (n.text[:60] if n else "<missing>").replace("\n", " ")
            print(
                f"  {sid}: S_h_E={d.get('S_h_E', 0):.3f}  S_h_F={d.get('S_h_F', 0):.3f}  "
                f"S_f={d.get('S_f', 0):.3f}  final={d.get('final_score', 0):.3f}  "
                f"echo={d.get('echo_penalty', 0):.3f}  ev={d.get('event_id', '?')[:8]}  "
                f"txt={txt!r}"
            )


if __name__ == "__main__":
    main()

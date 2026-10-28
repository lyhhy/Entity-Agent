"""Q5 trace with real DeepSeek client + sparse=False (matches seed=20)."""
from __future__ import annotations
import json, os, random, sys, io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import spacy
import eval.quick20_eval as q20
from src.hyper_edge import HyperEdgeRegistry, merge_paragraph_hyperedges_into
from src.llm_client import DeepSeekClient


def run_one(qdata, nlp, use_merge, use_llm):
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
        he.build_hyperedges_for_chunk(
            chunk_id=str(p_title), containers=p_containers,
            time_word="", paragraph_title=str(p_title),
        )
    if use_merge:
        merge_paragraph_hyperedges_into(he)

    from src.dispatcher import build_dispatcher
    client = DeepSeekClient() if use_llm else None
    disp = build_dispatcher(builder, client=client, nlp=nlp,
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
    print(f"Q5: {qdata['question']}\nGold: {qdata['answer']}")
    nlp = spacy.load("en_core_web_lg")

    for use_merge in (False, True):
        label = "MERGE-OFF" if not use_merge else "MERGE-ON"
        print(f"\n{'='*60}\n {label}\n{'='*60}")
        # use_llm=False to skip DeepSeek and just see rerank candidates
        builder, cm, he, res = run_one(qdata, nlp, use_merge, use_llm=False)
        print(f"\n# Hyperedges ({len(he)}):")
        for e in sorted(he.get_all_hyperedges(), key=lambda x: x.id):
            ents = [cm.get_container(cid).entity for cid in e.container_ids
                    if cm.get_container(cid)]
            print(f"  {e.id} title={e.event_name!r:42s} containers={len(e.container_ids)} "
                  f"merged_from={e.merged_from} ents={ents[:3]}")


if __name__ == "__main__":
    main()

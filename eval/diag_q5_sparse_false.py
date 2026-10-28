"""Q5: check sid → containers mapping under sparse=False + DeepSeek client."""
from __future__ import annotations
import json, os, random, sys, io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import spacy
import eval.quick20_eval as q20
from src.hyper_edge import HyperEdgeRegistry
from src.llm_client import DeepSeekClient


def main():
    os.environ.setdefault("DEEPSEEK_API_KEY",
        (ROOT / "eval" / "api_key.txt").read_text().strip())
    rng = random.Random(20)
    with open(ROOT / "eval" / "data" / "hotpot_dev_distractor_v1.json",
              encoding="utf-8") as fh:
        data = json.load(fh)
    indices = rng.sample(range(len(data)), 20)
    indices.sort()
    sample = [data[i] for i in indices]
    qdata = next(q for q in sample if q["_id"].startswith("5ab42b24"))
    nlp = spacy.load("en_core_web_lg")

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

    print(f"Hyperedges: {len(he)}")
    # sample sentence → containers
    print("\nFirst 5 sentence → containers:")
    for i, (sid, node) in enumerate(builder.sentence_registry._nodes.items()):
        if i >= 5:
            break
        cids = getattr(node, "containers", []) or []
        ents = [cm.get_container(c).entity for c in cids if cm.get_container(c)]
        print(f"  {sid}: containers={cids}  entities={ents}")

    # Now check he.hyperedges back-link
    print("\nContainer → he back-link (first 5):")
    for i, c in enumerate(cm.get_all_containers()):
        if i >= 5:
            break
        print(f"  {c.id} entity={c.entity!r:40} hyperedges={c.hyperedges}")

    from src.dispatcher import build_dispatcher
    client = DeepSeekClient()
    print(f"\nDeepSeek client loaded: {client is not None}")
    disp = build_dispatcher(builder, client=client, nlp=nlp,
                            hyperedge_registry=he, attr_soft_match=False)
    res = disp.run(qdata["question"], bridge_titles=bridge_titles)
    pr = res.pipeline_result
    print(f"\nstep7_top_n_sids: {len(pr.step7_top_n_sids or [])}")
    # dump ef_scores
    ef = pr.debug.get("ef_scores", [])
    print(f"ef_scores: {len(ef)}")
    for d in ef[:5]:
        print(f"  {d.get('sentence_id')}: S_h_E={d.get('S_h_E', 0):.3f} "
              f"event_id={d.get('event_id', '?')}")


if __name__ == "__main__":
    main()

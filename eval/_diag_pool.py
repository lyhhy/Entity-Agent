"""Targeted diagnostic: for questions whose gold answer was NOT
retrieved, is the gold-bearing sentence (a) absent from the step6
candidate pool, or (b) present in the pool but ranked below top_k_final?

This decides the fix:
  (a) missing from pool  -> fix candidate generation (seed containers /
      hyperedge expansion)
  (b) in pool, ranked low -> fix recall ceiling (top_k_final) and/or the
      LLM context length

Usage:
    python eval/_diag_pool.py [--n 20] [--seed 137]
"""
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(REPO / "eval"))

import logging
logging.basicConfig(level=logging.WARNING)

from src.graph_builder import _load_spacy_shared
from src.hyper_edge import HyperEdgeRegistry
from src.offline_hyperedge_store import OfflineHyperEdgeStore
from src.dispatcher import build_dispatcher
import quick20_eval as q20

_NORM = q20._normalize_answer


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 20
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 137

    nlp = _load_spacy_shared("en_core_web_lg")
    with open(REPO / "eval/data/hotpot_dev_distractor_v1.json", encoding="utf-8") as f:
        all_data = json.load(f)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(all_data)), min(n, len(all_data))))
    sample = [all_data[i] for i in idx]

    offline = OfflineHyperEdgeStore.from_parquet(
        str(REPO / "eval/offline_hyperedges/hyperedges.parquet"))

    tally = {
        "A_gold_in_final": 0,          # gold in final top-k
        "B_in_pool_not_final": 0,     # gold in step6 pool but cut by top_k
        "C_not_in_pool": 0,           # gold absent from pool entirely
        "D_gold_string_absent": 0,    # gold string not in ANY context sentence
    }
    details = []

    for qdata in sample:
        qid = qdata["_id"]
        gold = qdata["answer"]
        gold_norm = _NORM(gold)

        builder, sentences, titles, para_sents, bridge_titles = \
            q20.build_graph_from_hotpot(qdata, nlp)
        cm = builder.container_manager
        for c in cm.get_all_containers():
            cm.update_container_index(c.id)
        builder.compute_tfidf_index()

        # Does the gold string exist in ANY sentence of the context?
        all_sids = list(builder.sentence_registry._nodes.keys())
        gold_sids = [
            s for s in all_sids
            if gold_norm in _NORM(builder.sentence_registry.get_node(s).text or "")
        ]

        he = HyperEdgeRegistry(container_manager=cm)
        for t in titles:
            cids = offline.get_containers_for_paragraph(qid, t)
            if not cids:
                cids = (bridge_titles or {}).get(t.lower().strip(), [])
            conts = [cm._containers.get(c) for c in cids
                     if cm._containers.get(c) is not None]
            if conts:
                he.build_hyperedges_for_chunk(
                    chunk_id=str(t), containers=conts, time_word="",
                    paragraph_title=str(t))

        disp = build_dispatcher(
            builder, client=None, nlp=nlp, hyperedge_registry=he,
            attr_soft_match=True,
        )
        res = disp.run(qdata["question"], bridge_titles=bridge_titles)
        pr = res.pipeline_result
        final = list(getattr(pr, "step10_final_sids", []) or [])
        pool = list(pr.debug.get("step6_pool_sids", []) or [])

        if not gold_sids:
            tally["D_gold_string_absent"] += 1
            details.append((qid, "D_gold_string_absent", gold, [], pool, final))
            continue
        if any(s in final for s in gold_sids):
            tally["A_gold_in_final"] += 1
            details.append((qid, "A_gold_in_final", gold, gold_sids, pool, final))
        elif any(s in pool for s in gold_sids):
            tally["B_in_pool_not_final"] += 1
            details.append((qid, "B_in_pool_not_final", gold, gold_sids, pool, final))
        else:
            tally["C_not_in_pool"] += 1
            details.append((qid, "C_not_in_pool", gold, gold_sids, pool, final))

    print("=" * 84)
    print(f"seed={seed} n={len(sample)}   top_k_final=8")
    print("=" * 84)
    for k, v in tally.items():
        print(f"  {k:<26} {v:>3}/{len(sample)}")
    print()
    print("=" * 84)
    print("DETAIL (non-A cases)")
    print("=" * 84)
    for qid, kind, gold, gsids, pool, final in details:
        if kind == "A_gold_in_final":
            continue
        rank = [i for i, s in enumerate(final) if s in gsids]
        prank = [i for i, s in enumerate(pool) if s in gsids]
        print(f"  [{kind}] {qid[:8]}  gold={gold!r}")
        print(f"    gold_sids={gsids}  final_rank={rank}  pool_rank={prank}")
        print(f"    pool_size={len(pool)}  final_size={len(final)}")

    # Recall ceiling analysis
    print()
    print("=" * 84)
    print("RECALL CEILING")
    print("=" * 84)
    in_pool = tally["A_gold_in_final"] + tally["B_in_pool_not_final"]
    exists = in_pool + tally["C_not_in_pool"]
    print(f"  gold string present in context : {exists}/{len(sample)}")
    print(f"  gold reached the candidate pool : {in_pool}/{len(sample)}")
    print(f"  gold survived to top-8          : "
          f"{tally['A_gold_in_final']}/{len(sample)}")
    if exists:
        print(f"  -> pool recall      = {in_pool/exists:.3f}")
        print(f"  -> top-8 retention  = "
              f"{tally['A_gold_in_final']/max(1,in_pool):.3f}")


if __name__ == "__main__":
    main()

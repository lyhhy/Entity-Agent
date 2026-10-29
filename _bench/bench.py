"""Baseline / regression harness.

Runs the real main flow over N HotpotQA questions and reports the
metrics that must not regress when dead code is removed:
  R@1 / R@5 / R@10   (retrieval recall)
  pool size, final sid count
  wall time

Usage:  python _bench/bench.py --n 20 --seed 7 [--tag baseline]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import logging
logging.disable(logging.CRITICAL)


def run(n, seed, data_path):
    import quick20_eval as q20
    from src.graph_builder import _load_spacy_shared
    from src.dispatcher import build_dispatcher
    from src.hyper_edge import HyperEdgeRegistry

    data = json.load(open(data_path, encoding="utf-8"))
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(data)), min(n, len(data))))
    sample = [data[i] for i in idx]

    nlp = _load_spacy_shared("en_core_web_lg")

    r1 = r5 = r10 = 0
    pools, finals, times = [], [], []
    per_q = []

    for qdata in sample:
        q = qdata["question"]
        gold = qdata["answer"]
        t0 = time.time()
        builder, sentences, titles, para_sents, bridge_titles = q20.build_graph_from_hotpot(qdata, nlp)
        cm = builder.container_manager
        for c in cm.get_all_containers():
            cm.update_container_index(c.id)
        builder.compute_tfidf_index()
        he = HyperEdgeRegistry(container_manager=cm)
        for _pt, _pc in (bridge_titles or {}).items():
            cs = [cm._containers.get(_c) for _c in _pc if cm._containers.get(_c) is not None]
            if cs:
                he.build_hyperedges_for_chunk(chunk_id=str(_pt), containers=cs,
                                             time_word="", paragraph_title=str(_pt))
        disp = build_dispatcher(builder, client=None, nlp=nlp, hyperedge_registry=he)
        res = disp.run(q, bridge_titles=bridge_titles)
        pr = res.pipeline_result
        dt = (time.time() - t0) * 1000

        gold_norm = q20._normalize_answer(gold)
        sids = list(getattr(pr, "step10_final_sids", []) or [])

        def has_gold(sid):
            n = builder.sentence_registry.get_node(sid)
            return n is not None and gold_norm in q20._normalize_answer(n.text or "")

        h1 = 1 if (sids and has_gold(sids[0])) else 0
        h5 = 1 if any(has_gold(s) for s in sids[:5]) else 0
        h10 = 1 if any(has_gold(s) for s in sids[:10]) else 0
        r1 += h1; r5 += h5; r10 += h10
        pools.append(pr.step6_pool_size)
        finals.append(len(sids))
        times.append(dt)
        per_q.append({"qid": qdata["_id"], "r1": h1, "r5": h5, "r10": h10,
                       "pool": pr.step6_pool_size, "n_final": len(sids),
                       "a_cids": len(pr.step2_a_containers),
                       "b_cids": len(pr.step4_b_containers),
                       "n_he": len(pr.step3_hyperedges)})

    N = max(1, len(per_q))
    return {
        "n": len(per_q),
        "r1": r1 / N, "r5": r5 / N, "r10": r10 / N,
        "pool_avg": statistics.mean(pools) if pools else 0,
        "final_avg": statistics.mean(finals) if finals else 0,
        "ms_avg": statistics.mean(times) if times else 0,
        "per_q": per_q,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--data", default=str(ROOT / "eval" / "data" / "hotpot_dev_distractor_v1.json"))
    args = ap.parse_args()

    res = run(args.n, args.seed, args.data)
    out = Path(__file__).resolve().parent / f"bench_{args.tag}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n=== BENCH [{args.tag}]  n={res['n']} seed={args.seed} ===")
    print(f"  R@1  = {res['r1']:.3f}")
    print(f"  R@5  = {res['r5']:.3f}")
    print(f"  R@10 = {res['r10']:.3f}")
    print(f"  pool_avg = {res['pool_avg']:.2f}")
    print(f"  final_avg = {res['final_avg']:.2f}")
    print(f"  ms_avg = {res['ms_avg']:.0f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

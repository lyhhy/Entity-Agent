"""
Smoke-test the NEW S_h_E formula on a few questions.

Prints, for each candidate sentence:
  S_h_E_name_cos  = cos(event_name, query)      (weight 0.7)
  S_h_E_sent_cos  = cos(sentence_vec, query)     (weight 0.3)
  S_h_E           = 0.7*name + 0.3*sent
  S_f, final_score

and verifies the arithmetic matches the recomputed values.

No LLM. Retrieval only. Run:
    python eval/_smoke_new_shE.py --n 3 --seed 137
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, str(REPO / "eval"))

import logging
logging.basicConfig(level=logging.WARNING)

from src.graph_builder import _load_spacy_shared
from src.hyper_edge import HyperEdgeRegistry
import quick20_eval as q20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=137)
    args = ap.parse_args()

    nlp = _load_spacy_shared("en_core_web_lg")
    if nlp is None:
        print("[FATAL] spaCy unavailable")
        return 1

    with open(REPO / "eval/data/hotpot_dev_distractor_v1.json", encoding="utf-8") as f:
        import json
        all_data = json.load(f)
    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(len(all_data)), min(args.n, len(all_data))))
    sample = [all_data[i] for i in idx]

    for qi, qdata in enumerate(sample):
        qid = qdata["_id"]
        question = qdata["question"]
        gold = qdata.get("answer", "")
        builder, sentences, titles, para_sents, bridge_titles = (
            q20.build_graph_from_hotpot(qdata, nlp))
        cm = builder.container_manager
        for c in cm.get_all_containers():
            cm.update_container_index(c.id)
        builder.compute_tfidf_index()

        he = HyperEdgeRegistry(container_manager=cm)
        for _p_title, _p_cids in (bridge_titles or {}).items():
            _p_containers = [cm._containers.get(_cid) for _cid in _p_cids
                             if cm._containers.get(_cid) is not None]
            if _p_containers:
                he.build_hyperedges_for_chunk(
                    chunk_id=str(_p_title), containers=_p_containers,
                    time_word="", paragraph_title=str(_p_title))

        from src.dispatcher import build_dispatcher
        disp = build_dispatcher(
            builder, client=None, nlp=nlp, hyperedge_registry=he,
            strict_entity_match=False, attr_soft_match=True,
        )
        q20._apply_hybrid_config(disp.pipeline, "hybrid", 0.5)
        res = disp.run(question, bridge_titles=bridge_titles)
        pr = res.pipeline_result
        final_sids = list(getattr(pr, "step10_final_sids", []) or [])
        ef = pr.debug.get("ef_scores", []) or []

        gold_norm = q20._normalize_answer(gold)

        def is_gold(sid):
            n = builder.sentence_registry.get_node(sid)
            return n is not None and gold_norm in q20._normalize_answer(n.text or "")

        print("=" * 78)
        print(f"[{qi+1}/{len(sample)}] {qid}  pool={getattr(pr,'step6_pool_size',-1)} "
              f"final={len(final_sids)}")
        print(f"  Q: {question}")
        print(f"  G: {gold}")

        by_sid = {d["sentence_id"]: d for d in ef}
        max_err = 0.0
        n_checked = 0
        for rank, sid in enumerate(final_sids[:8], 1):
            d = by_sid.get(sid)
            if d is None:
                continue
            nc = d.get("S_h_E_name_cos", 0.0)
            sc = d.get("S_h_E_sent_cos", 0.0)
            expect = 0.7 * nc + 0.3 * sc
            err = abs(expect - d.get("S_h_E", 0.0))
            max_err = max(max_err, err)
            n_checked += 1
            node = builder.sentence_registry.get_node(sid)
            txt = (node.text if node else "")[:64]
            flag = "*GOLD*" if is_gold(sid) else ""
            print(f"   r{rank:<2} name={nc:.3f} sent={sc:.3f} "
                  f"S_h_E={d.get('S_h_E',0):.3f} (exp {expect:.3f}) "
                  f"S_f={d.get('S_f',0):.3f} final={d.get('final_score',0):.3f} "
                  f"{flag} | {txt}")

        # Cross-check: S_f == event_term + fact_term
        for d in ef[:20]:
            et = d.get("event_term", 0.0)
            ft = d.get("fact_term", 0.0)
            if abs((et + ft) - d.get("S_f", 0.0)) > 1e-6:
                print(f"   !! S_f mismatch sid={d.get('sentence_id')}: "
                      f"{et}+{ft} != {d.get('S_f')}")

        gold_rank = next(
            (r for r, s in enumerate(final_sids[:10], 1) if is_gold(s)), None)
        print(f"   -> formula check: n={n_checked} max_err={max_err:.2e} "
              f"| gold_rank_in_top10={gold_rank}")
        # Sanity: how much event signal is actually flowing?
        sh_es = [d.get("S_h_E", 0.0) for d in ef]
        if sh_es:
            print(f"   -> S_h_E over pool: min={min(sh_es):.3f} "
                  f"max={max(sh_es):.3f} mean={sum(sh_es)/len(sh_es):.3f} "
                  f"zeros={sum(1 for x in sh_es if x==0)}/{len(sh_es)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

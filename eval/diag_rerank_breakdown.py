"""
Diagnose: dump 6-signal breakdown for R@1 failure cases.
Loads api_key.txt directly (single file read at startup).
"""
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load API key from file (avoids shell env issues)
key_path = _REPO_ROOT / "eval" / "api_key.txt"
if key_path.exists():
    os.environ["DEEPSEEK_API_KEY"] = key_path.read_text().strip()

import argparse
from src.graph_builder import _load_spacy_shared
from src.dispatcher import build_dispatcher
from src.offline_hyperedge_store import OfflineHyperEdgeStore
from src.relation_retriever import load_hotpot_qdata

# Find a recent diagnose run for the qid list
DEFAULT_QIDS = [
    "5abe689a55429976d4830b16",   # Q2 French Open   (gold rank 6)
    "5ababbe9554299232ef4a38a",   # Q4 Margery       (gold rank 6)
    "5ae1e9fd5542997f29b3c1a1",   # Q7 Kelly         (gold rank 3)
    "5ae6479a55429929b0807b1b",   # Q9 Bizarre Girl  (gold rank 2)
    "5ab9d2e255429955dce3ed90",   # Q11 102,984      (gold rank 2)
    "5abce3015542993a06baf995",   # Q12 85 nations   (gold rank 2)
    "5a74cfed5542993748c896ea",   # Q13 Matthew Good (gold rank 5)
    "5a75642c5542996c70cfaeed",   # Q20 Bain Capital (gold rank 3)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", nargs="*", default=DEFAULT_QIDS)
    ap.add_argument("--n-max-pool", type=int, default=10,
                    help="top-N to dump from final_sids")
    args = ap.parse_args()

    # Load HotpotQA dev distractor
    qdata = load_hotpot_qdata(_REPO_ROOT / "eval" / "data" / "hotpot_dev_distractor_v1.json")
    by_qid = {q["_id"]: q for q in qdata}
    selected = [by_qid[q] for q in args.qids if q in by_qid]
    print(f"[diag] {len(selected)} questions selected", flush=True)

    # spaCy
    nlp = _load_spacy_shared("en_core_web_lg")
    if nlp is None:
        print("[FATAL] spaCy en_core_web_lg not available")
        return 1

    # Offline hyperedges
    offline_store = OfflineHyperEdgeStore.from_parquet(
        str(_REPO_ROOT / "eval" / "offline_hyperedges" / "hyperedges.parquet")
    )

    # DeepSeek client (off; we don't need LLM for breakdown)
    from src.llm_client import DeepSeekClient
    client = DeepSeekClient()
    if not client.enabled:
        client = None

    import quick20_eval as q20

    out = []
    for i, q in enumerate(selected, 1):
        qid = q["_id"]
        question = q["question"]
        gold = q.get("answer", "")
        t0 = time.time()
        builder, sentences, titles, para_sents, bridge_titles = q20.build_graph_from_hotpot(q, nlp)
        cm = builder.container_manager
        # bridge / hybrid: minimal
        he = offline_store.get_hyperedges_for_qid(qid)
        disp = build_dispatcher(
            builder, client=client, nlp=nlp,
            hyperedge_registry=he,
            strict_entity_match=False,
        )
        q20._apply_hybrid_config(disp.pipeline, "hybrid", 0.5)
        res = disp.run(question, bridge_titles=bridge_titles)
        if res is None or res.pipeline_result is None:
            print(f"[{i}/{len(selected)}] {qid}  NO_RESULT")
            continue
        final_sids = list(getattr(res.pipeline_result, "step10_final_sids", []) or [])
        gold_norm = q20._normalize_answer(gold)
        def contains_gold(sid):
            n = builder.sentence_registry.get_node(sid)
            return n is not None and gold_norm in q20._normalize_answer(n.text or "")

        # Re-score final_sids with breakdown hook on
        pipeline = disp.pipeline
        pipeline._collect_breakdown = True
        pipeline._breakdown_buffer = {}
        _ = pipeline._score_sentences_v8(question, final_sids[:args.n_max_pool])

        top10 = []
        for rank, sid in enumerate(final_sids[:args.n_max_pool], 1):
            n = builder.sentence_registry.get_node(sid)
            top10.append({
                "rank": rank,
                "sid": sid,
                "is_gold": contains_gold(sid),
                "text": (n.text if n else "")[:200],
                "score": pipeline._breakdown_buffer.get(sid, {}),
            })

        gold_rank = next((p["rank"] for p in top10 if p["is_gold"]), None)
        print(f"[{i}/{len(selected)}] {qid}  gold_rank={gold_rank}  ({time.time()-t0:.1f}s)", flush=True)

        out.append({
            "qid": qid,
            "question": question,
            "gold": gold,
            "n_final_sids": len(final_sids),
            "top10": top10,
            "gold_in_rank": gold_rank,
        })

    out_path = _REPO_ROOT / "eval" / "diag" / "diagnose_breakdown3" / "r1_breakdown.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] dumped → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

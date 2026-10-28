"""
A7: Dispatcher-as-reranker strict HotpotQA evaluation.

Pipeline per question:
  1.  Build graph (same as A4).
  2.  Run baseline Retriever → top-K sentence_ids with their BM25 scores.
  3.  Run Dispatcher (BFS+DFS+hyperedge) → get dual.path_sentence_ids
      (the dispatcher's curated chain).  Build a per-sentence score:
        *  any sentence in dual.path_sentence_ids      →  dual.confidence
        *  any sentence in concept.final_order         →  concept-confidence (0.5)
        *  else                                       →  0.0
  4.  Re-rank baseline top-K sentences by this dispatcher score (desc),
      break ties by original baseline score.
  5.  Feed the re-ranked sentence list into the LLM (strict; no soft
      post-processing).  Same prompt as A4.

Goal: test whether dispatcher's *reordering* (not its prose formatting)
helps the LLM answer quality under strict HotpotQA scoring.

Usage:
    python eval/_run_with_key.py eval/dispatcher_rerank_eval.py \\
        --split val --max-examples 20 --seed 33 --use-llm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random as _random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.dispatcher_strict_eval import _build_dispatcher_for_builder
from eval.hotpot_eval import (
    build_graph,
    generate_answer_via_llm,
    hotpot_to_sentences,
    load_hotpot,
    _em_score,
    _f1_score,
)
from src.graph_builder import _load_spacy_shared

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dispatcher_rerank_eval")


def _score_sentences_from_dispatch(dispatch_result, baseline_top_ids: List[str]) -> Dict[str, float]:
    """
    For each baseline sentence_id, return a dispatcher-derived relevance.
    Uses dual.confidence for path sentences, 0.5 for concept path sentences,
    0.0 otherwise.
    """
    scores: Dict[str, float] = {sid: 0.0 for sid in baseline_top_ids}

    if dispatch_result.dual is not None and dispatch_result.dual.path_sentence_ids:
        conf = float(dispatch_result.dual.confidence or 0.0)
        # Position-weighted: earlier in path → slightly higher
        n = len(dispatch_result.dual.path_sentence_ids)
        for i, sid in enumerate(dispatch_result.dual.path_sentence_ids):
            if sid in scores:
                # first gets conf, last gets conf*0.7
                w = 1.0 - 0.3 * (i / max(1, n - 1)) if n > 1 else 1.0
                scores[sid] = max(scores[sid], conf * w)

    if dispatch_result.concept is not None and dispatch_result.concept.final_order:
        for i, item in enumerate(dispatch_result.concept.final_order):
            if item.sentence_id in scores:
                w = 0.5 * (1.0 - 0.3 * (i / max(1, len(dispatch_result.concept.final_order) - 1)))
                scores[item.sentence_id] = max(scores[item.sentence_id], w)

    return scores


def _rerank(
    baseline_top: List[Dict[str, Any]],
    dispatch_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Re-rank baseline candidates by dispatcher score desc, breaking ties
    by baseline score desc.
    """
    for c in baseline_top:
        c["dispatch_score"] = dispatch_scores.get(c["sentence_id"], 0.0)

    return sorted(
        baseline_top,
        key=lambda c: (c["dispatch_score"], c.get("baseline_score", 0.0)),
        reverse=True,
    )


def _evaluate_one(
    qdata: Dict[str, Any],
    shared_nlp,
    use_llm: bool,
) -> Dict[str, Any]:
    qid = qdata.get("_id", "?")
    question = qdata.get("question", "")
    gold_answer = qdata.get("answer", "")

    t0 = time.time()

    sentences = hotpot_to_sentences(qdata, nlp=shared_nlp)
    builder = build_graph(sentences, shared_nlp=shared_nlp)

    # Baseline retriever (same as A4): use Retriever.evidence_chain
    # (top-1 chain AFTER sentence-level rerank) and read each sentence's
    # "id" field.  Earlier A7 erroneously used "sentence_ids" which does
    # not exist in chain.to_dict() output.
    from src.retriever import Retriever
    base = Retriever(builder).retrieve(question)
    top_chain = base.get("evidence_chain") or (base.get("evidence_chains", [{}])[0])
    top_sents = top_chain.get("sentences", []) if top_chain else []
    top_k = 5
    baseline_top = []
    for i, sent in enumerate(top_sents[:top_k]):
        sid = sent.get("id")
        text = sent.get("text") or sent.get("complemented_text") or ""
        if sid and text:
            baseline_top.append({
                "sentence_id": sid,
                "text": text,
                "baseline_score": 1.0 - 0.05 * i,
            })

    # Dispatcher run (default: with hyperedge)
    dispatcher = _build_dispatcher_for_builder(builder, shared_nlp=shared_nlp)
    dispatch_result = dispatcher.run(query=question)

    # Re-rank
    baseline_ids = [c["sentence_id"] for c in baseline_top]
    dispatch_scores = _score_sentences_from_dispatch(dispatch_result, baseline_ids)
    reranked = _rerank(baseline_top, dispatch_scores)

    sent_texts = [c["text"] for c in reranked]
    # Empty edges so the prompt template does not embed "rerank(score=…)"
    # metadata that confuses the LLM.  This matches A4 strict which feeds
    # raw baseline evidence_chains via Retriever → generate_answer_via_llm.
    edges = []

    latency_ms = (time.time() - t0) * 1000

    # LLM answer (strict)
    gen_answer = ""
    em = False
    f1 = 0.0
    llm_lat = 0.0
    if use_llm:
        chain_for_llm = {
            "sentences": [{"text": t} for t in sent_texts],
            "edges": edges,
        }
        gen_answer, llm_lat = generate_answer_via_llm(
            question, evidence_chain=chain_for_llm
        )
        em = _em_score(gen_answer, gold_answer)
        f1 = _f1_score(gen_answer, gold_answer)

    return {
        "qid": qid,
        "question": question,
        "gold_answer": gold_answer,
        "dispatch_path": dispatch_result.path,
        "baseline_top": [c["sentence_id"] for c in baseline_top],
        "baseline_texts": [c["text"][:80] for c in baseline_top],
        "reranked_top": [c["sentence_id"] for c in reranked],
        "reranked_texts": [c["text"][:80] for c in reranked],
        "dispatch_scores": dispatch_scores,
        "rerank_changed": [c["sentence_id"] for c in baseline_top] != [c["sentence_id"] for c in reranked],
        "generated_answer": gen_answer,
        "em": em,
        "f1": f1,
        "llm_latency_s": llm_lat,
        "latency_ms": latency_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--data-dir", default="eval/data")
    args = parser.parse_args()

    filename_aliases = {
        "val": ["dev", "validation", "val"],
        "train": ["train"],
    }
    aliases = filename_aliases.get(args.split, [args.split])
    candidates = []
    for a in aliases:
        for ext in (".parquet", ".json"):
            for suffix in ("", "_v1", "_v1.1"):
                candidates.append(
                    os.path.join(args.data_dir, f"hotpot_{a}_distractor{suffix}{ext}")
                )
    candidates.append(os.path.join(args.data_dir, f"hotpot_{args.split}_distractor_v1.json"))
    candidates.append(os.path.join(args.data_dir, f"hotpot_{args.split}_distractor.parquet"))
    candidates.append(os.path.join(args.data_dir, f"hotpot_{args.split}_distractor_v1.parquet"))
    data_file = next((c for c in candidates if os.path.exists(c)), None)
    if data_file is None:
        log.error("No HotpotQA file in %s", args.data_dir)
        return 1
    log.info("Data: %s", data_file)

    hotpot_data = load_hotpot(data_file, args.split)
    n_total = len(hotpot_data)
    if args.no_shuffle:
        hotpot_data = hotpot_data[: args.max_examples]
    else:
        rng = _random.Random(args.seed)
        if args.max_examples < n_total:
            idxs = rng.sample(range(n_total), args.max_examples)
            idxs.sort()
            hotpot_data = [hotpot_data[i] for i in idxs]
    log.info("Sampled: %d (seed=%d)", len(hotpot_data), args.seed)

    shared_nlp = _load_spacy_shared("en_core_web_lg")

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = out_dir / f"dispatcher_rerank_{args.split}_{len(hotpot_data)}_seed{args.seed}_{timestamp}.json"
    summary_path = out_dir / f"dispatcher_rerank_{args.split}_{len(hotpot_data)}_seed{args.seed}_{timestamp}_summary.json"

    all_results: List[Dict[str, Any]] = []
    ems: List[float] = []
    f1s: List[float] = []

    log.info("Dispatcher rerank strict eval start: n=%d use_llm=%s",
             len(hotpot_data), args.use_llm)
    for i, qdata in enumerate(hotpot_data):
        result = _evaluate_one(qdata, shared_nlp, args.use_llm)
        all_results.append(result)
        if args.use_llm:
            ems.append(float(result["em"]))
            f1s.append(float(result["f1"]))
        if not args.quiet:
            log.info(
                "[%d/%d] qid=%s path=%s rerank_changed=%s em=%s f1=%.3f lat=%.1fms",
                i + 1, len(hotpot_data), result["qid"],
                result["dispatch_path"], result["rerank_changed"],
                "Y" if result["em"] else "N",
                result["f1"], result["latency_ms"],
            )
            if args.use_llm and not result["em"]:
                log.info(
                    "          Q : %s\n          Gold: %s\n          Got : %s",
                    result["question"][:200],
                    result["gold_answer"],
                    result["generated_answer"],
                )

    summary: Dict[str, Any] = {
        "split": args.split,
        "n": len(all_results),
        "seed": args.seed,
        "use_llm": args.use_llm,
        "path_counts": {},
        "rerank_changed_count": 0,
    }
    if args.use_llm:
        summary["em"] = float(np.mean(ems)) if ems else 0.0
        summary["f1"] = float(np.mean(f1s)) if f1s else 0.0
    for r in all_results:
        p = r.get("dispatch_path", "?")
        summary["path_counts"][p] = summary["path_counts"].get(p, 0) + 1
        if r.get("rerank_changed"):
            summary["rerank_changed_count"] += 1

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info("─" * 60)
    log.info("DISPATCHER RERANK STRICT SUMMARY (n=%d seed=%d)",
             len(all_results), args.seed)
    if args.use_llm:
        log.info("  EM = %.3f", summary["em"])
        log.info("  F1 = %.3f", summary["f1"])
    log.info("  Path distribution: %s", summary["path_counts"])
    log.info("  Rerank changed: %d / %d", summary["rerank_changed_count"], len(all_results))
    log.info("Detail → %s", detail_path)
    log.info("Summary → %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
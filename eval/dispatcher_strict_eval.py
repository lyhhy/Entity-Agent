"""
A5: StenRAG Dispatcher (BFS+DFS+hyperedge) strict HotpotQA evaluation.

Same 20 questions / seed=33 as A1/A3/A4 so results are directly comparable.

Pipeline per question:
  1.  Build the sentence-level logic graph (same as hotpot_eval).
  2.  Build the Dispatcher (QueryAnalyzer -> DualEntityRetriever
      (BFS+DFS+hyperedge) -> SingleEntityRetriever -> ConceptRetriever).
  3.  Dispatcher.run(question) returns a DispatchResult whose
      final_answer is the formatted evidence chain text.  Feed that
      chain text into the LLM (DeepSeek) as the only context.
  4.  Strict HotpotQA scoring: NO _coref_pick / _short_answer_gating /
      _strip_parens / _strip_colon_prefix post-processing.  Just standard
      SQuAD normalize on (a, an, the, punctuation, ws).

Usage:
    python eval/_run_with_key.py eval/dispatcher_strict_eval.py \\
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.hotpot_eval import (
    build_graph,
    generate_answer_via_llm,
    gold_supporting_ids,
    gold_supporting_titles,
    hotpot_to_sentences,
    load_hotpot,
    _em_score,
    _f1_score,
)
from src.dispatcher import build_dispatcher
from src.graph_builder import _load_spacy_shared

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dispatcher_strict_eval")


def _build_dispatcher_for_builder(builder, shared_nlp=None, use_hyperedge: bool = True):
    """Build a Dispatcher using the same container_manager graph nodes."""
    return build_dispatcher(
        graph_builder=builder,
        nlp=shared_nlp,
        use_hyperedge=use_hyperedge,
    )


def _chain_text_from_dispatch(dispatch_dict: Dict[str, Any]) -> str:
    """
    The dispatcher's _format_uses_*/_format_concept path already produces a
    human-readable evidence chain (with edge-type markers and source tags).
    That's exactly what we want to feed the LLM.
    """
    return dispatch_dict.get("final_answer", "") or ""


def _evaluate_one(
    qdata: Dict[str, Any],
    shared_nlp,
    use_llm: bool,
    use_hyperedge: bool = True,
) -> Dict[str, Any]:
    qid = qdata.get("_id", "?")
    question = qdata.get("question", "")
    gold_answer = qdata.get("answer", "")
    gold_ids = gold_supporting_ids(qdata)
    gold_titles = gold_supporting_titles(qdata)

    t0 = time.time()

    # 1. Build graph
    sentences = hotpot_to_sentences(qdata, nlp=shared_nlp)
    builder = build_graph(sentences, shared_nlp=shared_nlp)

    # 2. Build dispatcher (BFS+DFS+hyperedge full pipeline; A6 can disable
    # hyperedge via --no-hyperedge)
    dispatcher = _build_dispatcher_for_builder(
        builder, shared_nlp=shared_nlp, use_hyperedge=use_hyperedge,
    )

    # 3. Run dispatch → evidence chain text
    dispatch_result = dispatcher.run(query=question)
    dispatch_dict = dispatch_result.to_dict()
    chain_text = _chain_text_from_dispatch(dispatch_dict)

    # Also retrieve baseline-style sentence list for retrieval-metric scoring
    # (we still use the original retriever for metric alignment with A1/A4).
    from src.retriever import Retriever
    base_retriever = Retriever(builder)
    base_result = base_retriever.retrieve(question)
    base_chains = base_result.get("evidence_chains", [])[:5]

    # Map dispatcher sentence_ids to texts (for metric scoring)
    dispatched_sent_ids = set()
    if dispatch_result.dual is not None:
        dispatched_sent_ids.update(dispatch_result.dual.path_sentence_ids or [])

    # Build a flat ordered list of sentence texts (dispatcher order first,
    # then baseline fillers) for the LLM prompt.  We pass this via
    # evidence_chain={"sentences":[{"text":..}], "edges":[...]} so the LLM
    # sees the dispatcher's curated chain first, then the baseline fillers
    # up to ~10 total.
    registry = builder.sentence_registry
    sent_texts: List[str] = []
    edges: List[str] = []
    seen_sids: set = set()
    for sid in list(dispatched_sent_ids):
        if sid in seen_sids:
            continue
        seen_sids.add(sid)
        node = registry.get_node(sid)
        if node is not None and node.text:
            sent_texts.append(node.text)
            edges.append("dispatch")
    for chain in base_chains:
        for sid in chain.get("sentence_ids", []):
            if sid in seen_sids:
                continue
            seen_sids.add(sid)
            node = registry.get_node(sid)
            if node is not None and node.text:
                sent_texts.append(node.text)
                edges.append(chain.get("primary_edge", "") or "base")
            if len(sent_texts) >= 10:
                break
        if len(sent_texts) >= 10:
            break

    latency_ms = (time.time() - t0) * 1000

    # 4. LLM answer (strict; no soft post-processing)
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
        # A5: strict HotpotQA scoring.  No _coref_pick / _short_answer_gating /
        # _strip_parens / _strip_colon_prefix — just SQuAD normalize inside
        # _em_score / _f1_score.
        em = _em_score(gen_answer, gold_answer)
        f1 = _f1_score(gen_answer, gold_answer)

    return {
        "qid": qid,
        "question": question,
        "gold_answer": gold_answer,
        "dispatch_path": dispatch_dict.get("path", ""),
        "dispatch_chain_text": chain_text,
        "dispatch_sentence_count": len(dispatched_sent_ids),
        "llm_context_sentences": sent_texts,
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
    parser.add_argument("--no-hyperedge", action="store_true",
                        help="A6: disable hyperedge track; BFS+DFS only.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--data-dir", default="eval/data")
    args = parser.parse_args()

    # Load data (same loader as hotpot_eval)
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
    detail_path = out_dir / f"dispatcher_strict_{args.split}_{len(hotpot_data)}_seed{args.seed}_{timestamp}.json"
    summary_path = out_dir / f"dispatcher_strict_{args.split}_{len(hotpot_data)}_seed{args.seed}_{timestamp}_summary.json"

    all_results: List[Dict[str, Any]] = []
    ems: List[float] = []
    f1s: List[float] = []

    log.info("Dispatcher strict eval start: n=%d use_llm=%s", len(hotpot_data), args.use_llm)
    for i, qdata in enumerate(hotpot_data):
        result = _evaluate_one(
            qdata, shared_nlp, args.use_llm, use_hyperedge=not args.no_hyperedge,
        )
        all_results.append(result)
        if args.use_llm:
            ems.append(float(result["em"]))
            f1s.append(float(result["f1"]))
        if not args.quiet:
            log.info(
                "[%d/%d] qid=%s path=%s em=%s f1=%.3f lat=%.1fms",
                i + 1, len(hotpot_data), result["qid"],
                result.get("dispatch_path", "?"),
                "Y" if result["em"] else "N",
                result["f1"],
                result["latency_ms"],
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
        "use_hyperedge": not args.no_hyperedge,
        "path_counts": {},
    }
    if args.use_llm:
        summary["em"] = float(np.mean(ems)) if ems else 0.0
        summary["f1"] = float(np.mean(f1s)) if f1s else 0.0
    # path distribution
    for r in all_results:
        p = r.get("dispatch_path", "?")
        summary["path_counts"][p] = summary["path_counts"].get(p, 0) + 1

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info("─" * 60)
    log.info("DISPATCHER STRICT SUMMARY (n=%d seed=%d use_hyperedge=%s)",
             len(all_results), args.seed, not args.no_hyperedge)
    if args.use_llm:
        log.info("  EM = %.3f", summary["em"])
        log.info("  F1 = %.3f", summary["f1"])
    log.info("  Path distribution: %s", summary["path_counts"])
    log.info("Detail → %s", detail_path)
    log.info("Summary → %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
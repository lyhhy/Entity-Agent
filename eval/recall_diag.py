"""
Recall@k diagnostic: does the dispatcher actually surface the
gold supporting sentences from HotpotQA?

For each question in a small sample, build the graph, dispatch, then
check whether any of the top-k ranked sentences matches a gold
supporting-fact sentence (by normalized text).

Three recall numbers are reported per question:
  * recall@1: top-1 sentence is a gold support
  * recall@3: any of the top-3 sentences is a gold support
  * recall@5: any of the top-5 sentences is a gold support

We also report the *joint* recall (both gold supports covered) and
the partial recall (at least one).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure project root on path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import spacy  # noqa: E402

from eval.quick20_eval import build_graph_from_hotpot, build_text_to_flat_idx  # noqa: E402
from src.dispatcher import build_dispatcher  # noqa: E402
from src.hyper_edge import HyperEdgeRegistry  # noqa: E402

log = logging.getLogger("recall_diag")


# ---------------------------------------------------------------------------
# Text normalization for sentence comparison.
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s or "").strip().lower()


def _gold_sentence_texts(qdata: Dict[str, Any]) -> List[str]:
    """Return the gold supporting-fact sentence strings."""
    out: List[str] = []
    ctx_dict: Dict[str, List[str]] = {}
    for title, sents in qdata.get("context", []):
        ctx_dict[title] = list(sents)
    for title, sent_idx in qdata.get("supporting_facts", []):
        sents = ctx_dict.get(title) or []
        if 0 <= sent_idx < len(sents):
            out.append(sents[sent_idx])
    return out


def _gather_ranked_sentences(dispatch_result) -> List[str]:
    """Collect every sentence the dispatcher surfaced, ordered by rank."""
    ordered: List[str] = []
    seen: Set[str] = set()

    def add(text: str) -> None:
        n = _norm(text)
        if n and n not in seen:
            seen.add(n)
            ordered.append(text)

    # Path = dual → use the path sentences in order.
    dual = getattr(dispatch_result, "dual", None)
    if dual is not None and getattr(dual, "path_sentence_ids", None):
        # We don't have direct text access here; reconstruct from registry.
        # The caller pre-loads the builder, so we do nothing here.
        # Instead, look at sent_ids list length to know "how many surfaced".
        pass

    # Concept path
    concept = getattr(dispatch_result, "concept", None)
    if concept is not None and getattr(concept, "final_order", None):
        for c in concept.final_order:
            add(getattr(c, "text", "") or "")

    return ordered


def _gather_ranked_sentence_ids(dispatch_result) -> List[str]:
    """All sentence IDs the dispatcher touched, in dispatch order."""
    ids: List[str] = []
    seen: Set[str] = set()
    dual = getattr(dispatch_result, "dual", None)
    if dual is not None and getattr(dual, "path_sentence_ids", None):
        for sid in dual.path_sentence_ids:
            if sid not in seen:
                seen.add(sid)
                ids.append(sid)
    concept = getattr(dispatch_result, "concept", None)
    if concept is not None and getattr(concept, "final_order", None):
        for c in concept.final_order:
            sid = getattr(c, "sentence_id", None)
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return ids


# ---------------------------------------------------------------------------
# Main recall computation.
# ---------------------------------------------------------------------------
def recall_for_question(
    qdata: Dict[str, Any],
    builder,
    registry,
    nlp,
    client,
) -> Dict[str, Any]:
    question = qdata["question"]
    gold_sents = _gold_sentence_texts(qdata)
    gold_norm = {_norm(s) for s in gold_sents if s}

    # Build dispatcher fresh per question so state is clean.
    dispatcher = build_dispatcher(
        builder,
        nlp=nlp,
        hyperedge_registry=registry,
        client=client,
    )
    dispatch_result = dispatcher.run(question)

    # Collect ranked sentences (as text) from the dispatched answer
    # order.  Dual path sentences first (in DFS order), then concept
    # reranked items.
    ranked: List[str] = []
    seen: Set[str] = set()

    def add_text(t: str) -> None:
        n = _norm(t)
        if n and n not in seen:
            seen.add(n)
            ranked.append(t)

    # Dual: take sentence texts via builder's registry.
    dual = dispatch_result.dual
    if dual is not None:
        for sid in dual.path_sentence_ids or []:
            node = builder.sentence_registry.get_node(sid)
            if node is not None:
                add_text(node.text)

    # Concept
    concept = dispatch_result.concept
    if concept is not None:
        for c in concept.final_order or []:
            add_text(c.text or "")

    # Recall@k on text overlap.
    def _hit(topk: int) -> int:
        return 1 if any(_norm(s) in gold_norm for s in ranked[:topk]) else 0

    # Joint recall: all gold sents surfaced.
    ranked_norm = {_norm(s) for s in ranked}
    joint = (
        1
        if gold_norm and all(g in ranked_norm for g in gold_norm)
        else 0
    )
    partial = 1 if any(g in ranked_norm for g in gold_norm) else 0

    # Path chosen
    path = dispatch_result.path

    # Number of gold sents hit in top-k
    def _count_in(topk: int) -> int:
        return sum(1 for s in ranked[:topk] if _norm(s) in gold_norm)

    return {
        "qid": qdata.get("_id", ""),
        "question": question,
        "path": path,
        "num_surfaced": len(ranked),
        "num_gold": len(gold_norm),
        "recall@1": _hit(1),
        "recall@3": _hit(3),
        "recall@5": _hit(5),
        "recall@10": _hit(10),
        "joint_recall": joint,
        "partial_recall": partial,
        "hit_count@5": _count_in(5),
        "gold_sents": gold_sents,
        "top5": ranked[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/data/hotpot_dev_distractor_v1.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading %s", args.data)
    data = json.load(open(args.data, encoding="utf-8"))
    log.info("Total questions: %d", len(data))

    rng_sample = data[: args.n]

    nlp = spacy.load("en_core_web_lg")

    client = None
    if args.use_llm:
        try:
            from src.llm_client import DeepSeekClient
            client = DeepSeekClient()
            log.info("LLM client enabled: %s", "yes" if client.enabled else "no")
        except Exception as e:
            log.warning("LLM init failed: %s", e)

    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(rng_sample, 1):
        builder, _sentences, _t, _p = build_graph_from_hotpot(q, nlp)
        registry = HyperEdgeRegistry()
        row = recall_for_question(q, builder, registry, nlp, client)
        rows.append(row)

        if args.verbose or i <= 5:
            log.info(
                "  [%2d/%-2d] %s path=%s surfaced=%d r@1=%d r@3=%d r@5=%d joint=%d partial=%d",
                i, args.n, row["qid"], row["path"], row["num_surfaced"],
                row["recall@1"], row["recall@3"], row["recall@5"],
                row["joint_recall"], row["partial_recall"],
            )
            log.info("    Q: %s", row["question"][:90])
            for g in row["gold_sents"][:2]:
                log.info("    G: %s", (g or "")[:120])
            for s in row["top5"][:3]:
                log.info("    T: %s", (s or "")[:120])

    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / max(n, 1)  # noqa: E731
    log.info("=" * 60)
    log.info("Recall diagnostic (n=%d)", n)
    log.info("  recall@1        = %.3f", avg("recall@1"))
    log.info("  recall@3        = %.3f", avg("recall@3"))
    log.info("  recall@5        = %.3f", avg("recall@5"))
    log.info("  recall@10       = %.3f", avg("recall@10"))
    log.info("  partial recall  = %.3f", avg("partial_recall"))
    log.info("  joint recall    = %.3f", avg("joint_recall"))
    log.info("  hit_count@5     = %.2f / %d",
             sum(r["hit_count@5"] for r in rows) / max(n, 1),
             max((r["num_gold"] for r in rows), default=0))
    log.info("  avg surfaced    = %.1f",
             sum(r["num_surfaced"] for r in rows) / max(n, 1))

    by_path: Dict[str, int] = {}
    for r in rows:
        by_path[r["path"]] = by_path.get(r["path"], 0) + 1
    log.info("  path breakdown  = %s", by_path)

    # Save results
    out_path = Path("eval/results/recall_diag.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "n": n,
                "summary": {
                    "recall@1": avg("recall@1"),
                    "recall@3": avg("recall@3"),
                    "recall@5": avg("recall@5"),
                    "recall@10": avg("recall@10"),
                    "partial_recall": avg("partial_recall"),
                    "joint_recall": avg("joint_recall"),
                },
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("Results → %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

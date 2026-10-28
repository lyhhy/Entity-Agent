"""
Why is DualEntityRetriever's `accepted` always False?
Diagnose each stage:
  1. Did the analyzer extract two entities? (entity_count)
  2. Did each entity match at least one container?
  3. Were intermediate containers discovered?
  4. Did the DFS find candidate paths?
  5. What was the top confidence? Was it below threshold?
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import spacy  # noqa: E402

from eval.quick20_eval import build_graph_from_hotpot  # noqa: E402
from src.dispatcher import build_dispatcher  # noqa: E402
from src.hyper_edge import HyperEdgeRegistry  # noqa: E402

log = logging.getLogger("dual_diag")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _gold_sentence_texts(qdata):
    out = []
    ctx_dict = {t: list(s) for t, s in qdata.get("context", [])}
    for t, idx in qdata.get("supporting_facts", []):
        sents = ctx_dict.get(t) or []
        if 0 <= idx < len(sents):
            out.append(sents[idx])
    return out


def diag_one(qdata, nlp):
    """Run the dispatcher and instrument the dual track."""
    question = qdata["question"]
    gold = _gold_sentence_texts(qdata)
    gold_norm = {_norm(g) for g in gold if g}

    builder, _, _, _ = build_graph_from_hotpot(qdata, nlp)
    registry = HyperEdgeRegistry()
    dispatcher = build_dispatcher(builder, nlp=nlp, hyperedge_registry=registry)

    # Manually walk the same steps the dispatcher does.
    from src.query_analyzer import make_query_analyzer
    analyzer = make_query_analyzer(client=None)
    analysis = analyzer(question)

    info: Dict[str, Any] = {
        "qid": qdata.get("_id", ""),
        "question": question,
        "type": qdata.get("type", ""),
        "level": qdata.get("level", ""),
        "gold_count": len(gold_norm),
        "entity_count": analysis.entity_count,
        "entity_names": list(analysis.entity_names),
        "answer_type": "",
    }

    if analysis.entity_count < 2:
        info["reason"] = "analyzer_found_<2_entities"
        return info

    a, b = analysis.entity_names[0], analysis.entity_names[1]
    dual_ret = dispatcher.dual

    containers_a = dual_ret._containers_for_entity(a)
    containers_b = dual_ret._containers_for_entity(b)
    info["containers_a"] = len(containers_a)
    info["containers_b"] = len(containers_b)
    info["container_a_ids"] = list(containers_a[:5])
    info["container_b_ids"] = list(containers_b[:5])
    # Surface their canonical entities for debugging
    info["container_a_entities"] = [
        (dual_ret.cm.get_container(cid).entity
         if dual_ret.cm.get_container(cid) else None)
        for cid in containers_a[:5]
    ]
    info["container_b_entities"] = [
        (dual_ret.cm.get_container(cid).entity
         if dual_ret.cm.get_container(cid) else None)
        for cid in containers_b[:5]
    ]

    if not containers_a or not containers_b:
        info["reason"] = "entity_not_linked_to_container"
        info["reason_detail"] = (
            f"a={a!r}→{len(containers_a)} "
            f"b={b!r}→{len(containers_b)}"
        )
        return info

    # Try the real retriever.
    result = dual_ret.retrieve(query=question, entity_a=a, entity_b=b, analysis=analysis)
    info["path_sentence_ids"] = list(result.path_sentence_ids)
    info["path_container_ids"] = list(result.path_container_ids)
    info["intermediate_container_ids"] = list(result.intermediate_container_ids)
    info["confidence"] = float(result.confidence)
    info["accepted"] = bool(result.accepted)

    # Text of surfaced sentences
    surfaced_text = []
    for sid in result.path_sentence_ids:
        n = builder.sentence_registry.get_node(sid)
        if n is not None:
            surfaced_text.append(n.text)
    info["surfaced_text"] = surfaced_text
    info["surfaced_n"] = len(surfaced_text)

    # Did it cover gold?
    ranked_norm = {_norm(t) for t in surfaced_text}
    info["gold_hit"] = sum(1 for g in gold_norm if g in ranked_norm)
    info["joint_gold_hit"] = (
        int(all(g in ranked_norm for g in gold_norm)) if gold_norm else 0
    )

    if not result.path_sentence_ids:
        info["reason"] = "no_path_found"
    elif not result.accepted:
        info["reason"] = "confidence_below_threshold"
    else:
        info["reason"] = "accepted"

    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/data/hotpot_dev_distractor_v1.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=None,
                    help="If set, recompute accepted manually with this threshold")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    data = json.load(open(args.data, encoding="utf-8"))
    nlp = spacy.load("en_core_web_lg")

    sample = data[: args.n]
    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(sample, 1):
        info = diag_one(q, nlp)
        rows.append(info)
        if i <= 5 or args.n <= 5:
            log.info(
                "[%2d] entities=%s reason=%s confidence=%.3f "
                "surfaced=%d gold_hit=%d/%d",
                i, info.get("entity_count", 0), info.get("reason", ""),
                info.get("confidence", 0.0),
                info.get("surfaced_n", 0),
                info.get("gold_hit", 0), info.get("gold_count", 0),
            )
            log.info("     A-containers: %s", info.get("container_a_entities"))
            log.info("     B-containers: %s", info.get("container_b_entities"))

    # Aggregate
    reasons: Dict[str, int] = {}
    for r in rows:
        reasons[r.get("reason", "unknown")] = reasons.get(r.get("reason", ""), 0) + 1
    log.info("=" * 60)
    log.info("Dual-entity failure breakdown (n=%d)", len(rows))
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        log.info("  %-40s  %d", k, v)

    # Examples for each reason
    for r in rows[:3]:
        log.info("-" * 60)
        log.info("Q: %s", r["question"])
        log.info("  type=%s level=%s", r["type"], r["level"])
        log.info("  entities=%s", r.get("entity_names"))
        log.info("  reason=%s  conf=%.3f", r.get("reason"), r.get("confidence", 0.0))
        if r.get("reason_detail"):
            log.info("  %s", r["reason_detail"])
        if r.get("container_a_entities"):
            log.info("  a-containers: %s", r.get("container_a_entities"))
        if r.get("container_b_entities"):
            log.info("  b-containers: %s", r.get("container_b_entities"))
        log.info("  path_sentence_ids: %s", r.get("path_sentence_ids"))
        log.info("  surfaced: %s", [t[:80] for t in r.get("surfaced_text", [])])

    out = Path("eval/results/dual_diag.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("→ %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

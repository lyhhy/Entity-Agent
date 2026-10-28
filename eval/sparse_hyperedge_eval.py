"""
Sparse Attribute Graph Hyperedge Runner
======================================

Runs the 20-question evaluation with **sparse attribute triggers**:
for each paragraph (chunk), we look at the `attribute_mounts` of every
sentence node, count attribute occurrences per container, and feed
`attribute_triggers={cid: [attr, ...]}` into
`HyperEdgeRegistry.build_hyperedges_for_chunk`.

This is the "Phase-Sparse" path described in hyper_edge.py docstring
that the legacy / offline builders never exercised.

Output:
    eval/results/clean_results_sparse_hyperedge_test.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import spacy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse utilities from quick20_eval (F1, normalize, etc.)
import eval.quick20_eval as q20  # noqa: E402
from src.hyper_edge import HyperEdgeRegistry  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sparse_he")
_F1 = q20._f1_score
_BUILD_GRAPH = q20.build_graph_from_hotpot
_EXTRACT_ANSWER = q20.extract_answer_from_dispatch
_NORMALIZE = q20._normalize_answer


# ---------------------------------------------------------------------------
# Sparse trigger builder
# ---------------------------------------------------------------------------
def build_sparse_triggers(
    builder,
    cm,
    paragraph_cids: List[str],
) -> Dict[str, List[str]]:
    """
    For each container in the paragraph, collect the attribute names that
    its sentences actually mount (from `node.attribute_mounts`).

    Returns:
        {cid: sorted_list_of_attribute_names}

    Strategy:
      - Walk every sentence node in the builder.
      - For each node, for every mount, if the mount's container is in
        this paragraph → record (cid, attr_name).
      - Drop the "other" bucket (it's a fallback that would defeat
        the sparsity).
    """
    triggers: Dict[str, set] = {cid: set() for cid in paragraph_cids}
    for sid, node in builder.sentence_registry._nodes.items():
        node_cid_set = set(node.containers)
        for mount in getattr(node, "attribute_mounts", []) or []:
            attr = mount.attribute_name
            if not attr or attr == "other":
                continue
            # Find which container in this paragraph owns this mount.
            for cid in node_cid_set & set(paragraph_cids):
                # The mount is keyed on entity name, not cid; we can
                # only be sure of (cid, attr) when the mount's entity
                # matches the container's entity. But entity matching
                # is already encoded in node.containers itself, so we
                # accept any mount whose container is in the paragraph.
                triggers[cid].add(attr)
    return {cid: sorted(s) for cid, s in triggers.items() if s}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _safe_print(s: str) -> None:
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(s.encode("gbk", "replace").decode("gbk"), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/data/hotpot_dev_distractor_v1.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--use-llm", action="store_true",
                    help="Use LLM analyzer (otherwise keyword-only)")
    ap.add_argument("--tag", default="sparse_hyperedge_test")
    args = ap.parse_args()

    # ── Setup ────────────────────────────────────────────────────────
    with open(args.data, encoding="utf-8") as fh:
        all_data = json.load(fh)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(all_data)), min(args.n, len(all_data)))
    sample = [all_data[i] for i in indices]

    log_dir = ROOT / "eval/diag" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      "%H:%M:%S"))
    log.addHandler(fh)

    client = None
    if args.use_llm:
        from src.llm_client import DeepSeekClient
        client = DeepSeekClient()
        if not client.enabled:
            _safe_print("[FATAL] --use-llm but DEEPSEEK_API_KEY not set")
            return 1

    log.info("n=%d seed=%d use_llm=%s", args.n, args.seed, args.use_llm)
    _safe_print(f"[setup] n={args.n} seed={args.seed} use_llm={args.use_llm}\n")
    _safe_print("=" * 80)

    nlp = spacy.load("en_core_web_lg")

    ems: List[bool] = []
    f1s: List[float] = []
    r1s: List[int] = []
    r5s: List[int] = []
    r10s: List[int] = []
    records: List[Dict[str, Any]] = []
    sparse_stats: List[Dict[str, Any]] = []

    for i, qdata in enumerate(sample, 1):
        qid = qdata["_id"]
        question = qdata["question"]
        gold = qdata.get("answer", "")

        t0 = time.time()
        # ── Build graph ────────────────────────────────────────────
        builder, sentences, titles, para_sents, bridge_titles = \
            _BUILD_GRAPH(qdata, nlp)
        cm = builder.container_manager
        for c in cm._containers.values():
            cm.update_container_index(c.id)
        builder.compute_tfidf_index()

        # ── Sparse hyperedges ──────────────────────────────────────
        he = HyperEdgeRegistry(container_manager=cm)
        total_triggers = 0
        non_empty_para = 0
        for _p_title, _p_cids in (bridge_titles or {}).items():
            _p_containers = [
                cm._containers.get(_cid)
                for _cid in _p_cids
                if cm._containers.get(_cid) is not None
            ]
            if not _p_containers:
                continue
            non_empty_para += 1
            triggers = build_sparse_triggers(builder, cm, _p_cids)
            total_triggers += sum(len(v) for v in triggers.values())
            he.build_hyperedges_for_chunk(
                chunk_id=str(_p_title),
                containers=_p_containers,
                time_word="",
                paragraph_title=str(_p_title),
                attribute_triggers=triggers,
            )
        sparse_stats.append({
            "qid": qid,
            "n_paragraphs": non_empty_para,
            "total_triggers": total_triggers,
        })
        log.info("[%d] %s: %d paragraphs, %d sparse triggers",
                 i, qid[:8], non_empty_para, total_triggers)

        # ── Build pipeline ─────────────────────────────────────────
        from src.dispatcher import build_dispatcher
        disp = build_dispatcher(
            builder, client=client, nlp=nlp,
            hyperedge_registry=he,
            attr_soft_match=True,
        )

        res = disp.run(question, bridge_titles=bridge_titles)
        answer = _EXTRACT_ANSWER(res, builder, gold or "", nlp, client, question) if res else ""
        em_hit = answer.lower() == (gold or "").lower()
        f1 = _F1(answer, gold or "")
        latency_ms = (time.time() - t0) * 1000
        ems.append(em_hit); f1s.append(f1)

        # R@K
        gold_norm = _NORMALIZE(gold)
        final_sids: List[str] = []
        if res is not None and res.pipeline_result is not None:
            final_sids = list(
                getattr(res.pipeline_result, "step10_final_sids", []) or []
            )

        def _contains_gold(sid: str) -> bool:
            n = builder.sentence_registry.get_node(sid)
            return n is not None and gold_norm in _NORMALIZE(n.text or "")

        r1 = 1 if (final_sids and _contains_gold(final_sids[0])) else 0
        r5 = 1 if any(_contains_gold(s) for s in final_sids[:5]) else 0
        r10 = 1 if any(_contains_gold(s) for s in final_sids[:10]) else 0
        r1s.append(r1); r5s.append(r5); r10s.append(r10)

        records.append({
            "qid": qid, "question": question, "gold": gold, "answer": answer,
            "em": em_hit, "f1": f1, "r1": r1, "r5": r5, "r10": r10,
            "latency_ms": latency_ms,
            "final_sids": final_sids,
            "final_texts": [
                (builder.sentence_registry.get_node(s).text or "")
                for s in final_sids if builder.sentence_registry.get_node(s)
            ],
        })
        mark = "Y" if em_hit else "N"
        _safe_print(
            f"[{i:>2}/{args.n}] {qid[:8]}  EM={mark} F1={f1:.3f} "
            f"R@1={r1} R@5={r5} R@10={r10}  ({latency_ms:.0f}ms)"
        )
        _safe_print(f"  Q : {question}")
        _safe_print(f"  G : {gold}")
        _safe_print(f"  A : {answer}")
        # DEBUG: show what the LLM actually saw as final evidence
        for j, txt in enumerate(records[-1]["final_texts"]):
            mark = "★" if gold_norm in _NORMALIZE(txt) else " "
            _safe_print(f"  {mark} f{j}: {txt[:200]}")

    em_avg = sum(ems) / len(ems) if ems else 0
    f1_avg = sum(f1s) / len(f1s) if f1s else 0
    r1_avg = sum(r1s) / len(r1s) if r1s else 0
    r5_avg = sum(r5s) / len(r5s) if r5s else 0
    r10_avg = sum(r10s) / len(r10s) if r10s else 0
    _safe_print("=" * 80)
    _safe_print(
        f"SUMMARY  n={len(records)}  EM={em_avg:.3f}  F1={f1_avg:.3f}  "
        f"R@1={r1_avg:.3f}  R@5={r5_avg:.3f}  R@10={r10_avg:.3f}"
    )

    out = ROOT / f"eval/results/clean_results_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "n": len(records),
            "em": em_avg, "f1": f1_avg,
            "r1": r1_avg, "r5": r5_avg, "r10": r10_avg,
            "records": records,
            "sparse_trigger_stats": sparse_stats,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _safe_print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

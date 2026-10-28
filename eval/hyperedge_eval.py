"""
Hyperedge-augmented retrieval experiment for HotpotQA.

This script runs a side-by-side comparison on the same set of HotpotQA
questions:

    Variant A (baseline):
        Build the sentence-level logic graph, run the existing
        `Retriever.retrieve()` (CAUSAL/TEMPORAL/CONTRAST edges +
        SAME_ENTITY hop).  This is identical to what
        `eval/hotpot_eval.py` already does.

    Variant B (baseline + hyperedge):
        Same as A, but additionally:
          1. After the graph is built, build container-level
             hyperedges via `HyperEdgeRegistry.build_hyperedges_for_chunk`.
          2. From the question, extract the 2 most-likely-title
             entities (heuristic NER + score against container entity
             inventory).
          3. Run `DualEntityRetriever` to find a bridge sentence
             chain that connects the two entity containers via the
             hyperedge track.
          4. Append the bridge chain's sentences to the baseline
             candidate pool **before** the cross-encoder rerank, so
             the downstream ranker can promote them if they're truly
             relevant.

We deliberately keep the comparison focused on retrieval metrics
(R@1/R@2/R@5/MRR, sentence + paragraph) — no LLM answer generation —
so the experiment runs in ~minutes, not hours, and isolates the
hyperedge signal cleanly.

USAGE:
    python eval/hyperedge_eval.py --max-examples 50 --seed 7

Results are written to `eval/results/hyperedge_*_detail.json` and
`*_summary.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random as _random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make the project root importable when run as `python eval/hyperedge_eval.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.hotpot_eval import (
    build_graph,
    gold_supporting_ids,
    gold_supporting_titles,
    hotpot_to_sentences,
    load_hotpot,
    _build_text_to_flat_idx,
    _sentence_to_paragraph_title,
)
from src.retriever import Retriever
from src.hyper_edge import HyperEdgeRegistry, parse_time_word
from src.dual_entity_retriever import DualEntityRetriever
from src.graph_builder import _load_spacy_shared


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hyperedge_eval")


# ─────────────────────────────────────────────────────────────────────
# Helpers: question → top-2 entities in graph
# ─────────────────────────────────────────────────────────────────────

# Multi-word titles should be matched as contiguous phrases. We sort
# candidate titles by length (longest first) and try them in order so
# that longer titles win over shorter substrings ("American Beauty
# (1999 film)" before "American Beauty").
#
# FIX-1b [entity-name dedup]: strip parenthetical disambiguators and
# "group N" suffixes before the "already chosen?" check.  Otherwise
# "Falling Stars (video game) (group 2)" (the bucket-A entity) and
# "Falling Stars" (the bucket-B entity) would both be accepted, and
# the bridge logic would have to deal with two distinct containers
# that the question's surface form doesn't differentiate between.
def _normalize_entity_label(label: str) -> str:
    """
    Reduce an entity string to its "core noun phrase" for the
    purpose of de-duplicating multi-form variants of the same
    referent.

    Examples
    --------
    >>> _normalize_entity_label("Falling Stars (video game) (group 2)")
    'falling stars'
    >>> _normalize_entity_label("American Beauty (1999 film)")
    'american beauty'
    >>> _normalize_entity_label("John Francis Kelly")
    'john francis kelly'
    """
    if not label:
        return ""
    s = label.lower().strip()
    # Strip one or more parenthesised tails.
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    # Collapse "group N" / "part N" / "volume N" suffixes that show
    # up when multiple buckets in the corpus share a title.
    s = re.sub(r"\b(group|part|volume|vol|chapter|section)\s+\d+\b",
               " ", s)
    # Strip year-style tail: "American Beauty (1999 film)" → handled
    # by the parenthetical strip above; this is a backstop.
    s = re.sub(r"\b\d{4}\b", " ", s)
    # Collapse whitespace.
    s = " ".join(s.split())
    return s


def _select_two_entities_from_question(
    question: str,
    container_manager,
) -> List[str]:
    """
    Pick up to 2 entities from the question that match a real
    container. Strategy:
      1.  NER-light: split on word boundaries, lowercase each token
          sequence of length >= 2 that overlaps with a known
          container entity.
      2.  Prefer the longest match.
    The runner returns 0, 1, or 2 strings (entity canonical names).

    FIX-1b: variants of the same entity that only differ in
    disambiguating parentheses / "group N" tails collapse to one.
    """
    q_lower = question.lower()
    # Collect candidate container entities (longest first).
    candidates = []
    seen = set()
    for c in container_manager.get_all_containers():
        e = (c.entity or "").strip()
        if not e or e in seen:
            continue
        seen.add(e)
        candidates.append(e)
    candidates.sort(key=lambda x: -len(x))

    # Precompute normalised labels for fast subspan check.
    norm_map: Dict[str, str] = {c: _normalize_entity_label(c) for c in candidates}

    chosen: List[str] = []
    used_norm: set = set()  # normalised forms already chosen
    for cand in candidates:
        if cand.lower() not in q_lower:
            continue
        cand_norm = norm_map[cand]
        if not cand_norm:
            continue
        # FIX-1b [dedup — equivalent forms]: skip if another
        # already-chosen entity has the same normalised form
        # ("Falling Stars (group 2)" vs "Falling Stars" both → "falling
        # stars").  This is the only relaxation vs the original.
        if cand_norm in used_norm:
            continue
        # [original Fix1 — subspan filter]: also skip if the
        # candidate's normalised form is a strict subspan of an
        # already-chosen entity's normalised form (e.g. don't pick
        # "derrickson" when "scott derrickson" is already in).  The
        # *reverse* direction is kept: if the candidate is the
        # LONGER one and the previously chosen is a strict subspan
        # of the candidate, we drop the shorter one in favour of the
        # longer one (and continue to the next candidate).
        subspan_of_prior = any(
            cand_norm in prior_norm and prior_norm != cand_norm
            for prior_norm in used_norm
        )
        if subspan_of_prior:
            continue
        # Reverse: a previously chosen entity is a strict subspan of
        # the candidate.  Replace the shorter one with the longer
        # one to prefer the more specific label.
        reverse_drop: Optional[int] = None
        for idx, prior in enumerate(chosen):
            prior_norm = norm_map[prior]
            if prior_norm in cand_norm and prior_norm != cand_norm:
                reverse_drop = idx
                break
        if reverse_drop is not None:
            dropped = chosen[reverse_drop]
            used_norm.discard(norm_map[dropped])
            chosen[reverse_drop] = cand
            used_norm.add(cand_norm)
            continue
        chosen.append(cand)
        used_norm.add(cand_norm)
        if len(chosen) >= 2:
            break
    return chosen


# ─────────────────────────────────────────────────────────────────────
# Bridge extraction: turn DualEntityRetriever output into ranked sids
# ─────────────────────────────────────────────────────────────────────

def _bridge_to_sentence_chain(
    der: DualEntityRetriever,
    builder,
    question: str,
    entity_a: str,
    entity_b: str,
) -> List[Tuple[str, float]]:
    """
    Run DualEntityRetriever and turn its result into a list of
    (sentence_id, score) tuples. The score is `chain.confidence` if
    accepted, else 0.5 (so the bridge still gets exposure to the
    downstream ranker but is unlikely to beat a confident baseline).
    """
    chain = der.retrieve(question, entity_a, entity_b, analysis=None)
    if not chain.path_sentence_ids:
        return []
    # Confidence: accepted chains use their score, FALLBACK_BRIDGE / unaccepted
    # ones get 0.5 so they can still lift recall but won't dominate.
    base_score = chain.confidence if chain.accepted else 0.5
    out = [(sid, base_score) for sid in chain.path_sentence_ids]
    return out


# ─────────────────────────────────────────────────────────────────────
# Merge baseline chains + bridge chain into a unified ranked list
# ─────────────────────────────────────────────────────────────────────

def _merge_with_bridge(
    baseline_chains: List[Dict],
    bridge_sids_with_score: List[Tuple[str, float]],
    builder,
) -> List[Dict]:
    """
    Add the bridge sentences as an *additional* evidence chain that
    sits at the end of the baseline chain list. We don't disturb the
    baseline ordering — we just expose more sentences to the downstream
    rerank by appending a final chain tagged as 'bridge'.
    """
    if not bridge_sids_with_score:
        return baseline_chains
    bridge_sentences = []
    avg_score = float(np.mean([s for _, s in bridge_sids_with_score])) if bridge_sids_with_score else 0.0
    for sid, sc in bridge_sids_with_score:
        node = builder.sentence_registry.get_node(sid)
        if node is None:
            continue
        bridge_sentences.append({
            "id": sid,
            "text": node.text,
            "entities": list(node.entities),
            "score": sc,
        })
    if not bridge_sentences:
        return baseline_chains
    new_chain = {
        "sentences": bridge_sentences,
        "edges": [],
        "score": avg_score,
        "source": "hyperedge_bridge",
    }
    return list(baseline_chains) + [new_chain]


# ─────────────────────────────────────────────────────────────────────
# Compute retrieval metrics on a chain list
# ─────────────────────────────────────────────────────────────────────

def _compute_metrics(
    chains: List[Dict],
    builder,
    question_data: Dict[str, Any],
    gold_ids: set,
    gold_titles: set,
    max_chains: int = 5,
) -> Dict[str, Any]:
    """
    Same metric set as hotpot_eval.retrieve_and_score but parameterised
    over the chains list (so we can score baseline and
    baseline+bridge on identical ground truth).
    """
    text_to_flat = _build_text_to_flat_idx(question_data)
    seen_ids = set()
    retrieved_ids = []
    retrieved_flat_idxs: List[int] = []
    for chain in chains[:max_chains]:
        for sent_info in chain.get("sentences", []):
            sid = sent_info.get("id", "")
            if not sid or sid in seen_ids:
                continue
            seen_ids.add(sid)
            retrieved_ids.append(sid)
            node = builder.sentence_registry.get_node(sid)
            if node is not None:
                key = " ".join((node.text or "").split())
                flat_idx = text_to_flat.get(key)
                if flat_idx is not None:
                    retrieved_flat_idxs.append(flat_idx)

    retrieved_texts = []
    for sid in retrieved_ids:
        node = builder.sentence_registry.get_node(sid)
        retrieved_texts.append(node.text if node else "")

    rank_of_first_hit = None
    for rank, flat_idx in enumerate(retrieved_flat_idxs, start=1):
        if flat_idx in gold_ids:
            rank_of_first_hit = rank
            break

    hit_at_1 = bool(retrieved_flat_idxs and retrieved_flat_idxs[0] in gold_ids)
    hit_at_2 = bool(
        retrieved_flat_idxs and any(s in gold_ids for s in retrieved_flat_idxs[:2])
    )
    hit_at_5 = bool(
        retrieved_flat_idxs and any(s in gold_ids for s in retrieved_flat_idxs[:5])
    )
    mrr = 1.0 / rank_of_first_hit if rank_of_first_hit else 0.0

    title_seq: List[str] = []
    seen_titles: set = set()
    for sid in retrieved_ids:
        title = _sentence_to_paragraph_title(builder, sid)
        if title and title not in seen_titles:
            seen_titles.add(title)
            title_seq.append(title)

    para_rank = None
    for rank, t in enumerate(title_seq, start=1):
        if t in gold_titles:
            para_rank = rank
            break
    para_hit_at_1 = bool(title_seq and title_seq[0] in gold_titles)
    para_hit_at_2 = bool(title_seq and any(t in gold_titles for t in title_seq[:2]))
    para_hit_at_5 = bool(title_seq and any(t in gold_titles for t in title_seq[:5]))
    para_mrr = 1.0 / para_rank if para_rank else 0.0

    return {
        "retrieved_ids": retrieved_ids,
        "retrieved_flat_idxs": retrieved_flat_idxs,
        "retrieved_titles": title_seq,
        "retrieved_sentences": retrieved_texts,
        "recall_at_1": float(hit_at_1),
        "recall_at_2": float(hit_at_2),
        "recall_at_5": float(hit_at_5),
        "mrr": float(mrr),
        "para_recall_at_1": float(para_hit_at_1),
        "para_recall_at_2": float(para_hit_at_2),
        "para_recall_at_5": float(para_hit_at_5),
        "para_mrr": float(para_mrr),
        "n_chains": min(len(chains), max_chains),
    }


# ─────────────────────────────────────────────────────────────────────
# Per-question: run both variants, return side-by-side metric dict
# ─────────────────────────────────────────────────────────────────────

def _process_one(
    qdata: Dict[str, Any],
    shared_nlp,
    max_chains: int,
) -> Dict[str, Any]:
    qid = qdata.get("_id", "?")
    question = qdata.get("question", "")
    gold_ids = gold_supporting_ids(qdata)
    gold_titles = gold_supporting_titles(qdata)

    # ── Build graph ────────────────────────────────────────────────
    sentences = hotpot_to_sentences(qdata, nlp=shared_nlp)
    builder = build_graph(sentences, shared_nlp=shared_nlp)
    cm = builder.container_manager

    # Make sure each container has its `entity_inventory` populated
    # (otherwise HyperEdgeRegistry and DualEntityRetriever both skip
    # them as "no inventory"). update_container_index already does
    # this; we just need to make sure it has run for every container.
    for c in cm.get_all_containers():
        cm.update_container_index(c.id)

    # ── Baseline retrieval ─────────────────────────────────────────
    retriever = Retriever(builder)
    baseline_result = retriever.retrieve(question)
    baseline_chains = baseline_result.get("evidence_chains", [])[:max_chains]
    base_metrics = _compute_metrics(
        baseline_chains, builder, qdata, gold_ids, gold_titles, max_chains,
    )

    # ── Hyperedge path construction ─────────────────────────────────
    he_metrics = {}
    augmented_chains = baseline_chains
    bridge_info: Dict[str, Any] = {
        "ran": False,
        "n_entities_matched": 0,
        "bridge_sentence_count": 0,
        "n_hyperedges": 0,
    }

    # 1. Build hyperedges for this chunk (= the whole question's
    #    context). Per spec, one chunk = one hyperedge build call.
    registry = HyperEdgeRegistry(container_manager=cm)
    containers_all = cm.get_all_containers()
    # `time_word` defaults to ""; if you want to inject a date,
    # do it here.
    he_list = registry.build_hyperedges_for_chunk(
        chunk_id=qid,
        containers=containers_all,
        time_word="",
        doc_ids=[qid],
    )
    bridge_info["n_hyperedges"] = len(he_list)

    # 2. Pick 2 entities from the question.
    ents = _select_two_entities_from_question(question, cm)
    bridge_info["n_entities_matched"] = len(ents)

    if len(ents) >= 2 and he_list:
        # 3. Run DualEntityRetriever
        der = DualEntityRetriever(
            graph=builder,
            container_manager=cm,
            hyperedge_registry=registry,
            nlp=shared_nlp,
        )
        bridge_sids = _bridge_to_sentence_chain(
            der, builder, question, ents[0], ents[1],
        )
        bridge_info["bridge_sentence_count"] = len(bridge_sids)
        bridge_info["ran"] = True

        # 4. Merge into the baseline chain list.
        augmented_chains = _merge_with_bridge(
            baseline_chains, bridge_sids, builder,
        )
        he_metrics = _compute_metrics(
            augmented_chains, builder, qdata, gold_ids, gold_titles, max_chains,
        )
    else:
        # Hyperedge path didn't fire — same metrics as baseline
        he_metrics = dict(base_metrics)
        he_metrics["hyperedge_skipped_reason"] = (
            "fewer_than_2_entities_matched" if len(ents) < 2
            else "no_hyperedges"
        )

    return {
        "qid": qid,
        "question": question,
        "gold_answer": qdata.get("answer", ""),
        "gold_titles": sorted(gold_titles),
        "question_entities": ents,
        "baseline": base_metrics,
        "hyperedge": he_metrics,
        "bridge_info": bridge_info,
    }


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def _find_data_file(data_dir: str, split: str) -> Optional[str]:
    filename_aliases = {"val": ["dev", "validation", "val"], "train": ["train"]}
    aliases = filename_aliases.get(split, [split])
    candidates = []
    for a in aliases:
        for ext in (".parquet", ".json"):
            for suffix in ("", "_v1", "_v1.1"):
                candidates.append(
                    os.path.join(data_dir, f"hotpot_{a}_distractor{suffix}{ext}")
                )
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hyperedge-augmented retrieval experiment"
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--data-dir", default="eval/data")
    parser.add_argument("--max-chains", type=int, default=5,
                        help="Max evidence chains per retrieval (default 5).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data_file = _find_data_file(args.data_dir, args.split)
    if data_file is None:
        log.error("No HotpotQA file found in %s", args.data_dir)
        return 1
    log.info("Using data file: %s", data_file)

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

    log.info("Loaded %d questions (max=%d seed=%d)",
             len(hotpot_data), args.max_examples, args.seed)

    shared_nlp = _load_spacy_shared("en_core_web_lg")
    if shared_nlp is None:
        log.warning("spaCy model unavailable; degraded retrieval signals")

    all_results: List[Dict[str, Any]] = []
    base_r1, base_r2, base_r5, base_mrr = [], [], [], []
    base_p_r1, base_p_r2, base_p_r5, base_p_mrr = [], [], [], []
    he_r1, he_r2, he_r5, he_mrr = [], [], [], []
    he_p_r1, he_p_r2, he_p_r5, he_p_mrr = [], [], [], []
    n_hyperedge_fired = 0

    log.info("Starting side-by-side retrieval experiment...")
    t0 = time.time()
    for i, qdata in enumerate(hotpot_data):
        t_q = time.time()
        rec = _process_one(qdata, shared_nlp, args.max_chains)
        all_results.append(rec)

        b = rec["baseline"]
        h = rec["hyperedge"]
        if rec["bridge_info"]["ran"]:
            n_hyperedge_fired += 1

        for tgt, src in (
            (base_r1, b["recall_at_1"]), (base_r2, b["recall_at_2"]),
            (base_r5, b["recall_at_5"]), (base_mrr, b["mrr"]),
            (base_p_r1, b["para_recall_at_1"]),
            (base_p_r2, b["para_recall_at_2"]),
            (base_p_r5, b["para_recall_at_5"]),
            (base_p_mrr, b["para_mrr"]),
            (he_r1, h["recall_at_1"]), (he_r2, h["recall_at_2"]),
            (he_r5, h["recall_at_5"]), (he_mrr, h["mrr"]),
            (he_p_r1, h["para_recall_at_1"]),
            (he_p_r2, h["para_recall_at_2"]),
            (he_p_r5, h["para_recall_at_5"]),
            (he_p_mrr, h["para_mrr"]),
        ):
            tgt.append(src)

        if not args.quiet:
            elapsed = time.time() - t_q
            log.info(
                "  [%d/%d] qid=%s bridge=%s  base R@1=%.3f R@5=%.3f MRR=%.3f | "
                "he R@1=%.3f R@5=%.3f MRR=%.3f  (%.1fs)",
                i + 1, len(hotpot_data), rec["qid"],
                "✓" if rec["bridge_info"]["ran"] else "·",
                b["recall_at_1"], b["recall_at_5"], b["mrr"],
                h["recall_at_1"], h["recall_at_5"], h["mrr"],
                elapsed,
            )

    total = time.time() - t0

    summary = {
        "split": args.split,
        "n": len(all_results),
        "n_hyperedge_fired": n_hyperedge_fired,
        "max_chains": args.max_chains,
        "baseline": {
            "R@1": float(np.mean(base_r1)),
            "R@2": float(np.mean(base_r2)),
            "R@5": float(np.mean(base_r5)),
            "MRR": float(np.mean(base_mrr)),
            "para_R@1": float(np.mean(base_p_r1)),
            "para_R@2": float(np.mean(base_p_r2)),
            "para_R@5": float(np.mean(base_p_r5)),
            "para_MRR": float(np.mean(base_p_mrr)),
        },
        "hyperedge": {
            "R@1": float(np.mean(he_r1)),
            "R@2": float(np.mean(he_r2)),
            "R@5": float(np.mean(he_r5)),
            "MRR": float(np.mean(he_mrr)),
            "para_R@1": float(np.mean(he_p_r1)),
            "para_R@2": float(np.mean(he_p_r2)),
            "para_R@5": float(np.mean(he_p_r5)),
            "para_MRR": float(np.mean(he_p_mrr)),
        },
        "delta": {},
        "wall_clock_s": round(total, 1),
    }
    # Per-metric delta
    for k in ("R@1", "R@2", "R@5", "MRR", "para_R@1", "para_R@2",
              "para_R@5", "para_MRR"):
        summary["delta"][k] = summary["hyperedge"][k] - summary["baseline"][k]

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = results_dir / f"hyperedge_{args.split}_{args.max_examples}_seed{args.seed}_{ts}.json"
    summary_path = results_dir / f"hyperedge_{args.split}_{args.max_examples}_seed{args.seed}_{ts}_summary.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Detail   → %s", detail_path)
    log.info("Summary  → %s", summary_path)

    # Pretty print
    log.info("=" * 70)
    log.info("HYPEREDGE EXPERIMENT  [%s, n=%d, hyperedge_fired=%d/%d]",
             args.split, len(all_results), n_hyperedge_fired, len(all_results))
    log.info("--- Sentence-level ---")
    log.info("                R@1      R@2      R@5      MRR")
    log.info("  baseline   : %.4f  %.4f  %.4f  %.4f",
             summary["baseline"]["R@1"], summary["baseline"]["R@2"],
             summary["baseline"]["R@5"], summary["baseline"]["MRR"])
    log.info("  hyperedge  : %.4f  %.4f  %.4f  %.4f",
             summary["hyperedge"]["R@1"], summary["hyperedge"]["R@2"],
             summary["hyperedge"]["R@5"], summary["hyperedge"]["MRR"])
    log.info("  Δ         : %+.4f %+.4f %+.4f %+.4f",
             summary["delta"]["R@1"], summary["delta"]["R@2"],
             summary["delta"]["R@5"], summary["delta"]["MRR"])
    log.info("--- Paragraph-level (HotpotQA official) ---")
    log.info("              para_R@1   para_R@2   para_R@5   para_MRR")
    log.info("  baseline : %.4f   %.4f   %.4f   %.4f",
             summary["baseline"]["para_R@1"], summary["baseline"]["para_R@2"],
             summary["baseline"]["para_R@5"], summary["baseline"]["para_MRR"])
    log.info("  hyperedge: %.4f   %.4f   %.4f   %.4f",
             summary["hyperedge"]["para_R@1"], summary["hyperedge"]["para_R@2"],
             summary["hyperedge"]["para_R@5"], summary["hyperedge"]["para_MRR"])
    log.info("  Δ        : %+.4f  %+.4f  %+.4f  %+.4f",
             summary["delta"]["para_R@1"], summary["delta"]["para_R@2"],
             summary["delta"]["para_R@5"], summary["delta"]["para_MRR"])
    log.info("Wall clock : %.1fs", total)
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
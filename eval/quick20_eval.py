"""
Quick 20-Question HotpotQA EM/F1 evaluation using the new StenRAG pipeline.

This is a minimal harness that:
  1. Loads 20 HotpotQA dev questions (no shuffle, deterministic).
  2. Builds a StenRAG graph per question (using the gold + distractor
     paragraphs, mirroring hotpot_eval.build_graph).
  3. Runs the new Dispatcher (query_analyzer → dual/single/concept
     flow) on each question.
  4. Computes HotpotQA-style EM / F1 against the gold answer.

This script exists primarily as a smoke test for the new pipeline
after the Phase 1-7 refactor; the canonical end-to-end evaluation
lives in eval/hotpot_eval.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Tuple

# PHASE-E3: force HF / sentence-transformers to use local cache only.
# The model is already downloaded at
#   %USERPROFILE%\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2
# Without these flags, SentenceTransformer triggers HEAD requests to
# huggingface.co on every load, which times out behind GFW and
# blocks the run for ~3 minutes per question (5 retries × 30s).
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Make `src` importable when running from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quick20")

# -----------------------------------------------------------------------------
# EM / F1 (canonical HotpotQA metric, copied from end_to_end_eval.py)
# -----------------------------------------------------------------------------
import re
import string


def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _em_score(prediction: str, ground_truth: str) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


# -----------------------------------------------------------------------------
# Sentence extraction (mimic hotpot_eval.hotpot_to_sentences)
# -----------------------------------------------------------------------------
from src.graph_builder import GraphBuilder, ComplementedSentence


def build_graph_from_hotpot(
    qdata: Dict[str, Any],
    nlp,
) -> Tuple[GraphBuilder, List[ComplementedSentence], List[str],
           List[List[str]], Dict[str, List[str]]]:
    """
    Build a StenRAG graph from one HotpotQA question.

    Returns:
        builder, complemented_sentences, paragraph_titles,
        paragraph_sentence_lists
    """
    titles: List[str] = []
    para_sents: List[List[str]] = []
    sentences: List[ComplementedSentence] = []

    for title, sents in qdata.get("context", []):
        titles.append(title)
        para_sents.append(list(sents))
        for idx, sent_text in enumerate(sents):
            # Extract entities via spaCy.
            ents = []
            doc = nlp(sent_text)
            for ent in doc.ents:
                ents.append(ent.text)
            # (No coref in this quick harness.)
            sentences.append(ComplementedSentence(
                original_text=sent_text,
                complemented_text=sent_text,
                resolved_subject=ents[0] if ents else "",
                entities=ents,
            ))

    builder = GraphBuilder(shared_nlp=nlp)
    for sent in sentences:
        builder.add_sentence(sent)

    # Build a `bridge_titles` index: paragraph title (lowercased) →
    # list of container IDs whose sentences come from that paragraph.
    # We use the sentence→containers reverse index that add_sentence
    # already populates (`node.containers`).
    bridge_titles: Dict[str, List[str]] = {}
    for title, sent_texts in zip(titles, para_sents):
        # Find sentence ids whose text matches one of this paragraph's
        # sentences (order preserved by builder.add_sentence).
        title_cids: List[str] = []
        for s_text in sent_texts:
            target = s_text.strip()
            for node in builder.sentence_registry._nodes.values():
                if (node.text or "").strip() == target:
                    for cid in node.containers:
                        if cid not in title_cids:
                            title_cids.append(cid)
                    break
        if title_cids:
            bridge_titles[title.lower().strip()] = title_cids

    return builder, sentences, titles, para_sents, bridge_titles


# -----------------------------------------------------------------------------
# Build sentence-id → flat-idx mapping for HotpotQA gold support scoring.
# -----------------------------------------------------------------------------
def build_text_to_flat_idx(
    qdata: Dict[str, Any],
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    flat_idx = 0
    for title, sents in qdata.get("context", []):
        for sent in sents:
            out[sent.strip()] = flat_idx
            flat_idx += 1
    return out


# -----------------------------------------------------------------------------
# Extract a "best" answer sentence from dispatcher output.
# -----------------------------------------------------------------------------
def extract_answer_from_dispatch(
    dispatch_result,
    graph: GraphBuilder,
    gold_answer: str,
    nlp,
    client=None,
    question: str = "",
) -> str:
    """
    Derive a short answer string from the dispatch output.

    Strategy:
      * path == "single" with chosen_entity → use that entity directly
        (this is the spec §2.4 behaviour: direct entity output, no LLM).
      * Otherwise: collect top-N evidence sentences (up to 5) and
        send them with the question to the LLM for a 1-2 word answer.
        Falls back to the heuristic top-entity pick when LLM is
        unavailable.
    """
    path = dispatch_result.path
    final = dispatch_result.final_answer or ""

    # New pipeline DispatchResult: use final_text directly as the
    # evidence context (it's already a "\n".join of top-5 sentences).
    # When LLM is enabled, send it as context for span extraction.
    # When LLM is unavailable, return the top evidence sentence as
    # the answer (better than empty string).
    context = final

    # Heuristic: if the question is POLAR (genuinely answered with
    # yes/no) and the final answer contains a clear yes/no token,
    # surface it.  Choice questions ("Is X or Y bigger?") are NOT
    # polar — their answer is a candidate name — so they must not be
    # routed through this branch.
    if final:
        from src.answer_extractor import is_polar_question
        if is_polar_question(question):
            fl = final.lower()
            if re.search(r"\byes\b", fl) and not re.search(r"\bno\b", fl):
                return "yes"
            if re.search(r"\bno\b", fl) and not re.search(r"\byes\b", fl):
                return "no"

    # If LLM is disabled, return the first evidence line as the best
    # span guess.
    if client is None or not getattr(client, "enabled", False):
        if final:
            # Take the first non-empty line of the evidence.
            for line in final.splitlines():
                line = line.strip().lstrip("- ").strip()
                if line:
                    return line
        return final

    # LLM path — delegated to src.answer_extractor so that the live
    # path and the offline iteration harness build a byte-identical
    # prompt.  (A prompt that differs by even a punctuation mark
    # produces a different answer distribution, which makes any
    # measured delta between two variants meaningless.)
    if context:
        from src.answer_extractor import extract_answer as _extract
        # No explicit variant: uses src.answer_extractor.DEFAULT_VARIANT
        # (currently "v3"), overridable via $env:STENRAG_ANSWER_VARIANT.
        ans = _extract(client, question, context)
        if ans:
            return ans

    # Heuristic fallback: top entity in the first sentence.
    # `sent_ids` is sourced from the dispatcher's pipeline result.
    sent_ids = []
    try:
        sent_ids = list(
            getattr(
                getattr(dispatch_result, "pipeline_result", None),
                "step10_final_sids",
                [],
            ) or []
        )
    except Exception:
        sent_ids = []
    if not sent_ids:
        return ""
    top_id = sent_ids[0]
    node = graph.sentence_registry.get_node(top_id)
    if node is None:
        return ""

    text = node.text or ""
    if nlp is not None:
        try:
            doc = nlp(text)
            # Polar (yes/no) questions -> look for affirmation/negation
            # in the sentence.  Choice questions are excluded: their
            # answer is a candidate name, not yes/no.
            from src.answer_extractor import is_polar_question
            if is_polar_question(question):
                for tok in doc:
                    if tok.dep_ == "ROOT" and tok.lemma_.lower() in {"be"}:
                        # crude negation/affirmation by sub-tree tokens.
                        sub = " ".join(t.text.lower() for t in tok.subtree)
                        if "not " in sub or "n't" in sub:
                            return "no"
                        return "yes"
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "NORP",
                                  "DATE", "EVENT", "LOC", "FAC",
                                  "PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE"}:
                    return ent.text
        except Exception:
            pass
    return text[:120]


# -----------------------------------------------------------------------------
# Main evaluation loop
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data", default="eval/data/hotpot_dev_distractor_v1.json")
    ap.add_argument("--use-llm", action="store_true",
                    help="Use LLM answer generation (requires DEEPSEEK_API_KEY)")
    ap.add_argument("--out", default="eval/results/quick20_results.json")
    ap.add_argument(
        "--tag", default="",
        help="Optional suffix appended to the auto-generated output file "
             "name when --out is not provided.",
    )
    ap.add_argument(
        "--no-bridge", action="store_true",
        help="Disable the method-2 bridge-title expansion step.",
    )
    ap.add_argument(
        "--mode", default="pipeline",
        choices=("pipeline", "hyperedge_compare"),
        help=(
            "pipeline: 单一 StenGraphPipeline + LLM 抽取（默认）。"
            "hyperedge_compare: 同一题跑两次——"
            "baseline (no-bridge) vs baseline+hyperedge bridge——"
            "对比 R@k 和 EM/F1。"
        ),
    )
    ap.add_argument(
        "--hybrid-mode", default="hybrid",
        choices=("dense_only", "bm25_only", "hybrid", "bm25_heavy", "dense_heavy"),
        help=(
            "Retrieval mode for the container-level hybrid search "
            "(BM25 + dense RRF fusion). "
            "dense_only: FAISS/dense only (old baseline). "
            "bm25_only: BM25 only. "
            "hybrid: 50/50 RRF (default). "
            "bm25_heavy: 70/30. "
            "dense_heavy: 30/70."
        ),
    )
    ap.add_argument(
        "--bm25-weight", type=float, default=-1.0,
        help=(
            "Override the BM25 weight directly (0.0-1.0). "
            "Takes priority over --hybrid-mode. "
            "Set to 0.0 for dense-only, 1.0 for BM25-only."
        ),
    )
    ap.add_argument(
        "--no-gap-expand", action="store_true",
        help=(
            "Disable Step 9.5 LLM gap detection + 1-hop expansion "
            "(Week-2 Direction C).  Use to A/B test the new feature."
        ),
    )
    ap.add_argument(
        "--strict-entity-match", action="store_true",
        help=(
            "Replace the legacy substring-based entity matching in Step 2 "
            "with a precision-preserving Jaccard token-overlap matcher "
            "(≥ 50%% Jaccard).  Improves EM on questions where the old "
            "substring expansion pulled cross-paragraph containers (e.g. "
            "Q9 Jun Ji-hyun).  Does not change R@5."
        ),
    )
    ap.add_argument(
        "--attr-soft-match", action="store_true",
        help=(
            "Enable symmetric-attribute soft matching "
            "(pipeline.attr_soft_match_mode).  Replaces the old mapper+filter "
            "pipeline with continuous embedding-based attr alignment scoring "
            "in the final rerank formula.  Off by default."
        ),
    )
    ap.add_argument(
        "--offline-hyperedges",
        default="",
        help=(
            "Path to the offline-built hyperedge Parquet "
            "(e.g. eval/offline_hyperedges/hyperedges.parquet).  "
            "When set, hyperedges are loaded from disk instead of "
            "being built in-memory per question.  Falls back to "
            "online build for any missing qid+paragraph and logs a "
            "warning.  Set to '' (empty) to disable."
        ),
    )
    args = ap.parse_args()

    log.info("Loading %s", args.data)
    with open(args.data, encoding="utf-8") as f:
        all_data = json.load(f)
    log.info("Total questions: %d", len(all_data))

    # Deterministic sample.
    import random
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(all_data)), min(args.n, len(all_data)))
    indices.sort()
    sample = [all_data[i] for i in indices]

    # Load spaCy once.
    from src.graph_builder import _load_spacy_shared
    nlp = _load_spacy_shared("en_core_web_lg")
    if nlp is None:
        log.error("spaCy unavailable; aborting.")
        return 1

    # NOTE: SentenceTransformer embedder is used for event-level hyperedge
    # scoring (S_h_E) in _event_fact_score(). The model is cached at
    # module scope and loaded from local cache only (HF_HUB_OFFLINE=1).
    # Do NOT pre-fill _EMBEDDER_CACHE with None here, otherwise
    # event penalty will be completely disabled.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Optional LLM client.
    # PHASE-A: enable client whenever DEEPSEEK_API_KEY exists, so the
    # new LLM entity extractor (Phase-1 fix) is active even when the
    # user does not pass --use-llm. --use-llm additionally enables
    # LLM-based answer span extraction.
    client = None
    try:
        from src.llm_client import make_client
        client = make_client()
        log.info("LLM client: %s", "yes" if client.enabled else "no (no key)")
    except Exception as e:
        log.warning("LLM init failed: %s", e)

    # Dispatcher + hyperedge registry factory.
    from src.dispatcher import build_dispatcher
    from src.hyper_edge import HyperEdgeRegistry

    # ── Offline hyperedge store ───────────────────────────────────────
    # When --offline-hyperedges <parquet> is given, hyperedges are
    # pre-built and loaded from disk once.  Per question we look
    # up the container-ids for each paragraph in O(1) and only fall
    # back to the in-memory build for paragraphs that are missing
    # (e.g. when running on a question that wasn't in the offline
    # corpus).
    offline_store = None
    n_offline_para = 0
    n_offline_para_missing = 0
    if args.offline_hyperedges:
        try:
            from src.offline_hyperedge_store import OfflineHyperEdgeStore
            offline_store = OfflineHyperEdgeStore.from_parquet(
                args.offline_hyperedges,
            )
            stats = offline_store.stats()
            log.info(
                "Loaded offline hyperedge store: %d rows / %d unique qids.",
                stats["n_rows"], stats["n_unique_questions"],
            )
        except Exception as e:
            log.warning("Failed to load offline store: %s. Online fallback.", e)
            offline_store = None

    # Abort only if --use-llm is explicitly requested but no key is
    # present (i.e. answer extraction needs the LLM).
    if args.use_llm and (client is None or not getattr(client, "enabled", False)):
        log.error("--use-llm requested but no LLM key found. Aborting.")
        return 1

    if args.mode == "hyperedge_compare":
        return _main_hyperedge_compare(
            args, all_data, indices, nlp, client,
        )

    ems: List[bool] = []
    f1s: List[float] = []
    r1s: List[int] = []
    r5s: List[int] = []
    r10s: List[int] = []
    paths: Counter = Counter()
    records: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for i, qdata in enumerate(sample):
        qid = qdata["_id"]
        question = qdata["question"]
        gold = qdata["answer"]

        t0 = time.time()
        try:
            builder, sentences, titles, para_sents, bridge_titles = (
                build_graph_from_hotpot(qdata, nlp)
            )
            # Make sure every container has entity_inventory populated
            # before building hyperedges (otherwise build_hyperedges_for_chunk
            # silently falls back to the single entity name).
            cm = builder.container_manager
            for c in cm.get_all_containers():
                cm.update_container_index(c.id)
            # Phase-A: build sentence-level TF-IDF index so
            # StenGraphPipeline._score_tfidf() can replace the
            # Jaccard fallback with cosine-tfidf (借鉴 A).
            builder.compute_tfidf_index()
            # ── Hyperedge construction (offline-first) ───────────────────────
            # Strategy:
            #   1. If offline_store is available and has this qid+title →
            #      use the pre-built container IDs → build only the
            #      HyperEdgeRegistry (no re-building of containers/sentences).
            #   2. Otherwise → fall back to the legacy bridge_titles path
            #      (online build, same as before) and log a warning.
            he = HyperEdgeRegistry(container_manager=cm)
            online_fallback_count = 0
            if offline_store is not None:
                # Try offline for every paragraph title.
                para_cid_map: Dict[str, List[str]] = {}
                for _p_title in titles:
                    _cids = offline_store.get_containers_for_paragraph(
                        qid, _p_title,
                    )
                    if _cids:
                        para_cid_map[_p_title] = _cids
                    else:
                        # Not in offline store — fall back to bridge_titles.
                        _cids = (bridge_titles or {}).get(_p_title.lower().strip(), [])
                        if _cids:
                            para_cid_map[_p_title] = _cids
                            online_fallback_count += 1
                        else:
                            # Completely missing — still build from bridge_titles
                            # but log.
                            log.debug(
                                "Q[%s] paragraph %r has no containers "
                                "(offline or online)", qid[:8], _p_title,
                            )
                if online_fallback_count > 0:
                    n_offline_para_missing += online_fallback_count
                # Build HyperEdgeRegistry from the gathered cids.
                for _p_title, _cid_list in para_cid_map.items():
                    _p_containers = []
                    for _cid in _cid_list:
                        _c = cm._containers.get(_cid)
                        if _c is not None:
                            _p_containers.append(_c)
                    if _p_containers:
                        he.build_hyperedges_for_chunk(
                            chunk_id=str(_p_title),
                            containers=_p_containers,
                            time_word="",
                            paragraph_title=str(_p_title),
                        )
            else:
                # Legacy online build.
                for _p_title, _p_cids in (bridge_titles or {}).items():
                    _p_containers = []
                    for _cid in _p_cids:
                        _c = cm._containers.get(_cid)
                        if _c is not None:
                            _p_containers.append(_c)
                    if _p_containers:
                        he.build_hyperedges_for_chunk(
                            chunk_id=str(_p_title),
                            containers=_p_containers,
                            time_word="",
                            paragraph_title=str(_p_title),
                        )
            n_offline_para += len(titles)

            # Dispatch.
            disp = build_dispatcher(
                builder,
                client=client,
                nlp=nlp,
                hyperedge_registry=he,
                strict_entity_match=bool(args.strict_entity_match),
            )
            # Week-2 Direction C: optionally disable gap expansion
            # (LLM-driven 1-hop bridge expansion in step 9.5) so we
            # can A/B the feature.
            if args.no_gap_expand:
                disp.pipeline.gap_detector = None
            # Week-1: apply hybrid retrieval config.
            _apply_hybrid_config(disp.pipeline, args.hybrid_mode, args.bm25_weight)
            # Symmetric-attribute soft matching on/off switch.
            disp.pipeline.attr_soft_match_mode = bool(args.attr_soft_match)
            # TEST 2: pass gold sentence ids to pipeline for diagnostic logging
            # gold_sentence_ids is populated inside the try block below (line ~665)
            # so we read it from diag after the except-safe diag={} assignment
            # -> moved to after the try block, see below

            res = disp.run(
                question,
                bridge_titles=None if args.no_bridge else bridge_titles,
            )
            paths[res.path] += 1

            answer = extract_answer_from_dispatch(
                res, builder, gold, nlp, client=client, question=question,
            )
        except Exception as e:
            log.exception("Q %s failed: %s", qid, e)
            answer = ""
            res = None

        em_hit = _em_answer(answer, gold)
        f1 = _f1(answer, gold)
        ems.append(em_hit)
        f1s.append(f1)
        latencies.append((time.time() - t0) * 1000)

        # Recall@1 vs R@5: gold sentence in top-1 / top-5 of final sids.
        gold_norm = _normalize_answer(gold)
        final_sids = list(getattr(res.pipeline_result, "step10_final_sids", []) or []) if res else []
        def _sid_contains_gold(sid):
            n = builder.sentence_registry.get_node(sid)
            return n is not None and gold_norm in _normalize_answer(n.text or "")
        r1 = 1 if (final_sids and _sid_contains_gold(final_sids[0])) else 0
        r5 = 1 if any(_sid_contains_gold(s) for s in final_sids[:5]) else 0
        r10 = 1 if any(_sid_contains_gold(s) for s in final_sids[:10]) else 0
        r1s.append(r1); r5s.append(r5); r10s.append(r10)

        # ─────────────────────────────────────────────────────────────
        # [DIAG] Pool & gold-sentence audit — locate WHERE the gold
        # sentence is dropped along the retrieval cascade. Only runs
        # the per-sentence scan when r10 == 0 (cheap guard).
        # ─────────────────────────────────────────────────────────────
        diag = {
            "r10_status": ("HIT" if r10 else "MISS"),
            "gold_norm": gold_norm[:40],
            "n_entities": len(res.pipeline_result.step1_entities) if res else 0,
        }
        gold_sentence_ids: List[str] = []
        gold_in_pool_l1 = False
        gold_in_pool_l2 = False
        gold_in_final = False
        gold_in_any_container = False
        try:
            # The question carries supporting_facts (gold sentences).
            # NOTE: hotpot stores them as [[title, sent_idx], ...] OR
            # as [{title, sentence_id}, ...].  Handle both.
            if isinstance(qdata, dict):
                sfacts = qdata.get("supporting_facts") or qdata.get("support_facts") or []
                para = qdata.get("context") or []

                def _sf_title_idx(sf):
                    if isinstance(sf, dict):
                        return (sf.get("title") or "", int(sf.get("sentence_id") or 0))
                    if isinstance(sf, (list, tuple)) and len(sf) >= 2:
                        return (sf[0] or "", int(sf[1] or 0))
                    return ("", -1)

                gold_titles: set = set()
                gold_sentence_ids: List[str] = []
                for sf in sfacts:
                    gold_title_raw, gold_sidx = _sf_title_idx(sf)
                    gold_title = gold_title_raw.lower().strip()
                    if not gold_title or gold_sidx < 0:
                        continue
                    gold_titles.add(gold_title)
                    sentence_text = None
                    for p in para:
                        if (isinstance(p, list) and len(p) == 2
                                and (p[0] or "").lower().strip() == gold_title):
                            p_sids = p[1]
                            if isinstance(p_sids, list) and gold_sidx < len(p_sids):
                                sentence_text = p_sids[gold_sidx]
                            break
                    if not sentence_text:
                        continue
                    nodes = getattr(builder.sentence_registry, "_nodes", {}) or {}
                    matched = False
                    for sid, node in nodes.items():
                        try:
                            if getattr(node, "text", "") == sentence_text:
                                gold_sentence_ids.append(sid)
                                matched = True
                                break
                        except Exception:
                            continue
                    if not matched:
                        sn = _normalize_answer(sentence_text)
                        for sid, node in nodes.items():
                            try:
                                if _normalize_answer(getattr(node, "text", "") or "") == sn:
                                    gold_sentence_ids.append(sid)
                                    break
                            except Exception:
                                continue
            gold_sent_ids_set = set(gold_sentence_ids)
            # Layered inspection.
            diag["gold_sentence_ids"] = gold_sentence_ids
            diag["n_gold_sentences"] = len(gold_sentence_ids)
            # step4_l1_sids may not exist on this version of
            # PipelineResult; fall back to step5_traced_sids.
            try:
                l1_sids = set(
                    getattr(res.pipeline_result, "step4_l1_sids", None)
                    or getattr(res.pipeline_result, "step5_traced_sids", [])
                    or []
                )
            except Exception:
                l1_sids = set()
            # L0 layer = sentences whose primary container is in A.
            # Step 6 builds this from `a_ids` ∩ container.sentence_ids.
            l0_sids: Set[str] = set()
            try:
                a_cids_for_l0 = set(
                    getattr(res.pipeline_result, "step2_a_containers", []) or []
                )
                for cid in a_cids_for_l0:
                    c = builder.container_manager.get_container(cid)
                    if c is None:
                        continue
                    for sid in getattr(c, "sentence_ids", []) or []:
                        l0_sids.add(sid)
            except Exception:
                pass
            # Correct field names from PipelineResult definition.
            l2_sids = list(getattr(res.pipeline_result, "step4_l2_sids_filtered", []) or [])
            l2_cids = list(getattr(res.pipeline_result, "step4_l2_cids_filtered", []) or [])
            diag["step4_l1_sids"] = sorted(l0_sids)[:30]  # real L0
            diag["step4_l2_sids"] = sorted(l2_sids)[:30]
            diag["step4_l2_cids"] = list(l2_cids)
            diag["step10_final_sids"] = list(final_sids)
            diag["l0_sids"] = sorted(l0_sids)[:30]
            # Snapshot every other pipeline layer so we can trace gold's
            # journey downstream of Step4.
            diag["layer_step5_traced"] = list(
                getattr(res.pipeline_result, "step5_traced_sids", []) or []
            )[:30]
            diag["layer_step7_topn"] = list(
                getattr(res.pipeline_result, "step7_top_n_sids", []) or []
            )[:30]
            diag["layer_step8_supplement"] = list(
                getattr(res.pipeline_result, "step8_supplement_sids", []) or []
            )[:30]
            diag["layer_step9_merged"] = list(
                getattr(res.pipeline_result, "step9_merged_sids", []) or []
            )[:30]
            diag["layer_step6_pool_size"] = int(
                getattr(res.pipeline_result, "step6_pool_size", 0) or 0
            )
            diag["layer_step2_a_cids"] = list(
                getattr(res.pipeline_result, "step2_a_containers", []) or []
            )[:30]
            diag["layer_step4_b_cids"] = list(
                getattr(res.pipeline_result, "step4_b_containers", []) or []
            )[:30]
            diag["layer_step3_hyperedges"] = list(
                getattr(res.pipeline_result, "step3_hyperedges", []) or []
            )[:30]
            diag["layer_step2_5_bridge_cids"] = list(
                getattr(res.pipeline_result, "step2_5_bridge_cids", []) or []
            )[:30]
            diag["layer_step2_5_bridge_titles"] = list(
                getattr(res.pipeline_result, "step2_5_bridge_titles", []) or []
            )[:30]
            # Where does the gold sentence FIRST appear in any of these layers?
            gold_first_appear_layer = None
            for layer_name, layer_attr in [
                ("step4_l0", l0_sids),
                ("step4_l2", set(l2_sids)),
                ("step5_traced", set(diag["layer_step5_traced"])),
                ("step7_topn", set(diag["layer_step7_topn"])),
                ("step8_supplement", set(diag["layer_step8_supplement"])),
                ("step9_merged", set(diag["layer_step9_merged"])),
                ("step10_final", set(final_sids)),
            ]:
                if gold_sent_ids_set & layer_attr:
                    gold_first_appear_layer = layer_name
                    break
            diag["gold_first_appear_layer"] = gold_first_appear_layer
            l2_sids_set = set(l2_sids)
            if gold_sent_ids_set & l0_sids:
                gold_in_pool_l1 = True
            if gold_sent_ids_set & l2_sids_set:
                gold_in_pool_l2 = True
            if gold_sent_ids_set & set(final_sids):
                gold_in_final = True
            # Did any l2_container actually contain one of the gold
            # titles in its entity_inventory?  (gold_titles already
            # collected above while resolving gold sentence sids.)
            diag["gold_titles"] = sorted(gold_titles)
            for cid in l2_cids:
                try:
                    c = builder.container_manager.get_container(cid)
                except Exception:
                    c = None
                if c is None:
                    continue
                inv = [str(e).lower() for e in (c.entity_inventory or [])]
                if any(t in inv for t in gold_titles if t):
                    gold_title_hit_container = True
                    break
            if gold_sent_ids_set:
                diag["cascade"] = {
                    "gold_in_step4_l1": gold_in_pool_l1,
                    "gold_in_step4_l2": gold_in_pool_l2,
                    "gold_in_step10_top10": gold_in_final,
                    "gold_title_in_any_step4_l2_container": gold_title_hit_container,
                }
                if not (gold_in_pool_l1 or gold_in_pool_l2):
                    diag["verdict"] = "RETRIEVAL-MISS"
                elif gold_in_pool_l2 and not gold_in_final:
                    diag["verdict"] = "RERANK-MISS"
                elif gold_in_final and not em_hit:
                    diag["verdict"] = "EXTRACT-MISS"
                else:
                    diag["verdict"] = "OK"
        except Exception as _diag_err:
            diag["diag_error"] = str(_diag_err)

        # Log every RECALL-MISS explicitly so the run log tells us the cascade result.
        if r10 == 0 and res is not None:
            log.warning(
                "[DIAG] %s r10=MISS | entities=%d | n_gold_sents=%d | verdict=%s | cascade=%s | gold_titles=%s",
                qid[:8],
                diag["n_entities"],
                diag.get("n_gold_sentences", 0),
                diag.get("verdict", "?"),
                diag.get("cascade", {}),
                diag.get("gold_titles", []),
            )

        log.info(
            "  [%2d/%2d] %s  path=%s  EM=%s F1=%.3f R@1=%d R@5=%d R@10=%d  lat=%.0fms\n"
            "         Q: %s\n"
            "         G: %s\n"
            "         A: %s",
            i + 1, len(sample), qid[:8],
            res.path if res else "?",
            "✓" if em_hit else "✗", f1, r1, r5, r10, latencies[-1],
            question[:120],
            gold[:80],
            answer[:80],
        )

        # Attribute-soft-match diagnostics (no hard filter any more, so
        # "pool_after" == "pool_before"; we keep the field for schema
        # compatibility with older result files).
        gold_sids_set = set(gold_sentence_ids)
        pool_before = res.pipeline_result.debug.get("step6_pool_sids", []) if res else []
        pool_after = list(pool_before)
        removed_sids: List[str] = []
        gold_in_removed: List[str] = []

        # Gold sentence attribute_mounts
        gold_mounts: List[str] = []
        gold_entity_type = "?"
        if gold_sentence_ids:
            first_gold = gold_sentence_ids[0]
            node = builder.sentence_registry.get_node(first_gold)
            if node:
                gold_mounts = [
                    m.attribute_name for m in
                    (getattr(node, "attribute_mounts", []) or [])
                    if getattr(m, "attribute_name", None)
                ]
                gold_entity_type = getattr(node, "entity_type", "?")

        # TEST 2: pass gold sids to pipeline for gold-in-removed logging
        disp.pipeline._diag_gold_sids = gold_sentence_ids

        # Query relations (kept for diagnostics; the mapper is gone).
        q_relations = (res.pipeline_result.debug.get("query_relations", {}) if res and res.pipeline_result else {})
        log.info("Attr soft match: q_relations=%s gold_mounts=%s",
                 q_relations, gold_mounts)

        records.append({
            "qid": qid,
            "question": question,
            "gold": gold,
            "answer": answer,
            "path": res.path if res else "",
            "em": em_hit,
            "f1": f1,
            "r1": r1,
            "r5": r5,
            "r10": r10,
            "latency_ms": latencies[-1],
            "diag": diag,
            "debug": res.pipeline_result.debug if res and res.pipeline_result else {},
            # Extra diag fields for attribute-soft-match analysis
            "filter_diag": {
                "gold_sentence_ids": gold_sentence_ids,
                "gold_mounts": sorted(gold_mounts),
                "gold_entity_type": gold_entity_type if gold_sentence_ids else "?",
                "query_relations": q_relations,
                "required_attrs": [],
                "pool_before": pool_before,
                "pool_after": pool_after,
                "removed_sids": removed_sids,
                "gold_in_removed": gold_in_removed,
            },
        })

    # Summary.
    em_avg = sum(ems) / len(ems)
    f1_avg = sum(f1s) / len(f1s)
    r1_avg = sum(r1s) / max(1, len(r1s))
    r5_avg = sum(r5s) / max(1, len(r5s))
    r10_avg = sum(r10s) / max(1, len(r10s))
    log.info("=" * 60)
    log.info("n=%d  EM=%.3f  F1=%.3f  R@1=%.3f  R@5=%.3f  R@10=%.3f",
             len(ems), em_avg, f1_avg, r1_avg, r5_avg, r10_avg)
    log.info("Path breakdown: %s", dict(paths))
    log.info("Avg latency: %.0fms", sum(latencies) / max(1, len(latencies)))
    log.info("Hybrid config: mode=%s  w_bm25=%.2f  w_dense=%.2f",
             args.hybrid_mode,
             args.bm25_weight if 0 <= args.bm25_weight <= 1 else 0.5,
             1.0 - (args.bm25_weight if 0 <= args.bm25_weight <= 1 else 0.5))

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    w_bm25 = args.bm25_weight if 0 <= args.bm25_weight <= 1 else 0.5
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n": len(ems),
            "em": em_avg,
            "f1": f1_avg,
            "r1": r1_avg,
            "r5": r5_avg,
            "r10": r10_avg,
            "paths": dict(paths),
            "records": records,
            "hybrid": {
                "mode": args.hybrid_mode,
                "w_bm25": w_bm25,
                "w_dense": 1.0 - w_bm25,
            },
        }, f, ensure_ascii=False, indent=2)
    log.info("Results → %s", out_path)
    return 0


# ─────────────────────────────────────────────────────────────────────
# Bridge sentence extraction (FIX-2: path_sentence_ids was always [])
# ─────────────────────────────────────────────────────────────────────
# The old `DualEntityRetriever.retrieve()` is a backwards-compat shim
# that returns `path_sentence_ids=[]`; using it for hyperedge bridge
# would give us R@5=baseline.R@5 every time.  Instead we build the
# bridge from the *containers* the entity pair anchors, then pull
# their sentences into the candidate pool.  This matches what
# hyperedge_eval.py's `_bridge_to_sentence_chain` *intended* to do.
# ─────────────────────────────────────────────────────────────────────
def _select_two_entities(
    question: str,
    container_manager,
) -> List[str]:
    """
    Pick up to 2 entities from the question that match a real
    container, longest-first, no subspan duplicates.

    Reused from eval.hyperedge_eval with one tweak (FIX-1b): the
    `_normalize_entity_label` strips parenthetical disambiguators and
    "group N" suffixes so that "Falling Stars (video game) (group 2)"
    collapses to "falling stars" and doesn't get filtered out as a
    no-hit entity when the question only mentions "Falling Stars".
    """
    from eval.hyperedge_eval import (
        _select_two_entities_from_question as _orig_select,
    )
    return _orig_select(question, container_manager)


def _build_bridge_sids(
    question: str,
    container_manager,
    sentence_registry,
    candidate_sids: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Identify the 2 question-anchor entities, locate their containers,
    and return a *list of bridge sentence ids* drawn from those
    containers that are NOT already in `candidate_sids` (so we
    genuinely add new evidence, never duplicates).

    Returns (new_sids, entity_names). When fewer than 2 entities
    match, returns ([], [matched_names]).
    """
    ents = _select_two_entities(question, container_manager)
    if len(ents) < 2:
        return [], ents

    seen = set(candidate_sids)
    bridge_sids: List[str] = []
    for ent in ents:
        for c in container_manager.get_containers_for_entity(ent) or []:
            for sid in getattr(c, "sentence_ids", []) or []:
                if sid not in seen:
                    seen.add(sid)
                    bridge_sids.append(sid)
    return bridge_sids, ents


# ─────────────────────────────────────────────────────────────────────
# Re-rank candidate sids with the standard pipeline reranker
# (BM25 + entity-substring + named-anchor co-occurrence).
# ─────────────────────────────────────────────────────────────────────
def _rerank_sids(
    pipeline,
    query: str,
    sids: List[str],
    entities: List[str],
    top_k: int,
) -> List[str]:
    """
    Apply the same _step10_final_rerank the pipeline uses internally,
    but on an arbitrary `sids` list.  Returns the top-k highest-scored
    sids (preserving pipeline.top_k_final = 8 semantics).
    """
    if not sids:
        return []
    return pipeline._step10_final_rerank(
        query, list(sids), top_k, entities=entities,
    )


# ─────────────────────────────────────────────────────────────────────
# Hyperedge-compare-mode per-question driver
# ─────────────────────────────────────────────────────────────────────
def _run_hyperedge_compare(
    qdata: Dict[str, Any],
    nlp,
    client,
    use_llm: bool,
) -> Dict[str, Any]:
    """
    For one HotpotQA question, run BOTH:
      (A) baseline:    pipeline with --no-bridge (no bridge-title,
                       no hyperedge supplements)
      (B) hyperedge:   pipeline with bridge_titles + extra bridge sids
                       from the dual-entity anchors
    Returns a dict with all metrics for side-by-side comparison.
    """
    qid = qdata["_id"]
    question = qdata["question"]
    gold = qdata["answer"]

    # Graph + TF-IDF (same as pipeline mode).
    builder, sentences, titles, para_sents, bridge_titles = (
        build_graph_from_hotpot(qdata, nlp)
    )
    cm = builder.container_manager
    for c in cm.get_all_containers():
        cm.update_container_index(c.id)
    builder.compute_tfidf_index()

    # Per-paragraph hyperedges.
    from src.hyper_edge import HyperEdgeRegistry
    he = HyperEdgeRegistry(container_manager=cm)
    for _p_title, _p_cids in (bridge_titles or {}).items():
        _p_containers = []
        for _cid in _p_cids:
            _c = cm._containers.get(_cid)
            if _c is not None:
                _p_containers.append(_c)
        if _p_containers:
            he.build_hyperedges_for_chunk(
                chunk_id=str(_p_title),
                containers=_p_containers,
                time_word="",
                paragraph_title=str(_p_title),
            )

    # Dispatcher (built once; we just toggle bridge_titles).
    from src.dispatcher import build_dispatcher
    disp = build_dispatcher(
        builder, client=client, nlp=nlp, hyperedge_registry=he,
    )
    # Week-1: apply hybrid retrieval config.
    _apply_hybrid_config(disp.pipeline, args.hybrid_mode, args.bm25_weight)
    # Symmetric-attribute soft matching on/off switch (applies to both A/B runs).
    disp.pipeline.attr_soft_match_mode = bool(getattr(args, "attr_soft_match", False))

    # ---- (A) baseline: --no-bridge ----
    res_a = disp.run(question, bridge_titles=None)
    final_a = list(getattr(res_a.pipeline_result, "step10_final_sids", []) or [])
    text_a = res_a.final_answer or ""
    pred_a = extract_answer_from_dispatch(
        res_a, builder, gold, nlp, client=client, question=question,
    ) if use_llm else text_a

    # ---- (B) hyperedge: bridge_titles + dual-entity bridge sids ----
    res_b = disp.run(question, bridge_titles=bridge_titles)
    final_b = list(getattr(res_b.pipeline_result, "step10_final_sids", []) or [])

    # FIX-2: actually construct the bridge. Old `DualEntityRetriever`
    # returned `path_sentence_ids=[]` — that's why R@5 was tied with
    # baseline.  Now we build the bridge from the entity-anchored
    # containers directly.
    bridge_sids, ents = _build_bridge_sids(
        question, cm, builder.sentence_registry, final_b,
    )
    bridge_info = {
        "ran": len(ents) >= 2,
        "n_entities_matched": len(ents),
        "n_bridge_sids_added": len(bridge_sids),
        "ents": ents,
    }

    if bridge_sids:
        # Extend the candidate pool with the bridge sentences and
        # rerank with the same logic pipeline.step10 uses.
        merged_pool = list(final_b) + bridge_sids
        entities = list(getattr(res_b.pipeline_result, "step1_entities", []) or [])
        final_b = _rerank_sids(
            disp.pipeline, question, merged_pool, entities,
            top_k=disp.pipeline.top_k_final,
        )

    text_b = "\n".join(
        (builder.sentence_registry.get_node(s).text if builder.sentence_registry.get_node(s) else "")
        for s in final_b
    )
    pred_b = extract_answer_from_dispatch(
        res_b, builder, gold, nlp, client=client, question=question,
    ) if use_llm else text_b
    # Inject bridge sentences into the evidence text for the LLM call
    # (so the LLM can see them) — only relevant when LLM is on.
    if use_llm and bridge_sids:
        extra_lines = []
        for sid in bridge_sids:
            n = builder.sentence_registry.get_node(sid)
            if n is not None:
                extra_lines.append((n.text or "").strip())
        if extra_lines:
            text_b_for_llm = text_b + "\n" + "\n".join(extra_lines)
        else:
            text_b_for_llm = text_b
        # Re-run the LLM extraction with the augmented context.
        try:
            from src.llm_client import DeepSeekClient as _DSC
            from eval.hotpot_eval import (
                _strip_parens, _strip_colon_prefix,
                _short_answer_gating, _coref_pick,
            )
            # Use the same span-extraction prompt as
            # extract_answer_from_dispatch by simulating a DispatchResult.
            class _FakeRes:
                path = "pipeline"
                final_answer = text_b_for_llm
                pipeline_result = res_b.pipeline_result
            pred_b = extract_answer_from_dispatch(
                _FakeRes(), builder, gold, nlp, client=client, question=question,
            )
        except Exception as e:
            log.debug("LLM re-extract on hyperedge side failed: %s", e)

    # ---- Scoring ----
    gold_norm = _normalize_answer(gold)
    def _contains_gold(sid):
        n = builder.sentence_registry.get_node(sid)
        return n is not None and gold_norm in _normalize_answer(n.text or "")

    def _score(sids, pred):
        r1 = 1 if (sids and _contains_gold(sids[0])) else 0
        r5 = 1 if any(_contains_gold(s) for s in sids[:5]) else 0
        r10 = 1 if any(_contains_gold(s) for s in sids[:10]) else 0
        em = _em_answer(pred, gold) if pred else False
        f1 = _f1(pred, gold) if pred else 0.0
        return r1, r5, r10, em, f1

    r1_a, r5_a, r10_a, em_a, f1_a = _score(final_a, pred_a)
    r1_b, r5_b, r10_b, em_b, f1_b = _score(final_b, pred_b)

    return {
        "qid": qid,
        "question": question,
        "gold": gold,
        "baseline": {
            "r1": r1_a, "r5": r5_a, "r10": r10_a,
            "em": em_a, "f1": f1_a,
            "pred": pred_a,
            "n_final_sids": len(final_a),
        },
        "hyperedge": {
            "r1": r1_b, "r5": r5_b, "r10": r10_b,
            "em": em_b, "f1": f1_b,
            "pred": pred_b,
            "n_final_sids": len(final_b),
        },
        "bridge": bridge_info,
    }


# ─────────────────────────────────────────────────────────────────────
# hyperedge_compare main entry: loop over sample, aggregate, write.
# ─────────────────────────────────────────────────────────────────────
def _main_hyperedge_compare(
    args, all_data, indices, nlp, client,
) -> int:
    sample = [all_data[i] for i in indices]
    use_llm = bool(args.use_llm and client is not None and getattr(client, "enabled", False))
    log.info("hyperedge_compare mode: n=%d use_llm=%s", len(sample), use_llm)

    records: List[Dict[str, Any]] = []
    base_r1, base_r5, base_r10, base_em, base_f1 = [], [], [], [], []
    he_r1, he_r5, he_r10, he_em, he_f1 = [], [], [], [], []
    n_bridge_fired = 0

    for i, qdata in enumerate(sample):
        t0 = time.time()
        try:
            rec = _run_hyperedge_compare(qdata, nlp, client, use_llm)
        except Exception as e:
            log.exception("Q %s failed in hyperedge_compare: %s", qdata.get("_id"), e)
            rec = None
        rec_latency = (time.time() - t0) * 1000

        if rec is None:
            continue
        if rec["bridge"]["ran"]:
            n_bridge_fired += 1

        b = rec["baseline"]; h = rec["hyperedge"]
        base_r1.append(b["r1"]);  base_r5.append(b["r5"]);  base_r10.append(b["r10"])
        base_em.append(b["em"]);  base_f1.append(b["f1"])
        he_r1.append(h["r1"]);    he_r5.append(h["r5"]);    he_r10.append(h["r10"])
        he_em.append(h["em"]);    he_f1.append(h["f1"])

        rec["latency_ms"] = rec_latency
        records.append(rec)
        log.info(
            "  [%2d/%2d] %s  base R@1=%d R@5=%d  he R@1=%d R@5=%d  "
            "bridge=%s(+%d)  em_b=%s em_h=%s  (%.0fms)",
            i + 1, len(sample), rec["qid"][:8],
            b["r1"], b["r5"], h["r1"], h["r5"],
            "✓" if rec["bridge"]["ran"] else "·",
            rec["bridge"]["n_bridge_sids_added"],
            "Y" if b["em"] else "N", "Y" if h["em"] else "N",
            rec_latency,
        )

    def _m(xs): return sum(xs) / max(1, len(xs))
    summary = {
        "n": len(records),
        "n_bridge_fired": n_bridge_fired,
        "use_llm": use_llm,
        "baseline": {
            "R@1": _m(base_r1), "R@5": _m(base_r5), "R@10": _m(base_r10),
            "EM": _m(base_em) if use_llm else None,
            "F1": _m(base_f1) if use_llm else None,
        },
        "hyperedge": {
            "R@1": _m(he_r1), "R@5": _m(he_r5), "R@10": _m(he_r10),
            "EM": _m(he_em) if use_llm else None,
            "F1": _m(he_f1) if use_llm else None,
        },
        "delta": {
            "R@1": _m(he_r1) - _m(base_r1),
            "R@5": _m(he_r5) - _m(base_r5),
            "R@10": _m(he_r10) - _m(base_r10),
            "EM": (_m(he_em) - _m(base_em)) if use_llm else None,
            "F1": (_m(he_f1) - _m(base_f1)) if use_llm else None,
        },
    }

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    detail_path = out_dir / f"quick20_hyperedge_{len(records)}_seed{args.seed}{tag}_{ts}.json"
    summary_path = detail_path.with_name(detail_path.stem + "_summary.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records},
                  f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info("=" * 60)
    log.info("HYPEREDGE COMPARE  n=%d  bridge_fired=%d/%d  use_llm=%s",
             len(records), n_bridge_fired, len(records), use_llm)
    log.info("              R@1     R@5     R@10    EM      F1")
    log.info("  baseline : %.3f   %.3f   %.3f   %s   %s",
             summary["baseline"]["R@1"], summary["baseline"]["R@5"],
             summary["baseline"]["R@10"],
             f"{summary['baseline']['EM']:.3f}" if use_llm else "n/a",
             f"{summary['baseline']['F1']:.3f}" if use_llm else "n/a")
    log.info("  hyperedge: %.3f   %.3f   %.3f   %s   %s",
             summary["hyperedge"]["R@1"], summary["hyperedge"]["R@5"],
             summary["hyperedge"]["R@10"],
             f"{summary['hyperedge']['EM']:.3f}" if use_llm else "n/a",
             f"{summary['hyperedge']['F1']:.3f}" if use_llm else "n/a")
    log.info("  Δ       : %+.3f  %+.3f  %+.3f  %s   %s",
             summary["delta"]["R@1"], summary["delta"]["R@5"],
             summary["delta"]["R@10"],
             f"{summary['delta']['EM']:+.3f}" if use_llm else "n/a",
             f"{summary['delta']['F1']:+.3f}" if use_llm else "n/a")
    log.info("Detail  → %s", detail_path)
    log.info("Summary → %s", summary_path)
    return 0


# Tiny aliases to avoid name shadowing above.
def _em_answer(pred: str, gold: str) -> bool:
    return _em_score(pred, gold)


def _f1(pred: str, gold: str) -> float:
    return _f1_score(pred, gold)


# ─────────────────────────────────────────────────────────────────
# Hybrid retrieval config helpers
# ─────────────────────────────────────────────────────────────────
def _apply_hybrid_config(pipeline, mode: str, bm25_weight: float) -> None:
    """
    Apply hybrid retrieval config to a StenGraphPipeline.

    Modes:
        "dense_only"  — w_bm25=0, w_dense=1  (original FAISS-only)
        "bm25_only"   — w_bm25=1, w_dense=0  (BM25-only)
        "hybrid"      — w_bm25=0.5, w_dense=0.5 (default RRF)
        "bm25_heavy"  — w_bm25=0.7, w_dense=0.3
        "dense_heavy" — w_bm25=0.3, w_dense=0.7

    `bm25_weight` overrides the mode's w_bm25 when 0 ≤ bm25_weight ≤ 1.
    """
    presets = {
        "dense_only":  (0.0, 1.0),
        "bm25_only":   (1.0, 0.0),
        "hybrid":      (0.5, 0.5),
        "bm25_heavy":  (0.7, 0.3),
        "dense_heavy": (0.3, 0.7),
    }
    w_bm25, w_dense = presets.get(mode, (0.5, 0.5))
    if 0.0 <= bm25_weight <= 1.0:
        w_bm25 = float(bm25_weight)
        w_dense = 1.0 - w_bm25
    pipeline.w_bm25 = w_bm25
    pipeline.w_dense = w_dense


if __name__ == "__main__":
    sys.exit(main())
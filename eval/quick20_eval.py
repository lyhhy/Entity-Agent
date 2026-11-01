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
from typing import Any, Dict, List, Optional, Tuple

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

# R@k 的 gold 句子判定（唯一真相源，见 eval/recall_utils.py 的说明）。
# 旧的子串判定会系统性低估 recall —— gold 为 'yes' 的题 R@k 恒为 0。
from eval.recall_utils import (
    GoldMatcher,
    build_sentence_to_paragraph,
    gold_passage_titles,
    gold_sentence_matcher,
    normalize_answer as _normalize_answer_recall,
    passage_metrics,
)


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
    # ------------------------------------------------------------------
    # Attribute-agent (rolling attribute-table optimisation)
    # ------------------------------------------------------------------
    ap.add_argument(
        "--attr-agent", action="store_true",
        help=(
            "Run the attribute agent on the sampled questions BEFORE "
            "the EM/F1 measurement, and let the accepted delta take "
            "effect on the graphs that are then evaluated."
        ),
    )
    ap.add_argument(
        "--attr-agent-ab", action="store_true",
        help=(
            "A/B mode: evaluate the sample TWICE — once with the "
            "current attribute table (baseline) and once after the "
            "agent's delta has been applied — and log the delta of "
            "every metric.  Implies --attr-agent."
        ),
    )
    ap.add_argument(
        "--attr-agent-n", type=int, default=20,
        help="Questions drawn per agent round (default 20).",
    )
    ap.add_argument(
        "--attr-agent-rounds", type=int, default=10,
        help="Agent sampling rounds (default 10).",
    )
    ap.add_argument(
        "--attr-agent-seed", type=int, default=None,
        help=(
            "Base seed of the agent's sampling.  OMITTED by default: a "
            "fresh random seed is drawn for every run so each run "
            "explores the attribute table from a different angle.  The "
            "seed that was used is printed and stored in the report."
        ),
    )
    ap.add_argument(
        "--attr-agent-schema", default="",
        help=(
            "Path to an existing attribute-schema delta to roll forward "
            "from (e.g. eval/results/attribute_schema.json)."
        ),
    )
    ap.add_argument(
        "--attr-agent-out", default="eval/results/attribute_schema.json",
        help="Where to persist the accepted schema delta.",
    )
    ap.add_argument(
        "--attr-agent-report", default="eval/results/attribute_agent_report.json",
        help="Where to persist the agent's full run report.",
    )
    ap.add_argument(
        "--attr-agent-no-llm", action="store_true",
        help="Disable the agent's LLM fallbacks (gap confirmation / "
             "ambiguous-owner arbitration).",
    )
    ap.add_argument(
        "--attr-agent-dry-run", action="store_true",
        help="Detect and report gaps but do NOT apply the delta.",
    )
    ap.add_argument(
        "--attr-agent-no-prune", action="store_true",
        help=(
            "Disable the prune pass.  By default the agent DROPS the "
            "table entries whose mount efficiency collapsed below the "
            "gate, so the table does not only ever grow."
        ),
    )
    ap.add_argument(
        "--attr-agent-prune-scope", choices=["delta", "all"], default="delta",
        help=(
            "What the prune pass may touch.  'delta' (default) only "
            "prunes entries the agent added; 'all' also prunes the "
            "hand-designed base table."
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
    #
    # `make_client()` auto-loads the repo-root `.env`, so a key written
    # there is picked up with no extra setup step.
    client = None
    try:
        from src.llm_client import make_client, describe_llm_config
        client = make_client()
        log.info("LLM client: %s", "yes" if client.enabled else "no (no key)")
        log.info("LLM config: %s", describe_llm_config())
    except Exception as e:
        log.warning("LLM init failed: %s", e)

    # Dispatcher + hyperedge registry are constructed per question inside
    # `_evaluate_sample()`, so they are imported there.

    # ── Offline hyperedge store ───────────────────────────────────────
    # When --offline-hyperedges <parquet> is given, hyperedges are
    # pre-built and loaded from disk once.  Per question we look
    # up the container-ids for each paragraph in O(1) and only fall
    # back to the in-memory build for paragraphs that are missing
    # (e.g. when running on a question that wasn't in the offline
    # corpus).
    offline_store = None
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

    # ------------------------------------------------------------------
    # Attribute agent (rolling attribute-table optimisation)
    # ------------------------------------------------------------------
    #
    # The agent can run in three ways:
    #   (none)    — the current attribute table is used as-is.
    #   --attr-agent      — the agent runs FIRST, its delta is applied to
    #                    the module tables, and the evaluation then runs
    #                    on graphs that inherit the new attributes.
    #   --attr-agent-ab   — the evaluation runs TWICE: once with the
    #                    current table (baseline) and once after the
    #                    delta has been applied (treatment).  The metric
    #                    delta is logged at the end.
    run_agent = bool(args.attr_agent or args.attr_agent_ab)
    agent_report = None
    agent_schema = None

    # ------------------------------------------------------------------
    # A/B BASELINE — must run BEFORE the agent patches the module tables.
    # ------------------------------------------------------------------
    baseline_summary = None
    baseline_records: List[Dict[str, Any]] = []
    if args.attr_agent_ab:
        log.info("=" * 68)
        log.info("A/B BASELINE — current attribute table")
        log.info("=" * 68)
        from src.progress import ConsoleReporter
        base_progress = ConsoleReporter(prefix="baseline")
        baseline_summary, baseline_records = _evaluate_sample(
            args, sample, nlp, client, offline_store,
            progress=base_progress, label="baseline",
        )
        base_progress.finish()

    if run_agent:
        # `eval/` is not a package (no __init__.py), so the helper module
        # is imported by path rather than as `eval.attribute_agent_eval`.
        import importlib.util
        _agent_eval_path = Path(__file__).resolve().parent / "attribute_agent_eval.py"
        _spec = importlib.util.spec_from_file_location(
            "attribute_agent_eval", _agent_eval_path,
        )
        _agent_eval = importlib.util.module_from_spec(_spec)
        sys.modules["attribute_agent_eval"] = _agent_eval
        _spec.loader.exec_module(_agent_eval)
        run_attribute_agent = _agent_eval.run_attribute_agent
        log_report = _agent_eval.log_report

        from src.attribute_schema import AttributeSchema
        from src.progress import ConsoleReporter

        # Progress reporter — the agent samples 200 questions by
        # default, so the run needs a live "how far are we" line.
        agent_progress = ConsoleReporter(prefix="agent")

        # Roll forward from an existing delta, if one was given.
        if args.attr_agent_schema:
            agent_schema = AttributeSchema.load(args.attr_agent_schema)
            log.info(
                "Loaded attribute-schema delta: %d attribute(s), %d class(es) "
                "from %s",
                len(agent_schema.attributes), len(agent_schema.classes),
                args.attr_agent_schema,
            )
            agent_schema.apply_to_modules()

        agent_report = run_attribute_agent(
            sample,
            n_questions=args.attr_agent_n,
            n_rounds=args.attr_agent_rounds,
            seed=args.attr_agent_seed,
            use_llm=not args.attr_agent_no_llm,
            prune=not args.attr_agent_no_prune,
            prune_scope=args.attr_agent_prune_scope,
            apply=not args.attr_agent_dry_run,
            client=client,
            schema=agent_schema,
            nlp=nlp,
            progress=agent_progress,
            report_path=args.attr_agent_report,
            schema_path=args.attr_agent_out,
        )
        log_report(agent_report)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    if args.mode == "hyperedge_compare":
        return _main_hyperedge_compare(
            args, all_data, indices, nlp, client,
        )

    if args.attr_agent_ab:
        log.info("=" * 68)
        log.info("A/B TREATMENT — attribute table + agent delta")
        log.info("=" * 68)

    from src.progress import ConsoleReporter
    eval_progress = ConsoleReporter(
        prefix="treatment" if args.attr_agent_ab else "eval",
    )
    summary, records = _evaluate_sample(
        args, sample, nlp, client, offline_store,
        progress=eval_progress,
    )
    eval_progress.finish({
        "EM": f"{summary.get('em', 0.0):.3f}",
        "F1": f"{summary.get('f1', 0.0):.3f}",
        "R5": f"{summary.get('r5', 0.0):.3f}",
    })

    # ------------------------------------------------------------------
    # A/B delta
    # ------------------------------------------------------------------
    if baseline_summary is not None:
        _log_ab_delta(baseline_summary, summary)
        # Persist both sides next to the main result file.
        out_path = Path(args.out)
        ab_path = out_path.with_name(out_path.stem + "_ab.json")
        with open(ab_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "baseline": baseline_summary,
                    "treatment": summary,
                    "delta": {
                        k: summary[k] - baseline_summary[k]
                        for k in ("em", "f1", "r1", "r5", "r10")
                    },
                    "agent": {
                        "n_gaps": len(agent_report.gaps) if agent_report else 0,
                        "n_accepted": (
                            len(agent_report.accepted) if agent_report else 0
                        ),
                        "n_rejected": (
                            len(agent_report.rejected) if agent_report else 0
                        ),
                        "n_pruned": (
                            len(agent_report.pruned) if agent_report else 0
                        ),
                        "llm_calls": (
                            agent_report.llm_calls if agent_report else 0
                        ),
                        "accepted_attributes": sorted(
                            g.attribute
                            for g in (agent_report.accepted if agent_report else [])
                            if g.attribute
                        ),
                        "accepted_classes": sorted(
                            g.entity_type
                            for g in (agent_report.accepted if agent_report else [])
                            if g.kind == "missing_class"
                        ),
                        "removed_attributes": sorted(
                            agent_report.removed_attributes
                            if agent_report else []
                        ),
                        "removed_classes": sorted(
                            agent_report.removed_classes
                            if agent_report else []
                        ),
                    },
                },
                f, ensure_ascii=False, indent=2,
            )
        log.info("A/B detail → %s", ab_path)

    # ------------------------------------------------------------------
    # Persist the treatment run.
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "records": records,
                "attr_agent": {
                    "ran": run_agent,
                    "ab": bool(args.attr_agent_ab),
                    "n_gaps": len(agent_report.gaps) if agent_report else 0,
                    "n_accepted": (
                        len(agent_report.accepted) if agent_report else 0
                    ),
                    "n_rejected": (
                        len(agent_report.rejected) if agent_report else 0
                    ),
                    "n_pruned": (
                        len(agent_report.pruned) if agent_report else 0
                    ),
                    "accepted_attributes": sorted(
                        g.attribute
                        for g in (agent_report.accepted if agent_report else [])
                        if g.attribute
                    ),
                    "accepted_classes": sorted(
                        g.entity_type
                        for g in (agent_report.accepted if agent_report else [])
                        if g.kind == "missing_class"
                    ),
                    # The entries the prune pass dropped from the table.
                    "removed_attributes": sorted(
                        agent_report.removed_attributes
                        if agent_report else []
                    ),
                    "removed_classes": sorted(
                        agent_report.removed_classes
                        if agent_report else []
                    ),
                },
            },
            f, ensure_ascii=False, indent=2,
        )
    log.info("Results → %s", out_path)

    return 0


# ─────────────────────────────────────────────────────────────────────
# The evaluation loop (extracted so A/B mode can run it twice)
# ─────────────────────────────────────────────────────────────────────
def _evaluate_sample(
    args,
    sample: List[Dict[str, Any]],
    nlp,
    client,
    offline_store,
    progress=None,
    label: str = "eval",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run the EM/F1 evaluation over `sample` and return (summary, records).

    The attribute table that is live in `src.sentence_graph` at the time
    this is called determines the attribute slots every container gets —
    so calling this twice (before and after the agent's delta) yields a
    clean A/B comparison.

    Args:
        progress: Optional progress reporter (see `src/progress.py`).
            When given, a live line shows how many questions are done
            and the running EM/F1/R@k tally.
        label:    The phase name shown on the progress line.
    """
    from src.dispatcher import build_dispatcher
    from src.hyper_edge import HyperEdgeRegistry

    n_offline_para = 0
    n_offline_para_missing = 0

    ems: List[bool] = []
    f1s: List[float] = []
    r1s: List[int] = []
    r5s: List[int] = []
    r10s: List[int] = []
    # R@k 的分母：极性题（gold 是 yes/no）不计入，因为 yes/no 不是
    # 句子片段，没有任何句子能"包含"它。把它记成 0 会系统性压低
    # recall，把它算作命中会虚高 —— 正确做法是排除并报数。
    r_scorable = 0
    r_polar = 0
    r_match_kinds: Counter = Counter()
    # 段落级指标的累加器。key 形如 "passage_recall@5" / "hit@10"。
    #
    # **cut-off 集合是一处单一真相源**（`PASSAGE_KS`）——
    # 累加器、每题计算、汇总表、JSON 落盘全部从它派生。
    # 之前这几处各写各的 `@1/@5/@10` vs `@3/@5/@10`，结果汇总时读的
    # key 根本不存在（`vals = []`），被 `max(3, ...)` 兜成 0 ——
    # **指标静默变成 0，没有任何报错**。这类 bug 必须靠单一真相源根治。
    PASSAGE_KS = (3, 5, 10)
    passage_acc: Dict[str, List[float]] = {
        f"passage_recall@{k}": [] for k in PASSAGE_KS
    }
    passage_acc.update({f"hit@{k}": [] for k in PASSAGE_KS})
    p_scorable = 0
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

        # ─────────────────────────────────────────────────────────────
        # Recall@1 / R@5 / R@10 — gold SUPPORTING-SENTENCE recall.
        #
        # **The old judgement was a substring test**:
        #     gold_norm in _normalize_answer(node.text)
        # which is wrong in BOTH directions (see eval/recall_utils.py
        # for the full analysis):
        #
        #   · False NEGATIVES — gold 'yes' (HotpotQA's "same X?"
        #     questions) appears in no context sentence at all, so those
        #     questions were FORCED to R@k = 0 regardless of retrieval
        #     quality.  In the seed-7 sample that is 1 of 20 questions,
        #     so R@10 = 0.800 was really 0.850.
        #   · False POSITIVES — gold 'Matthew Good' matched
        #     "…on Matthew Good's own imprint…", a sentence about an
        #     unrelated fact.
        #
        # The fix uses the dataset's OWN `supporting_facts` annotation
        # (authoritative), falling back to word-boundary answer
        # containment.
        #
        # Polar questions ('yes'/'no') are EXCLUDED from the R@k
        # denominator rather than scored 0 — a yes/no answer is not a
        # sentence span, so no sentence can "contain" it.
        # ─────────────────────────────────────────────────────────────
        matcher = gold_sentence_matcher(qdata, builder)

        def _sid_contains_gold(sid):
            n = builder.sentence_registry.get_node(sid)
            if n is None:
                return False
            return matcher.contains_answer(sid, n.text or "")

        final_sids = list(
            getattr(res.pipeline_result, "step10_final_sids", []) or []
        ) if res else []

        if matcher.is_scorable:
            r1 = 1 if (final_sids and _sid_contains_gold(final_sids[0])) else 0
            r5 = 1 if any(_sid_contains_gold(s) for s in final_sids[:5]) else 0
            r10 = 1 if any(_sid_contains_gold(s) for s in final_sids[:10]) else 0
            r1s.append(r1); r5s.append(r5); r10s.append(r10)
            r_scorable += 1
            r_match_kinds[matcher.match_kind] += 1
        else:
            # Polar question — R@k is undefined here.  Count it
            # separately so the summary can report "N of M questions
            # were scorable" instead of silently diluting the mean.
            r_polar += 1
            r1 = r5 = r10 = 0
            log.info(
                "  [RECALL-SKIP] %s polar gold=%r — excluded from R@k "
                "(a yes/no answer is not a sentence span)",
                qid[:8], gold,
            )
        r_match_kind = matcher.match_kind

        # ─────────────────────────────────────────────────────────────
        # 段落级指标（Passage Recall@k / Hit@k）
        #
        # 句子级 R@k 衡量的是"gold 支撑句有没有被检索到"，但多跳 QA 的
        # 真正瓶颈是**证据段落** —— HotpotQA 每题有 2 个 gold 段落，
        # 检索器要的是把这两个段落捞进上下文。
        #
        # 论文里的两种标准口径（都是段落粒度）：
        #
        #   口径 A —— Passage Recall@k（MDR / Xiong 2020）
        #     |top-k ∩ gold| / |gold|，允许只抓到一半证据（贡献 0.5）
        #
        #   口径 B —— Hit@k / Pair Recall@k（GraphRAG / LongRAG）
        #     必须全部 gold 段落都在 top-k，缺一个直接 0
        #
        # 详见 eval/recall_utils.py 的说明。两个口径都报，才能和不同
        # 论文的数字对齐。
        #
        # 注意：极性题**不排除**在这里 —— 段落级指标和答案是不是
        # yes/no 无关，它只看证据段落有没有被捞到。
        # ─────────────────────────────────────────────────────────────
        _sent_to_para = build_sentence_to_paragraph(qdata, builder)
        _gold_paras = gold_passage_titles(qdata)
        pmetrics = passage_metrics(
            final_sids, _sent_to_para, _gold_paras, ks=PASSAGE_KS,
        )
        for _k, _v in pmetrics.items():
            if _k == "n_gold_passages":
                continue
            # The key MUST exist in the accumulator.  If it doesn't, the
            # cut-off sets have drifted apart — appending to a missing
            # list would raise KeyError, which is what we want (a loud
            # failure beats a silently-zero metric).
            passage_acc[_k].append(_v)
        p_n_gold = int(pmetrics.get("n_gold_passages", 0))
        if p_n_gold:
            p_scorable += 1

        # ─────────────────────────────────────────────────────────────
        # [DIAG] Pool & gold-sentence audit — locate WHERE the gold
        # sentence is dropped along the retrieval cascade. Only runs
        # the per-sentence scan when r10 == 0 (cheap guard).
        # ─────────────────────────────────────────────────────────────
        diag = {
            "r10_status": ("HIT" if r10 else "MISS"),
            "gold_norm": matcher.gold_norm[:40],
            "gold_match_kind": matcher.match_kind,
            "gold_r_scorable": matcher.is_scorable,
            "n_entities": len(res.pipeline_result.step1_entities) if res else 0,
        }
        gold_sentence_ids: List[str] = []
        gold_in_pool_l1 = False
        gold_in_pool_l2 = False
        gold_in_final = False
        gold_in_any_container = False
        try:
            # The gold supporting sentences come from the dataset's own
            # `supporting_facts` annotation.
            #
            # This used to be a SECOND, independent implementation of the
            # same logic that now lives in `eval/recall_utils.py` (which
            # R@k uses).  Two copies of the same parsing rule is how they
            # drift apart — the R@k path and the DIAG path would then
            # disagree about which sentence is gold, and the diagnostics
            # would contradict the metrics they explain.  One
            # implementation, both callers.
            gold_sentence_ids = sorted(matcher.gold_sids)
            gold_titles = set()
            for _sf in (
                qdata.get("supporting_facts")
                or qdata.get("support_facts") or []
            ):
                _t = (
                    _sf.get("title") if isinstance(_sf, dict)
                    else (_sf[0] if isinstance(_sf, (list, tuple)) and _sf else "")
                )
                if _t:
                    gold_titles.add(str(_t).lower().strip())
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
        # Polar questions are skipped — their R@10 is undefined, so a
        # "MISS" verdict would be a false alarm.
        if r10 == 0 and res is not None and matcher.is_scorable:
            log.warning(
                "[DIAG] %s r10=MISS | entities=%d | n_gold_sents=%d | verdict=%s | cascade=%s | gold_titles=%s",
                qid[:8],
                diag["n_entities"],
                diag.get("n_gold_sentences", 0),
                diag.get("verdict", "?"),
                diag.get("cascade", {}),
                diag.get("gold_titles", []),
            )

        # Polar questions have no defined R@k — show "n/a" rather than a
        # misleading 0, which would look like a retrieval miss.
        def _rk(v):
            return "n/a" if not matcher.is_scorable else str(v)
        log.info(
            "  [%2d/%2d] %s  path=%s  EM=%s F1=%.3f R@1=%s R@5=%s R@10=%s  lat=%.0fms\n"
            "         passage: PR@%d=%.2f Hit@%d=%.0f (gold paras=%d)  |  "
            "Q: %s\n"
            "         G: %s\n"
            "         A: %s",
            i + 1, len(sample), qid[:8],
            res.path if res else "?",
            "✓" if em_hit else "✗", f1, _rk(r1), _rk(r5), _rk(r10),
            latencies[-1],
            PASSAGE_KS[1], pmetrics.get(f"passage_recall@{PASSAGE_KS[1]}", 0.0),
            PASSAGE_KS[1], pmetrics.get(f"hit@{PASSAGE_KS[1]}", 0.0),
            int(pmetrics.get("n_gold_passages", 0)),
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

        # ─────────────────────────────────────────────────────────────
        # [RANK-DIAG] 精排诊断：gold 句子是在哪一环被挤掉的？
        #
        # 用户实测的现象是"recall↑ 但 EM↓" —— 属性把 gold 句子带进了
        # 候选池，却没把它排到第一。这个诊断直接回答：gold 句子
        #   · 在候选池里吗？          （不在 → 是召回层的问题）
        #   · 过了属性门吗？          （没过 → 是属性门校准的问题）
        #   · 排第几？                （排后面 → 是精排公式的问题）
        #   · 五个分量各是多少？      （定位到具体哪个信号失灵）
        #
        # 没有这个诊断，EM 下降只能靠猜。
        # ─────────────────────────────────────────────────────────────
        rank_diag: Dict[str, Any] = {}
        try:
            ef_scores = (
                res.pipeline_result.debug.get("ef_scores", [])
                if res and res.pipeline_result else []
            )
            if ef_scores and gold_sentence_ids:
                gold_set = set(gold_sentence_ids)
                # 按 final_score 排出的名次
                ordered = sorted(
                    ef_scores, key=lambda d: -d.get("final_score", 0.0)
                )
                for rank, d in enumerate(ordered, 1):
                    if d.get("sentence_id") in gold_set:
                        rank_diag = {
                            "gold_in_pool": True,
                            "gold_rank": rank,
                            "gold_gate_kept": d.get("attr_gate_kept"),
                            "gold_gate_level": d.get("attr_gate_level"),
                            "gold_final": round(d.get("final_score", 0.0), 4),
                            # 五个归一化分量 —— 看哪个信号把 gold 压下去了
                            "gold_norm": {
                                "name": round(d.get("norm_name", 0.0), 3),
                                "sent": round(d.get("norm_sent", 0.0), 3),
                                "fact": round(d.get("norm_fact", 0.0), 3),
                                "attr": round(d.get("norm_attr", 0.0), 3),
                                "lex": round(d.get("norm_lex", 0.0), 3),
                            },
                            # 排第一的那个候选人的同一组分量 —— 直接对比
                            "top_norm": (
                                {
                                    "name": round(ordered[0].get("norm_name", 0.0), 3),
                                    "sent": round(ordered[0].get("norm_sent", 0.0), 3),
                                    "fact": round(ordered[0].get("norm_fact", 0.0), 3),
                                    "attr": round(ordered[0].get("norm_attr", 0.0), 3),
                                    "lex": round(ordered[0].get("norm_lex", 0.0), 3),
                                }
                                if ordered else {}
                            ),
                            "n_scored": len(ordered),
                        }
                        break
                if not rank_diag:
                    rank_diag = {
                        "gold_in_pool": False,
                        "n_scored": len(ordered),
                    }
        except Exception as _rd_err:
            rank_diag = {"rank_diag_error": str(_rd_err)}
        if rank_diag:
            log.info(
                "[RANK-DIAG] %s  %s",
                qid[:8],
                json.dumps(rank_diag, ensure_ascii=False),
            )

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
            # 段落级指标（逐题）：passage_recall@k 是口径 A，hit@k 是口径 B。
            "passage": pmetrics,
            "latency_ms": latencies[-1],
            "diag": diag,
            # [RANK-DIAG] gold 句子在精排里的位置与五个分量的取值。
            # 这是定位"recall↑ 但 EM↓"的直接证据：gold_rank 很大 →
            # 精排问题；gold_in_pool=False → 召回问题；
            # gold_gate_kept=False → 属性门校准问题。
            "rank_diag": rank_diag,
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

        # Live progress: how far through the sample we are, plus the
        # running metric tally.  Without this the user sees nothing
        # between the start banner and the final summary.
        if progress is not None:
            n_done = len(ems)
            progress.update(
                n_done, len(sample),
                f"EM={sum(ems) / max(1, n_done):.3f} "
                f"R@5={sum(r5s) / max(1, n_done):.3f}",
            )
            progress.set_extra(
                EM=f"{sum(ems) / max(1, n_done):.3f}",
                F1=f"{sum(f1s) / max(1, n_done):.3f}",
                R5=f"{sum(r5s) / max(1, n_done):.3f}",
            )

    # Summary.
    if not ems:
        log.warning("No questions were evaluated (sample is empty).")
        return {"n": 0, "em": 0.0, "f1": 0.0, "r1": 0.0, "r5": 0.0,
                "r10": 0.0, "paths": {}, "hybrid": {}}, records
    em_avg = sum(ems) / len(ems)
    f1_avg = sum(f1s) / len(f1s)
    # The R@k denominators are the SCORABLE question count — polar
    # questions ('yes'/'no' gold) are excluded because no sentence can
    # "contain" a yes/no answer.  `len(r1s)` already holds exactly that
    # count, so the mean is over the right denominator.
    r1_avg = sum(r1s) / max(1, len(r1s))
    r5_avg = sum(r5s) / max(1, len(r5s))
    r10_avg = sum(r10s) / max(1, len(r10s))
    log.info("=" * 60)
    log.info("n=%d  EM=%.3f  F1=%.3f  R@1=%.3f  R@5=%.3f  R@10=%.3f",
             len(ems), em_avg, f1_avg, r1_avg, r5_avg, r10_avg)
    log.info(
        "Recall denominator: %d scorable / %d total "
        "(%d polar excluded, %s)",
        r_scorable, len(ems), r_polar, dict(r_match_kinds),
    )

    # ─────────────────────────────────────────────────────────────
    # 段落级指标汇总
    #
    # 口径 A —— Passage Recall@k（MDR / Xiong 2020）：
    #     |top-k ∩ gold_passages| / |gold_passages|，逐题平均。
    #     允许只抓到一半证据就得分。
    #
    # 口径 B —— Hit@k（GraphRAG / LongRAG 的 Pair Recall@k）：
    #     必须全部 gold 段落都在 top-k 才算成功，缺一个直接 0。
    #
    # 两者都是段落粒度，这才是多跳检索论文里可比的口径。
    # ─────────────────────────────────────────────────────────────
    passage_summary: Dict[str, float] = {}
    if p_scorable:
        log.info("-" * 60)
        log.info("Passage-level (n=%d queries):", p_scorable)
        # The header is derived from PASSAGE_KS so it can never drift
        # out of sync with the values printed below it.
        log.info(
            "  %-28s %s",
            "metric",
            "".join(f"{('@' + str(k)):>8}" for k in PASSAGE_KS),
        )
        for family, label in (
            ("passage_recall", "Passage Recall@k (口径A)"),
            ("hit", "Hit@k (口径B)"),
        ):
            row = []
            for k in PASSAGE_KS:
                key = f"{family}@{k}"
                vals = passage_acc.get(key) or []
                # The denominator is the number of SCORED questions.
                # Do NOT floor it at 3 — that hides an empty accumulator
                # behind a plausible-looking number.  If the list is
                # empty, report 0 and say so.
                row.append(sum(vals) / len(vals) if vals else 0.0)
            passage_summary[family] = {
                str(k): row[i] for i, k in enumerate(PASSAGE_KS)
            }
            log.info(
                "  %-28s %s",
                label,
                "".join(f"{v:8.3f}" for v in row),
            )
        log.info(
            "  (Hit@k = 所有 gold 段落都在 top-k 才算 1；"
            "Passage Recall@k 允许部分命中)"
        )
    log.info("Path breakdown: %s", dict(paths))
    log.info("Avg latency: %.0fms", sum(latencies) / max(1, len(latencies)))
    log.info("Hybrid config: mode=%s  w_bm25=%.2f  w_dense=%.2f",
             args.hybrid_mode,
             args.bm25_weight if 0 <= args.bm25_weight <= 1 else 0.5,
             1.0 - (args.bm25_weight if 0 <= args.bm25_weight <= 1 else 0.5))

    # ─────────────────────────────────────────────────────────────
    # [RANK-DIAG] 汇总：EM 掉在哪一环？
    #
    # 把每题归到三个互斥的桶里，一眼看出瓶颈：
    #   · 召回丢失  gold 根本没进候选池        → 召回层的问题
    #   · 门杀掉了  gold 进池了但被属性门删掉    → 属性门校准的问题
    #   · 排序丢失  gold 活着但没排到第一      → 精排公式的问题
    # ─────────────────────────────────────────────────────────────
    n_diag = 0
    n_not_in_pool = 0
    n_gated_out = 0
    n_ranked_out = 0
    n_rank1 = 0
    rank_sum = 0
    comp_sum: Dict[str, float] = {
        "name": 0.0, "sent": 0.0, "fact": 0.0, "attr": 0.0, "lex": 0.0,
    }
    top_comp_sum: Dict[str, float] = {
        "name": 0.0, "sent": 0.0, "fact": 0.0, "attr": 0.0, "lex": 0.0,
    }
    for rec in records:
        rd = rec.get("rank_diag") or {}
        if not rd or "gold_in_pool" not in rd:
            continue
        n_diag += 1
        if not rd.get("gold_in_pool"):
            n_not_in_pool += 1
            continue
        if rd.get("gold_gate_kept") is False:
            n_gated_out += 1
        if rd.get("gold_rank") == 1:
            n_rank1 += 1
        if isinstance(rd.get("gold_rank"), int):
            rank_sum += rd["gold_rank"]
        for k in comp_sum:
            comp_sum[k] += float((rd.get("gold_norm") or {}).get(k, 0.0))
            top_comp_sum[k] += float((rd.get("top_norm") or {}).get(k, 0.0))
    if n_diag:
        denom = max(1, n_diag)
        log.info("=" * 60)
        log.info(
            "[RANK-DIAG] n=%d | 召回丢失=%d  门删掉=%d  排序丢失=%d  "
            "gold排第1=%d | gold平均名次=%.1f",
            n_diag,
            n_not_in_pool,
            n_gated_out,
            max(0, n_diag - n_not_in_pool - n_gated_out - n_rank1),
            n_rank1,
            rank_sum / denom,
        )
        log.info(
            "[RANK-DIAG] gold 平均分量: name=%.3f sent=%.3f fact=%.3f "
            "attr=%.3f lex=%.3f",
            comp_sum["name"] / denom, comp_sum["sent"] / denom,
            comp_sum["fact"] / denom, comp_sum["attr"] / denom,
            comp_sum["lex"] / denom,
        )
        log.info(
            "[RANK-DIAG] top1 平均分量: name=%.3f sent=%.3f fact=%.3f "
            "attr=%.3f lex=%.3f",
            top_comp_sum["name"] / denom, top_comp_sum["sent"] / denom,
            top_comp_sum["fact"] / denom, top_comp_sum["attr"] / denom,
            top_comp_sum["lex"] / denom,
        )
        log.info(
            "[RANK-DIAG] gold 相对 top1 的差距: "
            "name=%+.3f sent=%+.3f fact=%+.3f attr=%+.3f lex=%+.3f",
            (comp_sum["name"] - top_comp_sum["name"]) / denom,
            (comp_sum["sent"] - top_comp_sum["sent"]) / denom,
            (comp_sum["fact"] - top_comp_sum["fact"]) / denom,
            (comp_sum["attr"] - top_comp_sum["attr"]) / denom,
            (comp_sum["lex"] - top_comp_sum["lex"]) / denom,
        )
        log.info("=" * 60)

    w_bm25 = args.bm25_weight if 0 <= args.bm25_weight <= 1 else 0.5
    summary: Dict[str, Any] = {
        "n": len(ems),
        "em": em_avg,
        "f1": f1_avg,
        "r1": r1_avg,
        "r5": r5_avg,
        "r10": r10_avg,
        "paths": dict(paths),
        "hybrid": {
            "mode": args.hybrid_mode,
            "w_bm25": w_bm25,
            "w_dense": 1.0 - w_bm25,
        },
        # R@k 的分母信息 —— 极性题被排除，必须随指标一起落盘，
        # 否则跨运行的 R@k 数字不可比（分母不同）。
        "recall_denominator": {
            "n_scorable": r_scorable,
            "n_polar_excluded": r_polar,
            "match_kinds": dict(r_match_kinds),
        },
        # 段落级指标（多跳检索论文的标准口径）。
        #   passage_recall@k —— 口径 A（MDR / Xiong 2020）
        #   hit@k           —— 口径 B（GraphRAG / LongRAG）
        "passage": {
            "n_scorable": p_scorable,
            **passage_summary,
        },
        # 精排瓶颈定位（见上面的 [RANK-DIAG] 汇总日志）
        "rank_diag": {
            "n_diag": n_diag,
            "n_not_in_pool": n_not_in_pool,
            "n_gated_out": n_gated_out,
            "n_rank1": n_rank1,
            "avg_gold_rank": (rank_sum / n_diag) if n_diag else 0.0,
        },
    }
    return summary, records


def _log_ab_delta(
    baseline: Dict[str, Any],
    treatment: Dict[str, Any],
) -> None:
    """Log the metric delta between the baseline and the treatment run."""
    log.info("=" * 68)
    log.info("A/B RESULT   (baseline → treatment)")
    log.info("            EM      F1      R@1     R@5     R@10")
    for key, label in (
        ("em", "EM"), ("f1", "F1"), ("r1", "R@1"),
        ("r5", "R@5"), ("r10", "R@10"),
    ):
        b = baseline.get(key, 0.0)
        t = treatment.get(key, 0.0)
        log.info("  %-6s %.3f → %.3f   (%+.3f)", label, b, t, t - b)
    log.info("=" * 68)


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

    Inlined from the former `eval/hyperedge_eval.py` so that this
    module stays self-contained.
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
    used_norm: set = set()
    for cand in candidates:
        if cand.lower() not in q_lower:
            continue
        cand_norm = norm_map[cand]
        if not cand_norm:
            continue
        # Skip if the candidate's normalised form is already used
        # ("Falling Stars (group 2)" vs "Falling Stars" both collapse).
        if cand_norm in used_norm:
            continue
        # Skip if the candidate is a strict subspan of an already-chosen
        # entity's normalised form.
        subspan_of_prior = any(
            cand_norm in prior_norm and prior_norm != cand_norm
            for prior_norm in used_norm
        )
        if subspan_of_prior:
            continue
        # Reverse: a previously chosen entity is a strict subspan of the
        # candidate.  Replace the shorter one with the longer one.
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


def _normalize_entity_label(label: str) -> str:
    """
    Reduce an entity string to its "core noun phrase" for the purpose
    of de-duplicating multi-form variants of the same referent.

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
    # Collapse "group N" / "part N" / "volume N" suffixes.
    s = re.sub(r"\b(group|part|volume|vol|chapter|section)\s+\d+\b",
               " ", s)
    # Strip year-style tail as a backstop.
    s = re.sub(r"\b\d{4}\b", " ", s)
    # Collapse whitespace.
    s = " ".join(s.split())
    return s


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
    # Same gold-matching rules as the main path (see eval/recall_utils.py):
    # supporting_facts first, word-boundary answer containment as fallback,
    # polar questions excluded from the R@k denominator.
    matcher = gold_sentence_matcher(qdata, builder)

    def _contains_gold(sid):
        n = builder.sentence_registry.get_node(sid)
        if n is None:
            return False
        return matcher.contains_answer(sid, n.text or "")

    def _score(sids, pred):
        if not matcher.is_scorable:
            # Polar question — R@k undefined.  Report -1 so the caller
            # can tell "excluded" apart from "scored 0".
            em = _em_answer(pred, gold) if pred else False
            f1 = _f1(pred, gold) if pred else 0.0
            return -1, -1, -1, em, f1
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
        # -1 = polar question, excluded from R@k.  Show "n/a" rather
        # than a misleading -1 in the per-question log line.
        def _rr(v):
            return "n/a" if v is None or v < 0 else str(v)
        log.info(
            "  [%2d/%2d] %s  base R@1=%s R@5=%s  he R@1=%s R@5=%s  "
            "bridge=%s(+%d)  em_b=%s em_h=%s  (%.0fms)",
            i + 1, len(sample), rec["qid"][:8],
            _rr(b["r1"]), _rr(b["r5"]), _rr(h["r1"]), _rr(h["r5"]),
            "✓" if rec["bridge"]["ran"] else "·",
            rec["bridge"]["n_bridge_sids_added"],
            "Y" if b["em"] else "N", "Y" if h["em"] else "N",
            rec_latency,
        )

    def _m(xs):
        """Mean over the SCORABLE entries only.

        Polar questions ('yes'/'no' gold) are recorded as -1 by
        `_score` — they are excluded from the R@k denominator rather
        than counted as a miss.  Averaging them in would silently
        depress the mean by an amount that depends on how many polar
        questions the sample happens to contain.
        """
        vals = [x for x in xs if x >= 0]
        return sum(vals) / max(1, len(vals))

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
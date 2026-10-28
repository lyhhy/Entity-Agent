"""
Offline HyperEdge Builder
=========================

Pre-builds container-level hyperedges for the entire HotpotQA dev
set, serialises them to Parquet (zstd), and dumps a small summary
JSON so the live pipeline can load them in O(1) per paragraph.

Output schema (one row per paragraph):
    qid             string  -- HotpotQA question id
    paragraph_idx   int     -- 0-based index in q.context
    paragraph_title string  -- canonical title (lowercased)
    container_ids   list[string]
    event_name      string  -- = paragraph_title (PHASE-E1)
    n_sentences     int     -- total sentences in this paragraph
    n_containers    int     -- number of containers produced

Why offline?
    - Eliminates per-question LLM/spaCy cost during evaluation.
    - Eliminates non-determinism (graph builder is stable but
      complex).
    - Removes the dependency on the in-memory graph at retrieval
      time, so dispatchers can be stateless.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Repo imports (run from repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.graph_builder import (
    ComplementedSentence, GraphBuilder, _load_spacy_shared,
)
from src.hyper_edge import HyperEdgeRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("offline_he")


def build_for_one_question(
    qid: str,
    qdata: Dict,
    nlp,
) -> List[Dict]:
    """
    Build hyperedges for one question. Returns a list of row-dicts,
    one per paragraph.
    """
    titles: List[str] = []
    para_sents: List[List[str]] = []
    sentences: List[ComplementedSentence] = []

    for title, sents in qdata.get("context", []):
        titles.append(title)
        para_sents.append(list(sents))
        for sent_text in sents:
            ents: List[str] = []
            doc = nlp(sent_text)
            for ent in doc.ents:
                ents.append(ent.text)
            sentences.append(ComplementedSentence(
                original_text=sent_text,
                complemented_text=sent_text,
                resolved_subject=ents[0] if ents else "",
                entities=ents,
            ))

    if not sentences:
        return []

    builder = GraphBuilder(shared_nlp=nlp)
    for sent in sentences:
        builder.add_sentence(sent)
    cm = builder.container_manager
    for c in cm.get_all_containers():
        cm.update_container_index(c.id)
    builder.compute_tfidf_index()

    # Map each sentence_id back to its paragraph index (the order
    # we appended them).
    sid_to_para: Dict[str, int] = {}
    sid_iter = iter(builder.sentence_registry._nodes.keys())
    for p_idx, sents in enumerate(para_sents):
        for _ in sents:
            try:
                sid = next(sid_iter)
            except StopIteration:
                break
            sid_to_para[sid] = p_idx

    # Build a fresh HyperEdgeRegistry per paragraph (mirrors
    # the legacy quick20_eval flow).
    para_to_cids: Dict[int, List[str]] = {i: [] for i in range(len(titles))}

    he = HyperEdgeRegistry(container_manager=cm)
    for p_idx, title in enumerate(titles):
        # Find containers whose sentences all belong to this paragraph.
        p_containers = []
        seen_cids: set = set()
        for sid, p in sid_to_para.items():
            if p != p_idx:
                continue
            node = builder.sentence_registry.get_node(sid)
            if not node:
                continue
            for cid in node.containers:
                if cid in seen_cids:
                    continue
                seen_cids.add(cid)
                c_obj = cm.get_container(cid)
                if c_obj is not None:
                    p_containers.append(c_obj)
        if not p_containers:
            continue
        new_edges = he.build_hyperedges_for_chunk(
            chunk_id=str(title),
            containers=p_containers,
            time_word="",
            paragraph_title=str(title),
        )
        for edge in new_edges:
            for cid in edge.container_ids:
                if cid not in para_to_cids[p_idx]:
                    para_to_cids[p_idx].append(cid)

    # Emit one row per paragraph.
    rows: List[Dict] = []
    for p_idx, title in enumerate(titles):
        cids = para_to_cids.get(p_idx, [])
        rows.append({
            "qid": qid,
            "paragraph_idx": p_idx,
            "paragraph_title": title,
            "container_ids": cids,
            "event_name": title,
            "n_sentences": len(para_sents[p_idx]) if p_idx < len(para_sents) else 0,
            "n_containers": len(cids),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline hyperedge builder (writes Parquet).",
    )
    ap.add_argument(
        "--data",
        default="eval/data/hotpot_dev_distractor_v1.json",
        help="HotpotQA JSON path (default: dev distractor v1).",
    )
    ap.add_argument(
        "--out-dir",
        default="eval/offline_hyperedges",
        help="Output directory for the Parquet + summary JSON.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only process the first N questions (smoke test).",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based, inclusive) for batching.",
    )
    ap.add_argument(
        "--end",
        type=int,
        default=0,
        help="End index (0-based, exclusive) for batching. 0 means len(data).",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="Append a suffix to the parquet file name (for batch builds).",
    )
    ap.add_argument(
        "--spacy-model",
        default="en_core_web_lg",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    parquet_path = out_dir / f"hyperedges{suffix}.parquet"
    summary_path = out_dir / f"summary{suffix}.json"

    log.info("Loading %s ...", args.data)
    with open(args.data, "r", encoding="utf-8") as fh:
        all_data = json.load(fh)
    if args.limit > 0:
        all_data = all_data[: args.limit]
    # Apply --start/--end (batching).
    start = max(0, args.start)
    end = args.end if args.end > 0 else len(all_data)
    all_data = all_data[start:end]
    log.info("Range [%d:%d] → %d questions.", start, end, len(all_data))

    log.info("Loading spaCy %s ...", args.spacy_model)
    nlp = _load_spacy_shared(args.spacy_model)

    rows: List[Dict] = []
    t0 = time.time()
    n_fail = 0
    for i, qdata in enumerate(all_data):
        qid = qdata.get("_id", "")
        try:
            r = build_for_one_question(qid, qdata, nlp)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            log.warning("Q[%d] %s failed: %s", i, qid[:8], exc)
            r = []
        rows.extend(r)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            log.info(
                "  [%d/%d] rows=%d  fail=%d  %.1fs",
                i + 1, len(all_data), len(rows), n_fail, elapsed,
            )

    log.info("Total rows: %d (failures: %d)", len(rows), n_fail)
    log.info("Writing Parquet (zstd) to %s ...", parquet_path)
    df = pd.DataFrame(rows)
    # column order
    df = df[[
        "qid", "paragraph_idx", "paragraph_title", "container_ids",
        "event_name", "n_sentences", "n_containers",
    ]]
    df.to_parquet(
        parquet_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    # Summary stats
    n_questions = df["qid"].nunique()
    avg_para = df.groupby("qid").size().mean() if n_questions else 0.0
    avg_containers = df["n_containers"].mean() if len(df) else 0.0
    total_containers = df["container_ids"].apply(len).sum()
    summary = {
        "n_questions": int(n_questions),
        "n_paragraphs": int(len(df)),
        "n_containers_total": int(total_containers),
        "avg_paragraphs_per_question": float(avg_para),
        "avg_containers_per_paragraph": float(avg_containers),
        "n_failures": int(n_fail),
        "elapsed_seconds": float(time.time() - t0),
        "parquet_path": str(parquet_path),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log.info("Summary: %s", json.dumps(summary, ensure_ascii=False, indent=2))
    log.info("Done. Files in %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

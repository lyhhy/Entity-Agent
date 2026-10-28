"""
diag_fuzzy_retrieval.py — end-to-end trace of the retrieval path
==================================================================

Answers three questions for a given seed, per question:

  1. **Is fuzzy matching actually used, and does it fire?**
     Reports, for every question: the config flags
     (`fuzzy_match`, `strict_entity_match`, `w_bm25`/`w_dense`,
     `faiss_threshold`), the query entities, how many containers each
     retrieval channel contributed (substring / strict / fuzzy /
     hybrid), and — for each fuzzy hit — *why* it matched (signal type,
     score, matched surface).

  2. **Are the gold container and the gold sentence recalled?**
     Gold containers are derived from HotpotQA's `supporting_facts`
     (paragraph title → the containers that hold that paragraph's
     sentences).  Gold sentences are the sentences at the supporting
     fact indices.  For each question we report:
       * gold containers hit by A (the seed set)
       * gold containers hit anywhere in the candidate pool
       * gold sentence present in the pool
       * gold sentence present in the final top-K
     This isolates *recall loss* from *ranking loss*.

  3. **Why is the final answer wrong?**
     For every failure we classify the root cause:
       * RETRIEVAL_MISS      — the gold sentence never entered the pool
       * RANK_MISS           — it was in the pool but not in the top-K
       * ANSWER_EXTRACTION   — the gold sentence WAS in the top-K but
                               the LLM answered something else
       * NO_GOLD_SENTENCE    — the answer string does not literally
                               appear in any supporting sentence

Usage:
    python eval/diag_fuzzy_retrieval.py --seed 137 --n 20 --use-llm \\
        --llm-base-url http://127.0.0.1:8013/v1 --llm-model llama-3.1-8b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import quick20_eval as q20  # noqa: E402

_NORMALIZE = q20._normalize_answer
_EM = q20._em_score
_F1 = q20._f1_score
_BUILD_GRAPH = q20.build_graph_from_hotpot
_EXTRACT_ANSWER = q20.extract_answer_from_dispatch
_APPLY_HYBRID = q20._apply_hybrid_config


def _p(s: str) -> None:
    """Print that survives a GBK console."""
    try:
        sys.stdout.write(s)
        sys.stdout.flush()
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        sys.stdout.write(
            s.encode(enc, errors="replace").decode(enc, errors="replace")
        )
        sys.stdout.flush()


def _gold_paragraph_titles(qdata: Dict[str, Any]) -> List[str]:
    """Paragraph titles that carry a supporting fact."""
    out: List[str] = []
    for sf in qdata.get("supporting_facts") or []:
        if isinstance(sf, (list, tuple)) and sf:
            title = str(sf[0])
            if title not in out:
                out.append(title)
    return out


def _gold_sentence_texts(qdata: Dict[str, Any]) -> Set[str]:
    """Exact texts of the supporting sentences."""
    out: Set[str] = set()
    facts = qdata.get("supporting_facts") or []
    ctx = qdata.get("context") or []
    for sf in facts:
        if not (isinstance(sf, (list, tuple)) and len(sf) >= 2):
            continue
        title, idx = str(sf[0]), int(sf[1])
        for t, sents in ctx:
            if str(t) == title and 0 <= idx < len(sents):
                out.add((sents[idx] or "").strip())
    return out


def _build_gold_container_map(builder, qdata: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    gold paragraph title -> set of container ids holding that paragraph.

    Resolves paragraph sentences to graph nodes by exact text match,
    the same way the harness's `bridge_titles` index is built.
    """
    ctx = qdata.get("context") or []
    out: Dict[str, Set[str]] = {}
    nodes = builder.sentence_registry._nodes.values()
    for title, sents in ctx:
        cids: Set[str] = set()
        for s_text in sents:
            target = (s_text or "").strip()
            for node in nodes:
                if (node.text or "").strip() == target:
                    cids.update(node.containers or [])
                    break
        if cids:
            out[str(title)] = cids
    return out


def _instrument_channels(pipeline, entities: List[str], query: str) -> Dict[str, Any]:
    """Run each Step-2 channel separately and record what each found."""
    sub_ids: Set[str] = set()
    strict_ids: Set[str] = set()
    for ent in entities:
        for c in pipeline.cm.get_containers_for_entity(ent) or []:
            cid = getattr(c, "id", None)
            if cid:
                sub_ids.add(cid)
        for cid, _s in pipeline._strict_entity_match(ent) or []:
            strict_ids.add(cid)

    fuzzy_pairs = pipeline._fuzzy_match_containers(
        entities, threshold=pipeline.fuzzy_threshold,
    )
    fuzzy_ids = {cid for cid, _ in fuzzy_pairs}

    hybrid = pipeline._hybrid_search(query, top_k=pipeline.faiss_top_k)
    hybrid_ids = {cid for cid, *_ in hybrid}

    bm25 = pipeline.cm._bm25_search(query, top_k=50)
    dense = pipeline.cm._faiss_search(
        query, top_k=pipeline.faiss_top_k, threshold=pipeline.faiss_threshold,
    )

    return {
        "substring_ids": sub_ids,
        "strict_ids": strict_ids,
        "fuzzy_ids": fuzzy_ids,
        "fuzzy_pairs": fuzzy_pairs,
        "hybrid_ids": hybrid_ids,
        "bm25_n": len(bm25),
        "dense_n": len(dense),
        "dense_threshold": pipeline.faiss_threshold,
    }


def _classify_failure(
    gold_in_pool: bool,
    gold_in_final: bool,
    em_hit: bool,
    gold_text_found_anywhere: bool,
) -> str:
    if em_hit:
        return "OK"
    if not gold_text_found_anywhere:
        return "NO_GOLD_SENTENCE"
    if not gold_in_pool:
        return "RETRIEVAL_MISS"
    if not gold_in_final:
        return "RANK_MISS"
    return "ANSWER_EXTRACTION"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=137)
    ap.add_argument(
        "--data", default="eval/data/hotpot_dev_distractor_v1.json"
    )
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--llm-base-url", default="")
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-fuzzy", action="store_true",
                    help="Disable the full fuzzy matcher (A/B against it).")
    ap.add_argument("--strict-entity-match", action="store_true")
    ap.add_argument("--hybrid-mode", default="hybrid")
    ap.add_argument("--bm25-weight", type=float, default=-1.0)
    args = ap.parse_args()

    logging.getLogger().handlers = []
    logging.getLogger().setLevel(logging.ERROR)
    for nm in ("src", "src.stengraph_pipeline", "src.hyper_edge",
               "src.dispatcher", "faiss", "sentence_transformers"):
        lg = logging.getLogger(nm)
        lg.handlers = []
        lg.propagate = False
        lg.setLevel(logging.ERROR)

    from src.graph_builder import _load_spacy_shared
    from src.hyper_edge import HyperEdgeRegistry
    from src.dispatcher import build_dispatcher

    with open(args.data, encoding="utf-8") as f:
        all_data = json.load(f)
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(all_data)), min(args.n, len(all_data))))
    sample = [all_data[i] for i in indices]

    nlp = _load_spacy_shared("en_core_web_lg")
    if nlp is None:
        _p("[FATAL] spaCy unavailable\n")
        return 1

    client = None
    if args.use_llm:
        from src.llm_client import make_client
        base_url = args.llm_base_url or os.environ.get(
            "STENRAG_LLM_BASE_URL", "").strip()
        if base_url:
            from src.local_llm_server import ensure_server
            try:
                base_url = ensure_server(
                    base_url=base_url, model=args.llm_model or "")
            except Exception as e:
                _p(f"[FATAL] local LLM failed: {e}\n")
                return 1
        client = make_client(
            base_url=base_url or None, model=args.llm_model or None)
        if not client.enabled:
            _p("[FATAL] no usable LLM config\n")
            return 1
        _p(f"[setup] llm={client.model} @ {client.base_url}\n")

    _p(f"[setup] seed={args.seed} n={len(sample)} "
       f"fuzzy={not args.no_fuzzy} strict={args.strict_entity_match}\n")
    _p("=" * 100 + "\n")

    tally: Dict[str, int] = {}
    n_em = n_f1 = 0.0
    r1 = r5 = r10 = 0.0
    g_c_in_a = g_c_in_pool = g_s_in_pool = 0.0
    fuzz_fired = 0
    fuzz_only = 0
    per_q: List[Dict[str, Any]] = []

    for qi, qdata in enumerate(sample, 1):
        qid = qdata["_id"]
        question = qdata["question"]
        gold = qdata["answer"]

        t0 = time.time()
        builder, sentences, titles, para_sents, bridge_titles = _BUILD_GRAPH(
            qdata, nlp)
        cm = builder.container_manager
        for c in cm.get_all_containers():
            cm.update_container_index(c.id)
        builder.compute_tfidf_index()

        he = HyperEdgeRegistry(container_manager=cm)
        for p_title, p_cids in (bridge_titles or {}).items():
            p_containers = [
                cm._containers.get(cid) for cid in p_cids
                if cm._containers.get(cid) is not None
            ]
            if p_containers:
                he.build_hyperedges_for_chunk(
                    chunk_id=str(p_title),
                    containers=p_containers,
                    time_word="",
                    paragraph_title=str(p_title),
                )

        disp = build_dispatcher(
            builder, client=client, nlp=nlp,
            hyperedge_registry=he,
            strict_entity_match=bool(args.strict_entity_match),
            fuzzy_match=not args.no_fuzzy,
        )
        _APPLY_HYBRID(disp.pipeline, args.hybrid_mode, args.bm25_weight)
        pipeline = disp.pipeline

        entities = pipeline._step1_query_analysis(question)
        chans = _instrument_channels(pipeline, entities, question)

        res = disp.run(question, bridge_titles=bridge_titles)
        pr = res.pipeline_result
        final_sids = list(getattr(pr, "step10_final_sids", []) or [])
        pool_sids = list(getattr(pr, "debug", {}).get("step6_pool_sids", []) or [])
        a_cids = list(getattr(pr, "step2_a_containers", []) or [])

        answer = _EXTRACT_ANSWER(
            res, builder, gold, nlp, client=client, question=question)
        em_hit = bool(_EM(answer, gold))
        f1 = float(_F1(answer, gold))

        # ---- gold containers / sentences ----
        gold_cmap = _build_gold_container_map(builder, qdata)
        gold_titles = _gold_paragraph_titles(qdata)
        gold_cids: Set[str] = set()
        for t in gold_titles:
            gold_cids |= gold_cmap.get(t, set())

        gold_texts = _gold_sentence_texts(qdata)
        node_by_sid = builder.sentence_registry._nodes

        a_set = set(a_cids)

        # The pool holds *sentence* ids; map them back to containers so we
        # can ask "did the gold container contribute any sentence?".
        pool_cids: Set[str] = set()
        for sid in pool_sids:
            n = node_by_sid.get(sid)
            if n is not None:
                pool_cids.update(n.containers or [])
        gold_c_in_a = len(gold_cids & a_set)
        gold_c_in_pool = len(gold_cids & pool_cids)

        def sid_text(sid: str) -> str:
            n = node_by_sid.get(sid)
            return (n.text or "") if n else ""

        gold_sids_pool = {
            sid for sid in pool_sids if sid_text(sid).strip() in gold_texts
        }
        gold_sids_final = [
            sid for sid in final_sids if sid_text(sid).strip() in gold_texts
        ]
        gold_s_in_pool = 1 if gold_sids_pool else 0
        gold_rank = (
            final_sids.index(gold_sids_final[0]) + 1
            if gold_sids_final else None
        )
        gold_in_final = 1 if gold_sids_final else 0

        gold_norm = _NORMALIZE(gold)
        gold_text_found_anywhere = any(
            gold_norm in _NORMALIZE(n.text or "")
            for n in node_by_sid.values()
        )

        cause = _classify_failure(
            bool(gold_sids_pool), bool(gold_sids_final),
            em_hit, gold_text_found_anywhere,
        )
        tally[cause] = tally.get(cause, 0) + 1

        # ---- channel contribution ----
        sub, fz = chans["substring_ids"], chans["fuzzy_ids"]
        strict = chans["strict_ids"]
        if fz:
            fuzz_fired += 1
        if (fz - sub) and not (sub | strict):
            fuzz_only += 1

        n_em += em_hit
        n_f1 += f1
        this_r1 = 1 if (final_sids and final_sids[0] in gold_sids_final) else 0
        this_r5 = 1 if any(s in gold_sids_final for s in final_sids[:5]) else 0
        this_r10 = 1 if any(s in gold_sids_final for s in final_sids[:10]) else 0
        r1 += this_r1
        r5 += this_r5
        r10 += this_r10
        g_c_in_a += gold_c_in_a
        g_c_in_pool += gold_c_in_pool
        g_s_in_pool += gold_s_in_pool

        # ---- fuzzy detail ----
        fz_detail = []
        for cid, score in chans["fuzzy_pairs"][:6]:
            meta = pipeline._last_fuzzy_debug.get(cid, {})
            c = cm.get_container(cid)
            fz_detail.append({
                "cid": cid,
                "entity": meta.get("entity"),
                "score": meta.get("score"),
                "type": meta.get("type"),
                "surface": meta.get("surface"),
                "container_entity": getattr(c, "entity", "") if c else "",
                "is_gold_container": cid in gold_cids,
            })

        _p(f"[{qi:>2}/{len(sample)}] {qid[:8]}  EM={'Y' if em_hit else 'N'} "
           f"F1={f1:.3f} R@1={this_r1} R@5={this_r5} R@10={this_r10}  "
           f"cause={cause}\n")
        _p(f"   Q: {question}\n")
        _p(f"   G: {gold}   A: {answer}\n")
        _p(f"   ents={entities}\n")
        _p(f"   channels: substring={len(sub)} strict={len(strict)} "
           f"fuzzy={len(fz)} hybrid={len(chans['hybrid_ids'])} | "
           f"bm25_hits={chans['bm25_n']} "
           f"dense_hits(thr={chans['dense_threshold']})={chans['dense_n']}\n")
        _p(f"   gold_c: in_A={gold_c_in_a}/{len(gold_cids)} "
           f"in_pool={gold_c_in_pool}/{len(gold_cids)} | "
           f"gold_s: in_pool={gold_s_in_pool} in_final={gold_in_final} "
           f"rank={gold_rank} pool={len(pool_sids)} final={len(final_sids)}\n")
        if fz_detail:
            top = ", ".join(
                f"{d['cid']}[{d['type']}={d['score']} '{d['surface']}'"
                f"{'*' if d['is_gold_container'] else ''}]"
                for d in fz_detail[:4]
            )
            _p(f"   fuzzy hits: {top}\n")
        _p("\n")

        # ---- rerank breakdown for failures ----
        ef = list(getattr(pr, "debug", {}).get("ef_scores", []) or [])
        ef_by_sid = {d["sentence_id"]: d for d in ef}
        dump: List[Dict[str, Any]] = []
        if cause in ("RANK_MISS", "ANSWER_EXTRACTION"):
            for rank, sid in enumerate(final_sids[:6], 1):
                d = ef_by_sid.get(sid, {})
                dump.append({
                    "rank": rank, "sid": sid,
                    "is_gold": sid in gold_sids_final,
                    "text": sid_text(sid)[:150],
                    "S_h_E": d.get("S_h_E"), "S_h_F": d.get("S_h_F"),
                    "event_term": d.get("event_term"),
                    "fact_term": d.get("fact_term"),
                    "attr_term": d.get("attr_term"),
                    "echo_penalty": d.get("echo_penalty"),
                    "final_score": d.get("final_score"),
                })
            if cause == "RANK_MISS" and gold_sids_pool:
                for sid in list(gold_sids_pool)[:3]:
                    d = ef_by_sid.get(sid, {})
                    dump.append({
                        "rank": None, "sid": sid, "is_gold": True,
                        "text": sid_text(sid)[:150],
                        "S_h_E": d.get("S_h_E"), "S_h_F": d.get("S_h_F"),
                        "event_term": d.get("event_term"),
                        "fact_term": d.get("fact_term"),
                        "attr_term": d.get("attr_term"),
                        "echo_penalty": d.get("echo_penalty"),
                        "final_score": d.get("final_score"),
                    })
            _p("   -- rerank breakdown (failure) --\n")
            for d in dump:
                mark = "*" if d["is_gold"] else " "
                _p(f"    {mark} rank={str(d['rank']):>4} "
                   f"S_h_E={d['S_h_E']} S_h_F={d['S_h_F']} "
                   f"ev={d['event_term']} fact={d['fact_term']} "
                   f"attr={d['attr_term']} echo={d['echo_penalty']} "
                   f"FINAL={d['final_score']} | {d['text'][:90]}\n")
            _p("\n")

        per_q.append({
            "qid": qid, "question": question, "gold": gold,
            "answer": answer, "em": em_hit, "f1": f1, "cause": cause,
            "entities": entities,
            "channels": {
                "substring": len(sub), "strict": len(strict),
                "fuzzy": len(fz), "hybrid": len(chans["hybrid_ids"]),
                "bm25_hits": chans["bm25_n"], "dense_hits": chans["dense_n"],
                "dense_threshold": chans["dense_threshold"],
            },
            "fuzzy_detail": fz_detail,
            "gold_stats": {
                "n_gold_containers": len(gold_cids),
                "gold_containers_in_A": gold_c_in_a,
                "gold_containers_in_pool": gold_c_in_pool,
                "gold_sentence_in_pool": gold_s_in_pool,
                "gold_sentence_in_final": gold_in_final,
                "gold_rank": gold_rank,
            },
            "pool_size": len(pool_sids),
            "final_size": len(final_sids),
            "rerank_breakdown": dump,
            "latency_ms": (time.time() - t0) * 1000,
        })

    n = len(sample)
    _p("=" * 100 + "\n")
    _p(f"SUMMARY seed={args.seed} n={n}\n")
    _p(f"  EM={n_em/n:.3f}  F1={n_f1/n:.3f}  "
       f"R@1={r1/n:.3f}  R@5={r5/n:.3f}  R@10={r10/n:.3f}\n")
    _p(f"  gold containers in A   : {g_c_in_a/n:.3f} per question\n")
    _p(f"  gold containers in pool: {g_c_in_pool/n:.3f} per question\n")
    _p(f"  gold sentence in pool  : {g_s_in_pool/n:.3f}\n")
    _p(f"  fuzzy fired on {fuzz_fired}/{n} questions; "
       f"fuzzy-only rescue on {fuzz_only}/{n}\n")
    _p("  failure causes:\n")
    for k in sorted(tally, key=lambda x: -tally[x]):
        _p(f"    {k:<20} {tally[k]:>3}/{n}\n")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump({
                "seed": args.seed, "n": n,
                "em": n_em / n, "f1": n_f1 / n,
                "r1": r1 / n, "r5": r5 / n, "r10": r10 / n,
                "tally": tally,
                "fuzzy_fired": fuzz_fired,
                "fuzzy_only_rescue": fuzz_only,
                "questions": per_q,
            }, f, ensure_ascii=False, indent=2)
        _p(f"  -> {outp}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

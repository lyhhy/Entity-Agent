"""
Isolate the S_h_E formula change from LLM variance.

For each seed-137 question:
  1. Build the graph, run the pipeline ONCE to get the candidate pool
     (step6) — retrieval only, NO LLM.
  2. Score that same pool with the NEW formula  → gold ranks (new).
  3. Score that same pool with the OLD formula  → gold ranks (old).
  4. Compare gold-rank movement and R@1/R@5/R@10.

Same pool, same embeddings, same hyperedges — the ONLY difference is the
scoring formula.  This removes LLM variance entirely.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, str(REPO / "eval"))

import logging
logging.basicConfig(level=logging.WARNING)

from src.graph_builder import _load_spacy_shared
from src.hyper_edge import HyperEdgeRegistry
import quick20_eval as q20


def old_formula_scores(pipeline, query, sids):
    """Recompute the OLD S_h_E ranking for the same pool.

    OLD:  S_h(E,Q) = 0.5*cos(name,q) + 0.5*cos(mean_member_vec,q)
          event_term = W_EVENT * S_h(E,Q) * penalty
          fact_term  = (1-W_EVENT) * cos(sentence_vec,q)

    Returns list of (sid, final_score) using the same attr term.
    """
    import numpy as np
    import src.stengraph_pipeline as sgp

    q_vec = pipeline._encode_query(query)
    embedder = pipeline._get_embedder()
    if q_vec is None:
        return None

    # S_h_F per sentence.
    s_h_f = {sid: float(sim) for sid, sim in
             pipeline._score_cosine(query, list(sids), embedder)}

    # Build hyperedge index exactly like _event_fact_score does.
    events = {}
    sid_events = {sid: [] for sid in sids}
    for sid in sids:
        node = pipeline.registry.get_node(sid)
        if node is None:
            continue
        for cid in getattr(node, "containers", []) or []:
            try:
                he_list = pipeline.he_registry.get_hyperedges_for_container(cid) or []
            except Exception:
                he_list = []
            for he in he_list:
                he_id = getattr(he, "id", None) or f"he_{len(events)}"
                if he_id not in events:
                    events[he_id] = {"name": (getattr(he, "event_name", "") or "").strip(),
                                     "members": [], "penalty": 1.0}
                if sid not in events[he_id]["members"]:
                    events[he_id]["members"].append(sid)
                if he_id not in sid_events[sid]:
                    sid_events[sid].append(he_id)

    for he_id, ev in events.items():
        # name vec
        name_vec = None
        if ev["name"] and embedder is not None:
            try:
                nv = np.asarray(embedder.encode([ev["name"]],
                                                normalize_embeddings=True)[0],
                                dtype=np.float32).reshape(-1)
                n = np.linalg.norm(nv)
                if n > 0:
                    name_vec = nv / n
            except Exception:
                name_vec = None
        # mean member vec
        mvecs = []
        for m_sid in ev["members"]:
            mn = pipeline.registry.get_node(m_sid)
            sv = getattr(mn, "summary_vector", None) if mn else None
            if sv is None:
                continue
            try:
                v = np.asarray(sv, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if v.size:
                mvecs.append(v)
        mean_vec = None
        if mvecs:
            mv = np.stack(mvecs, axis=0).mean(axis=0)
            n = np.linalg.norm(mv)
            if n > 0:
                mean_vec = mv / n

        # penalty (same computation as the pipeline).
        try:
            he_obj = pipeline.he_registry.get_hyperedge(he_id)
            all_sents = set()
            if he_obj is not None and getattr(he_obj, "container_ids", None):
                for cid in he_obj.container_ids:
                    cobj = pipeline.cm.get_container(cid)
                    if cobj is None:
                        continue
                    for msid in (getattr(cobj, "sentence_ids", []) or []):
                        all_sents.add(msid)
            if not all_sents:
                all_sents = set(ev["members"])
            sz = len(all_sents)
            ev["penalty"] = 1.0 if sz <= 1 else float(1.0 / ((sz - 1) ** sgp.GAMMA_E))
        except Exception:
            ev["penalty"] = 1.0

        if q_vec is None:
            ev["S_h_E"] = 0.0
            continue
        parts = []
        if name_vec is not None:
            parts.append((sgp._cos(name_vec, q_vec), 0.5))
        if mean_vec is not None:
            parts.append((sgp._cos(mean_vec, q_vec), 0.5 if name_vec is not None else 1.0))
        if not parts:
            ev["S_h_E"] = 0.0
        else:
            num = sum(c * w for c, w in parts)
            den = sum(w for _, w in parts)
            ev["S_h_E"] = float(num / den) if den > 0 else 0.0

    # Compose final score with the OLD formula.
    # The attr term is identical for both formulas, so we re-derive it
    # from the pipeline's own breakdown (already computed for NEW) to
    # keep the comparison apples-to-apples.
    new_scored = pipeline._last_ef_scored or []
    attr_by_sid = {d["sentence_id"]: d for d in new_scored}

    out = []
    for sid in sids:
        sh_f = s_h_f.get(sid, 0.0)
        # ORIGINAL selection semantics: pick the event with the highest
        # RAW S_h_E (no penalty), THEN apply the penalty.  Selecting on
        # the penalised value instead would be a different formula.
        best_raw = 0.0
        best_pen = 1.0
        for eid in sid_events.get(sid, []):
            ev = events[eid]
            raw = ev["S_h_E"]
            if raw > best_raw:
                best_raw = raw
                best_pen = ev.get("penalty", 1.0)
        event_term = sgp.W_EVENT * best_raw * best_pen
        fact_term = (1.0 - sgp.W_EVENT) * sh_f
        s_f = event_term + fact_term
        d = attr_by_sid.get(sid, {})
        attr_term = d.get("attr_term", 0.0)
        out.append((sid, s_f + attr_term))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=137)
    args = ap.parse_args()

    nlp = _load_spacy_shared("en_core_web_lg")
    if nlp is None:
        print("[FATAL] spaCy unavailable")
        return 1

    with open(REPO / "eval/data/hotpot_dev_distractor_v1.json", encoding="utf-8") as f:
        all_data = json.load(f)
    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(len(all_data)), min(args.n, len(all_data))))
    sample = [all_data[i] for i in idx]

    from src.dispatcher import build_dispatcher

    rows = []
    for qi, qdata in enumerate(sample):
        qid = qdata["_id"]
        question = qdata["question"]
        gold = qdata.get("answer", "")
        builder, sentences, titles, para_sents, bridge_titles = (
            q20.build_graph_from_hotpot(qdata, nlp))
        cm = builder.container_manager
        for c in cm.get_all_containers():
            cm.update_container_index(c.id)
        builder.compute_tfidf_index()

        he = HyperEdgeRegistry(container_manager=cm)
        for _p_title, _p_cids in (bridge_titles or {}).items():
            _p_containers = [cm._containers.get(_cid) for _cid in _p_cids
                             if cm._containers.get(_cid) is not None]
            if _p_containers:
                he.build_hyperedges_for_chunk(
                    chunk_id=str(_p_title), containers=_p_containers,
                    time_word="", paragraph_title=str(_p_title))

        disp = build_dispatcher(
            builder, client=None, nlp=nlp, hyperedge_registry=he,
            strict_entity_match=False, attr_soft_match=True,
        )
        q20._apply_hybrid_config(disp.pipeline, "hybrid", 0.5)
        res = disp.run(question, bridge_titles=bridge_titles)
        pr = res.pipeline_result
        pool = list(getattr(pr, "debug", {}).get("step6_pool_sids", []) or [])
        new_final = list(getattr(pr, "step10_final_sids", []) or [])

        old_scored = old_formula_scores(disp.pipeline, question, pool)
        if old_scored is None:
            print(f"[{qi+1}/{len(sample)}] {qid[:8]} SKIP (no embedder)")
            continue
        old_final = [sid for sid, _ in sorted(old_scored,
                                             key=lambda x: -x[1])][:len(new_final)]

        gold_norm = q20._normalize_answer(gold)

        def grank(final):
            for i, sid in enumerate(final, 1):
                n = builder.sentence_registry.get_node(sid)
                if n is not None and gold_norm in q20._normalize_answer(n.text or ""):
                    return i
            return None

        def rk(final, k):
            g = grank(final)
            return 1 if (g is not None and g <= k) else 0

        gb, gn = grank(old_final), grank(new_final)
        rows.append({
            "qid": qid, "question": question, "gold": gold,
            "pool": len(pool),
            "gold_old": gb, "gold_new": gn,
            "old": {"r1": rk(old_final, 1), "r5": rk(old_final, 5),
                    "r10": rk(old_final, 10)},
            "new": {"r1": rk(new_final, 1), "r5": rk(new_final, 5),
                    "r10": rk(new_final, 10)},
        })
        print(f"[{qi+1}/{len(sample)}] {qid[:8]} pool={len(pool):<3} "
              f"gold_old={str(gb):<5} gold_new={str(gn):<5} "
              f"R@1 {rk(old_final,1)}→{rk(new_final,1)}  "
              f"R@5 {rk(old_final,5)}→{rk(new_final,5)}  "
              f"R@10 {rk(old_final,10)}→{rk(new_final,10)}",
              flush=True)

    # ── Aggregate ──
    n = len(rows)
    if not n:
        return 1
    print("\n" + "=" * 78)
    print("ISOLATED FORMULA A/B  (same pool, same embeddings, NO LLM)")
    print("=" * 78)
    for k in ("r1", "r5", "r10"):
        o = sum(r["old"][k] for r in rows) / n
        w = sum(r["new"][k] for r in rows) / n
        print(f"  R@{k:<3} old={o:.3f}  new={w:.3f}  delta={w-o:+.3f}")
    improved = sum(1 for r in rows
                   if r["gold_old"] is not None and r["gold_new"] is not None
                   and r["gold_new"] < r["gold_old"])
    worsened = sum(1 for r in rows
                   if r["gold_old"] is not None and r["gold_new"] is not None
                   and r["gold_new"] > r["gold_old"])
    only_old = sum(1 for r in rows
                   if r["gold_old"] is not None and r["gold_new"] is None)
    only_new = sum(1 for r in rows
                   if r["gold_old"] is None and r["gold_new"] is not None)
    both = sum(1 for r in rows
               if r["gold_old"] is not None and r["gold_new"] is not None)
    print(f"\n  gold rank improved : {improved}")
    print(f"  gold rank worsened : {worsened}")
    print(f"  gold only in OLD   : {only_old}   (regression)")
    print(f"  gold only in NEW   : {only_new}   (gain)")
    print(f"  gold in both       : {both}")
    out = REPO / "eval" / "diag" / "formula_ab_seed137.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Minimal sparse filter diagnostic - no external imports needed."""
import json, random, sys, os

# Use the eval/ dir as root
_ROOT = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.dirname(_ROOT)   # StenRAG/
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from src.graph_builder import GraphBuilder, ComplementedSentence
import spacy

def build_graph(qdata, nlp):
    context = qdata.get("context", [])
    sentences = []
    for title, para in context:
        for sent in para:
            sentences.append(ComplementedSentence(
                original_text=sent,
                complemented_text=sent,
                resolved_subject=title,
                entities=[]))
    builder = GraphBuilder(shared_nlp=nlp)
    for s in sentences:
        builder.add_sentence(s)
    return builder

def main():
    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_lg")

    print("Loading hotpot data...")
    with open(os.path.join(_PKG, "eval", "data", "hotpot_dev_distractor_v1.json"), encoding="utf-8") as f:
        all_data = json.load(f)
    rng = random.Random(137)
    idx = rng.sample(range(len(all_data)), 20)
    idx.sort()
    sample = [all_data[i] for i in idx]
    hotpot = {r["_id"]: r for r in sample}

    print("Loading eval results (smoke20_diag_sparse.json)...")
    d = json.load(open(os.path.join(_PKG, "eval", "results", "smoke20_diag_sparse.json"), encoding="utf-8"))
    records = d.get("records", [])
    n = len(records)
    print(f"n={n}\n")

    # ── TEST 1: Mount quality ─────────────────────────────────────
    print("=" * 70)
    print("TEST 1  Mount Quality  (gold sentence -> attribute_mounts)")
    print("=" * 70)

    no_mount, has_mount = 0, 0
    for rec in records:
        qid = rec["qid"]
        gold_sids = rec.get("diag", {}).get("gold_sentence_ids", [])
        gold_text = rec.get("gold", "")
        if not gold_sids:
            continue
        gold_sid = gold_sids[0]
        qdata = hotpot.get(qid)
        if not qdata:
            continue
        builder = build_graph(qdata, nlp)
        node = builder.sentence_registry.get_node(gold_sid)
        if not node:
            continue
        mounts = [m.attribute_name for m in getattr(node, "attribute_mounts", []) or []
                  if getattr(m, "attribute_name", None)]
        etype = getattr(node, "entity_type", "?")
        ok = bool(mounts)
        if ok:
            has_mount += 1
            mark = "PASS"
        else:
            no_mount += 1
            mark = "FAIL"
        print(f"  [{mark}] {qid}  entity={etype}  mounts={mounts}")
        print(f"         gold: {gold_text!r}")

    print(f"\n  -- Mount: {no_mount}/{no_mount+has_mount} gold sentences have NO mount --\n")

    # ── TEST 2: Mapper coverage ────────────────────────────────────
    print("=" * 70)
    print("TEST 2  Query Attribute Profile  (query -> attr similarity)")
    print("=" * 70)

    from src.stengraph_pipeline import ATTRIBUTE_NAMES
    from src.stengraph_pipeline import StenGraphPipeline as _SGP

    embedder = None
    try:
        from src.graph_loader import load_graph
        from src.container_manager import ContainerManager
        g = load_graph("graphs/hotpotqa_graph.pkl")
        embedder = getattr(ContainerManager(g), "embedder", None)
    except Exception as _e:
        print(f"  [WARN] embedder unavailable: {_e}")

    prof_ok, prof_fail = 0, 0
    for rec in records:
        qid = rec["qid"]
        q_text = rec.get("question", "")
        if embedder is None:
            break
        try:
            import numpy as _np
            qv = _np.asarray(embedder.encode([q_text], normalize_embeddings=True)[0],
                            dtype=_np.float32).reshape(-1)
            av = embedder.encode(ATTRIBUTE_NAMES, normalize_embeddings=True)
            sims = []
            for i, an in enumerate(ATTRIBUTE_NAMES):
                a = _np.asarray(av[i], dtype=_np.float32).reshape(-1)
                s = float(_np.dot(a, qv) / (_np.linalg.norm(a) * _np.linalg.norm(qv)))
                sims.append((an, s))
            top = sorted(sims, key=lambda t: -t[1])[:4]
            print(f"  >> {qid}  top_attrs={[(a, round(s, 3)) for a, s in top]}")
            print(f"        Q: {q_text[:65]}")
            prof_ok += 1
        except Exception as _e:
            prof_fail += 1

    print(f"\n  -- Query attr profile: {prof_ok}/{n} queries profiled --\n")

    # ── TEST 3: Soft-match contribution ───────────────────────────
    print("=" * 70)
    print("TEST 3  Soft-Match Contribution  (attr_term in final score)")
    print("=" * 70)

    has_ef = any(r.get("debug", {}) for r in records)
    if has_ef:
        scored = 0
        for rec in records:
            ef = (rec.get("debug", {}) or {}).get("ef_scores") or []
            if not ef:
                continue
            scored += 1
            top = sorted(ef, key=lambda d: -d.get("final_score", 0))[:3]
            for d in top:
                print(f"  {d.get('sentence_id','?')[:18]:<20} "
                      f"attr_match={d.get('attr_match', 0):.4f} "
                      f"attr_term={d.get('attr_term', 0):.4f} "
                      f"final={d.get('final_score', 0):.4f}")
            print()
        print(f"  -- Soft-match: {scored}/{n} questions with EF breakdown --\n")
    else:
        print("  [WARN] No EF breakdown in results (need re-run with the new pipeline)")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Mount quality:   {no_mount}/{no_mount+has_mount} gold sentences have NO mount")
    print(f"  Query profile:   {prof_ok}/{n} queries profiled")
    print(f"  Filter:          removed (mapper/filter deleted; pool is unfiltered)")

if __name__ == "__main__":
    main()

"""Direct A/B diagnostic of --attr-soft-match on the real pipeline.

Runs the seed137 sample twice (soft-match OFF / ON) on the SAME built
graph + hyperedges, and compares:
  - the EF breakdown (attr_match / attr_term per sentence)
  - the final top-K sid ordering
No LLM is involved, so the comparison is deterministic.

Usage:
    python eval/_ab_attr_softmatch.py [--n 4]
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import logging
logging.basicConfig(level=logging.WARNING)

from src.graph_builder import _load_spacy_shared
from src.hyper_edge import HyperEdgeRegistry
from src.dispatcher import build_dispatcher
from src.offline_hyperedge_store import OfflineHyperEdgeStore

DATA = "eval/data/hotpot_dev_distractor_v1.json"
PARQUET = "eval/offline_hyperedges/hyperedges.parquet"


def build_graph(qdata, nlp):
    """Mirror quick20_clean._BUILD_GRAPH."""
    from src.graph_builder import GraphBuilder, ComplementedSentence
    context = qdata.get("context", [])
    sentences = []
    para_sents = {}
    titles = []
    for title, para in context:
        titles.append(title)
        para_sents[title] = []
        for sent in para:
            cs = ComplementedSentence(
                original_text=sent,
                complemented_text=sent,
                resolved_subject=title,
                entities=[],
            )
            sentences.append(cs)
            para_sents[title].append(cs)
    builder = GraphBuilder(shared_nlp=nlp)
    for s in sentences:
        builder.add_sentence(s)
    return builder, sentences, titles, para_sents


def bridge_titles_from(titles, para_sents):
    """Very rough bridge-title guess: titles sharing a word."""
    out = {}
    norm = {t: t.lower().strip() for t in titles}
    for t in titles:
        cands = []
        for t2 in titles:
            if t2 == t:
                continue
            if set(norm[t].split()) & set(norm[t2].split()):
                cands.append(t2)
        if cands:
            out[t] = cands
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=137)
    args = ap.parse_args()

    nlp = _load_spacy_shared("en_core_web_lg")
    if nlp is None:
        print("[FATAL] spaCy unavailable")
        return 1

    with open(REPO / DATA, encoding="utf-8") as f:
        all_data = json.load(f)
    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(len(all_data)), min(args.n, len(all_data))))
    sample = [all_data[i] for i in idx]
    print(f"[setup] n={len(sample)} seed={args.seed}")

    offline_store = OfflineHyperEdgeStore.from_parquet(str(REPO / PARQUET))
    print(f"[setup] offline rows: {offline_store.stats()}")

    tot_changed = 0
    for qi, qdata in enumerate(sample):
        qid = qdata["_id"]
        question = qdata["question"]
        gold = qdata["answer"]

        builder, sentences, titles, para_sents = build_graph(qdata, nlp)
        cm = builder.container_manager
        for c in cm.get_all_containers():
            cm.update_container_index(c.id)
        builder.compute_tfidf_index()

        he = HyperEdgeRegistry(container_manager=cm)
        para_cid_map = {}
        for p_title in titles:
            cids = offline_store.get_containers_for_paragraph(qid, p_title)
            if cids:
                para_cid_map[p_title] = cids
        for p_title, cids in para_cid_map.items():
            conts = [cm._containers.get(c) for c in cids if cm._containers.get(c) is not None]
            if conts:
                he.build_hyperedges_for_chunk(
                    chunk_id=str(p_title), containers=conts,
                    time_word="", paragraph_title=str(p_title),
                )

        bt = bridge_titles_from(titles, para_sents)

        results = {}
        for mode in (False, True):
            disp = build_dispatcher(
                builder, client=None, nlp=nlp, hyperedge_registry=he,
                attr_soft_match=mode,
            )
            try:
                res = disp.run(question, bridge_titles=bt)
            except Exception as e:
                print(f"  [{qid[:8]}] mode={mode} ERROR {e}")
                results[mode] = None
                continue
            pr = res.pipeline_result
            ef = pr.debug.get("ef_scores") or []
            results[mode] = {
                "final": list(getattr(pr, "step10_final_sids", []) or []),
                "ef": ef,
                "answer_ctx": None,
            }

        off, on = results.get(False), results.get(True)
        print()
        print("=" * 78)
        print(f"[{qi+1}/{len(sample)}] {qid}  gold={gold!r}")
        print(f"  Q: {question}")
        if not off or not on:
            continue

        # --- attr signal ---
        am_on = [d.get("attr_match", 0.0) or 0.0 for d in on["ef"]]
        am_off = [d.get("attr_match", 0.0) or 0.0 for d in off["ef"]]
        nz_on = [a for a in am_on if a > 1e-9]
        print(f"  OFF: attr_match nonzero {sum(1 for a in am_off if a > 1e-9)}/{len(am_off)}"
              f"  max={max(am_off) if am_off else 0:.4f}")
        print(f"  ON : attr_match nonzero {len(nz_on)}/{len(am_on)}"
              f"  max={max(am_on) if am_on else 0:.4f}"
              f"  mean={sum(nz_on)/len(nz_on) if nz_on else 0:.4f}")

        # --- ordering ---
        same = off["final"] == on["final"]
        if same:
            print("  final sids: IDENTICAL")
        else:
            tot_changed += 1
            print(f"  final sids: DIFFERENT")
            print(f"    OFF: {off['final']}")
            print(f"    ON : {on['final']}")

        # --- score shift on the shared candidates ---
        fo = {d["sentence_id"]: d for d in off["ef"]}
        fn = {d["sentence_id"]: d for d in on["ef"]}
        shared = [s for s in fo if s in fn]
        deltas = []
        for s in shared:
            deltas.append((fn[s]["final_score"] - fo[s]["final_score"], s))
        deltas.sort(reverse=True)
        if deltas:
            print(f"  final_score delta: max=+{deltas[0][0]:.5f} "
                  f"min={deltas[-1][0]:.5f}")
            top_d, top_s = deltas[0]
            if top_d > 1e-9:
                print(f"    biggest gain {top_s}: "
                      f"off={fo[top_s]['final_score']:.5f} "
                      f"on={fn[top_s]['final_score']:.5f} "
                      f"(attr_term={fn[top_s].get('attr_term', 0):.5f})")

    print()
    print("=" * 78)
    print(f"questions whose final ordering changed: {tot_changed}/{len(sample)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

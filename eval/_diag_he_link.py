"""Instrument the REAL quick20_clean build path and report why
sentences never link to hyperedges (S_h_E == 0 for every candidate).

Usage:
    python eval/_diag_he_link.py [--n 3] [--seed 137]
"""
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
from src.offline_hyperedge_store import OfflineHyperEdgeStore
import quick20_eval as q20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
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

    offline = OfflineHyperEdgeStore.from_parquet(
        str(REPO / "eval/offline_hyperedges/hyperedges.parquet"))

    for qi, qdata in enumerate(sample):
        qid = qdata["_id"]
        builder, sentences, titles, para_sents, bridge_titles = (
            q20.build_graph_from_hotpot(qdata, nlp))
        cm = builder.container_manager
        for c in cm.get_all_containers():
            cm.update_container_index(c.id)
        builder.compute_tfidf_index()

        print("=" * 80)
        print(f"[{qi+1}/{len(sample)}] {qid}")
        print(f"  Q: {qdata['question']}")

        # ---- paragraph -> offline container ids ----
        n_hit = n_miss = 0
        miss_titles = []
        for t in titles:
            cids = offline.get_containers_for_paragraph(qid, t)
            live = [c for c in cids if cm._containers.get(c) is not None]
            if not cids:
                n_miss += 1
                miss_titles.append(f"{t}(no-offline)")
            elif not live:
                n_miss += 1
                miss_titles.append(f"{t}(offline-ids-dead:{len(cids)})")
            else:
                n_hit += 1
        print(f"  offline paragraphs: hit={n_hit} miss={n_miss} of {len(titles)}")
        if miss_titles:
            print(f"    misses: {miss_titles[:6]}")

        # ---- build hyperedges exactly like quick20_clean ----
        he = HyperEdgeRegistry(container_manager=cm)
        built = 0
        for t in titles:
            cids = offline.get_containers_for_paragraph(qid, t)
            if not cids:
                cids = (bridge_titles or {}).get(t.lower().strip(), [])
            conts = [cm._containers.get(c) for c in cids
                     if cm._containers.get(c) is not None]
            if conts:
                he.build_hyperedges_for_chunk(
                    chunk_id=str(t), containers=conts, time_word="",
                    paragraph_title=str(t))
                built += 1
        print(f"  hyperedges built for {built}/{len(titles)} paragraphs; "
              f"total edges={len(he.get_all_hyperedges())}")

        # ---- container coverage ----
        all_cids = list(cm._containers.keys())
        with_he = [cid for cid in all_cids
                   if cm._containers[cid].hyperedges]
        print(f"  containers: total={len(all_cids)} with_hyperedge={len(with_he)}")

        # ---- sentence -> container -> hyperedge linkage ----
        n_sid = 0
        n_sid_no_cont = 0
        n_sid_cont_no_he = 0
        n_sid_linked = 0
        examples = []
        for node in builder.sentence_registry._nodes.values():
            n_sid += 1
            conts = getattr(node, "containers", []) or []
            if not conts:
                n_sid_no_cont += 1
                continue
            linked = False
            for cid in conts:
                try:
                    hs = he.get_hyperedges_for_container(cid)
                except Exception:
                    hs = []
                if hs:
                    linked = True
                    break
            if linked:
                n_sid_linked += 1
            else:
                n_sid_cont_no_he += 1
                if len(examples) < 4:
                    txt = (getattr(node, "text", "") or "")[:70]
                    examples.append((txt, list(conts)[:3]))
        print(f"  sentences: total={n_sid} no_container={n_sid_no_cont} "
              f"container_without_he={n_sid_cont_no_he} LINKED={n_sid_linked}")
        for txt, conts in examples:
            print(f"    e.g. {txt!r} conts={conts}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

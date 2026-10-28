"""Decisive test: can the symmetric-attribute soft match produce a
non-zero attr_match on the real seed137 questions?

Reuses the exact real build path (quick20_eval.build_graph_from_hotpot),
then:
  1. dumps the attribute_mounts actually produced
  2. computes query_attr_vec the same way the pipeline does
  3. reports the resulting attr_match / attr_term per sentence
"""
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO / "eval"))

import logging
logging.basicConfig(level=logging.WARNING)

from src.graph_builder import _load_spacy_shared
from src.stengraph_pipeline import ATTRIBUTE_NAMES, W_ATTR
import quick20_eval as q20

import numpy as np


def cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    nlp = _load_spacy_shared("en_core_web_lg")
    with open(REPO / "eval/data/hotpot_dev_distractor_v1.json", encoding="utf-8") as f:
        all_data = json.load(f)
    rng = random.Random(137)
    idx = sorted(rng.sample(range(len(all_data)), 3))
    sample = [all_data[i] for i in idx]

    from src.container_manager import _get_embedder
    embedder = _get_embedder("all-MiniLM-L6-v2")
    print(f"embedder present: {embedder is not None}")

    attr_embs = embedder.encode(ATTRIBUTE_NAMES, normalize_embeddings=True)
    print(f"ATTRIBUTE_NAMES ({len(ATTRIBUTE_NAMES)}): {ATTRIBUTE_NAMES}")

    for qdata in sample:
        qid = qdata["_id"]
        q = qdata["question"]
        builder, sentences, titles, para_sents, bridge = q20.build_graph_from_hotpot(
            qdata, nlp)

        # ---- what mounts exist? ----
        mount_counter = {}
        n_nodes = 0
        n_with_mount = 0
        for node in builder.sentence_registry._nodes.values():
            n_nodes += 1
            ms = getattr(node, "attribute_mounts", []) or []
            if ms:
                n_with_mount += 1
            for m in ms:
                mount_counter[m.attribute_name] = mount_counter.get(m.attribute_name, 0) + 1

        print("=" * 80)
        print(f"{qid}   Q: {q}")
        print(f"  nodes={n_nodes}  nodes_with_mounts={n_with_mount}")
        print(f"  mount histogram: {mount_counter}")
        real = {k: v for k, v in mount_counter.items() if k != "other"}
        print(f"  NON-'other' mounts: {real}  "
              f"(-> {'USABLE' if real else 'ALL EXCLUDED BY SOFT MATCH'})")

        # ---- query_attr_vec ----
        qv = embedder.encode([q], normalize_embeddings=True)[0]
        raw = [max(0.0, cos(np.asarray(attr_embs[i], dtype=np.float32).reshape(-1),
                            np.asarray(qv, dtype=np.float32).reshape(-1)))
               for i in range(len(ATTRIBUTE_NAMES))]
        top = max(raw) if raw else 0.0
        qav = [r / top for r in raw] if top > 1e-6 else raw
        ranked = sorted(zip(qav, ATTRIBUTE_NAMES), reverse=True)[:5]
        print(f"  query_attr_vec top5: {[(n, round(v,3)) for v, n in ranked]}")

        # ---- per-sentence attr_match ----
        best_overall = 0.0
        for node in builder.sentence_registry._nodes.values():
            ms = getattr(node, "attribute_mounts", []) or []
            w_by_attr = {}
            for m in ms:
                an = getattr(m, "attribute_name", "") or ""
                if not an or an == "other" or an not in ATTRIBUTE_NAMES:
                    continue
                w = min(max(float(getattr(m, "weight", 1.0) or 0.0), 0.0), 1.0)
                w_by_attr[an] = max(w_by_attr.get(an, 0.0), w)
            if not w_by_attr:
                continue
            num = den = best_c = 0.0
            for an, w in w_by_attr.items():
                k = ATTRIBUTE_NAMES.index(an)
                c = w * qav[k]
                num += c
                den += w
                best_c = max(best_c, c)
            cov = (num / den) if den > 0 else 0.0
            am = best_c * (0.7 + 0.3 * cov)
            best_overall = max(best_overall, am)
            if am > 0.05:
                print(f"    am={am:.4f} attr_term={W_ATTR*am:.4f} "
                      f"mounts={list(w_by_attr)} txt={(node.text or '')[:55]!r}")
        print(f"  >>> max attr_match over all sentences = {best_overall:.6f} "
              f"(attr_term={W_ATTR*best_overall:.6f})")


if __name__ == "__main__":
    main()

"""Offline smoke test for hyperedge paragraph→event merge.

Builds a tiny HyperEdgeRegistry with 4 fake hyperedges:
  - he_a, he_b share a low-freq entity "kelly" and have similar titles
  - he_c shares nothing with he_a/he_b but is in the same chunk
  - he_d shares an entity with he_a but very different title (should NOT merge)

Then runs merge_paragraph_hyperedges_into and prints the stats.
"""
import os
import sys
from types import SimpleNamespace

ROOT = r"c:\Users\zsf\Desktop\lyh\StenRAG"
sys.path.insert(0, ROOT)

from src.hyper_edge import (
    HyperEdge,
    HyperEdgeRegistry,
    merge_paragraph_hyperedges_into,
    LOW_FREQ_THRESHOLD,
    TITLE_SIM_THRESHOLD,
    MAX_MERGED_SIZE,
)


class _FakeCM:
    """Minimal stand-in for ContainerManager."""
    def __init__(self):
        self._containers = {}

    def get_container(self, cid):
        return self._containers.get(cid)

    def get_all_containers(self):
        return list(self._containers.values())


def _mk_container(cid, ent):
    return SimpleNamespace(id=cid, entity=ent, aliases=[], hyperedges=[])


def main():
    print(f"[smoke] LOW_FREQ_THRESHOLD={LOW_FREQ_THRESHOLD} "
          f"TITLE_SIM_THRESHOLD={TITLE_SIM_THRESHOLD} "
          f"MAX_MERGED_SIZE={MAX_MERGED_SIZE}")
    cm = _FakeCM()
    cm._containers = {
        "c_k1": _mk_container("c_k1", "kelly"),
        "c_k2": _mk_container("c_k2", "kelly"),
        "c_d1": _mk_container("c_d1", "discovery"),
        "c_x1": _mk_container("c_x1", "x"),
    }

    he = HyperEdgeRegistry(container_manager=cm)
    # Manually inject 4 hyperedges (bypassing build_hyperedges_for_chunk so
    # we can control id/event_name/time_range directly).
    he._counter = 0
    a = HyperEdge(id="h_0001", container_ids=["c_k1"], event_name="Kelly Award",
                  time_range=("2020-01", "2020-12"),
                  source_chunk_id="chunk_kelly", chunk_doc_ids=["doc1"])
    b = HyperEdge(id="h_0002", container_ids=["c_k2"], event_name="Kelly Award 2020",
                  time_range=("2020-06", "2020-12"),
                  source_chunk_id="chunk_kelly2", chunk_doc_ids=["doc2"])
    c = HyperEdge(id="h_0003", container_ids=["c_d1"], event_name="Discovery",
                  time_range=("", ""),
                  source_chunk_id="chunk_d", chunk_doc_ids=["doc3"])
    d = HyperEdge(id="h_0004", container_ids=["c_x1"], event_name="Random X",
                  time_range=("1999-01", "1999-12"),
                  source_chunk_id="chunk_x", chunk_doc_ids=["doc4"])
    for e in (a, b, c, d):
        he._edges[e.id] = e

    # entity_freq: kelly appears 2x (< 20 low-freq), discovery 1x
    entity_freq = {"kelly": 2, "discovery": 1, "x": 1}
    stats = he.merge_paragraph_hyperedges(
        entity_freq=entity_freq,
        embedder=None,  # token-Jaccard fallback for titles
    )
    print("[smoke] edges after merge:")
    for e in he.get_all_hyperedges():
        print(f"  - id={e.id}  event_name={e.event_name!r}  "
              f"containers={e.container_ids}  "
              f"time_range={e.time_range}  "
              f"source_chunk_id={e.source_chunk_id}  "
              f"merged_from={e.merged_from}")

    # Edge-cases test 1: only 1 edge → no merge
    print("\n[smoke] only 1 edge:")
    he2 = HyperEdgeRegistry(container_manager=cm)
    he2._counter = 0
    e0 = HyperEdge(id="h_0001", container_ids=["c_x1"], event_name="Solo",
                   time_range=("", ""), source_chunk_id="chunk_solo",
                   chunk_doc_ids=[])
    he2._edges[e0.id] = e0
    stats2 = he2.merge_paragraph_hyperedges(entity_freq=entity_freq)
    print("[smoke] solo stats:", stats2)

    # Edge-case test 2: title exactly equal → trivially merges (if entities ok)
    print("\n[smoke] identical title pair:")
    he3 = HyperEdgeRegistry(container_manager=cm)
    he3._counter = 0
    a1 = HyperEdge(id="h_0001", container_ids=["c_k1"],
                   event_name="Discovery Award", time_range=("2020-01", "2020-12"),
                   source_chunk_id="c1", chunk_doc_ids=[])
    a2 = HyperEdge(id="h_0002", container_ids=["c_d1"],
                   event_name="Discovery Award", time_range=("2021-01", "2021-12"),
                   source_chunk_id="c2", chunk_doc_ids=[])
    he3._edges[a1.id] = a1
    he3._edges[a2.id] = a2
    # discovery and kelly: no shared low-freq entity → no merge
    stats3 = he3.merge_paragraph_hyperedges(entity_freq=entity_freq)
    print("[smoke] equal-title but no shared entity stats:", stats3)


if __name__ == "__main__":
    main()

"""
Relation Retrieval Pipeline
==========================

Five-phase retrieval for "A and B are related" questions:

  Phase 1 – Sub-question Decomposition + Recall
    Split "A 和 B 是什么关系" into sub-queries "A" and "B".
    Each sub-query returns Top-2 sentences via sentence-level FAISS.

  Phase 2 – BFS Sub-graph Expansion
    From each seed sentence, BFS-expand 2 hops across all edge types.
    Nodes with semantic similarity < 0.6 are pruned (not expanded further).

  Phase 3 – Sub-graph Intersection Validation
    Attempt intersections A1∩B1, A1∩B2, A2∩B1, A2∩B2.
    A valid intersection requires shared entities, shared sentences, and
    those entities appearing inside those sentences.
    First successful group proceeds; all-fail triggers global fallback.

  Phase 4 – Constraint Extraction
    From shared sentences: collect shared entities (no limit) + top-3
    by frequency as supplemental constraints.  Deduplicate.
    Keep only containers whose degree ranks in the top 30 % (high-freq hubs).

  Phase 5 – DFS Path Search
    From A → B within 4 hops, must pass ≥1 constraint container.
    Confidence = min_edge_weight × cross_container_ratio.
    Paths with min_edge < 0.7 are discarded.  Paths that stay within
    a single container are discarded.  Return the highest-confidence path.

  SATELLITE handling:
    • BFS/DFS includes SATELLITE sentences normally in the sub-graph.
    • SATELLITE sentences that appear in shared sentences are upgraded
      to CORE role.
    • SATELLITE sentences are excluded from confidence scoring and
      ranking; they are attached as side context at output time.

  Output format:
    sentences ordered: A-subgraph → B-subgraph → intersection.
    container summaries: only seed-container chains.
    cross-subgraph edge weights boosted ×1.2 then normalised [0,1].
    duplicate sentences assigned to the sub-graph with the higher weight.

Usage:
    from src.relation_retriever import RelationRetriever
    from eval.hotpot_eval import build_graph, hotpot_to_sentences

    builder = build_graph(sentences)
    rr = RelationRetriever(builder)
    result = rr.retrieve("What is the relationship between A and B?",
                         entity_a="A", entity_b="B")
"""

from __future__ import annotations

import heapq
import logging
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .sentence_graph import LogicEdge, SentenceNode
from .container_manager import Container, ContainerManager
from .graph_builder import GraphBuilder

log = logging.getLogger("relation_retriever")

# ─────────────────────────────────────────────────────────────────────────────
# Embedder cache (same pattern as container_manager)
# ─────────────────────────────────────────────────────────────────────────────

_EMBEDDER_CACHE: Dict[str, object] = {}
_EMBEDDER_LOCK = threading.Lock()


def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    if model_name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model_name]
    with _EMBEDDER_LOCK:
        if model_name in _EMBEDDER_CACHE:
            return _EMBEDDER_CACHE[model_name]
        try:
            from sentence_transformers import SentenceTransformer
            from .model_paths import ensure_model
            # Prefer the local model store (downloading it if missing) so
            # the eval runs fully offline.
            emb = SentenceTransformer(ensure_model(model_name))
        except Exception:
            emb = None
        _EMBEDDER_CACHE[model_name] = emb
        return emb


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SeedResult:
    """A sub-question seed: which sub-query it came from and FAISS rank."""
    sub_query: str          # "A" or "B"
    sentence_id: str
    faiss_score: float = 1.0
    role: str = "CORE"


@dataclass
class SubGraph:
    """Local sub-graph around a seed sentence."""
    seed: SeedResult
    sentence_ids: Set[str] = field(default_factory=set)
    core_ids: Set[str] = field(default_factory=set)   # role == "CORE"
    sat_ids: Set[str] = field(default_factory=set)     # role == "SATELLITE"
    edge_weights: Dict[str, float] = field(default_factory=dict)
    # sentence_id -> sentence for fast lookup
    _sent_map: Dict[str, SentenceNode] = field(default_factory=dict, repr=False)

    def add_sentence(self, sid: str, node: SentenceNode, role: str = "CORE") -> None:
        self.sentence_ids.add(sid)
        self._sent_map[sid] = node
        if role == "SATELLITE":
            self.sat_ids.add(sid)
        else:
            self.core_ids.add(sid)


@dataclass
class IntersectionResult:
    """Result of Phase 3 intersection validation."""
    valid: bool
    shared_sids: Set[str] = field(default_factory=set)
    shared_entities: Set[str] = field(default_factory=set)
    group_label: str = ""   # e.g. "A1∩B1"


@dataclass
class DFSPath:
    """A candidate A→B path found in Phase 5."""
    sentence_ids: List[str]
    edges: List[LogicEdge]
    min_edge_weight: float
    cross_container_ratio: float
    confidence: float
    constraint_container_ids: Set[str] = field(default_factory=set)

    @property
    def hops(self) -> int:
        return len(self.edges)


@dataclass
class RetrievalResult:
    """
    Final output of the full 5-phase pipeline.

    Fields:
        sentences          – ordered list of SentenceNode ids
        sentence_map       – id → SentenceNode
        containers         – Container objects relevant to this query
        edges              – LogicEdge objects in the output sub-graph
        constraint_entities – entity strings used as constraints
        constraint_containers – container ids in the constraint set
        paths              – all DFSPath objects found
        best_path          – highest-confidence DFSPath (or None)
        phase_used         – which intersection group succeeded, or "global_fallback"
        satellite_ids       – sentence ids that are SATELLITE (side context only)
        metrics             – diagnostic dict (for debugging)
    """
    sentences: List[str]
    sentence_map: Dict[str, SentenceNode]
    containers: List[Container]
    edges: List[LogicEdge]
    constraint_entities: List[str]
    constraint_containers: List[str]
    paths: List[DFSPath]
    best_path: Optional[DFSPath]
    phase_used: str
    satellite_ids: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentences": self.sentences,
            "sentence_count": len(self.sentences),
            "container_count": len(self.containers),
            "edge_count": len(self.edges),
            "constraint_entities": self.constraint_entities,
            "constraint_containers": self.constraint_containers,
            "phase_used": self.phase_used,
            "satellite_ids": self.satellite_ids,
            "best_path_hops": self.best_path.hops if self.best_path else 0,
            "best_path_confidence": round(self.best_path.confidence, 4) if self.best_path else 0.0,
            "metrics": self.metrics,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RelationRetriever
# ─────────────────────────────────────────────────────────────────────────────

class RelationRetriever:
    """
    Five-phase relation retrieval pipeline.

    Args:
        builder: GraphBuilder instance (already built).
        embedder_model: SentenceTransformer model name for FAISS.
        bfs_hops: BFS expansion depth (default 2).
        min_similarity: Semantic similarity floor for BFS expansion (default 0.6).
        min_edge_weight: Minimum edge weight for DFS path (default 0.7).
        max_path_hops: Maximum DFS depth (default 4).
        cross_entity_threshold: Fraction of path hops that must cross containers
            (default 0.0; a path must cross ≥2 different entity containers).
        top_k_per_sub: Top-K sentences per sub-query in Phase 1 (default 2).
    """

    ALL_EDGE_TYPES: List[str] = [
        LogicEdge.TYPE_CAUSAL,
        LogicEdge.TYPE_TEMPORAL,
        LogicEdge.TYPE_CONTRAST,
        LogicEdge.TYPE_ELABORATION,
        LogicEdge.TYPE_SAME_ENTITY,
        LogicEdge.TYPE_CONTEXT,
    ]

    LOGIC_EDGE_TYPES: List[str] = [
        LogicEdge.TYPE_CAUSAL,
        LogicEdge.TYPE_TEMPORAL,
        LogicEdge.TYPE_CONTRAST,
        LogicEdge.TYPE_ELABORATION,
        LogicEdge.TYPE_CONTEXT,
    ]

    def __init__(
        self,
        builder: GraphBuilder,
        embedder_model: str = "all-MiniLM-L6-v2",
        bfs_hops: int = 2,
        min_similarity: float = 0.6,
        min_edge_weight: float = 0.7,
        max_path_hops: int = 4,
        cross_entity_threshold: float = 0.0,
        top_k_per_sub: int = 2,
    ):
        self.builder = builder
        self.sentence_registry = builder.sentence_registry
        self.edge_registry = builder.edge_registry
        self.container_manager = builder.container_manager
        self.embedder_model = embedder_model
        self._embedder = _get_embedder(embedder_model)

        # Tunable parameters
        self.bfs_hops = bfs_hops
        self.min_similarity = min_similarity
        self.min_edge_weight = min_edge_weight
        self.max_path_hops = max_path_hops
        self.cross_entity_threshold = cross_entity_threshold
        self.top_k_per_sub = top_k_per_sub

        # ── Phase 0: Build sentence-level FAISS index ──────────────────────
        self._sentence_faiss_index: Any = None   # faiss.IndexFlatIP
        self._sentence_ids: List[str] = []        # index position → sentence_id
        self._sentence_id_to_idx: Dict[str, int] = {}
        self._sentence_id_to_embedding: Dict[str, np.ndarray] = {}
        self._build_sentence_faiss_index()

        # ── Phase 0: Build reverse index sentence_id → container_ids ───────
        self._reverse_index: Dict[str, List[str]] = {}
        self._build_reverse_index()

        # ── Phase 0: Build container degree map (for hub pruning in Phase 4) ─
        self._container_degrees: Dict[str, int] = {}
        self._build_container_degrees()

    # ─────────────────────────────────────────────────────────────────────
    # Phase 0 helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_sentence_faiss_index(self) -> None:
        """Build FAISS IndexFlatIP over all sentence embeddings."""
        self._sentence_faiss_index = None
        self._sentence_ids = []
        self._sentence_id_to_idx = {}
        self._sentence_id_to_embedding = {}

        all_nodes = self.sentence_registry.get_all_nodes()
        if not all_nodes:
            return

        texts = [n.complemented_text or n.text for n in all_nodes]
        sids = [n.id for n in all_nodes]

        if self._embedder is None:
            log.warning("Embedder unavailable; sentence-level FAISS disabled")
            return

        try:
            import faiss
            embeddings = self._embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

            self._sentence_faiss_index = index
            self._sentence_ids = sids
            self._sentence_id_to_idx = {sid: i for i, sid in enumerate(sids)}
            for i, sid in enumerate(sids):
                self._sentence_id_to_embedding[sid] = embeddings[i]
            log.debug("Built sentence FAISS index: %d sentences, dim=%d",
                      len(sids), dim)
        except Exception as e:
            log.warning("Failed to build sentence FAISS index: %s", e)

    def _build_reverse_index(self) -> None:
        """Build sentence_id → [container_ids]."""
        self._reverse_index = {}
        for container in self.container_manager.get_all_containers():
            for sid in container.sentence_ids:
                self._reverse_index.setdefault(sid, []).append(container.id)

    def _build_container_degrees(self) -> None:
        """Container degree = number of edges that pass through it."""
        self._container_degrees = {}
        for edge in self.edge_registry.get_all_edges():
            for cid in self._reverse_index.get(edge.source_sent_id, []):
                self._container_degrees[cid] = self._container_degrees.get(cid, 0) + 1
            for cid in self._reverse_index.get(edge.target_sent_id, []):
                self._container_degrees[cid] = self._container_degrees.get(cid, 0) + 1

    # ─────────────────────────────────────────────────────────────────────
    # Phase 1: Sub-question decomposition + FAISS recall
    # ─────────────────────────────────────────────────────────────────────

    def _extract_entities_from_question(
        self, question: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse "What is the relationship between A and B?" style questions.

        Returns (entity_a, entity_b).  Falls back to None if parsing fails.
        """
        text = question.strip()

        # Pattern 1: "relationship between X and Y"
        m = re.search(
            r"(?:relationship|relation|connection|link|association|how)\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            text, re.IGNORECASE
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()

        # Pattern 2: "What is the link between X and Y"
        m = re.search(
            r"link\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            text, re.IGNORECASE
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()

        # Pattern 3: "compare X and Y"
        m = re.search(
            r"(?:compare|contrast|relationship|relation)\s+(?:the\s+)?(.+?)\s+and\s+(.+?)(?:\?|$)",
            text, re.IGNORECASE
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()

        return None, None

    def _faiss_search_sentences(
        self,
        query: str,
        k: int = 2,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Search sentence-level FAISS index. Returns list of (sentence_id, score),
        sorted descending by score.
        """
        if self._sentence_faiss_index is None:
            return []

        if self._embedder is None:
            return []

        try:
            q_vec = self._embedder.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            k_capped = min(k, len(self._sentence_ids))
            if k_capped <= 0:
                return []
            scores, indices = self._sentence_faiss_index.search(q_vec, k_capped)
        except Exception:
            return []

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            sid = self._sentence_ids[int(idx)]
            sim = float(score)
            if sim >= min_score:
                results.append((sid, sim))
        return results

    def _phase1_subquestion_recall(
        self, question: str, entity_a: str, entity_b: str
    ) -> Tuple[List[SeedResult], List[SeedResult]]:
        """
        Phase 1: decompose question into sub-queries and recall Top-2 sentences each.

        Returns (seeds_a, seeds_b), each a list of SeedResult.
        """
        # Sub-query construction
        sub_a = entity_a
        sub_b = entity_b

        raw_a = self._faiss_search_sentences(sub_a, k=self.top_k_per_sub)
        raw_b = self._faiss_search_sentences(sub_b, k=self.top_k_per_sub)

        seeds_a = [
            SeedResult(
                sub_query="A",
                sentence_id=sid,
                faiss_score=score,
                role=self._get_sentence_role(sid),
            )
            for sid, score in raw_a
        ]
        seeds_b = [
            SeedResult(
                sub_query="B",
                sentence_id=sid,
                faiss_score=score,
                role=self._get_sentence_role(sid),
            )
            for sid, score in raw_b
        ]

        self._metrics_phase1 = {
            "entity_a": entity_a, "entity_b": entity_b,
            "seeds_a": [(s.sentence_id, round(s.faiss_score, 3)) for s in seeds_a],
            "seeds_b": [(s.sentence_id, round(s.faiss_score, 3)) for s in seeds_b],
        }
        return seeds_a, seeds_b

    def _get_sentence_role(self, sid: str) -> str:
        node = self.sentence_registry.get_node(sid)
        return getattr(node, "role", "CORE") if node else "CORE"

    # ─────────────────────────────────────────────────────────────────────
    # Phase 2: BFS sub-graph expansion
    # ─────────────────────────────────────────────────────────────────────

    def _phase2_bfs_expand(
        self, seed: SeedResult
    ) -> SubGraph:
        """
        BFS expand 2 hops from seed sentence across all edge types.
        Prune neighbours whose semantic similarity to seed < self.min_similarity.
        SATELLITE sentences are included but tracked separately.
        """
        subgraph = SubGraph(seed=seed)

        seed_node = self.sentence_registry.get_node(seed.sentence_id)
        if seed_node is None:
            return subgraph

        seed_emb = self._sentence_id_to_embedding.get(seed.sentence_id)
        if seed_emb is None:
            # Still add the seed
            subgraph.add_sentence(
                seed.sentence_id, seed_node,
                role=getattr(seed_node, "role", "CORE"),
            )
            return subgraph

        # Queue: (sentence_id, depth, edge_weight_from_parent)
        queue: List[Tuple[str, int, float]] = [(seed.sentence_id, 0, 1.0)]
        visited: Set[str] = {seed.sentence_id}

        subgraph.add_sentence(
            seed.sentence_id, seed_node,
            role=getattr(seed_node, "role", "CORE"),
        )

        while queue:
            sid, depth, parent_weight = queue.pop(0)

            if depth >= self.bfs_hops:
                continue

            # Get neighbours across all edge types
            for neighbour_sid, edge in self.edge_registry.get_neighbors(
                sid, edge_types=self.ALL_EDGE_TYPES
            ):
                if neighbour_sid in visited:
                    continue

                # Prune by semantic similarity
                neighbour_emb = self._sentence_id_to_embedding.get(neighbour_sid)
                if neighbour_emb is not None and seed_emb is not None:
                    sim = float(np.dot(seed_emb, neighbour_emb))
                    if sim < self.min_similarity:
                        continue
                    # Record edge weight with similarity bonus
                    edge_key = f"{sid}|{neighbour_sid}"
                    combined_weight = edge.weight * min(sim, 1.0)
                    subgraph.edge_weights[edge_key] = combined_weight
                else:
                    edge_key = f"{sid}|{neighbour_sid}"
                    subgraph.edge_weights[edge_key] = edge.weight

                neighbour_node = self.sentence_registry.get_node(neighbour_sid)
                if neighbour_node is None:
                    continue

                visited.add(neighbour_sid)
                subgraph.add_sentence(
                    neighbour_sid, neighbour_node,
                    role=getattr(neighbour_node, "role", "CORE"),
                )
                queue.append((neighbour_sid, depth + 1, edge.weight))

        return subgraph

    def _phase2_expand_all_seeds(
        self,
        seeds_a: List[SeedResult],
        seeds_b: List[SeedResult],
    ) -> Tuple[List[SubGraph], List[SubGraph]]:
        """Expand BFS sub-graphs for all seeds."""
        subgraphs_a = [self._phase2_bfs_expand(s) for s in seeds_a]
        subgraphs_b = [self._phase2_bfs_expand(s) for s in seeds_b]
        self._metrics_phase2 = {
            "subgraphs_a": [
                {"seed": s.seed.sentence_id, "n_core": len(s.core_ids),
                 "n_sat": len(s.sat_ids), "n_total": len(s.sentence_ids)}
                for s in subgraphs_a
            ],
            "subgraphs_b": [
                {"seed": s.seed.sentence_id, "n_core": len(s.core_ids),
                 "n_sat": len(s.sat_ids), "n_total": len(s.sentence_ids)}
                for s in subgraphs_b
            ],
        }
        return subgraphs_a, subgraphs_b

    # ─────────────────────────────────────────────────────────────────────
    # Phase 3: Sub-graph intersection validation
    # ─────────────────────────────────────────────────────────────────────

    def _compute_entity_sentence_overlap(
        self, sg_a: SubGraph, sg_b: SubGraph
    ) -> IntersectionResult:
        """
        Check whether two sub-graphs have a valid intersection.

        Valid intersection requires:
          1. At least one shared entity
          2. At least one shared sentence
          3. Those shared entities appear in those shared sentences
        """
        # Collect all nodes in each sub-graph
        nodes_a = {sid: sg_a._sent_map.get(sid) for sid in sg_a.sentence_ids}
        nodes_b = {sid: sg_b._sent_map.get(sid) for sid in sg_b.sentence_ids}

        shared_sids = sg_a.sentence_ids & sg_b.sentence_ids
        if not shared_sids:
            return IntersectionResult(valid=False)

        # Collect entities
        entities_a: Set[str] = set()
        for node in nodes_a.values():
            if node:
                entities_a.update(node.entities)
        entities_b: Set[str] = set()
        for node in nodes_b.values():
            if node:
                entities_b.update(node.entities)

        shared_entities = entities_a & entities_b
        if not shared_entities:
            return IntersectionResult(valid=False)

        # Condition 3: shared entities must appear in shared sentences
        valid_shared_sids = set()
        for sid in shared_sids:
            node = sg_a._sent_map.get(sid) or sg_b._sent_map.get(sid)
            if node:
                node_entities = set(node.entities)
                if shared_entities & node_entities:
                    valid_shared_sids.add(sid)

        if not valid_shared_sids:
            return IntersectionResult(valid=False)

        return IntersectionResult(
            valid=True,
            shared_sids=valid_shared_sids,
            shared_entities=shared_entities,
        )

    def _phase3_intersection_validation(
        self,
        subgraphs_a: List[SubGraph],
        subgraphs_b: List[SubGraph],
    ) -> Tuple[Optional[Tuple[SubGraph, SubGraph]], IntersectionResult, str]:
        """
        Try intersection groups in priority order.

        Returns (matched_pair, IntersectionResult, group_label).
        If no group matches, returns (None, result, "none").
        """
        groups = [
            (subgraphs_a[0], subgraphs_b[0], "A1∩B1"),
            (subgraphs_a[0], subgraphs_b[-1], "A1∩B2"),
            (subgraphs_a[-1], subgraphs_b[0], "A2∩B1"),
            (subgraphs_a[-1], subgraphs_b[-1], "A2∩B2"),
        ]

        for sg_a, sg_b, label in groups:
            ix = self._compute_entity_sentence_overlap(sg_a, sg_b)
            if ix.valid:
                self._metrics_phase3 = {
                    "group": label, "valid": True,
                    "shared_entities": list(ix.shared_entities),
                    "shared_sids": list(ix.shared_sids),
                }
                return (sg_a, sg_b), ix, label

        self._metrics_phase3 = {"group": "none", "valid": False}
        return None, IntersectionResult(valid=False), "none"

    # ─────────────────────────────────────────────────────────────────────
    # Phase 4: Constraint extraction
    # ─────────────────────────────────────────────────────────────────────

    def _phase4_extract_constraints(
        self,
        sg_a: SubGraph,
        sg_b: SubGraph,
        ix: IntersectionResult,
    ) -> Tuple[List[str], List[str]]:
        """
        Extract constraint entities and constraint container ids.

        Entities from shared sentences are taken unconditionally.
        Top-3 additional entities by frequency are added as supplemental.
        Containers filtered to top-30% degree (high-freq hubs).
        """
        constraint_entities: Set[str] = set(ix.shared_entities)
        entity_freq: Counter = Counter()

        for sid in (sg_a.sentence_ids | sg_b.sentence_ids):
            node = (sg_a._sent_map.get(sid) or
                    sg_b._sent_map.get(sid))
            if node:
                for e in node.entities:
                    entity_freq[e] += 1

        # Supplemental: top-3 by frequency (not already in constraint_entities)
        supplemental = [
            e for e, _ in entity_freq.most_common(10)
            if e not in constraint_entities
        ][:3]
        constraint_entities.update(supplemental)

        # Constraint containers: those whose degree is in top 30%
        all_constraint_sids: Set[str] = set()
        for sid in (sg_a.sentence_ids | sg_b.sentence_ids | ix.shared_sids):
            for cid in self._reverse_index.get(sid, []):
                all_constraint_sids.add(cid)

        if self._container_degrees:
            sorted_degrees = sorted(set(self._container_degrees.values()), reverse=True)
            pct30_idx = int(len(sorted_degrees) * 0.30)
            threshold = sorted_degrees[pct30_idx] if pct30_idx < len(sorted_degrees) else 0

            constraint_container_ids = [
                cid for cid in all_constraint_sids
                if self._container_degrees.get(cid, 0) >= threshold
            ]
        else:
            constraint_container_ids = list(all_constraint_sids)

        self._metrics_phase4 = {
            "constraint_entities": list(constraint_entities),
            "constraint_container_ids": constraint_container_ids,
            "threshold_degree": threshold if self._container_degrees else None,
        }
        return list(constraint_entities), constraint_container_ids

    # ─────────────────────────────────────────────────────────────────────
    # Phase 5: DFS path search A → B
    # ─────────────────────────────────────────────────────────────────────

    def _containers_of_sentence(self, sid: str) -> Set[str]:
        return set(self._reverse_index.get(sid, []))

    def _phase5_dfs_search(
        self,
        seed_sids: List[str],
        target_entity: str,
        constraint_container_ids: List[str],
    ) -> List[DFSPath]:
        """
        DFS search from seed sentences to any sentence whose container
        contains the target entity.  Within 4 hops, must pass at least
        one constraint container.  Confidence = min_edge × cross_container_ratio.
        Paths with min_edge < self.min_edge_weight are discarded.
        Paths confined to a single container are discarded.
        """
        constraint_set = set(constraint_container_ids)
        all_paths: List[DFSPath] = []

        # Collect target sentence ids (any sentence whose container entity matches)
        target_sids: Set[str] = set()
        for container in self.container_manager.get_all_containers():
            if target_entity.lower() in container.entity.lower():
                target_sids.update(container.sentence_ids)
        if not target_sids:
            # Fallback: any sentence mentioning the target entity
            for node in self.sentence_registry.get_all_nodes():
                if any(target_entity.lower() in e.lower() for e in node.entities):
                    target_sids.add(node.id)

        for start_sid in seed_sids:
            path = self._dfs_single_source(
                start_sid=start_sid,
                target_sids=target_sids,
                constraint_set=constraint_set,
                max_hops=self.max_path_hops,
            )
            if path:
                all_paths.extend(path)

        # Sort by confidence descending
        all_paths.sort(key=lambda p: -p.confidence)
        return all_paths

    def _dfs_single_source(
        self,
        start_sid: str,
        target_sids: Set[str],
        constraint_set: Set[str],
        max_hops: int,
    ) -> List[DFSPath]:
        """
        DFS from a single source sentence.  Returns all valid paths.
        """
        paths: List[DFSPath] = []

        # State: (current_sid, path_sids, path_edges, min_edge_weight,
        #         visited_sids, containers_visited, passes_constraint)
        StackItem = Tuple[str, List[str], List[LogicEdge],
                          float, FrozenSet[str], Set[str], bool]

        stack: List[StackItem] = [(
            start_sid,
            [start_sid],
            [],
            1.0,
            frozenset([start_sid]),
            self._containers_of_sentence(start_sid),
            bool(self._containers_of_sentence(start_sid) & constraint_set),
        )]

        while stack:
            (cur_sid, path_sids, path_edges,
             min_ew, visited_frozen,
             containers_seen, passes_constraint) = stack.pop()

            if cur_sid in target_sids:
                if not passes_constraint:
                    continue
                if len(containers_seen) < 2:
                    # Path confined to a single container → discard
                    continue
                cross_ratio = len(containers_seen) / max(len(path_edges), 1)
                confidence = min_ew * cross_ratio
                if confidence < self.min_edge_weight:
                    continue
                paths.append(DFSPath(
                    sentence_ids=path_sids,
                    edges=path_edges,
                    min_edge_weight=min_ew,
                    cross_container_ratio=cross_ratio,
                    confidence=confidence,
                    constraint_container_ids=containers_seen & constraint_set,
                ))
                continue

            if len(path_edges) >= max_hops:
                continue

            for neighbour_sid, edge in self.edge_registry.get_neighbors(
                cur_sid, edge_types=self.ALL_EDGE_TYPES
            ):
                if neighbour_sid in visited_frozen:
                    continue
                new_min_ew = min(min_ew, edge.weight)
                if new_min_ew < self.min_edge_weight:
                    continue
                new_visited = visited_frozen | {neighbour_sid}
                new_containers = containers_seen | self._containers_of_sentence(neighbour_sid)
                new_passes = passes_constraint or bool(new_containers & constraint_set)
                new_edges = path_edges + [edge]
                stack.append((
                    neighbour_sid,
                    path_sids + [neighbour_sid],
                    new_edges,
                    new_min_ew,
                    new_visited,
                    new_containers,
                    new_passes,
                ))

        return paths

    def _phase5_search_all_groups(
        self,
        subgraphs_a: List[SubGraph],
        subgraphs_b: List[SubGraph],
        entity_a: str,
        entity_b: str,
        constraint_container_ids: List[str],
        groups: List[Tuple[SubGraph, SubGraph, str]],
    ) -> Tuple[List[DFSPath], Optional[DFSPath], str]:
        """
        Try DFS for each intersection group in priority order.
        Returns (all_paths, best_path, group_used).
        """
        best_path: Optional[DFSPath] = None
        group_used = "none"
        all_paths: List[DFSPath] = []

        for sg_a, sg_b, label in groups:
            seed_sids_a = list(sg_a.core_ids) or list(sg_a.sentence_ids)
            paths = self._phase5_dfs_search(
                seed_sids=seed_sids_a,
                target_entity=entity_b,
                constraint_container_ids=constraint_container_ids,
            )
            all_paths.extend(paths)
            if paths:
                best_path = paths[0]
                group_used = label
                break

        all_paths.sort(key=lambda p: -p.confidence)
        self._metrics_phase5 = {
            "group_used": group_used,
            "total_paths_found": len(all_paths),
            "best_path_confidence": round(best_path.confidence, 4) if best_path else 0.0,
            "best_path_hops": best_path.hops if best_path else 0,
        }
        return all_paths, best_path, group_used

    # ─────────────────────────────────────────────────────────────────────
    # Global fallback
    # ─────────────────────────────────────────────────────────────────────

    def _global_fallback(
        self, question: str, entity_a: str, entity_b: str
    ) -> RetrievalResult:
        """
        When all four intersection groups fail, fall back to global
        FAISS search on the original question.
        """
        raw = self._faiss_search_sentences(question, k=20)
        all_sids = [sid for sid, _ in raw]

        sentence_map: Dict[str, SentenceNode] = {}
        for sid in all_sids:
            node = self.sentence_registry.get_node(sid)
            if node:
                sentence_map[sid] = node

        # Collect edges between retrieved sentences
        edges_out: List[LogicEdge] = []
        for sid in all_sids:
            for _, edge in self.edge_registry.get_neighbors(sid, self.ALL_EDGE_TYPES):
                if edge.target_sent_id in all_sids:
                    edges_out.append(edge)

        # Deduplicate edges
        seen_edge_keys = set()
        unique_edges = []
        for e in edges_out:
            key = (e.source_sent_id, e.target_sent_id, e.edge_type)
            if key not in seen_edge_keys:
                seen_edge_keys.add(key)
                unique_edges.append(e)

        containers_out = list({
            cid
            for sid in all_sids
            for cid in self._reverse_index.get(sid, [])
        })
        container_objs = [
            self.container_manager.get_container(cid)
            for cid in containers_out
            if self.container_manager.get_container(cid) is not None
        ]

        sat_ids = [sid for sid in all_sids
                   if self._get_sentence_role(sid) == "SATELLITE"]

        self._metrics_global = {
            "fallback": True,
            "entity_a": entity_a,
            "entity_b": entity_b,
            "n_sentences": len(all_sids),
        }

        return RetrievalResult(
            sentences=all_sids,
            sentence_map=sentence_map,
            containers=container_objs,
            edges=unique_edges,
            constraint_entities=[],
            constraint_containers=[],
            paths=[],
            best_path=None,
            phase_used="global_fallback",
            satellite_ids=sat_ids,
            metrics=self._metrics_global,
        )

    # ─────────────────────────────────────────────────────────────────────
    # SATELLITE upgrade
    # ─────────────────────────────────────────────────────────────────────

    def _upgrade_shared_satellites(
        self,
        shared_sids: Set[str],
        subgraphs_a: List[SubGraph],
        subgraphs_b: List[SubGraph],
    ) -> None:
        """
        If a SATELLITE sentence appears in shared_sids, upgrade it to CORE.
        This is done in-place on the sub-graph node maps.
        """
        for sg in subgraphs_a + subgraphs_b:
            if shared_sids & sg.sat_ids:
                for sid in list(shared_sids & sg.sat_ids):
                    sg.sat_ids.discard(sid)
                    sg.core_ids.add(sid)

    # ─────────────────────────────────────────────────────────────────────
    # Output assembly
    # ─────────────────────────────────────────────────────────────────────

    def _assemble_output(
        self,
        subgraphs_a: List[SubGraph],
        subgraphs_b: List[SubGraph],
        shared_sids: Set[str],
        constraint_entities: List[str],
        constraint_container_ids: List[str],
        all_paths: List[DFSPath],
        best_path: Optional[DFSPath],
        phase_used: str,
    ) -> RetrievalResult:
        """
        Assemble the final RetrievalResult:
          • Sentences ordered: A → B → intersection
          • SATELLITE ids tracked separately (side context only)
          • Cross-subgraph edge weights boosted ×1.2, normalised [0,1]
          • Duplicate sentences assigned to the higher-weight sub-graph
        """
        # Collect all core sentence ids per sub-graph
        a_core = {sid for sg in subgraphs_a for sid in sg.core_ids}
        b_core = {sid for sg in subgraphs_b for sid in sg.core_ids}
        intersection_core = shared_sids & (a_core | b_core)

        # Deduplicate: if a sentence appears in multiple sets,
        # keep it where its weight is highest
        seen: Set[str] = set()
        ordered: List[str] = []

        # Weight lookup: seed FAISS score + subgraph depth weight
        seed_scores: Dict[str, float] = {}
        for sg in subgraphs_a + subgraphs_b:
            seed_scores[sg.seed.sentence_id] = sg.seed.faiss_score

        def _sentence_weight(sid: str) -> float:
            # Prefer seed sentences (higher FAISS score)
            base = seed_scores.get(sid, 0.0)
            # Boost if in intersection
            if sid in intersection_core:
                base += 0.1
            return base

        def _add_with_dedup(sids: Set[str]) -> None:
            for sid in sorted(sids, key=lambda s: -_sentence_weight(s)):
                if sid not in seen:
                    seen.add(sid)
                    ordered.append(sid)

        _add_with_dedup(a_core - b_core)
        _add_with_dedup(b_core - a_core)
        _add_with_dedup(intersection_core)

        # Build sentence map
        sentence_map: Dict[str, SentenceNode] = {}
        for sid in ordered:
            node = self.sentence_registry.get_node(sid)
            if node:
                sentence_map[sid] = node

        # Collect all edges among output sentences
        output_sid_set = set(ordered)
        edges_raw: List[LogicEdge] = []
        for sid in ordered:
            for _, edge in self.edge_registry.get_neighbors(
                sid, self.ALL_EDGE_TYPES
            ):
                if edge.target_sent_id in output_sid_set:
                    edges_raw.append(edge)

        # Deduplicate edges
        seen_edge_keys: Set[Tuple[str, str, str]] = set()
        edges_out: List[LogicEdge] = []
        for e in edges_raw:
            key = (e.source_sent_id, e.target_sent_id, e.edge_type)
            if key not in seen_edge_keys:
                seen_edge_keys.add(key)
                edges_out.append(e)

        # Boost cross-subgraph edges ×1.2, normalise to [0,1]
        max_weight = max((e.weight for e in edges_out), default=1.0)
        if max_weight > 0:
            for e in edges_out:
                src_in_a = any(src in a_core for src in [e.source_sent_id])
                tgt_in_b = any(tgt in b_core for tgt in [e.target_sent_id])
                cross_a_to_b = src_in_a and tgt_in_b
                cross_b_to_a = any(src in b_core for src in [e.source_sent_id]) \
                    and any(tgt in a_core for tgt in [e.target_sent_id])
                if cross_a_to_b or cross_b_to_a:
                    e.weight = min(e.weight * 1.2 / max_weight, 1.0)
                else:
                    e.weight = e.weight / max_weight

        # Collect containers
        container_ids_out = list({
            cid
            for sid in ordered
            for cid in self._reverse_index.get(sid, [])
        })
        container_objs = [
            self.container_manager.get_container(cid)
            for cid in container_ids_out
            if self.container_manager.get_container(cid) is not None
        ]

        # SATELLITE ids (side context only)
        sat_ids = []
        for sg in subgraphs_a + subgraphs_b:
            for sid in sg.sat_ids:
                if sid not in ordered:
                    sat_ids.append(sid)

        metrics = {
            "phase1": getattr(self, "_metrics_phase1", {}),
            "phase2": getattr(self, "_metrics_phase2", {}),
            "phase3": getattr(self, "_metrics_phase3", {}),
            "phase4": getattr(self, "_metrics_phase4", {}),
            "phase5": getattr(self, "_metrics_phase5", {}),
            "global_fallback": getattr(self, "_metrics_global", {}),
        }

        return RetrievalResult(
            sentences=ordered,
            sentence_map=sentence_map,
            containers=container_objs,
            edges=edges_out,
            constraint_entities=constraint_entities,
            constraint_containers=constraint_container_ids,
            paths=all_paths,
            best_path=best_path,
            phase_used=phase_used,
            satellite_ids=sat_ids,
            metrics=metrics,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        entity_a: Optional[str] = None,
        entity_b: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Run the full 5-phase retrieval pipeline.

        If entity_a / entity_b are not provided they are auto-extracted from
        the question text.

        Returns a RetrievalResult.  If all groups fail, falls back to global
        FAISS search.
        """
        # Auto-extract entities if not provided
        if entity_a is None or entity_b is None:
            auto_a, auto_b = self._extract_entities_from_question(question)
            entity_a = entity_a or auto_a
            entity_b = entity_b or auto_b

        if not entity_a or not entity_b:
            log.warning(
                "Could not extract entity pair from question: %s", question
            )
            return self._global_fallback(question, "", "")

        # ── Phase 1 ──────────────────────────────────────────────────────
        seeds_a, seeds_b = self._phase1_subquestion_recall(
            question, entity_a, entity_b
        )
        if not seeds_a or not seeds_b:
            return self._global_fallback(question, entity_a, entity_b)

        # ── Phase 2 ──────────────────────────────────────────────────────
        subgraphs_a, subgraphs_b = self._phase2_expand_all_seeds(seeds_a, seeds_b)

        # ── Phase 3 ──────────────────────────────────────────────────────
        matched_pair, ix, group_label = self._phase3_intersection_validation(
            subgraphs_a, subgraphs_b
        )

        if not matched_pair:
            return self._global_fallback(question, entity_a, entity_b)

        sg_a, sg_b = matched_pair

        # ── SATELLITE upgrade ─────────────────────────────────────────────
        self._upgrade_shared_satellites(ix.shared_sids, subgraphs_a, subgraphs_b)

        # ── Phase 4 ──────────────────────────────────────────────────────
        constraint_entities, constraint_container_ids = self._phase4_extract_constraints(
            sg_a, sg_b, ix
        )

        # ── Phase 5 ──────────────────────────────────────────────────────
        groups = [
            (subgraphs_a[0], subgraphs_b[0], "A1∩B1"),
            (subgraphs_a[0], subgraphs_b[-1], "A1∩B2"),
            (subgraphs_a[-1], subgraphs_b[0], "A2∩B1"),
            (subgraphs_a[-1], subgraphs_b[-1], "A2∩B2"),
        ]
        all_paths, best_path, phase_used = self._phase5_search_all_groups(
            subgraphs_a, subgraphs_b,
            entity_a, entity_b,
            constraint_container_ids,
            groups,
        )

        if not best_path:
            return self._global_fallback(question, entity_a, entity_b)

        # ── Output assembly ───────────────────────────────────────────────
        return self._assemble_output(
            subgraphs_a=subgraphs_a,
            subgraphs_b=subgraphs_b,
            shared_sids=ix.shared_sids,
            constraint_entities=constraint_entities,
            constraint_container_ids=constraint_container_ids,
            all_paths=all_paths,
            best_path=best_path,
            phase_used=phase_used,
        )

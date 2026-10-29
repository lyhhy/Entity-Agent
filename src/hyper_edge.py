"""
Hyperedge Module
================

Container-level hyperedges connecting multiple containers via shared
events (noun-phrase clusters within a chunk).

Per spec (§1.5):
    - Hyperedges live at the container level (NOT sentence level).
    - Within a chunk, containers are grouped by shared noun-phrase
      (the "event name"). Each group emits one hyperedge.
    - Hyperedges with the same event name are merged.
    - A single container can belong to multiple hyperedges (overlap
      allowed).
    - Hyperedges do NOT cross chunks.
    - Each hyperedge records: container_ids, event_name, time_range,
      source_chunk_id.
    - Bidirectional bookkeeping: hyperedge records its containers,
      each container records its hyperedge IDs.

The hyperedge system is consumed by `DualEntityRetriever` (track 2
"hyperedge association check"): if a question contains two entities
A and B, hyperedges that contain both A's container and B's container
are candidate bridges. They are filtered by event-name similarity to
the question (>= 0.85) and (if the question is temporal) by time-range
overlap with the question's time reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import re
import threading


# ─────────────────────────────────────────────────────────────────
# Time-word → (start, end) range mapping for the hyperedge's
# `time_range` field.  Spec says "2022 年 → 2022-01 ~ 2022-12"; we
# follow the same convention: a single year expands to Jan-Dec, a
# single month expands to the full month.  Anything we can't parse
# falls back to ("", "") which means "no time constraint".
# ─────────────────────────────────────────────────────────────────
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_time_word(text: str) -> Tuple[str, str]:
    """
    Extract a coarse (start, end) date range from a textual time word.

    Examples:
        "2022"      -> ("2022-01", "2022-12")
        "2022-03"   -> ("2022-03", "2022-03")
        "March 2022"-> ("2022-03", "2022-03")
        "Q1 2022"   -> ("2022-01", "2022-03")
        ""            -> ("", "")
    """
    if not text:
        return ("", "")
    t = text.lower().strip()
    # Year only: e.g., "2022"
    m = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", t)
    if m:
        year = m.group(1)
        # Check for month
        month_match = None
        for name, num in _MONTHS.items():
            if name in t:
                month_match = num
                break
        if month_match:
            mm = f"{month_match:02d}"
            return (f"{year}-{mm}", mm)
        return (f"{year}-01", f"{year}-12")
    return ("", "")




@dataclass
class HyperEdge:
    """
    A container-level hyperedge grouping containers by a shared event.

    Attributes:
        id: Unique hyperedge ID (e.g., "h_0001")
        container_ids: Container IDs that participate in the hyperedge.
        event_name: The shared noun phrase that anchors the event.
        time_range: Tuple (start, end) in YYYY-MM / YYYY format.
        source_chunk_id: The chunk that produced this hyperedge.
        chunk_doc_ids: Document IDs that contributed containers (optional,
            used for retrieval filtering).
        attribute_triggers: Mapping container_id -> set of attribute
            names that this event "triggers" for that container.
            The hyperedge is the ONLY place where the cross-container
            association between specific attributes is recorded, which
            is what makes the property "hyperedge → container → attribute
            → sentence" navigable in the sparse graph.

            When the trigger set is empty for a container (legacy data),
            retrieval falls back to "all attributes of that container"
            so existing behaviour is preserved.
    """
    id: str
    container_ids: List[str] = field(default_factory=list)
    event_name: str = ""
    time_range: Tuple[str, str] = ("", "")
    source_chunk_id: str = ""
    chunk_doc_ids: List[str] = field(default_factory=list)
    # Phase-Sparse: per-container attribute triggers.
    # {container_id -> set(attribute_name)}
    attribute_triggers: Dict[str, Set[str]] = field(default_factory=dict)
    # 段落级超边融合：记录该事件级超边由哪些原段落超边 id 融合而成。
    # 仅融合产生的超边会设置该字段；原段落超边保持为空。
    merged_from: List[str] = field(default_factory=list)

    def add_container(self, cid: str) -> None:
        """Idempotent container add."""
        if cid not in self.container_ids:
            self.container_ids.append(cid)

    def add_trigger(self, cid: str, attribute_name: str) -> None:
        """Record that this event triggers `attribute_name` on `cid`."""
        if attribute_name:
            self.attribute_triggers.setdefault(cid, set()).add(attribute_name)

    def triggers_for(self, cid: str) -> Set[str]:
        """Return the set of attributes triggered for `cid` (empty = all)."""
        return set(self.attribute_triggers.get(cid, set()))

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "container_ids": list(self.container_ids),
            "event_name": self.event_name,
            "time_range": list(self.time_range),
            "source_chunk_id": self.source_chunk_id,
            "chunk_doc_ids": list(self.chunk_doc_ids),
            "attribute_triggers": {
                cid: sorted(attrs)
                for cid, attrs in self.attribute_triggers.items()
            },
            "merged_from": list(self.merged_from),
        }


# ─────────────────────────────────────────────────────────────────
# 段落级超边融合常量 + 工具
#  - 在段落级超边（每段一条，成员 2-4 container）构建完之后调用
#    `HyperEdgeRegistry.merge_paragraph_hyperedges()` 把描述同一
#    事件的多条段落超边融合为事件级超边。
#  - 融合判据 + 阈值全部用规则，不调 LLM。
# ─────────────────────────────────────────────────────────────────
LOW_FREQ_THRESHOLD = 20        # 实体的全局出现次数 < 该值视为"低频"
TITLE_SIM_THRESHOLD = 0.8      # 段落标题嵌入 cos 相似度阈值
MAX_MERGED_SIZE = 30           # 融合后成员数上限；超则整组回退为原段落超边


class _UnionFind:
    """轻量并查集，用作传递融合的连通分量。"""
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]  # 路径压缩
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


class HyperEdgeRegistry:
    """
    Registry for container-level hyperedges.

    Bidirectional bookkeeping is maintained automatically:
        * Adding a container to a hyperedge also appends the
          hyperedge ID to that container's `hyperedges` list.
        * Containers are looked up via `get_hyperedges_for_container`.

    Hyperedges are deduplicated by event_name within a single chunk —
    if two chunks of containers share the same event_name, the
    existing hyperedge is extended rather than duplicated.
    """

    def __init__(self, container_manager=None):
        # id -> HyperEdge
        self._edges: Dict[str, HyperEdge] = {}
        # event_name (within chunk_id) -> id  (for merge within chunk)
        self._event_index: Dict[Tuple[str, str], str] = {}
        self._counter: int = 0
        self._lock = threading.Lock()
        # Optional reference to register container-side bookkeeping.
        self._container_manager = container_manager

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def build_hyperedges_for_chunk(
        self,
        chunk_id: str,
        containers: List,
        time_word: str = "",
        doc_ids: Optional[List[str]] = None,
        paragraph_title: Optional[str] = None,
        attribute_triggers: Optional[Dict[str, List[str]]] = None,
    ) -> List[HyperEdge]:
        """
        Build hyperedges for one chunk.

        Algorithm:
          1. PHASE-E1 [event_name fix]: the event_name of a hyperedge
             is the chunk's PARAGRAPH TITLE (the noun-phrase cluster
             the spec calls the "event name"). It is NOT a single
             entity name.  This fixes the bug where the previous
             implementation treated each entity in `entity_inventory`
             as its own event, causing a single chunk to emit dozens
             of hyperedges that all shared only the entity name
             "discovery" / "the matrix" / "kelly", which then bridged
             completely unrelated paragraphs.

             Concretely:
             * If `paragraph_title` is provided, ONE hyperedge per
               chunk is emitted, with `event_name = paragraph_title`.
             * If `paragraph_title` is missing, we fall back to the
               UNION of titles' tokens of containers in the chunk
               (so legacy callers that pass only containers still get
               sensible events).

          2. The hyperedge connects EVERY container in the chunk.
             This is the spec's "all containers in a chunk are joined
             by the shared event" rule.  Cross-chunk connectivity
             comes from subsequent multi-hop expansion, not from a
             single chunk emitting many hyperedges.

          3. Phase-Sparse: per-container attribute triggers.
             The optional `attribute_triggers` dict maps container_id
             -> list of attribute names that this event "triggers"
             for that container. When the dict is provided, the
             hyperedge records which specific attributes are joined
             by the event, which makes the path
                hyperedge → container → attribute → sentence
             navigable.  If `attribute_triggers` is missing or a
             container has no entry, retrieval falls back to "all
             attributes of the container" (legacy behaviour).

        Args:
            chunk_id: Identifier of the chunk being processed.
            containers: List of Container objects in the chunk.
            time_word: Textual time reference (e.g. "2022"); expanded
                into a (start, end) tuple by `parse_time_word`.
            doc_ids: Document IDs in this chunk (optional).
            paragraph_title: Title of the paragraph (chunk). When set,
                this becomes the event_name. Preferred over the legacy
                entity-inventory path.
            attribute_triggers: Optional mapping container_id →
                list[attribute_name]. Records which attributes on
                each container this event actively involves.

        Returns:
            List of newly created or extended HyperEdge objects
            (typically length 1 when paragraph_title is set).
        """
        if not containers:
            return []

        time_range = parse_time_word(time_word)

        # 旧: 一条超边 = 整 chunk 全连接团, event_name = paragraph_title
        # (问题 3 是真的, 但 token 聚类在某些 chunk 退步; 先恢复原版).
        event_name = (paragraph_title or "").strip() or self._fallback_event_name(containers)

        edge = self._get_or_create_edge(
            chunk_id=chunk_id,
            event_name=event_name,
            time_range=time_range,
            doc_ids=doc_ids or [],
        )
        new_containers: List[str] = []
        triggers = attribute_triggers or {}
        for c in containers:
            cid = getattr(c, "id", None)
            if not cid or cid in edge.container_ids:
                continue
            edge.add_container(cid)
            new_containers.append(cid)
            # Record the per-container attribute triggers, if any.
            for attr_name in triggers.get(cid, []) or []:
                edge.add_trigger(cid, attr_name)
            if self._container_manager is not None:
                container_obj = self._container_manager.get_container(cid)
                if container_obj is not None:
                    container_obj.add_hyperedge(edge.id)
        return [edge] if new_containers else []

    @staticmethod
    def _fallback_event_name(containers: List) -> str:
        """
        Legacy fallback: when `paragraph_title` is not provided,
        derive a stable event name from the most-common token in the
        containers' entity names.  This is *not* a per-entity
        hyperedge (we do NOT emit one per entity any more); it is
        a single shared label so the chunk still produces one
        hyperedge.
        """
        from collections import Counter
        ctr: Counter = Counter()
        for c in containers:
            ent = (getattr(c, "entity", "") or "").lower().strip()
            if not ent:
                continue
            for tok in re.findall(r"[a-z0-9'\-]+", ent):
                if len(tok) >= 4:
                    ctr[tok] += 1
        if not ctr:
            # Last resort: pick the first non-empty entity name.
            for c in containers:
                ent = (getattr(c, "entity", "") or "").lower().strip()
                if ent:
                    return ent
            return "event"
        # Most common long token across all containers in this chunk.
        return ctr.most_common(1)[0][0]

    def _get_or_create_edge(
        self,
        chunk_id: str,
        event_name: str,
        time_range: Tuple[str, str],
        doc_ids: List[str],
    ) -> HyperEdge:
        key = (event_name, chunk_id)
        existing_id = self._event_index.get(key)
        if existing_id is not None:
            edge = self._edges.get(existing_id)
            if edge is not None:
                # Merge time_range / doc_ids.
                if not edge.time_range[0]:
                    edge.time_range = time_range
                for d in doc_ids:
                    if d not in edge.chunk_doc_ids:
                        edge.chunk_doc_ids.append(d)
                return edge

        with self._lock:
            self._counter += 1
            new_id = f"h_{self._counter:04d}"
            edge = HyperEdge(
                id=new_id,
                event_name=event_name,
                time_range=time_range,
                source_chunk_id=chunk_id,
                chunk_doc_ids=list(doc_ids),
            )
            self._edges[new_id] = edge
            self._event_index[key] = new_id
            return edge

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_hyperedge(self, he_id: str) -> Optional[HyperEdge]:
        """Direct lookup by id. O(1)."""
        return self._edges.get(he_id)

    def get_hyperedges_for_container(self, cid: str) -> List[HyperEdge]:
        """All hyperedges that include this container."""
        if not self._container_manager:
            return []
        c = self._container_manager.get_container(cid)
        if c is None:
            return []
        return [self._edges[h] for h in c.hyperedges if h in self._edges]

    def get_hyperedges_containing_both(
        self,
        cid_a: str,
        cid_b: str,
    ) -> List[HyperEdge]:
        """Hyperedges that include BOTH cid_a and cid_b."""
        edges_a = {e.id: e for e in self.get_hyperedges_for_container(cid_a)}
        return [e for eid, e in edges_a.items()
                if cid_b in e.container_ids]

    def get_all_hyperedges(self) -> List[HyperEdge]:
        return list(self._edges.values())

    def __len__(self) -> int:
        return len(self._edges)

    # ------------------------------------------------------------------
    # 段落级超边 → 事件级超边 融合
    # ------------------------------------------------------------------
    def merge_paragraph_hyperedges(
        self,
        entity_freq: Optional[Dict[str, int]] = None,
        embedder: Optional[Any] = None,
        low_freq_threshold: int = LOW_FREQ_THRESHOLD,
        title_sim_threshold: float = TITLE_SIM_THRESHOLD,
        max_merged_size: int = MAX_MERGED_SIZE,
    ) -> Dict[str, Any]:
        """
        把描述同一事件的多条段落超边融合为事件级超边（并查集传递融合）。

        融合判据 (三条 AND)：
          1. 共享低频实体：H_i/H_j 的实体集交集里，至少一个实体的
             `entity_freq[entity] < low_freq_threshold`。
          2. 标题相似：cos(embed(title_i), embed(title_j)) > title_sim_threshold；
             任一标题为空则跳过该判据，仅用 1 + 3。
          3. 时间重叠：若两条超边都有 time_range 且不重叠，则不融合；
             任一没有 time_range 则跳过该判据。

        融合后：
          - event_name = 融合组中 `len(container_ids)` 最多的那条原超边的 title
          - container_ids = 融合组所有超边 container 的并集（去重）
          - time_range = 融合组所有 time_range 的并集（最早开始，最晚结束）
          - source_chunk_id = 第一个原超边的 chunk_id（合并语义）；同时通过
            chunk_doc_ids 扩展多来源；merged_from 记录所有原超边 id
          - 若合并后 len(container_ids) > max_merged_size，整组回退为原
            段落超边（即不参与融合）

        Args:
            entity_freq: {entity_norm -> global_count}。None 时退化为全部
                实体视为"低频"（即仅靠标题 + 时间判据）。
            embedder: 有 .encode([str], normalize_embeddings=True) 的对象；
                None 时退化为基于 token 的 Jaccard 相似度。
            其余参数覆盖模块级默认值。

        Returns:
            stats dict with: before, after, size_buckets, fallback_groups,
            fallback_members, merged_groups。
        """
        edges = list(self._edges.values())
        n_before = len(edges)
        if n_before <= 1:
            stats = {
                "before": n_before,
                "after": n_before,
                "size_buckets": {"2-4": 0, "5-9": 0, "10-19": 0, "20-30": 0, ">30": 0},
                "fallback_groups": 0,
                "fallback_members": 0,
                "merged_groups": 0,
            }
            print(
                f"[hyperedge-merge] skip: {n_before} edge(s) (<=1), nothing to merge"
            )
            return stats

        # ---- 准备每条超边的签名 ----
        # 1) entity set
        edge_entities: List[Set[str]] = [self._entities_of_edge(e) for e in edges]
        # 2) time_range tuple (copy)
        edge_time: List[Tuple[str, str]] = [
            tuple(e.time_range or ("", "")) for e in edges
        ]
        # 3) title
        edge_title: List[str] = [(e.event_name or "").strip() for e in edges]
        # 4) title vector (cache)
        edge_title_vec: Dict[int, Any] = {}
        if embedder is not None:
            try:
                import numpy as _np
                titles_to_encode = [t if t else " " for t in edge_title]
                vecs = _np.asarray(
                    embedder.encode(titles_to_encode, normalize_embeddings=True),
                    dtype=_np.float32,
                )
                for i, v in enumerate(vecs):
                    edge_title_vec[i] = v
            except Exception as exc:  # noqa: BLE001
                print(f"[hyperedge-merge] title encode failed, fallback to Jaccard: {exc}")
                embedder = None

        # ---- 判据对 (i, j) 是否可融合 ----
        uf = _UnionFind(n_before)

        def _has_low_freq_intersection(i: int, j: int) -> bool:
            inter = edge_entities[i] & edge_entities[j]
            if not inter:
                return False
            if entity_freq is None:
                return True  # 退化模式：全视为低频
            for ent in inter:
                if int(entity_freq.get(ent, 0)) < low_freq_threshold:
                    return True
            return False

        def _title_similar(i: int, j: int) -> bool:
            ti, tj = edge_title[i], edge_title[j]
            if not ti or not tj:
                return True  # 标题为空 → 跳过该判据
            if ti.lower() == tj.lower():
                return True
            if embedder is not None and i in edge_title_vec and j in edge_title_vec:
                import numpy as _np
                vi, vj = edge_title_vec[i], edge_title_vec[j]
                ni = float(_np.linalg.norm(vi))
                nj = float(_np.linalg.norm(vj))
                if ni > 0 and nj > 0:
                    cos = float(_np.dot(vi, vj) / (ni * nj))
                    return cos > title_sim_threshold
            # 退化：包含式判据
            # 1) 任一标题的所有 token 都被另一标题包含 → 视为相似
            # 2) 否则退回 token Jaccard
            ta = set(ti.lower().split())
            tb = set(tj.lower().split())
            if not ta or not tb:
                return False
            if ta <= tb or tb <= ta:
                return True
            jac = len(ta & tb) / len(ta | tb)
            return jac > title_sim_threshold

        def _time_overlaps(i: int, j: int) -> bool:
            ti, tj = edge_time[i], edge_time[j]
            si_a, ei_a = ti
            si_b, ei_b = tj
            # 任一端时间字段为空 → 跳过该判据
            if not si_a or not ei_a or not si_b or not ei_b:
                return True
            # ISO-like 字符串字典序即可比较
            return not (ei_a < si_b or ei_b < si_a)

        for i in range(n_before):
            for j in range(i + 1, n_before):
                if uf.find(i) == uf.find(j):
                    continue
                if not _has_low_freq_intersection(i, j):
                    continue
                if not _title_similar(i, j):
                    continue
                if not _time_overlaps(i, j):
                    continue
                uf.union(i, j)

        # ---- 收集连通分量 ----
        groups: Dict[int, List[int]] = {}
        for i in range(n_before):
            r = uf.find(i)
            groups.setdefault(r, []).append(i)

        # ---- 决定哪些组实际融合 ----
        # 仅大小 >= 2 的组才考虑融合；超 max_merged_size 整组回退
        new_edges: List[HyperEdge] = []
        merged_group_count = 0
        fallback_group_count = 0
        fallback_member_total = 0
        size_buckets = {"2-4": 0, "5-9": 0, "10-19": 0, "20-30": 0, ">30": 0}

        for root, members in groups.items():
            if len(members) == 1:
                # 单独一条：原样保留
                new_edges.append(edges[members[0]])
                continue

            # 计算融合后成员数（去重 container_id）
            merged_containers: List[str] = []
            seen_cid: Set[str] = set()
            for idx in members:
                for cid in edges[idx].container_ids or []:
                    if cid not in seen_cid:
                        seen_cid.add(cid)
                        merged_containers.append(cid)

            if len(merged_containers) > max_merged_size:
                # 整组回退为原段落超边
                fallback_group_count += 1
                fallback_member_total += len(members)
                titles = [edges[m].event_name for m in members]
                print(
                    f"[hyperedge-merge]   fallback group {fallback_group_count}: "
                    f"size={len(merged_containers)}>{max_merged_size}  "
                    f"member_titles={titles}"
                )
                for idx in members:
                    new_edges.append(edges[idx])
                continue

            if len(merged_containers) <= 1:
                # 边界：融合后成员数 == 1；记录 warning 并保留
                print(
                    f"[hyperedge-merge] WARNING: merged group size=1 after dedup "
                    f"(raw members={len(members)}); keeping originals"
                )
                for idx in members:
                    new_edges.append(edges[idx])
                continue

            # event_name：取组内 len(container_ids) 最多的那条原超边
            best_idx = max(
                members, key=lambda k: len(edges[k].container_ids or [])
            )
            best_edge = edges[best_idx]

            # time_range：取最早 start、最晚 end
            starts = [edge_time[m][0] for m in members if edge_time[m][0]]
            ends = [edge_time[m][1] for m in members if edge_time[m][1]]
            new_time = (
                min(starts) if starts else "",
                max(ends) if ends else "",
            )

            # source_chunks：所有原超边 source_chunk_id 的并集
            chunk_ids: List[str] = []
            seen_chunks: Set[str] = set()
            for idx in members:
                cid = edges[idx].source_chunk_id or ""
                if cid and cid not in seen_chunks:
                    seen_chunks.add(cid)
                    chunk_ids.append(cid)

            # chunk_doc_ids：并集
            doc_ids: List[str] = []
            seen_docs: Set[str] = set()
            for idx in members:
                for d in edges[idx].chunk_doc_ids or []:
                    if d not in seen_docs:
                        seen_docs.add(d)
                        doc_ids.append(d)

            # 合并 attribute_triggers：每 container 的 attribute 集合并集
            merged_triggers: Dict[str, Set[str]] = {}
            for idx in members:
                trg = edges[idx].attribute_triggers or {}
                for cid, attrs in trg.items():
                    s = merged_triggers.setdefault(cid, set())
                    s.update(attrs or [])

            # 取第一个原超边的 source_chunk_id 作为主归属
            primary_chunk = (
                edges[members[0]].source_chunk_id or best_edge.source_chunk_id
            )

            # 分配新 id
            with self._lock:
                self._counter += 1
                new_id = f"h_{self._counter:04d}"

            merged_edge = HyperEdge(
                id=new_id,
                container_ids=list(merged_containers),
                event_name=(best_edge.event_name or "").strip(),
                time_range=new_time,
                source_chunk_id=primary_chunk,
                chunk_doc_ids=doc_ids,
                attribute_triggers=merged_triggers,
                merged_from=[edges[m].id for m in members],
            )
            new_edges.append(merged_edge)
            merged_group_count += 1
            size_buckets[self._bucket(len(merged_containers))] += 1
            print(
                f"[hyperedge-merge]   group {merged_group_count}: "
                f"event_name={best_edge.event_name!r}  "
                f"merged_size={len(merged_containers)}  "
                f"merged_from={merged_edge.merged_from}"
            )

        # ---- 替换 registry 内部状态 ----
        # 1) 重新构建 _edges / _event_index
        new_edge_map: Dict[str, HyperEdge] = {e.id: e for e in new_edges}
        self._edges = new_edge_map
        self._event_index = {
            (e.event_name, e.source_chunk_id): e.id for e in new_edges
        }

        # 2) 同步 container 侧的 hyperedges 列表。
        # 重要：
        #   - 先清空每个 container 的 hyperedges 列表（要逐个生成新 list，
        #     避免所有 container 共享同一个 list 对象导致 append 互相污染）
        #   - 再按新超边的 container_ids 累加 push
        if self._container_manager is not None:
            try:
                all_cids = [c.id for c in self._container_manager.get_all_containers()]
                # 1) 清空：每个 container 独立新 list
                for cid in all_cids:
                    cobj = self._container_manager.get_container(cid)
                    if cobj is not None:
                        cobj.hyperedges = []
                # 2) 累加
                for e in new_edges:
                    for cid in e.container_ids or []:
                        cobj = self._container_manager.get_container(cid)
                        if cobj is not None and e.id not in cobj.hyperedges:
                            cobj.hyperedges.append(e.id)
            except Exception as _sync_exc:  # noqa: BLE001
                print(f"[hyperedge-merge] container-side sync failed: {_sync_exc}")

        n_after = len(new_edges)
        stats = {
            "before": n_before,
            "after": n_after,
            "size_buckets": size_buckets,
            "fallback_groups": fallback_group_count,
            "fallback_members": fallback_member_total,
            "merged_groups": merged_group_count,
        }
        print(
            f"[hyperedge-merge] before={n_before} after={n_after} "
            f"merged_groups={merged_group_count} "
            f"fallback_groups={fallback_group_count} "
            f"fallback_members={fallback_member_total}"
        )
        print(
            f"[hyperedge-merge] size_buckets (post-merge only): {size_buckets}"
        )
        return stats

    def _entities_of_edge(self, edge: HyperEdge) -> Set[str]:
        """从超边的 container_ids 抽取归一化实体集（用 self._container_manager）。"""
        ents: Set[str] = set()
        if self._container_manager is None:
            return ents
        for cid in edge.container_ids or []:
            c = self._container_manager.get_container(cid)
            if c is None:
                continue
            for raw in [getattr(c, "entity", "")] + list(
                getattr(c, "aliases", []) or []
            ):
                norm = (raw or "").lower().strip()
                if norm:
                    ents.add(norm)
        return ents

    @staticmethod
    def _bucket(size: int) -> str:
        if size <= 4:
            return "2-4"
        if size <= 9:
            return "5-9"
        if size <= 19:
            return "10-19"
        if size <= 30:
            return "20-30"
        return ">30"





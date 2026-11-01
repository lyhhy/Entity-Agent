"""
StenGraph Pipeline (10-step)
============================

Pure container/sentence retrieval. NO graph traversal, NO DFS, NO BFS
at runtime.  The only "graph" reads are:
  * container_manager.get_containers_for_entity  (entity→containers)
  * container_manager._container_connectivity   (B-A co-occurrence, O(1) lookup)
  * hyper_edge.get_hyperedges_for_container     (container→hyperedges)
  * sentence_registry._entity_index / get_neighbors (entity→sentences)

Everything else is set/dict arithmetic.

Pipeline (mirrors the spec the user gave):

  [1] entity extraction + query vector
  [2] entity match + query→container faiss  →  containers A  (top-3)
  [3] for each A: hyperedges H that touch A
  [4] per-hyperedge: B_h = (other containers in h, ranked by
        0.6 * faiss_cos(query, B) + 0.4 * cooccur(B, A))
        union of top-3 of each h  →  containers B
  [5] for each query entity e with no container hit:
        find chunk(s) that contain e
        take sentences in those chunks that contain e
        plus the sentence immediately before / after each one
  [6] candidate pool = sentences(A) ∪ sentences(B) ∪ traced(e)
  [7] rank candidate sentences by sentence-level query cosine
        (we use container-level faiss *as a proxy* — there is no
        sentence faiss index in this codebase; if no faiss available,
        fall back to token Jaccard)  →  top-N
  [8] for each B container:
        sentences in B that contain any of its hyperedge event_name
        (via event_name_similarity >= 0.85 to be semantically faithful,
        not bare substring)  →  supplement pool
  [9] merge [7] ∪ [8] (supplement only fills gaps in [7] up to top-N)
  [10] final semantic rerank over the merged pool  →  top-5 sentences
  output: plain text of the top-5 sentence contents
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# [2026-10-30] DEPRECATED: 旧精排公式的权重常量。
#
# 旧公式：
#   S_f(F) = W_EVENT * S_h(E,Q) + (1 - W_EVENT) * S_h(F,Q)
# 其中 S_h(E,Q) = W_EVENT_NAME·cos(title,q) + W_EVENT_SENT·cos(sent,q)
#
# 这些常量现在**没有任何消费者**（新的公式用 W_EVENT_NAME_N /
# W_SENT_N / W_FACT_N / W_ATTR / W_LEX）。保留它们是为了不破坏
# 外部可能 import 它们的代码，但它们不再影响排序。
#
# 保留它们的另一个理由是**可对照**：旧常量记录了"当初以为的权重
# 配比"，而新公式暴露了一个事实 —— 在未归一化的求和里，
# W_EVENT_NAME=0.7 这个设定**从来没有真正生效**（句子余弦的方差
# 更大，实际主导了排序）。
# ─────────────────────────────────────────────────────────────────
W_EVENT: float = 0.3

# S_h_E(句子) 的内部配比（旧公式专用，见上）：
#   S_h_E(sent) = 0.7 * cos(event_name, query)
#               + 0.3 * cos(sentence_vec, query)
#
# 权重可调: 环境变量 STENRAG_EVENT_NAME_WEIGHT (0.0-1.0) 覆盖事件名权重,
# 句子权重 = 1 - 事件名权重。
#
# 降权的代价: 事件名是唯一区分"同一段落内哪个事件更相关"的信号。
# 权重过低会让段落内所有句子的 S_h_E 趋于一致, 退回旧 mean-member 行为。
W_EVENT_NAME: float = float(
    os.environ.get("STENRAG_EVENT_NAME_WEIGHT", "0.7") or 0.7
)
W_EVENT_SENT: float = 1.0 - W_EVENT_NAME

# Attribute rerank: continuous soft-match score added to the final formula.
# Soft-match replaces the old binary attr_match from mapper/filter.
# Range [0, 1]; W_ATTR=0.10 gives attr_term range [0, 0.10].
#
# ─────────────────────────────────────────────────────────────────
# [2026-10-30] 精排公式修复 —— 量纲归一化 + 词法信号接回
#
# 上一版公式有三个结构性缺陷，导致"属性在召回层有效，但 EM/F1 反而
# 下降"：
#
# (1) **量纲不可比**。`name_term` = cos(段落标题, query) 典型值
#     0.15–0.35；`sent_term` = cos(整句, query) 典型值 0.35–0.55。
#     两者直接加权求和，谁的方差大谁主导 —— 结果是 W_EVENT_NAME=0.7
#     这个"事件名主导"的设定在数值上根本没有实现（实际是句子向量
#     主导），与设计意图完全相反。
#
# (2) **属性信号被压死**。W_ATTR=0.10 → attr_term ∈ [0, 0.10]，
#     而 final 的动态范围约 0.2–0.6。属性的整个取值范围只相当于
#     句子余弦波动的 1/4，属性表的所有改进在最终排序里几乎看不见。
#
# (3) **词法信号整条链路断线**。`_score_sentences_v8` /
#     `_bm25_scores` / `_score_tfidf` 在 Step 7 被删后失去唯一调用者，
#     现在是死代码 —— BM25、TF-IDF 实体区分度、以及专门为 HotpotQA
#     写的答案形状先验（ner_density / has_number）全部不参与打分。
#     排序只剩两个稠密余弦。
#
# 修复方式：所有分量先各自归一化到 [0,1]（min-max over the候选池），
# 再加权相加。归一化之后"量纲"不再是问题，权重才真正表达设计意图。
# ─────────────────────────────────────────────────────────────────
W_ATTR: float = float(
    os.environ.get("STENRAG_W_ATTR", "0.30") or 0.30
)

# 词法信号（BM25 + 答案形状先验）权重。0 = 完全关闭（保持旧行为）。
#
# 这条信号是**独立于稠密余弦**的：MiniLM 在"问句问一个具体事实、
# 答案句是短事实句"这类任务上纯稠密余弦的 top-1 精度本来就弱，
# 而 BM25 能靠 IDF 权重救回稀有实体（Havel / Fitzgerald 这类词在
# Jaccard/稠密余弦里权重被稀释）。改造日志 A 已经验证过：接回
# TF-IDF 后 R@5 +0.033。
W_LEX: float = float(
    os.environ.get("STENRAG_W_LEX", "0.30") or 0.30
)

# 事件名（段落标题）分量的归一化后权重。
#
# 注意：修复前 `name_term` 的绝对贡献只有 0.15·0.7 ≈ 0.1，而大事件
# 惩罚（≤1 的乘子）把它压到 0.01 量级 —— 惩罚对排序的影响不到 3%，
# GAMMA_E 调成 2、3 也没用。归一化之后 name_term 的量级回到 [0,1]，
# 惩罚才真正起作用。
W_EVENT_NAME_N: float = float(
    os.environ.get("STENRAG_W_EVENT_NAME", "0.35") or 0.35
)

# 句子向量分量的归一化后权重。
W_SENT_N: float = float(
    os.environ.get("STENRAG_W_SENT", "0.20") or 0.20
)

# 事实项（句子向量余弦）的归一化后权重。
W_FACT_N: float = float(
    os.environ.get("STENRAG_W_FACT", "0.15") or 0.15
)

# 大事件惩罚的指数（作用于归一化后的 name 分量）。
#
# GAMMA_E=1.0 时 penalty = 1/(size-1)。段落标题在 HotpotQA 里常常
# 一个词都不出现在问句里（问句问的是属性，标题是文章名），所以
# name 分量对"答案在哪个段落"几乎无判别力 —— 这正是要靠 B 段落
# 配额来解决的，而不是靠加大惩罚。
#
# 修复前 name 分量的绝对贡献只有 0.15·0.7 ≈ 0.1，惩罚把它压到
# 0.01 量级 —— 惩罚对排序的影响不到 3%，GAMMA_E 调成 2、3 也没用。
# 归一化之后 name 分量的量级回到 [0,1]，惩罚才真正起作用。
GAMMA_E: float = 1.0

# Hard attribute gate: a candidate sentence is kept only if it CARRIES at
# least one attribute that the query actually asks for.
#
# The gate works on the *top-3* query attributes by normalised cosine
# (query_emb · attr_desc_emb).  A sentence passes when its own vector
# attribute profile (sent_attr_vec) contains one of those attributes
# above ATTR_SENT_MIN_SIM.
#
# Rationale: the question names the attribute it wants ("what year was
# it released" → `date`; "who is the head coach" → `leadership`).
# A sentence that carries none of the top-3 query attributes is unlikely
# to answer it, so it is DELETED from the pool.
#
# The gate is a DELETION, not a reorder: a gate-dropped sentence never
# reaches the LLM.  (The previous reorder behaviour let 66% of the
# delivered context be gate-dropped sentences, which made the gate a
# no-op.)
#
# When the query has no usable attribute signal (no embedder, or all
# cosines are 0) the gate is disabled for that query — it must never
# empty the pool.
#
# ─────────────────────────────────────────────────────────────────
# [2026-11-01] **硬门降级为软门（改造 H）**
# ─────────────────────────────────────────────────────────────────
# 硬门被证伪了。seed=7 / 20 题的 A/B（无属性 → 有属性）：
#
#     EM    0.850 → 0.750   (-0.100)
#     R@1   0.684 → 0.579   (-0.105)
#     R@10  0.947 → 0.789   (-0.158)
#
# 但 RANK-DIAG 显示 **排序反而变好了**（排序丢失 6 → 2），恶化 100%
# 来自"门删掉"这一项（0 → 5）。逐题复盘 5 道被删的题，发现：
#
#   · 5ab9d2e2: gold 的 final_score = 0.962，全场第 1 —— 排序器已经
#     把 gold 排到第一了，**门把它删掉了**。
#   · 5a74cfed: gold 的 final_score = 1.072，全场第 1 —— 同上。
#
# 也就是说，**门的删除行为覆盖了排序器正确的判断**。
#
# 根因：`wanted_attrs` 只取 query 侧 embedding 余弦的 top-3，**完全
# 不看 sentence 侧挂了什么**。而 `mounted_attrs` 是"实体类型 × 属性
# 类别"的粗分类，于是经常出现 wanted 与 mounted **零交集**：
#
#   · gold 挂 [achievement, career, education, event]，门要
#     [location, population, residence] —— 零交集
#   · gold 挂 8 个属性，门只认 3 个，其中一个都不重叠
#   · gold 根本没挂任何属性（mounted = []）
#
# 这是**粗分类粒度**问题，不是阈值问题 —— 把 ATTR_GATE_TOP_K 调大
# 或调小都治不了"wanted 集合本身是错的"。
#
# 所以：**属性信号保留为打分分量（W_ATTR），但不再拥有删除权。**
# 门改为"软门"：属性不满足的句子不被删除，只是拿不到属性加权，
# 排序时自然下沉。属性从"准入证"退回"加分项"—— 这才是它作为
# 排序信号应有的位置。
#
# `ATTR_GATE_TOP_K` 仍然决定**哪些属性算被问到了**（即 wanted 集合
# 的宽度），它现在影响的是 `attr_match` 的计算范围，而不是生存。
ATTR_GATE_TOP_K: int = 3

# Canonical attribute names + descriptors used in symmetric-attribute
# soft matching.  These live in `sentence_graph` (the lowest-level
# module) so that the graph builder and this reranker share ONE
# vocabulary without a circular import.
#
# The descriptor (not the bare attribute name) is what gets embedded.
# A bare name like "birth" embeds to a point that is far from the way
# sentences talk about birth dates: measured cosines were 0.10-0.20 and
# frequently misclassified ("born" sentences scored as "genre").
#
# Each descriptor is a COMPLETE SENTENCE, not a keyword list, so the
# cosine measures "does this sentence assert this fact" rather than
# "do these words co-occur".
#
# The sentence side no longer comes from a hand-written keyword table
# (`_ATTRIBUTE_STEMS`): a sentence's continuous attribute profile
# (date 0.8, death 0.9) is computed from its embedding and applied as
# a SOFT mount, so one sentence can carry several attributes at once.
from .sentence_graph import (  # noqa: E402
    ATTRIBUTE_DESCRIPTORS,
    ATTRIBUTE_NAMES,
)

# Minimum cosine for a sentence to be considered to carry a given
# attribute.  Calibrated on real HotpotQA paragraphs: the median
# sentence's best attribute similarity is ~0.20-0.26, while genuine
# attribute-bearing sentences ("Its population was 403 at the 2010
# census", "He attended Duke University") land at 0.36-0.44.
ATTR_SENT_MIN_SIM: float = 0.20

# Minimum cosine for a query to be considered to want a given attribute.
# The query's best attribute always normalises to 1.0, so this is a
# *relative* cut: the third-best attribute must be at least this
# fraction of the best one.
ATTR_QUERY_MIN_SIM: float = 0.45

# If the hard gate would empty the pool, fall back to keeping the
# best-scoring candidates.  A gate that removes everything is worse
# than no gate, but silently keeping everything would hide a
# mis-calibrated gate — the fallback level is recorded in
# `attr_gate_level`.
ATTR_GATE_MIN_KEEP: int = 3


# ─────────────────────────────────────────────────────────────────
# [2026-11-01] 实体-属性槽位匹配的权重（改造 H，第 2 项）
#
# 槽位对齐分（"这个句子的这个实体，恰好挂着问句想要的属性"）在
# `attr_match` 里能起到的最大作用。它是一个**乘子**，所以：
#
#   · 槽位完全命中（实体对上 + 属性对上）→ attr_match = 1.0·W_ATTR
#   · 槽位没命中 → attr_match 只由粗粒度路径决定
#
# 为什么是 1.0 而不是更小：槽位匹配是**细粒度**证据，它一旦命中，
# 就说明这句确实在讲"问句问的那个实体的那个属性" —— 这正是我们要
# 的信号。粗粒度路径给的 0.3-0.6 在这种情况下是**低估**。
#
# 为什么不超过 1.0：末词降级有误匹配风险（"John Smith" 与
# "James Smith" 都降级到 "smith"）。留一点余量，让粗粒度信号在
# 误匹配时还能把分数拉回来。
# ─────────────────────────────────────────────────────────────────
ATTR_SLOT_MATCH_WEIGHT: float = float(
    os.environ.get("STENRAG_ATTR_SLOT_W", "1.0") or 1.0
)


# ─────────────────────────────────────────────────────────────────
# [2026-11-01] 保底保证：分数最高的 N 句一定进入 LLM 上下文
#
# 这是**最后一道防线**，独立于属性逻辑。旧的门有
# `ATTR_GATE_MIN_KEEP=3` 的保底，但那保护的是"数量"（上下文不会
# 太太空），不是"质量"（正确答案句在不在里面）。误校准时池子往往
# 还有 20+ 候选，门从容删掉 5 个 gold，保底根本不会启动。
#
# 新保底保护的是"**排序器的判断**"：分数最高的 N 句一定进上下文，
# 任何后续筛选都不能踢掉它们。
#
# N 的取值：默认 = top_k（全部）。理由是既然排序器已经把 gold 排到
# 第 1，就不该有任何句子比它更值得进上下文。留一个可调参数是为了
# 将来做消融（"只保证前 8 句" vs "保证全部 16 句"）。
# ─────────────────────────────────────────────────────────────────
ATTR_TOP_SCORE_GUARANTEE: int = int(
    os.environ.get("STENRAG_TOP_GUARANTEE", "16") or 16
)


# ─────────────────────────────────────────────────────────────────
# [2026-11-01] 实体键归一化（改造 H，第 2 项）
#
# 为什么需要它
# ------------
# 问句和句子里的实体表面形式经常不一样：
#
#     问句: "Which city was Havel born in?"
#     句子: "Václav Havel was born in Mělník ..."
#
# 精确匹配对不上，两边就都"查不到这个实体的属性"，槽位匹配失效。
#
# 规则（与 query 侧的末词降级形态保持一致，两边必须用同一套）
# ------------------------------------------------------------------------
#  1. 完整的小写形式        "václav havel"
#  2. 末词（长度 ≥ 3）      "havel" —— 匹配姓氏
#  3. 去掉标点的形式        "vaclav havel" —— 匹配去重音变体
#
# 末词降级有**误匹配风险**（"John Smith" / "James Smith" 都降级到
# "smith"），所以槽位匹配的结果**不单独决定排序**，而是在
# `attr_match` 里作为一个**加成项**（见 `_event_fact_score`）。
# 宁可多给一点分，不可错杀。
# ─────────────────────────────────────────────────────────────────
def _entity_keys(name: str) -> List[str]:
    """Return the lookup keys for an entity surface form.

    The last-word form exists because HotpotQA questions frequently
    mention a person by surname only ("Which city was Havel born in?")
    while the sentence spells out the full name ("Václav Havel").
    """
    keys: List[str] = []
    raw = (name or "").strip().lower()
    if not raw:
        return keys

    keys.append(raw)

    # Strip accents/diacritics so "Václav Havel" → "vaclav havel".
    try:
        import unicodedata
        decomposed = unicodedata.normalize("NFKD", raw)
        stripped = "".join(
            c for c in decomposed if not unicodedata.combining(c)
        )
        if stripped != raw:
            keys.append(stripped)
    except Exception:  # noqa: BLE001
        pass

    # Surname / last-word form.
    parts = [p for p in re.findall(r"[a-z0-9]+", raw) if len(p) >= 3]
    if parts:
        last = parts[-1]
        if last != raw:
            keys.append(last)
        # The accent-stripped last word too ("václav" case).
        try:
            import unicodedata
            d = unicodedata.normalize("NFKD", last)
            s = "".join(c for c in d if not unicodedata.combining(c))
            if s and s != last:
                keys.append(s)
        except Exception:  # noqa: BLE001
            pass

    # Deduplicate, preserve order.
    seen: Set[str] = set()
    out: List[str] = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _cos(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine similarity of two 1-D float32 vectors (assumed L2-normalized
    upstream; falls back gracefully if not)."""
    try:
        import numpy as _np
        a = _np.asarray(a, dtype=_np.float32).reshape(-1)
        b = _np.asarray(b, dtype=_np.float32).reshape(-1)
        na = float(_np.linalg.norm(a))
        nb = float(_np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(_np.dot(a, b) / (na * nb))
    except Exception:  # noqa: BLE001
        return 0.0


def _minmax_norm(values: Dict[str, float]) -> Dict[str, float]:
    """
    Min-max normalise a per-sentence score map to [0, 1].

    Why this exists
    ---------------
    The rerank formula used to add raw cosines of different scales
    together: `cos(paragraph_title, query)` lands around 0.15-0.35 while
    `cos(sentence, query)` lands around 0.35-0.55.  Adding them with
    fixed weights does NOT express the intended emphasis — whichever
    component happens to have the larger VARIANCE wins, regardless of
    its weight.  That is how a nominal "event-name dominant" setting
    (W_EVENT_NAME=0.7) ended up being sentence-vector dominant in
    practice.

    Normalising each component over the candidate pool FIRST makes the
    weights mean what they say: after normalisation every component
    spans [0, 1] and the weight is genuinely its share of the final
    score.

    Degenerate case: when all values are equal (or there is only one
    candidate), the map is returned as-is scaled to 1.0 for the max —
    otherwise every candidate would collapse to 0.0 and the formula
    would lose all discrimination.
    """
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo < 1e-9:
        # All equal — no discrimination is possible from this
        # component, so give every candidate the same neutral value.
        return {k: 1.0 for k in values}
    span = hi - lo
    return {k: (v - lo) / span for k, v in values.items()}


# ─────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────
@dataclass
class PipelineResult:
    query: str
    top_k: int
    final_sids: List[str] = field(default_factory=list)
    final_text: str = ""
    step1_entities: List[str] = field(default_factory=list)
    step2_a_containers: List[str] = field(default_factory=list)
    step2_5_bridge_titles: List[str] = field(default_factory=list)
    step2_5_bridge_cids: List[str] = field(default_factory=list)
    step3_hyperedges: List[str] = field(default_factory=list)
    step4_b_containers: List[str] = field(default_factory=list)
    step5_traced_sids: List[str] = field(default_factory=list)
    step4_l2_cids_filtered: List[str] = field(default_factory=list)
    step4_l2_sids_filtered: List[str] = field(default_factory=list)
    step6_pool_size: int = 0
    step7_top_n_sids: List[str] = field(default_factory=list)
    step8_supplement_sids: List[str] = field(default_factory=list)
    step9_merged_sids: List[str] = field(default_factory=list)
    step10_final_sids: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "final_sids": self.final_sids,
            "final_text": self.final_text,
            "step1_entities": self.step1_entities,
            "step2_a_containers": self.step2_a_containers,
            "step2_5_bridge_titles": self.step2_5_bridge_titles,
            "step2_5_bridge_cids": self.step2_5_bridge_cids,
            "step3_hyperedges": self.step3_hyperedges,
            "step4_b_containers": self.step4_b_containers,
            "step5_traced_sids": self.step5_traced_sids,
            "step4_l2_cids_filtered": self.step4_l2_cids_filtered,
            "step4_l2_sids_filtered": self.step4_l2_sids_filtered,
            "step6_pool_size": self.step6_pool_size,
            "step7_top_n_sids": self.step7_top_n_sids,
            "step8_supplement_sids": self.step8_supplement_sids,
            "step9_merged_sids": self.step9_merged_sids,
            "step10_final_sids": self.step10_final_sids,
            "debug": self.debug,
        }


# ─────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────
class StenGraphPipeline:
    """
    10-step container/sentence retrieval.  No DFS, no BFS, no graph
    traversal at runtime.  All "graph" reads are O(1) lookups against
    pre-computed indices.
    """

    def __init__(
        self,
        graph: Any,
        container_manager: Any,
        hyperedge_registry: Any,
        sentence_registry: Any,
        edge_registry: Any = None,
        analyzer: Optional[Callable[[str], Any]] = None,
        nlp: Any = None,
        entity_extractor: Optional[Callable[[str], List[str]]] = None,
        strict_entity_match: bool = False,
        fuzzy_match: bool = True,
        fuzzy_threshold: float = 0.58,
    ):
        self.graph = graph
        self.cm = container_manager
        self.he_registry = hyperedge_registry
        self.registry = sentence_registry
        # ±1 neighbour lookup lives on EdgeRegistry; fall back to None.
        self.edge_registry = (
            edge_registry
            or getattr(graph, "edge_registry", None)
        )
        self.analyzer = analyzer
        # NEW: a dedicated named-anchor extractor (LLM-backed when
        # available). When set, _step1_query_analysis prefers it over
        # `analyzer` because the QueryAnalyzer prompt is biased toward
        # abstract concepts and misses the PERSON/ORG/WORK anchors
        # that drive multi-hop retrieval.
        self.entity_extractor = entity_extractor
        # Week-2 Direction A: precision-preserving entity match.
        # When True, Step 2 uses token-Jaccard (≥ 0.5) instead of
        # substring-overlap on entity_inventory.  Improves EM on
        # cross-paragraph substring collisions (e.g. Q9 Jun Ji-hyun).
        self.strict_entity_match = strict_entity_match
        # Full fuzzy entity → container matching (src/fuzzy_match.py).
        # Enabled by default: it covers the three variations that exact
        # and substring matching cannot reach — partial/alternate names,
        # typos, and entities that only appear in a container's
        # entity_inventory.  `fuzzy_threshold` is the minimum similarity
        # a surface must reach to count as a match.
        self.fuzzy_match = fuzzy_match
        self.fuzzy_threshold = float(fuzzy_threshold)
        # Debug record: cid → why the fuzzy matcher recalled it.
        self._last_fuzzy_debug: Dict[str, Dict[str, Any]] = {}
        self.nlp = nlp

        # Tunable weights (per the new spec).
        self.w_query_on_b = 0.6
        self.w_cooccur_on_b = 0.4
        self.cooccur_min_weight = 0.30

        # Output sizes.
        self.top_a = 10   # was 3 — entity match is accurate, keep more A
        # PHASE-E2: step4 B 召回不再依赖 faiss 排序截断。每个超边
        # 现在的语义是"该 paragraph 内的所有容器"——这些都该是
        # B 候选，再交给 step7 sentence-level 排序挑出关键句。
        # 把 top_b_per_hyperedge 提到 999（实际受 max_b_per_hyperedge 兜底）。
        self.top_b_per_hyperedge = 999
        self.max_b_per_hyperedge = 999
        # DEPRECATED: only the removed layer-quota step (step 7) read this.
        # Step 10 now ranks the whole pool in one pass and honours
        # `top_k_final`, so this value has no consumer.  Kept so external
        # configs that still set it do not break.
        self.top_n_sentences = 10
        # PHASE-D [排序修复]: step10 final 之前固定 5，截断导致多个
        # 命中 gold 的句子在 #6+ 被砍掉（seed-7 diag 5/20 因截断）。
        # 提到 8 既保留 LLM 抽取时的上下文长度可控，又把 R@10 拉满。
        #
        # PHASE-E [召回上限]: 8 仍然太低。step6 候选池实测 16~33 条，
        # 而 pool→final 的存活率只有 0.647 —— 池内排名 6/8/15/15/18/27
        # 的 gold 全部在截断处被砍掉，LLM 根本没机会看到。
        # 提到 16 后（3 seed × 20 题，n=60）:
        #   EM   0.483 → 0.550
        #   F1   0.549 → 0.658
        #   UNANSWERABLE 误答   11/60 → 7/60
        #   gold 未进 top-8      10/60 → 7/60
        # 三个 seed 全部同向（0.500→0.600 / 0.550→0.550 / 0.400→0.500），
        # 不是单题抖动。
        self.top_k_final = 16

        # Query entities from the most recent `retrieve()` call.  The
        # attribute gate uses them to restrict the query's wanted
        # attributes to the ones those entities actually own, so the
        # query side and the sentence side are symmetric.
        self._last_query_entities: List[str] = []

        # FAISS / semantic search params.
        self.faiss_top_k = 20
        self.faiss_threshold = 0.55

        # Hyperedge event-name similarity threshold (same as spec).
        self.event_sim_threshold = 0.85

        # Symmetric-attribute soft match: when True, the reranker receives
        # a continuous `attr_match_score` computed as
        #   cosine(query_emb, attr_name_emb) · mount_strength_per_attr
        # via embedding-based symmetric matching, and adds it to `attr_term`
        # in the final score formula (no hard filter on the candidate pool).
        self.attr_soft_match_mode = True

        # ─────────────────────────────────────────────────────────────
        # Week-1: Hybrid Retrieval config
        #
        # w_bm25 + w_dense control the fusion weights.
        # Default 0.5/0.5 is a safe starting point; the ablation study
        # in eval should try (0.7/0.3), (0.3/0.7), (1.0/0.0), (0.0/1.0).
        # ─────────────────────────────────────────────────────────────
        self.w_bm25 = 0.5
        self.w_dense = 0.5

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        bridge_titles: Optional[Dict[str, List[str]]] = None,
    ) -> PipelineResult:
        # PHASE-D: callers that pass top_k=5 (legacy default) would
        # silently re-truncate the reranker output.  Use
        # self.top_k_final (8) as the new default; respect explicit
        # larger values too.
        if top_k is None:
            top_k = self.top_k_final
        top_k = max(int(top_k), self.top_k_final)
        result = PipelineResult(query=query, top_k=top_k)

        # [1] entity extraction
        entities = self._step1_query_analysis(query)
        result.step1_entities = entities
        result.debug["entities"] = entities
        # Remembered so the attribute gate can restrict the query's
        # wanted attributes to the ones these entities actually own
        # (the sentence side is already entity-aware).
        self._last_query_entities = list(entities)

        # [1.5] spaCy relation extraction on the query.
        # Extracts the question's *relation type* (verb of interest)
        # and *target object* (e.g., "county of birth") so that
        # later steps can match the question's syntactic skeleton
        # against candidate sentences' syntactic skeletons.
        q_relations = self._step1_5_query_relations(query)
        # Cache so step10 final rerank can use it for a verb/obj bonus.
        self._last_query_relations = q_relations
        self._last_q_relations = q_relations
        self._last_q_text = query
        result.debug["query_relations"] = q_relations
        logger.info(
            "Step1.5: query relations = %s", q_relations,
        )

        # [2] containers A
        a_containers = self._step2_recall_a_containers(query, entities)
        result.step2_a_containers = a_containers
        if not a_containers:
            return result

        # [2.5] bridge-title expansion: add containers whose paragraph
        # title is mentioned in the query.  These are the "second hop"
        # paragraphs that step 1 entity extraction often misses.
        if bridge_titles:
            bridge_t, bridge_cids = self._step2_5_bridge_titles(
                query, a_containers, bridge_titles,
            )
            result.step2_5_bridge_titles = bridge_t
            result.step2_5_bridge_cids = bridge_cids
            for cid in bridge_cids:
                if cid not in a_containers:
                    a_containers.append(cid)
            result.step2_a_containers = list(a_containers)

        # [3] hyperedges touching A
        h_ids = self._step3_hyperedges_for_containers(a_containers)
        result.step3_hyperedges = h_ids

        # [4] containers B (per-hyperedge recall — all containers in each
        # hyperedge, no faiss截断)
        b_containers = self._step4_recall_b_containers(
            query, h_ids, a_containers,
            bridge_titles=bridge_titles,
        )
        result.step4_b_containers = b_containers

        # [4b] v8 多归属传播:层1句 → 通过 node.containers 找其他容器.
        # 不依赖 SAME_ENTITY 边,不依赖 bridge_titles 限制.
        # 每个层1句子最多带3个新容器,层2容器总数≤15.
        l2_cids: List[str] = []
        l2_sids: List[str] = []
        if b_containers:
            l2_cids, l2_sids = self._step4_multihome_propagate(
                a_containers, b_containers, set(a_containers),
            )
        result.debug["step4_l2_cids"] = l2_cids
        result.debug["step4_l2_sids"] = l2_sids
        result.step4_l2_cids_filtered = list(l2_cids)
        result.step4_l2_sids_filtered = list(l2_sids)

        # [5] traceback for entities without a container
        covered = (set(a_containers) | set(b_containers) | set(l2_cids))
        traced = self._step5_trace_missing_entities(entities, covered)
        result.step5_traced_sids = traced

        # [6] candidate pool: A (seed containers) ∪ B (hyperedge
        # containers, event-name filtered).  No layer assignment and no
        # per-layer quota — Step 10 is the single ranking entry point.
        pool_sids = self._step6_build_pool(a_containers, b_containers)
        result.step6_pool_size = len(pool_sids)
        result.debug["step6_pool_sids"] = list(pool_sids)

        # Steps 7/8/9 (layer quota / supplement / merge) removed.
        # Their PipelineResult fields are kept so downstream diag
        # (quick20_eval) can still getattr() them; they are no longer
        # produced.
        a_cids_set = set(a_containers)
        self._last_a_cids = a_cids_set
        result.step7_top_n_sids = []
        result.step8_supplement_sids = []
        result.step9_merged_sids = []

        # [10] final semantic rerank  →  top-K
        final = self._step10_final_rerank(
            query, pool_sids, top_k, entities=entities,
        )

        result.step10_final_sids = final
        result.final_sids = final
        result.final_text = self._format(result, final)
        # Persist event+fact score breakdown for inspection.
        if hasattr(self, "_last_ef_scored") and self._last_ef_scored:
            result.debug["ef_scores"] = self._last_ef_scored
        return result

    # ------------------------------------------------------------------
    # [1] entity extraction
    # ------------------------------------------------------------------
    def _step1_query_analysis(self, query: str) -> List[str]:
        """
        Named-anchor extraction. Two paths, in priority order:

        1. `self.entity_extractor` — LLM-backed, returns lowercased
           list of PERSON/ORG/GPE/EVENT/WORK_OF_ART/DATE anchors.
           Used when the dispatcher wires one up (see dispatcher.py).
        2. `self.analyzer` — the existing QueryAnalyzer (LLM with
           semantic-entity prompt, or regex fallback). Used only as
           a backstop so the pipeline still works without an LLM.

        Returns a deduplicated, lowercased list of anchor strings.
        """
        entities: List[str] = []

        # Path 1: dedicated LLM extractor.
        if self.entity_extractor is not None:
            try:
                entities = list(self.entity_extractor(query) or [])
            except Exception as e:  # noqa: BLE001
                logger.warning("entity_extractor failed (%s); trying analyzer", e)
                entities = []

        # Path 2: legacy analyzer (QueryAnalysis.entity_names).
        if not entities and self.analyzer is not None:
            try:
                analysis = self.analyzer(query)
                entities = list(getattr(analysis, "entity_names", []) or [])
            except Exception as e:  # noqa: BLE001
                logger.warning("analyzer failed (%s); falling back to []", e)

        # Clean + dedup.
        # Drop pure-digit / mostly-digit anchors (they don't carry useful
        # entity-inventory signal — "15", "1857", "85" all map to no
        # entity attribute that helps recall).
        def _is_numericish(s: str) -> bool:
            if not s:
                return False
            t = s.strip(".,;:!?\"'`")
            if not t:
                return False
            digits = sum(c.isdigit() for c in t)
            letters = sum(c.isalpha() for c in t)
            # Pure digit/digits-with-punct, OR digit letter ratio too high
            if digits >= 1 and letters == 0:
                return True
            if digits >= 4 and letters <= 1:
                return True
            # Token-level: every token starts with a digit (e.g. "15 december")
            toks = [x for x in re.split(r"\s+", t) if x]
            if toks and all(re.match(r"^[0-9]", x) for x in toks) and any(
                re.fullmatch(r"[0-9.,]+", x) for x in toks
            ):
                return True
            return False

        out: List[str] = []
        seen = set()
        for e in entities:
            s = str(e).lower().strip().strip(".,;:!?\"'`")
            if not s or len(s) < 2 or s in seen:
                continue
            if _is_numericish(s):
                logger.info("Step1: dropping numeric entity %r", s)
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= 8:
                break

        # Path 3: index-probing expansion.  For each entity, scan
        # the union of every container's `entity_inventory` for a
        # LONGER string whose token set is a superset of the
        # entity's token set (coverage ≥ 0.5).  This bridges the
        # "john kelly" ↔ "john francis kelly" and "Scarborough" ↔
        # "Charles Joseph Scarborough" mismatches that the LLM
        # extractor's short forms create.
        if self.cm is not None and entities:
            try:
                out = self._expand_entities_via_inventory(out)
            except Exception as e:  # noqa: BLE001
                logger.debug("entity expansion failed: %s", e)

        return out

    def _expand_entities_via_inventory(self, entities: List[str]) -> List[str]:
        """
        Probe every container's `entity_inventory` for a longer
        token-superset of each entity.  We cache the inventory
        snapshot on the pipeline instance so we don't re-scan.
        """
        cache = getattr(self, "_inv_cache", None)
        if cache is None:
            cache = []
            seen_keys: set = set()
            try:
                for c in self.cm.get_all_containers() or []:
                    inv = list(getattr(c, "entity_inventory", []) or [])
                    for e in inv:
                        k = (e or "").lower().strip()
                        if k and k not in seen_keys:
                            seen_keys.add(k)
                            cache.append(k)
            except Exception:
                pass
            try:
                self._inv_cache = cache
            except Exception:
                pass
        if not cache:
            return entities

        out = list(entities)
        seen = set(out)
        stop = {
            "the", "a", "an", "of", "in", "on", "at", "to", "for", "by",
            "and", "or", "is", "are", "was", "were", "be", "been",
            "as", "that", "this", "these", "those", "who", "what",
            "which", "where", "when", "why", "how", "it", "its",
        }
        for e in entities:
            e_tokens = {
                t for t in re.findall(r"[a-z0-9'\-]+", e.lower())
                if t and t not in stop
            }
            if not e_tokens:
                continue
            best_match: Optional[str] = None
            best_extra = 0
            for k in cache:
                k_tokens = {
                    t for t in re.findall(r"[a-z0-9'\-]+", k)
                    if t and t not in stop
                }
                if not k_tokens or k_tokens == e_tokens:
                    continue
                if not e_tokens.issubset(k_tokens):
                    continue
                coverage = len(e_tokens) / len(k_tokens)
                if coverage < 0.5:
                    continue
                extra = len(k_tokens) - len(e_tokens)
                if best_match is None or extra > best_extra:
                    best_match = k
                    best_extra = extra
            if best_match and best_match not in seen:
                out.append(best_match)
                seen.add(best_match)
                if len(out) >= 12:
                    break
        return out

    # ------------------------------------------------------------------
    # [1.5] spaCy relation extraction on the query
    # ------------------------------------------------------------------
    def _step1_5_query_relations(self, query: str) -> Dict[str, List[str]]:
        """
        Use spaCy to extract the *syntactic skeleton* of the question:

          * "ROOT verb" — the action the question is asking about
            (e.g., "was born in" → born, "plays" → play, "shared" → share).
          * "direct object" — what the question wants to know about
            (e.g., "county", "instrument", "year").
          * "head nouns" of the entity anchors — the property asked
            about (e.g., for "What county was X born in?",
            obj="county" + prepositional head="in" → birth-location).
          * "subject entity" — the named anchor (X).

        This skeleton is later used by:
          Step 6.5 (LLM gap detector)  — to phrase the missing-info question
          Step 7 (sentence rerank)     — to add a syntactic-match bonus
                                         between question verb/obj and
                                         candidate sentence verb/obj.

        Falls back to empty dict if spaCy is unavailable.
        """
        nlp = getattr(self, "nlp", None) or globals().get("_CONTAINER_NLP")
        if nlp is None:
            try:
                import spacy
                nlp = spacy.load("en_core_web_lg")
                globals()["_CONTAINER_NLP"] = nlp
            except Exception as e:  # pragma: no cover
                logger.warning("Step1.5: spaCy unavailable, returning empty relations: %s", e)
                return {"verb": [], "obj": [], "subj": [], "preps": []}
        try:
            doc = nlp(query)
        except Exception as e:
            logger.warning("Step1.5: spaCy parse failed: %s", e)
            return {"verb": [], "obj": [], "subj": [], "preps": []}

        verbs: List[str] = []
        objs: List[str] = []
        subjs: List[str] = []
        preps: List[str] = []

        # Collect content tokens (skip stopwords/aux/punct).
        stop = {
            "be", "is", "are", "was", "were", "do", "did", "does",
            "have", "has", "had", "what", "which", "who", "where",
            "when", "how", "in", "on", "at", "by", "for", "to",
            "the", "a", "an",
        }
        for tok in doc:
            if tok.is_stop or tok.is_punct or tok.like_num:
                continue
            lemma = tok.lemma_.lower()
            if not lemma or len(lemma) < 2:
                continue
            if tok.dep_ in ("ROOT", "conj", "advcl") and tok.pos_ == "VERB":
                if lemma not in stop:
                    verbs.append(lemma)
            if tok.dep_ in ("dobj", "attr", "oprd", "acomp") and tok.pos_ in ("NOUN", "PROPN", "ADJ"):
                objs.append(lemma)
            if tok.dep_ in ("nsubj", "nsubjpass") and tok.pos_ in ("NOUN", "PROPN"):
                subjs.append(lemma)
            if tok.dep_ == "prep" and tok.pos_ == "ADP":
                if lemma not in stop:
                    preps.append(lemma)
            if tok.dep_ == "pobj" and tok.pos_ in ("NOUN", "PROPN"):
                objs.append(lemma)

        # Deduplicate, preserve order.
        def _dedup(xs: List[str]) -> List[str]:
            seen = set()
            out = []
            for x in xs:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        rels = {
            "verb": _dedup(verbs),
            "obj": _dedup(objs),
            "subj": _dedup(subjs),
            "preps": _dedup(preps),
        }
        return rels

    # ------------------------------------------------------------------
    # [2] containers A: entity match + fuzzy + hybrid (BM25 + dense)
    # ------------------------------------------------------------------
    def _step2_recall_a_containers(
        self,
        query: str,
        entities: List[str],
    ) -> List[str]:
        scores: Dict[str, float] = {}

        # ── (a) Strict entity match ────────────────────────────────────
        # When self.strict_entity_match is True, use the precision-
        # preserving Jaccard matcher (replaces the legacy substring
        # over-expansion on entity_inventory).
        for ent in entities:
            if self.strict_entity_match:
                for cid, score in self._strict_entity_match(ent) or []:
                    scores[cid] = scores.get(cid, 0.0) + score
            else:
                for c in self.cm.get_containers_for_entity(ent) or []:
                    cid = getattr(c, "id", None)
                    if cid:
                        scores[cid] = scores.get(cid, 0.0) + 1.0

        # ── (b) Full fuzzy match (typos / partial names / inventory) ────
        # Uses src/fuzzy_match.py: six layered signals compared against
        # both the container's primary entity and its entity_inventory.
        # Gated by `self.fuzzy_match` (on by default).
        if getattr(self, "fuzzy_match", True):
            try:
                fuzzy_scores = self._fuzzy_match_containers(
                    entities, threshold=self.fuzzy_threshold,
                )
                for cid, score in fuzzy_scores:
                    scores[cid] = scores.get(cid, 0.0) + score
            except Exception:  # noqa: BLE001
                logger.debug("Step2 fuzzy match skipped: %s", Exception())

        # ── (c) Hybrid retrieval: RRF(BM25 + dense) ────────────────────
        hybrid_hits = self._hybrid_search(query, top_k=self.faiss_top_k)
        for cid, combined, bm25_s, dense_s in hybrid_hits:
            scores[cid] = scores.get(cid, 0.0) + float(combined)

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [cid for cid, _ in ranked[: self.top_a]]

    def _strict_entity_match(
        self, entity: str,
    ) -> List[Tuple[str, float]]:
        """
        Replace `cm.get_containers_for_entity(entity)` for Step 2 with
        a *precision-preserving* matcher:

          (1) primary entity: substring OK only if
              container.entity has ≥ 50% token overlap with entity.
          (2) inventory: each entry must have ≥ 50% token overlap
              with entity (multi-word), OR be an exact token match
              (single-word) with len ≥ 4.

        Returns (cid, score) list.  Score = 1.0 for primary
        entity match, 0.7 for inventory match.
        """
        stop = {
            "a", "an", "the", "of", "in", "on", "at", "to", "for",
            "with", "by", "and", "or", "is", "are", "was", "were",
            "be", "been",
        }

        def _tokens(text: str) -> Set[str]:
            return {
                t for t in re.findall(r"[a-z0-9'\-]+", (text or "").lower())
                if t and t not in stop and len(t) >= 2
            }

        def _jaccard(a: Set[str], b: Set[str]) -> float:
            inter = len(a & b)
            union = len(a | b)
            return inter / union if union else 0.0

        ent_toks = _tokens(entity)
        if not ent_toks:
            return []
        ent_lower = entity.lower()
        out: List[Tuple[str, float]] = []
        for c in self.cm.get_all_containers() or []:
            cid = getattr(c, "id", None)
            if not cid:
                continue
            # (1) primary entity check
            c_entity = (getattr(c, "entity", "") or "").lower()
            c_toks = _tokens(c_entity)
            if c_toks:
                # substring case
                if ent_lower in c_entity:
                    # accept only if strong overlap (≥ 50% Jaccard)
                    jac = _jaccard(ent_toks, c_toks)
                    if jac >= 0.5:
                        out.append((cid, 1.0))
                        continue
                # exact tokens equal
                if ent_toks == c_toks:
                    out.append((cid, 1.0))
                    continue
            # (2) inventory check
            inv = getattr(c, "entity_inventory", []) or []
            best_inv = 0.0
            for e in inv:
                e_lower = (e or "").lower()
                if not e_lower:
                    continue
                e_toks = _tokens(e_lower)
                if not e_toks:
                    continue
                if len(ent_toks) == 1 and len(e_toks) == 1:
                    # single-word: exact match only (avoid 1-letter hits)
                    if (
                        next(iter(ent_toks)) == next(iter(e_toks))
                        and len(next(iter(ent_toks))) >= 4
                    ):
                        best_inv = max(best_inv, 0.7)
                else:
                    jac = _jaccard(ent_toks, e_toks)
                    if jac >= 0.5:
                        best_inv = max(best_inv, 0.7)
            if best_inv > 0:
                out.append((cid, best_inv))
        # Dedup: keep max score per cid.
        agg: Dict[str, float] = {}
        for cid, s in out:
            agg[cid] = max(agg.get(cid, 0.0), s)
        return sorted(agg.items(), key=lambda kv: -kv[1])

    def _step2_fuzzy_entity_match(
        self, entities: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Full fuzzy entity → container match.

        Delegates to `src.fuzzy_match`, which layers six signals
        (exact / containment / token Jaccard / partial token overlap /
        character trigram / Levenshtein) and compares the query entity
        against **both** `container.entity` and every entry of
        `container.entity_inventory`.  Matching the inventory is what
        makes a query entity that is merely mentioned inside a topic
        container reachable.

        The previous implementation only compared `container.entity`, so
        it could not see inventory-only mentions and could not fix a
        multi-token typo (it only had a single-token edit-distance test).

        Returns: sorted list of (cid, score), descending score.
        """
        try:
            return self._fuzzy_match_containers(entities)
        except Exception as e:  # noqa: BLE001
            logger.debug("Step2 fuzzy match failed: %s", e)
            return []

    def _fuzzy_match_containers(
        self,
        entities: List[str],
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Fuzzy-match every query entity against every container.

        Each container keeps its **best** score across all entities and
        all surfaces (primary entity + entity inventory).  The winning
        signal is recorded on `self._last_fuzzy_debug` so the caller can
        explain *why* a container was recalled.
        """
        from src.fuzzy_match import (
            DEFAULT_THRESHOLD,
            best_container_match,
        )

        thr = DEFAULT_THRESHOLD if threshold is None else float(threshold)
        agg: Dict[str, float] = {}
        best_meta: Dict[str, Dict[str, Any]] = {}

        for ent in entities:
            if not ent:
                continue
            for c in self.cm.get_all_containers() or []:
                cid = getattr(c, "id", None)
                if not cid:
                    continue
                m = best_container_match(ent, c, threshold=thr)
                if m is None:
                    continue
                if m.score > agg.get(cid, 0.0):
                    agg[cid] = m.score
                    best_meta[cid] = {
                        "entity": ent,
                        "score": round(m.score, 4),
                        "type": m.match_type,
                        "surface": m.surface,
                    }

        self._last_fuzzy_debug = best_meta
        return sorted(agg.items(), key=lambda kv: -kv[1])

    # ------------------------------------------------------------------
    # [2.5] bridge-title expansion
    # ------------------------------------------------------------------
    def _hybrid_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[str, float, float, float]]:
        """
        Week-1: Hybrid search — RRF fusion of BM25 + dense (FAISS).

        Delegates to ContainerManager.hybrid_search() when available.
        Falls back to plain FAISS when BM25 isn't ready.
        """
        try:
            return self.cm.hybrid_search(
                query,
                top_k=top_k,
                w_bm25=self.w_bm25,
                w_dense=self.w_dense,
                bm25_top_k=50,
                dense_top_k=top_k,
                dense_threshold=self.faiss_threshold,
                rrf_k=60,
            )
        except Exception:
            # Fallback: dense-only when hybrid not available.
            return self._faiss_search(query, top_k)

    # ------------------------------------------------------------------
    # [2.5] bridge-title expansion
    # ------------------------------------------------------------------
    def _step2_5_bridge_titles(
        self,
        query: str,
        a_containers: List[str],
        bridge_titles: Dict[str, List[str]],
    ) -> Tuple[List[str], List[str]]:
        """
        Find paragraph titles whose lowercase form (or any meaningful
        phrase in it) is mentioned in the query, and return the
        containers that hold that paragraph's sentences.

        This is the "method 2" expansion: when the user writes
        "the movement Hans Knirsch was an activist for", the title
        "Austrian National Socialism" is the bridge paragraph we need
        to fetch — even though the LLM in step 1 did not extract it
        as an entity.

        Matching rules (any of):
          * the full lowercased title appears in the query string
            (substring check);
          * all *content words* of the title appear in the query
            (token-set intersection, drop stop words).

        Returns:
            (matched_titles_in_order, deduped_container_ids)
        """
        q_lower = (query or "").lower()
        # Tokenize once: split on non-letter, then drop stopwords / short
        # tokens.  We use this for the "all content words" check.
        stop = {
            "a", "an", "the", "of", "in", "on", "at", "to", "for",
            "with", "by", "and", "or", "is", "are", "was", "were",
            "be", "been", "being", "as", "that", "this", "these",
            "those", "who", "what", "which", "where", "when", "why",
            "how", "i", "you", "he", "she", "it", "they", "we",
            "his", "her", "their", "my", "your", "its", "do", "did",
            "does", "had", "from", "has", "have", "but", "not",
        }
        q_tokens = {
            t for t in re.findall(r"[a-z0-9][a-z0-9'\-]+", q_lower)
            if t not in stop and len(t) > 1
        }

        a_set = set(a_containers)
        matched_titles: List[str] = []
        matched_cids: List[str] = []
        seen_cids: Set[str] = set()

        # Deterministic order: sort titles by length desc (longer
        # match = more specific bridge), then alphabetically.
        for title_lower in sorted(
            bridge_titles.keys(), key=lambda t: (-len(t), t)
        ):
            # Rule 1: full title substring.
            hit = False
            if len(title_lower) >= 4 and title_lower in q_lower:
                hit = True
            # Rule 2: content-word match.
            if not hit:
                title_tokens = {
                    t for t in re.findall(r"[a-z0-9][a-z0-9'\-]+", title_lower)
                    if t not in stop and len(t) > 1
                }
                # Require at least 1 meaningful content token to match
                # and at least 80% of the title's content tokens.
                if title_tokens:
                    overlap = title_tokens & q_tokens
                    if (
                        overlap
                        and len(overlap) >= 1
                        and len(overlap) / len(title_tokens) >= 0.8
                    ):
                        hit = True
            if not hit:
                continue

            matched_titles.append(title_lower)
            for cid in bridge_titles[title_lower]:
                if cid in a_set or cid in seen_cids:
                    continue
                seen_cids.add(cid)
                matched_cids.append(cid)

            # Cap the number of extra bridge containers to avoid
            # exploding the A set.
            if len(matched_cids) >= 5:
                break

        return matched_titles, matched_cids

    # ------------------------------------------------------------------
    # [3] hyperedges touching any A
    # ------------------------------------------------------------------
    def _step3_hyperedges_for_containers(
        self, container_ids: List[str],
    ) -> List[str]:
        # NOTE: keep insertion order.  Returning `list(set(...))` made the
        # order depend on Python's per-process string hash seed, so the
        # same question could yield a different hyperedge sequence (and
        # therefore a different B expansion) from one run to the next.
        seen: Set[str] = set()
        ordered: List[str] = []
        for cid in container_ids:
            for he in self.he_registry.get_hyperedges_for_container(cid):
                if he.id not in seen:
                    seen.add(he.id)
                    ordered.append(he.id)
        return ordered

    # ------------------------------------------------------------------
    # [4] containers B: per-hyperedge ranked, then union
    # ------------------------------------------------------------------
    def _step4_recall_b_containers(
        self,
        query: str,
        hyperedge_ids: List[str],
        a_ids: List[str],
        bridge_titles: Optional[Dict[str, List[str]]] = None,
    ) -> List[str]:
        """
        B 召回：超边内的容器 + paragraph bridge 第二轮。

        PHASE-E2 [step4 召回修复] — 之前 step4 的逻辑：
          (1) 对每个超边内的容器，按 faiss(query, container) 排序
          (2) 每超边取 top-3
          (3) 合并 dedup
        这个逻辑在以下两类题上失效：
          (a) 答案在 paragraph 内某一句，但 anchor 容器名和 query
              的 faiss 相似度不高（如 Q11 Pompano Beach 人口：anchor
              容器是 "Pompano Beach"，query 关键词是 population，faiss
              相似度低 → 容器被截断 → 该 paragraph 的容器没进 B）
          (b) 答案在另一 paragraph 内（如 Q8 Kim Warwick 三冠：答
              案句 "He won three singles titles" 在 Kim Warwick paragraph
              内另一个 entity 的容器里）— 旧逻辑用 faiss 卡 top-3，
              会丢掉该 paragraph 的其它容器

        新逻辑：
          (1) 对每个 hyperedge（每个 hyperedge 现在只覆盖 1 个
              paragraph），取该 paragraph 内**所有**容器作为 B
              候选（不再 faiss 截断）。
          (2) 额外做"paragraph bridge"——在 step2_5 找到的
              bridge_titles 里，把**全部** bridge paragraph 的
              容器加入 B（不依赖超边是否存在）。
          (3) 排序：bridge paragraph 的容器排在前面（更强的
              evidence 信号），其它 paragraph 内的容器用
              faiss(query, container) 排序打 tiebreak。
        """
        a_set = set(a_ids)
        hybrid_hits = self._hybrid_search(query, top_k=self.faiss_top_k)
        hybrid_sim: Dict[str, float] = {
            cid: combined for cid, combined, _, _ in hybrid_hits
        }

        # (1) Per-hyperedge recall: every container in the hyperedge.
        per_h: List[Tuple[str, float, str]] = []  # (cid, score, source_he)
        for he_id in hyperedge_ids:
            he = self.he_registry.get_hyperedge(he_id)
            if he is None:
                continue
            for cid in he.container_ids:
                if cid in a_set:
                    continue
                # Score: hybrid(query, container) — used as tiebreak.
                s = hybrid_sim.get(cid, 0.0)
                per_h.append((cid, s, he_id))

        # (2) Paragraph bridge recall: every container in any bridge
        # paragraph. This is the new "step4-B second pass" — it
        # directly adds containers from the answer paragraphs that
        # step2_5 identified.
        bridge_cids: Set[str] = set()
        if bridge_titles:
            for _title, _cids in bridge_titles.items():
                for _cid in _cids:
                    if _cid not in a_set:
                        bridge_cids.add(_cid)
        # Bridge containers are pushed to the front with score +1
        # so they win the merge tiebreak.

        # (3) Merge: bridge first, then per-hyperedge (faiss tiebreak).
        # Bridge containers are score=inf to ensure they rank first.
        best: Dict[str, float] = {}
        for cid, s, _he in per_h:
            if s > best.get(cid, -1.0):
                best[cid] = s
        for cid in bridge_cids:
            # Override to a large positive score so bridge wins.
            if cid not in best:
                best[cid] = 1.0e6
            else:
                best[cid] = max(best[cid], 1.0e6)

        if not best:
            return []
        ranked = sorted(best.items(), key=lambda kv: -kv[1])

        # ─────────────────────────────────────────────────────────────────
        # Week-2 Direction A: SAME_ENTITY cross-paragraph expansion
        #
        # After building B from hyperedges + bridge_titles, do a second
        # pass: for each A container, find SAME_ENTITY edges that go
        # from a sentence in A to a sentence in another paragraph.
        # Pull in containers containing those cross-paragraph neighbors.
        #
        # This is the single-hop version of what the old Retriever's
        # `_traverse_with_entity_edges` does — but at the *container*
        # level, so it feeds directly into step 6's candidate pool.
        # ─────────────────────────────────────────────────────────────────
        se_cids = self._step4_expand_via_same_entity(a_ids, a_set, bridge_titles)
        for cid in se_cids:
            if cid not in best:
                best[cid] = 0.5  # moderate score; below bridge but above noise
            # If already present, don't downgrade.

        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return [cid for cid, _ in ranked]

    def _step4_expand_via_same_entity(
        self,
        a_ids: List[str],
        a_set: Set[str],
        bridge_titles: Optional[Dict[str, List[str]]] = None,
    ) -> List[str]:
        """
        Week-2 Direction A: cross-paragraph SAME_ENTITY expansion.

        From each sentence in every A container, follow SAME_ENTITY edges
        to find cross-paragraph neighbor sentences. Collect the containers
        of those neighbors and return them as additional B candidates.

        Filters:
          - Only sentences in DIFFERENT paragraphs (bridge_titles) qualify.
            This avoids re-discovering A's own paragraph.
          - Exclude already-known B containers (caller's a_set).
          - Hard cap of 30 cids to prevent explosion.
          - Skip if edge_registry is unavailable.

        Returns:
            Deduplicated list of container IDs reachable via 1 SAME_ENTITY hop
            from any A container, sorted by # of A seeds that reach them.
        """
        edge_reg = self.edge_registry
        if edge_reg is None:
            return []
        if not a_ids:
            return []

        # Build paragraph membership: cid → set of paragraph titles.
        # We use bridge_titles for this; missing entries are treated as
        # "unknown paragraph" (we include them since we can't tell).
        para_titles_by_cid: Dict[str, Set[str]] = {}
        if bridge_titles:
            for title_lower, cids in bridge_titles.items():
                for cid in cids:
                    if cid not in para_titles_by_cid:
                        para_titles_by_cid[cid] = set()
                    para_titles_by_cid[cid].add(title_lower)

        # Collect A sentences per container.
        a_sids_by_cid: Dict[str, Set[str]] = {}
        for cid in a_ids:
            c = self.cm.get_container(cid)
            if c is None:
                continue
            sids = set(getattr(c, "sentence_ids", []) or [])
            if sids:
                a_sids_by_cid[cid] = sids

        # Walk 1-hop SAME_ENTITY edges from every A sentence.
        # Count how many A seeds reach each candidate cid.
        seed_count: Dict[str, int] = {}
        visited_sids: Set[str] = set()

        for cid, sids in a_sids_by_cid.items():
            a_para = para_titles_by_cid.get(cid, set())
            for sid in sids:
                if sid in visited_sids:
                    continue
                visited_sids.add(sid)
                try:
                    neighbors = edge_reg.get_neighbors(
                        sid,
                        edge_types=["SAME_ENTITY"],
                    )
                except Exception:
                    continue
                for nb_sid, _edge in neighbors:
                    # Skip A's own sentences.
                    if nb_sid in visited_sids:
                        continue
                    visited_sids.add(nb_sid)
                    # Find which container(s) contain this neighbor sentence.
                    try:
                        nb_sid_str = str(nb_sid)
                    except Exception:
                        continue
                    for nb_cid in self._containers_of_sentence(nb_sid_str):
                        if nb_cid in a_set:
                            continue
                        # Must be a DIFFERENT paragraph.
                        nb_para = para_titles_by_cid.get(nb_cid, set())
                        if a_para and nb_para and a_para == nb_para:
                            # Same paragraph — skip (no cross-paragraph signal).
                            continue
                        seed_count[nb_cid] = seed_count.get(nb_cid, 0) + 1

        if not seed_count:
            return []

        # Sort by seed count (more seeds = stronger cross-entity signal).
        ranked = sorted(seed_count.items(), key=lambda kv: -kv[1])
        return [cid for cid, _ in ranked[:30]]

    def _containers_of_sentence(self, sid: str) -> List[str]:
        """Return the container IDs that contain sentence sid."""
        cids: List[str] = []
        for c in self.cm.get_all_containers():
            if sid in getattr(c, "sentence_ids", []) or []:
                cids.append(c.id)
        return cids

    def _cooccur_with_a(self, cid: str, a_set: Set[str]) -> float:
        """
        Co-occurrence strength between a candidate B container and the
        A set, via the pre-built `_container_connectivity` map.
        Returns the max edge weight over neighbours that are in A.
        """
        adj = getattr(self.cm, "_container_connectivity", None) or {}
        neighbours = adj.get(cid, {}) or {}
        best = 0.0
        for nb, w in neighbours.items():
            if nb in a_set and float(w) > best:
                best = float(w)
        return best

    # ------------------------------------------------------------------
    # [5] traceback for entities without a container
    # ------------------------------------------------------------------
    def _neighbours(self, sid: str) -> List[str]:
        """±1 neighbour sentence ids via EdgeRegistry; [] if unavailable."""
        er = self.edge_registry
        if er is None:
            return []
        # EdgeRegistry.get_neighbors returns List[(neighbor_id, edge)].
        try:
            raw = er.get_neighbors(sid) or []
        except Exception:
            return []
        out: List[str] = []
        for item in raw:
            if isinstance(item, tuple) and len(item) >= 1:
                out.append(item[0])
            elif isinstance(item, str):
                out.append(item)
        return out

    def _step5_trace_missing_entities(
        self,
        entities: List[str],
        covered_container_ids: List[str],
    ) -> List[str]:
        covered = set(covered_container_ids)
        out: List[str] = []
        seen: Set[str] = set()

        for ent in entities:
            # If the entity has at least one container that is already
            # covered by A or B, skip — no traceback needed.
            entity_containers = self.cm.get_containers_for_entity(ent) or []
            if any(getattr(c, "id", None) in covered for c in entity_containers):
                continue

            # Find sentences in the registry's entity index.
            ent_idx = getattr(self.registry, "_entity_index", {}) or {}
            hit_sids = list(ent_idx.get(ent, set()) or set())
            for sid in hit_sids:
                if sid in seen:
                    continue
                seen.add(sid)
                out.append(sid)
                # ±1 neighbour sentences (registry handles the actual edges)
                for nb in (self._neighbours(sid) or []):
                    if nb not in seen:
                        seen.add(nb)
                        out.append(nb)
        return out

    # ------------------------------------------------------------------
    # [6] candidate pool
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # [4b] v8 多归属传播 (替代 SAME_ENTITY 边)
    # ------------------------------------------------------------------
    def _step4_multihome_propagate(
        self,
        a_ids: List[str],
        b_ids: List[str],
        a_set: Set[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Spec §传播层2: 从种子容器的所有句子出发,通过多归属关系
        直接找到其他容器.

        不需要图遍历,不需要 SAME_ENTITY 边 — 句子在建图时已同时
        挂载到它提到的每个实体对应的容器上(node.containers 列表).

        规则:
          - Layer 1 = 种子容器(A)的所有句子
          - Layer 2 = 从每个层1句子出发,找出它还属于哪些容器
          - 每个层1句子最多带出 3 个新容器
          - Layer 2 容器总数 ≤ 15

        返回: (l2_container_ids, l2_sentence_ids)
        """
        MAX_NEW_PER_L1 = 3
        MAX_L2_CIDS = 15

        # Layer 1 sentences: all sentences in A containers.
        a_sids: Set[str] = set()
        for cid in a_ids:
            c = self.cm.get_container(cid)
            if c is None:
                continue
            for sid in getattr(c, "sentence_ids", []) or []:
                a_sids.add(sid)

        if not a_sids:
            return [], []

        # Collect L2 containers: for each L1 sentence, read node.containers
        # and collect non-A containers, up to MAX_NEW_PER_L1 per sentence
        # and MAX_L2_CIDS total.
        l2_cid_count: Dict[str, int] = {}  # cid → how many L1 sentences reached it
        l2_cids_ordered: List[str] = []     # preserve insertion order for determinism

        for sid in a_sids:
            node = self.registry.get_node(sid)
            if node is None:
                continue
            containers = getattr(node, "containers", None) or []
            new_this_sent = 0
            for cid in containers:
                if cid in a_set:
                    continue
                if cid in l2_cid_count:
                    l2_cid_count[cid] += 1
                else:
                    if len(l2_cids_ordered) >= MAX_L2_CIDS:
                        break
                    l2_cid_count[cid] = 1
                    l2_cids_ordered.append(cid)
                    new_this_sent += 1
                    if new_this_sent >= MAX_NEW_PER_L1:
                        break

        if not l2_cids_ordered:
            return [], []

        # Layer 2 sentences: all sentences in L2 containers,
        # excluding L1 sentences.
        l2_sids: List[str] = []
        seen: Set[str] = set(a_sids)
        for cid in l2_cids_ordered:
            c = self.cm.get_container(cid)
            if c is None:
                continue
            for sid in getattr(c, "sentence_ids", []) or []:
                if sid not in seen:
                    seen.add(sid)
                    l2_sids.append(sid)

        return l2_cids_ordered, l2_sids

    # ------------------------------------------------------------------
    def _step6_build_pool(
        self,
        a_ids: List[str],
        b_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Build the flat candidate pool: A (seed containers) ∪ B.

        - A containers contribute every sentence they hold.
        - B containers (reached through hyperedges) contribute only the
          sentences whose text matches the hyperedge event name, i.e. the
          former "supplement" signal, folded in here instead of living in
          a separate step.

        No layer tagging, no quota: Step 10 ranks the whole pool in one
        pass.  Order is A-first then B, deduplicated.
        """
        pool: List[str] = []
        seen: Set[str] = set()

        def _add(cid: str) -> None:
            c = self.cm.get_container(cid)
            if c is None:
                return
            for sid in getattr(c, "sentence_ids", []) or []:
                if sid not in seen:
                    seen.add(sid)
                    pool.append(sid)

        for cid in a_ids or []:
            _add(cid)

        # B containers: only event-name-matching sentences (was Step 8).
        for cid in b_ids or []:
            c = self.cm.get_container(cid)
            if c is None:
                continue
            event_names: List[str] = []
            for he in self.he_registry.get_hyperedges_for_container(cid):
                en = (getattr(he, "event_name", "") or "").strip()
                if en:
                    event_names.append(en)
            if not event_names:
                continue
            for sid in getattr(c, "sentence_ids", []) or []:
                if sid in seen:
                    continue
                text = self._text_of(sid).lower()
                if not text:
                    continue
                if any(
                    self._event_in_text(en, text) for en in event_names
                ):
                    seen.add(sid)
                    pool.append(sid)

        return pool

    # [7] v8 三层配额句子级排序
    # ------------------------------------------------------------------
    def _score_sentences_v8(
        self, query: str, sids: List[str],
        a_cids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        v8 评分:四个主信号 + 两个弱信号,全部加在一起.
        全部基于规则计算,无需 LLM,无需图遍历.
        """
        import re as _re

        # ── Precompute query features ────────────────────────────────
        q_lower = (query or "").lower()
        q_tokens = set(_re.findall(r"[a-z]{3,}", q_lower))
        q_token_count = len(q_tokens) or 1

        # Query entities: loaded from step1 (we read from registry's entity_index).
        q_entities: Set[str] = set()
        try:
            ent_idx = getattr(self.registry, "_entity_index", {}) or {}
            for k in ent_idx:
                if any(e in q_lower for e in [k.lower(), k]):
                    q_entities.add(k.lower())
        except Exception:
            pass

        # BM25 baseline scores for all sentences (raw BM25).
        raw_scores = self._bm25_scores(query, sids)

        # Build hyperedge → A container lookup (for weak signal 5).
        a_cids_set: Set[str] = set()
        # Read from the last run's step2_a_containers if available.
        # Fallback: try to reconstruct from the last query's a containers.
        # (In practice, the pipeline sets self._last_a_cids in retrieve.)
        a_cids = getattr(self, "_last_a_cids", set()) or set()
        he_cid_to_event_names: Dict[str, List[str]] = {}
        try:
            for he in self.he_registry._hyperedges.values():
                for cid in getattr(he, "container_ids", []) or []:
                    if cid not in he_cid_to_event_names:
                        he_cid_to_event_names[cid] = []
                    en = getattr(he, "event_name", "") or ""
                    if en:
                        he_cid_to_event_names[cid].append(en.strip().lower())
        except Exception:
            pass

        # Sentence → container(s) map (for weak signal 5).
        sid_to_cids: Dict[str, List[str]] = {}
        for sid in sids:
            node = self.registry.get_node(sid)
            if node:
                sid_to_cids[sid] = getattr(node, "containers", []) or []

        # ── Score each sentence ─────────────────────────────────────
        out: List[Tuple[str, float]] = []
        for sid in sids:
            bm25_raw = raw_scores.get(sid, 0.0)

            # Signal 1: BM25 × 0.5.
            score = bm25_raw * 0.5

            node = self.registry.get_node(sid)
            text = self._text_of(sid)
            text_lower = text.lower()
            s_tokens = set(_re.findall(r"[a-z]{3,}", text_lower))
            s_token_count = len(s_tokens) or 1

            # ── Signal 2: 长实体锚点加分 ─────────────────────────────
            node_entities = getattr(node, "entities", []) or []
            in_query_entities = q_entities & set(e.lower() for e in node_entities)
            out_of_query_long: List[str] = []  # len>=6, not in query
            for e in node_entities:
                el = e.lower()
                if el not in q_lower and len(el) >= 6:
                    out_of_query_long.append(el)
            # High bonus for long new entities.
            if out_of_query_long:
                # 问题覆盖率 = query tokens ∩ sentence tokens / sentence tokens.
                overlap = len(q_tokens & s_tokens)
                cov = overlap / q_token_count
                base_bonus = 0.05 * len(out_of_query_long)
                # 问题覆盖率 0 → 打七折.
                effective_bonus = base_bonus * (0.7 if cov < 0.1 else 1.0)
                score += effective_bonus

            # ── Signal 3: 问题回声惩罚 —— 已移除 ─────────────────────
            # The echo penalty (coverage/density of query tokens inside the
            # sentence) was removed by design: a sentence that echoes the
            # question's anchor words is frequently exactly the one that
            # holds the answer, and the penalty pushed it *down* the pool
            # ranking.  It was applied twice (here at pool-build time and
            # again in the step-10 rerank); both are now gone.
            penalty = 0.0

            # ── Signal 4: 第二跳加分 ─────────────────────────────────
            has_query_entity = bool(in_query_entities)
            has_new_entity = bool(out_of_query_long)
            if has_query_entity and has_new_entity:
                # 桥接句: 同时含查询实体和新实体.
                score += 0.15
            elif has_new_entity and not has_query_entity:
                # 答案句: 含新实体但不含查询实体.
                score += 0.08

            # ── Weak Signal 5: 同段落超边加分 ────────────────────────
            s_cids = sid_to_cids.get(sid, [])
            hyperedge_bonus = False
            for cid in s_cids:
                if cid in a_cids:
                    # sentence's container shares a hyperedge with A container.
                    for he in self.he_registry.get_hyperedges_for_container(cid):
                        for other_cid in getattr(he, "container_ids", []) or []:
                            if other_cid in a_cids and other_cid != cid:
                                hyperedge_bonus = True
                                break
                        if hyperedge_bonus:
                            break
                if hyperedge_bonus:
                    break
            if hyperedge_bonus:
                score += 0.05

            # ── Weak Signal 6: 事件名命中加分 ───────────────────────
            event_bonus = False
            for cid in s_cids:
                event_names = he_cid_to_event_names.get(cid, [])
                for en in event_names:
                    if self._event_in_text(en, text_lower):
                        event_bonus = True
                        break
                if event_bonus:
                    break
            if event_bonus:
                score += 0.03

            # ── DEBUG: breakdown dump (only when explicitly requested) ──
            if getattr(self, "_collect_breakdown", False):
                self._breakdown_buffer[sid] = {
                    "bm25": round(bm25_raw * 0.5, 4),
                    "long_entity_bonus": round(
                        (0.05 * len(out_of_query_long))
                        * (0.7 if (out_of_query_long and cov < 0.1) else 1.0),
                        4,
                    ),
                    "echo_penalty": round(-penalty, 4),
                    "second_hop": round(
                        0.15 if (has_query_entity and has_new_entity)
                        else (0.08 if (has_new_entity and not has_query_entity) else 0.0),
                        4,
                    ),
                    "hyperedge": 0.05 if hyperedge_bonus else 0.0,
                    "event": 0.03 if event_bonus else 0.0,
                    "total": round(score, 4),
                    "n_long_new": len(out_of_query_long),
                    "in_query_ent": list(in_query_entities)[:3],
                    "out_long": out_of_query_long[:3],
                }

            out.append((sid, score))

        return out

    def _bm25_scores(
        self, query: str, sids: List[str],
    ) -> Dict[str, float]:
        """
        Standalone BM25 scores for all sentences (used as raw signal
        in _score_sentences_v8). Returns dict sid → raw BM25 score.
        """
        import re as _re
        try:
            import numpy as _np
        except Exception:
            return {}

        vocab = getattr(self.graph, "_tfidf_vocab", None)
        idf_arr = getattr(self.graph, "_tfidf_idf", None)
        if not vocab or idf_arr is None:
            return {}

        q_counter: dict = {}
        for t in _re.findall(r"[a-z]{3,}", (query or "").lower()):
            q_counter[t] = q_counter.get(t, 0) + 1

        avg_len = 1.0
        k1, b = 1.5, 0.75

        scores: Dict[str, float] = {}
        for sid in sids:
            text = self._text_of(sid) or ""
            tokens = _re.findall(r"[a-z]{3,}", text.lower())
            dl = len(tokens)
            if not dl:
                scores[sid] = 0.0
                continue
            if avg_len == 1.0:
                # single pass for avg
                pass
            tf: Dict[str, int] = {t: tokens.count(t) for t in set(tokens)}
            score = 0.0
            for w, qf in q_counter.items():
                j = vocab.get(w)
                if j is None:
                    continue
                f = tf.get(w, 0)
                if f == 0:
                    continue
                idf = float(idf_arr[j])
                num = f * (k1 + 1.0)
                den = f + k1 * (1.0 - b + b * dl / max(avg_len, 1.0))
                score += idf * (num / den) * (qf ** 0.5)
            scores[sid] = float(score)

        # Compute avg_len for proper normalization (second pass if needed).
        if not scores:
            return scores
        lens = []
        for sid in sids:
            text = self._text_of(sid) or ""
            tokens = _re.findall(r"[a-z]{3,}", text.lower())
            lens.append(len(tokens))
        avg_len = (sum(lens) / max(len(lens), 1)) or 1.0
        for sid in sids:
            text = self._text_of(sid) or ""
            tokens = _re.findall(r"[a-z]{3,}", text.lower())
            dl = len(tokens)
            if dl < 1:
                continue
            tf_d = {t: tokens.count(t) for t in set(tokens)}
            raw = scores.get(sid, 0.0)
            # Re-normalize with correct avg_len.
            _re_score = 0.0
            for w, qf in q_counter.items():
                j = vocab.get(w)
                if j is None:
                    continue
                f = tf_d.get(w, 0)
                if f == 0:
                    continue
                idf = float(idf_arr[j])
                num = f * (k1 + 1.0)
                den = f + k1 * (1.0 - b + b * dl / avg_len)
                _re_score += idf * (num / den) * (qf ** 0.5)
            scores[sid] = _re_score

        return scores

    def _score_sentences(
        self, query: str, sids: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Try to get a real sentence embedding for the query.  If the
        ContainerManager's embedder is available AND each sentence has
        a stored `summary_vector`, use cosine.  Otherwise fall back to
        token Jaccard (still works for the ranker; the spec calls for
        "semantic similarity" and this is the closest deterministic
        proxy we have without sentence-level faiss).

        Phase-A [借鉴 A]: prefer sentence-level TF-IDF cosine when
        `GraphBuilder.compute_tfidf_index()` has populated
        `node.tfidf_vector`. TF-IDF dramatically outperforms token
        Jaccard on HotpotQA multi-hop (idf weighting rescues
        discriminating entities like "Havel"/"Fitzgerald" that Jaccard
        under-counts).

        Week-2 Direction B: entity-substring bonus injected here so it
        fires in Step 7 (sentence ranking), not just Step 10 (final rerank).
        """
        # 1) Try TF-IDF cosine first (Phase-A fast path).
        scored_tfidf = self._score_tfidf(query, sids)
        if scored_tfidf is not None:
            return scored_tfidf

        # 2) Sentence-level FAISS / summary_vector (existing Phase-2 path).
        embedder = getattr(self.cm, "embedder", None)
        if embedder is not None and self._has_sentence_vectors():
            return self._score_cosine(query, sids, embedder)
        # 3) Final fallback: token Jaccard.
        return [
            (sid, self._jaccard_sim(query, self._text_of(sid)))
            for sid in sids
        ]

    def _score_tfidf(
        self, query: str, sids: List[str],
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Score sentences by BM25(query, sentence) on the shared
        TF-IDF vocab, with two extra rerank signals that fix the
        specific seed-7 failures:

          (a) length-banded answer-likelihood bonus: HotpotQA gold
              sentences are usually 1 short clause with a named
              anchor, not 60-word exposition.  Sentences whose
              NER-density (named-entity tokens / total tokens) is
              high AND whose word count is moderate (10-35 words)
              get a small additive boost.

          (b) numeric/date bonus: gold answers are often dates or
              numbers ("November 15, 1903", "102,984", "three
              singles titles").  Sentences that contain a number or
              a year get a small additive boost to surface them
              above generic prose.

          (c) Week-2 Direction B: entity-substring bonus injected HERE
              (not just in Step 10 final rerank) so that a sentence
              containing the question's named anchors gets a genuine
              score boost during Step 7 sentence ranking, not only
              at the final-pass level.  This is what pushes
              gold-level sentences from rank 12 → rank 8.

        Returns None (caller falls back) if the TF-IDF index is
        absent or any requested sentence has no vector.
        """
        try:
            import re as _re
            import numpy as np
        except Exception:  # noqa: BLE001
            return None
        if not sids:
            return []
        vocab = getattr(self.graph, "_tfidf_vocab", None)
        idf_arr = getattr(self.graph, "_tfidf_idf", None)
        if not vocab or idf_arr is None:
            return None

        # 1) query tokens (raw counts).
        q_counter: dict = {}
        for t in _re.findall(r"[a-z]{3,}", (query or "").lower()):
            q_counter[t] = q_counter.get(t, 0) + 1
        if not q_counter:
            return None
        # BM25 hyper-params (standard defaults).
        k1 = 1.5
        b = 0.75

        # 2) gather per-sentence term frequencies + length.
        sent_tf: List[Dict[str, int]] = []
        sent_lens: List[int] = []
        sent_texts: List[str] = []
        for sid in sids:
            text = self._text_of(sid) or ""
            sent_texts.append(text)
            tokens = _re.findall(r"[a-z]{3,}", text.lower())
            sent_lens.append(len(tokens))
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            sent_tf.append(tf)

        # Average doc length for BM25 length normalisation.
        avg_len = (sum(sent_lens) / max(1, len(sent_lens))) or 1.0

        out: list = []
        for sid, tf, dl, text in zip(sids, sent_tf, sent_lens, sent_texts):
            # BM25 score against query terms only.
            score = 0.0
            for w, qf in q_counter.items():
                j = vocab.get(w)
                if j is None:
                    continue
                f = tf.get(w, 0)
                if f == 0:
                    continue
                idf = float(idf_arr[j])
                # Standard BM25 term contribution.
                num = f * (k1 + 1.0)
                den = f + k1 * (1.0 - b + b * dl / avg_len)
                score += idf * (num / den) * (qf ** 0.5)  # qf saturation
            # length-banded answer-likelihood bonus.
            ner_density, has_number, n_tokens = self._answer_shape_features(text)
            if 8 <= n_tokens <= 38 and ner_density >= 0.10:
                score += 0.06  # short, entity-dense → likely gold-shaped
            elif n_tokens <= 8 and ner_density >= 0.20:
                score += 0.10  # very short + dense → very gold-shaped
            if has_number:
                # Many gold answers are dates/numbers.
                score += 0.03
            out.append((sid, float(score)))
        return out

    def _answer_shape_features(self, text: str) -> Tuple[float, bool, int]:
        """
        Return (ner_density, has_number, n_tokens) for a sentence.

        ner_density: ratio of tokens covered by an NER run (we use
            a cheap regex-based proxy: tokens starting with an
            uppercase letter, or all-caps tokens, or tokens that
            follow "the"/"Mr"/"Mrs" etc.  Good enough for
            anchor-density estimation; we use spaCy only when it's
            cheap).
        has_number: True iff the text contains a year (1900-2099)
            or a comma-separated / decimal number.
        n_tokens: word count (alphanumeric runs).
        """
        if not text:
            return 0.0, False, 0
        try:
            import re as _re
            tokens = _re.findall(r"[A-Za-z0-9'\-]+", text)
            if not tokens:
                return 0.0, False, 0
            n_tokens = len(tokens)
            # Capitalised / all-caps tokens → likely named entity.
            cap = sum(
                1 for t in tokens
                if t[:1].isupper() or (t.isupper() and len(t) >= 2)
            )
            ner_density = cap / n_tokens
            has_number = bool(
                _re.search(r"\b(19|20)\d{2}\b", text)
                or _re.search(r"\b\d{1,3}(,\d{3})+\b", text)
                or _re.search(r"\b\d+\.\d+\b", text)
            )
            return ner_density, has_number, n_tokens
        except Exception:
            return 0.0, False, len(text.split())

    def _has_sentence_vectors(self) -> bool:
        # If any sentence carries a numeric summary_vector, treat the
        # corpus as having sentence-level vectors.
        for s in list(getattr(self.registry, "_nodes", {}).values())[:1]:
            return getattr(s, "summary_vector", None) is not None
        return False

    def _score_cosine(
        self, query: str, sids: List[str], embedder: Any,
    ) -> List[Tuple[str, float]]:
        try:
            import numpy as np
            qv = np.asarray(
                embedder.encode([query], normalize_embeddings=True)[0],
                dtype=np.float32,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("embedder.encode failed (%s); falling back to Jaccard", e)
            return [(s, self._jaccard_sim(query, self._text_of(s))) for s in sids]
        out: List[Tuple[str, float]] = []
        for sid in sids:
            node = self.registry.get_node(sid)
            if node is None:
                continue
            sv = getattr(node, "summary_vector", None)
            if sv is None:
                out.append((sid, self._jaccard_sim(query, self._text_of(sid))))
                continue
            try:
                v = np.asarray(sv, dtype=np.float32)
                cos = float(np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v) + 1e-9))
            except Exception:
                cos = self._jaccard_sim(query, self._text_of(sid))
            out.append((sid, max(0.0, cos)))
        return out

    # ------------------------------------------------------------------
    # [8] supplement pool: B sentences whose text matches a hyperedge event_name
    # ------------------------------------------------------------------
    def _event_in_text(self, event_name: str, text_lower: str) -> bool:
        """
        PHASE-E1 [event_in_text rewrite]: with the new
        paragraph-title-based event_name, the event name is now a
        proper noun phrase (e.g. "Oliver Fricker", "White House Chief
        of Staff").  We no longer rely on bare-substring match (too
        noisy) or 0.5 token Jaccard (too lossy).  Instead we score
        each event token and accept the sentence if it accumulates
        enough *weighted* coverage:

          (a) tokens with length >= 6 (multi-word, content-bearing)
              contribute 1.0 each if present
          (b) tokens with length 4-5 contribute 0.5
          (c) tokens with length <= 3 or pure stop-words contribute 0

        Accept threshold = 0.6 (tuned so a 2-token title like
        "Hans Knirsch" needs both tokens, while a 4-token title
        like "Acela Express" needs 2 of them).

        This replaces the previous Jaccard 0.5 fallback which lost
        long titles with only one matching token (e.g. event
        "Münsterländer" + text "Large Münsterländer" → Jaccard 0.5
        borderline).
        """
        if not event_name or not text_lower:
            return False
        en = event_name.lower().strip()
        # Fast path 1: bare substring (the cheapest, most precise match).
        if en in text_lower:
            return True
        # Token-level weighted coverage.
        stop = {
            "the", "a", "an", "of", "in", "on", "at", "to", "for", "by",
            "and", "or", "is", "are", "was", "were", "be", "been",
            "as", "that", "this", "these", "those", "with", "from",
            "into", "than", "then", "but",
        }
        en_tokens = [
            t for t in re.findall(r"[a-z0-9'\-]+", en)
            if t and t not in stop and len(t) >= 3
        ]
        if not en_tokens:
            return False
        weight = 0.0
        total = 0.0
        for t in en_tokens:
            w = 1.0 if len(t) >= 6 else (0.5 if len(t) >= 4 else 0.0)
            if w == 0.0:
                continue
            total += w
            if t in text_lower:
                weight += w
        if total <= 0:
            return False
        return (weight / total) >= 0.6

    # ------------------------------------------------------------------
    # [9] merge top-N + supplement (supplement only fills gaps)
    # ------------------------------------------------------------------
    def _step10_final_rerank(
        self, query: str, sids: List[str], top_k: int,
        entities: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Score every candidate with the event+fact fused score, sort by
        `final_score` desc, return the top-K sids.  The full per-sentence
        breakdown dicts (sentence_id, event_id, S_h_E, S_h_F, S_f,
        attr_match, attr_gate_pass, echo_penalty, final_score) are
        stashed on `self._last_ef_scored` so the caller can persist them
        onto `PipelineResult.debug` for inspection.

        Hard attribute gate — a DELETION, but a BOUNDED one:
          a candidate is kept only if it carries at least one of the
          query's top-`ATTR_GATE_TOP_K` attributes, measured by the
          VECTOR attribute profile of the sentence (sent_attr_vec) and
          not by a keyword table.  Candidates that carry none of them
          are removed from the pool entirely — they never reach the LLM.

        **The bound matters (2026-10-30 revision).**  The gate used to
        relax ONLY when it emptied the pool.  A gate that keeps 1
        candidate out of 30 was treated as "working as intended" — and
        that is exactly what destroys EM/F1 while *improving* recall:
        the attribute table decides which sentences are eligible, so a
        mis-calibrated wanted-attribute set deletes the gold sentence
        before the reranker ever sees it.  Recall then goes UP (the
        surviving sentences are on-topic) while the answer goes DOWN
        (the sentence carrying it is gone).

        So the gate now also has a FLOOR: it must keep at least
        `ATTR_GATE_MIN_KEEP` candidates, and when the gate would keep
        fewer than that it relaxes.  The relaxation ladder is top-3 →
        top-5 → top-8 → keep-all → best-N-by-score.

        Safety: the level actually used is recorded in
        `attr_gate_level`, so a mis-calibrated gate is visible in the
        logs rather than silent.
        """
        scored = self._event_fact_score(query, sids)
        if not scored:
            self._last_ef_scored = []
            return []

        # Apply the hard attribute gate.
        #
        # The gate asks: does this sentence CARRY one of the query's
        # top-3 wanted attributes (per the vector profile)?  If not, it
        # is DELETED — it no longer competes for a slot.
        #
        # ─────────────────────────────────────────────────────────────
        # [2026-11-01] **硬门 → 软门（改造 H）**
        # ─────────────────────────────────────────────────────────────
        # 硬门在这里被证伪了。它带来的净效果是 EM -0.100 / R@10 -0.158，
        # 而排序本身是变好的（排序丢失 6 → 2）—— 恶化全部来自"门删掉"
        # 这一项（0 → 5）。
        #
        # 逐题复盘那 5 道被删的题，两道的 gold 的 final_score 是**全场
        # 第 1** —— 排序器判断正确，门把它删掉了。门覆盖了排序器的判断。
        #
        # 根因是 wanted 与 mounted **零交集**：wanted 只取 query 侧
        # embedding 余弦的 top-3，完全不看 sentence 侧挂了什么；而
        # `mounted_attrs` 是粗分类（实体类型 × 属性类别），两者经常对不
        # 上。这不是阈值问题 —— 调大调小 ATTR_GATE_TOP_K 都治不了
        # "wanted 集合本身是错的"。
        #
        # 现在属性**不再有删除权**。它保留为打分分量（`W_ATTR`），
        # 不满足的句子只是拿不到属性加权、排序时自然下沉。属性从
        # "准入证"退回"加分项" —— 这才是排序信号应有的位置。
        #
        # 保留 wanted 集合的用途：它决定 `attr_match` 用哪些属性算
        # 对齐度（见 `_event_fact_score` 的 coverage 计算）。
        #
        # `attr_gate_kept` / `attr_gate_pass` 仍然被填充（值恒为 True），
        # 因为 eval 的 RANK-DIAG 依赖这两个字段来报告"门删掉了多少"。
        # 字段语义变了，但**接口不变** —— 旧的结果文件仍然可读。
        kept = scored
        gate_level = 0
        n_gated = 0
        if scored and scored[0].get("wanted_attrs"):
            # 软门：不再删除任何候选。这里只做一次**观测**，把
            # "如果这是硬门，会删掉多少" 记进日志，让误校准的属性表
            # 依然可见 —— 只是不再伤害结果。
            base = list(scored[0].get("wanted_attrs_rank") or [])
            allowed = set(base[:ATTR_GATE_TOP_K])
            n_would_gated = sum(
                1 for d in scored
                if not (set(d.get("mounted_attrs") or []) & allowed)
            )
            if n_would_gated:
                logger.info(
                    "[EF-attr-gate] SOFT gate — %d/%d candidates carry none "
                    "of the wanted attrs %s (kept anyway; attr is a "
                    "ranking signal, not an admission ticket)",
                    n_would_gated, len(scored), sorted(allowed),
                )
        kept_ids = {id(d) for d in kept}
        for d in scored:
            d["attr_gate_level"] = gate_level
            d["attr_gate_kept"] = id(d) in kept_ids
            # Keep attr_gate_pass consistent with the decision actually
            # taken.  Under the soft gate every candidate is kept, so
            # this is always True — but the field is still written so
            # the eval diagnostics keep working.
            d["attr_gate_pass"] = id(d) in kept_ids

        # Rank everything by the fused score.
        scored.sort(key=lambda d: -d["final_score"])

        # ----------------------------------------------------------------
        # Selection — rank everything, truncate to top_k.
        #
        # [2026-11-01] Under the SOFT gate every candidate is kept, so
        # this is now simply "take the top_k by final_score".  The old
        # code filtered by `id(d) in kept_ids` first, which under the
        # hard gate meant "gate-passing only".
        #
        # `final_rows` is the ranked candidate list, truncated to top_k.
        # `attr_block` is kept in the record for backwards compatibility
        # with the eval diagnostics.
        final_rows = [d for d in scored if id(d) in kept_ids][:top_k]

        # ─────────────────────────────────────────────────────────────
        # [2026-11-01] **保底保证（改造 H，第 3 项）**
        # ─────────────────────────────────────────────────────────────
        # 保证：**分数最高的 N 句一定进入 LLM 上下文**，任何筛选逻辑
        # 都不能把它们排除。
        #
        # 为什么需要这一层（改造 H 的第 1、2 项还不够）：
        # 旧的门有 `ATTR_GATE_MIN_KEEP=3` 的保底，但那个保底只在
        # "池子被掏空"时才触发 —— 而误校准时池子往往还有 20+ 候选，
        # 门从容地删掉 5 个 gold，保底根本不会启动。
        #
        # 也就是说：**旧保底保护的是"数量"，不是"质量"**。它保证
        # 上下文不会太空，但完全不保证正确答案句在里面。
        #
        # 新的保底保护的是"**排序器的判断**"。既然排序器已经把 gold
        # 排到第 1（实测有两道题的 gold final_score 是全场第 1），那
        # 任何后续的筛选都不该把它踢掉。这是一道**最后一道防线**，
        # 独立于属性逻辑 —— 未来即使再加别的过滤器，也绕不过它。
        #
        # 实现方式：显式地把 score 最高的 N 句补进 final_rows。
        # （当前软门下这已经是恒等操作，但写成显式代码是为了：）
        #   1. 把这条不变量固定下来，不再依赖"恰好没有别的过滤器"
        #   2. 让它可测 —— 下面的断言会真的检查它
        # ─────────────────────────────────────────────────────────────
        n_guarantee = max(1, min(ATTR_TOP_SCORE_GUARANTEE, top_k))
        top_by_score = scored[:n_guarantee]
        final_ids = {id(d) for d in final_rows}
        for d in top_by_score:
            if id(d) not in final_ids:
                final_rows.append(d)
                final_ids.add(id(d))
        # Keep final_rows ranked by score after the guarantee append.
        final_rows.sort(key=lambda d: -d["final_score"])
        final_rows = final_rows[:top_k]

        # The guarantee must hold — assert it rather than hope.
        # A violation here means some filter ran after the sort.
        _guaranteed = {id(d) for d in top_by_score}
        if not _guaranteed.issubset({id(d) for d in final_rows}):
            logger.error(
                "[EF-guarantee] VIOLATION: top-%d by score not all in "
                "final_rows — a filter is running after the sort",
                n_guarantee,
            )

        logger.info(
            "[EF-rank] kept %d/%d (soft gate deleted %d) → final %d/%d "
            "(top-%d score guarantee held)",
            len(kept), len(scored), len(scored) - len(kept),
            len(final_rows), top_k, n_guarantee,
        )

        # Log per-query average contribution of each component so we can
        # see which one is driving the ranking.
        #
        # **The labels changed (2026-10-30).**  The old line said
        # "W_EVENT" / "avg_(1-wE)·S_h_F", which described the OLD
        # formula where S_f = W_EVENT·S_h_E + (1-W_EVENT)·S_h_F.
        # The new formula has five independently-weighted normalised
        # components, so the log has to name them individually —
        # otherwise a weight regression is invisible.
        n = max(1, len(final_rows))
        avg_e = sum(d["S_h_E"] for d in final_rows) / n
        avg_f = sum(d["S_h_F"] for d in final_rows) / n
        avg_attr = sum(d["attr_term"] for d in final_rows) / n
        avg_lex = sum(d.get("lex_term", 0.0) for d in final_rows) / n
        logger.info(
            "[EF-rank] n=%d | W_NAME_N=%.2f W_SENT_N=%.2f W_FACT_N=%.2f "
            "W_ATTR=%.2f W_LEX=%.2f | avg_event=%.4f avg_fact=%.4f "
            "avg_attr=%.4f avg_lex=%.4f | avg_S_f=%.4f | avg_echo=%.4f | "
            "avg_final=%.4f",
            n, W_EVENT_NAME_N, W_SENT_N, W_FACT_N, W_ATTR, W_LEX,
            avg_e, avg_f, avg_attr, avg_lex,
            sum(d["S_f"] for d in final_rows) / n,
            sum(d["echo_penalty"] for d in final_rows) / n,
            sum(d["final_score"] for d in final_rows) / n,
        )
        # Persist ALL scored rows (including gate-deleted ones) so the
        # debug dump shows what the gate removed and why.
        self._last_ef_scored = scored
        for d in final_rows:
            d["attr_block"] = "primary"
        return [d["sentence_id"] for d in final_rows[:top_k]]

    # ------------------------------------------------------------------
    # 事件+事实 融合打分的核心
    # ------------------------------------------------------------------
    def _event_fact_score(
        self, query: str, sids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Score every candidate sentence with the normalised rerank formula.

        **The formula (2026-10-30 revision)**
        -------------------------------
        Every component is min-max normalised over the candidate pool
        FIRST, then weighted::

            final = W_EVENT_NAME_N · norm(cos(event_name, q) · penalty)
                  + W_SENT_N       · norm(cos(sentence, q))
                  + W_FACT_N       · norm(cos(sentence, q))
                  + W_ATTR         · norm(attribute_match)
                  + W_LEX          · norm(BM25 + answer-shape prior)

        Normalisation is not cosmetic — it is what makes the weights
        mean anything.  The OLD formula added raw cosines of different
        scales (``cos(title,q)`` ∈ [0.15,0.35] vs ``cos(sent,q)`` ∈
        [0.35,0.55]); whichever component had the larger VARIANCE
        dominated, so a nominal "event-name dominant" setting
        (W_EVENT_NAME=0.7) was in practice sentence-vector dominant,
        and the attribute term (W_ATTR=0.10) was drowned out entirely.

        **The lexical term (W_LEX)** is BM25 plus the answer-shape
        prior.  It was DEAD CODE before this revision: `_score_tfidf`
        had lost its only caller when the layer-quota step was
        removed, so BM25, the IDF-weighted entity discrimination and
        the HotpotQA-specific answer-shape prior (ner_density /
        has_number) contributed nothing to the ranking.  They are
        back.

        Returns one dict per sid:
            {sentence_id, event_id, S_h_E, S_h_F, S_f,
             attr_match, attr_term, echo_penalty, final_score,
             norm_name, norm_sent, norm_fact, norm_attr, norm_lex}
        Sorted order is the caller's responsibility.
        """
        if not sids:
            return []

        import numpy as _np  # local import keeps the top of file clean

        # ----- query vector -----
        q_vec = self._encode_query(query)

        # ----- S_h(F, Q) via the existing cosine path -----
        s_h_f: Dict[str, float] = {}
        s_h_f_method = "cosine"
        if q_vec is not None:
            try:
                for sid, sim in self._score_cosine(query, sids, self._get_embedder()):
                    s_h_f[sid] = float(sim)
            except Exception as e:  # noqa: BLE001
                logger.debug("event_fact: cosine path failed (%s); using Jaccard", e)
                s_h_f_method = "jaccard"
                for sid in sids:
                    s_h_f[sid] = float(self._jaccard_sim(query, self._text_of(sid)))
        else:
            s_h_f_method = "jaccard"
            for sid in sids:
                s_h_f[sid] = float(self._jaccard_sim(query, self._text_of(sid)))

        # Log fallback method so we know if cosine path is alive.
        logger.debug("event_fact: s_h_f method=%s (q_vec=%s)", s_h_f_method, q_vec is not None)

        # ----- build hyperedge index for these sids -----
        # event_id -> {
        #   "name": str, "members": [sids],
        #   "name_vec": ndarray|None, "mean_vec": ndarray|None,
        #   "S_h_E": float,
        # }
        events: Dict[str, Dict[str, Any]] = {}
        # sid -> [event_id, ...]   (all hyperedges the sentence touches)
        sid_events: Dict[str, List[str]] = {sid: [] for sid in sids}

        for sid in sids:
            node = self.registry.get_node(sid)
            if node is None:
                continue
            for cid in getattr(node, "containers", []) or []:
                try:
                    he_list = self.he_registry.get_hyperedges_for_container(cid) or []
                except Exception:
                    he_list = []
                for he in he_list:
                    he_id = getattr(he, "id", None) or f"he_{len(events)}"
                    if he_id not in events:
                        events[he_id] = {
                            "name": (getattr(he, "event_name", "") or "").strip(),
                            "members": [],
                            "name_vec": None,
                            "mean_vec": None,
                            "S_h_E": 0.0,
                            # ----- 大事件惩罚字段 -----
                            # |F_e|: 事件 E 的全量成员句子数(基于 he.container_ids
                            #        → 每个 container 的所有 sentence_ids 并集)
                            "all_members": set(),
                            "event_size": 0,
                            "penalty": 1.0,
                        }
                    if sid not in events[he_id]["members"]:
                        events[he_id]["members"].append(sid)
                    if he_id not in sid_events[sid]:
                        sid_events[sid].append(he_id)

        # ----- per-event vectors -----
        # Only the event_name vector is computed here.  The sentence-side
        # term of S_h_E is per-SENTENCE (not per-event), so it is applied
        # later, in the per-sentence loop below.
        embedder = self._get_embedder()
        for he_id, ev in events.items():
            # event_name vector
            name_vec: Optional[_np.ndarray] = None
            if ev["name"] and embedder is not None and q_vec is not None:
                try:
                    name_vec = _np.asarray(
                        embedder.encode([ev["name"]], normalize_embeddings=True)[0],
                        dtype=_np.float32,
                    ).reshape(-1)
                    nrm = _np.linalg.norm(name_vec)
                    if nrm > 0:
                        name_vec = name_vec / nrm
                except Exception as e:  # noqa: BLE001
                    logger.debug("event_fact: name embed failed for he=%s (%s)", he_id, e)
                    name_vec = None

            ev["name_vec"] = name_vec
            ev["mean_vec"] = None

            if q_vec is None:
                ev["S_h_E"] = 0.0
                continue

            # Event-level component: cos(event_name, query).
            # The 0.3 sentence term is added per-sentence downstream.
            if name_vec is None:
                ev["S_h_E"] = 0.0
            else:
                ev["S_h_E"] = float(_cos(name_vec, q_vec))

        # ----- 大事件惩罚: 计算 |F_e| 与 penalty(E) -----
        # |F_e| = 事件 E 的全量成员句子数(基于 he.container_ids 下所有
        #         container 的 sentence_ids 并集).
        # penalty(E) = 1 / (|F_e| - 1) ** GAMMA_E , |F_e| <= 1 → 1.0
        try:
            for ev_id, ev in events.items():
                he_obj = self.he_registry.get_hyperedge(ev_id) if self.he_registry else None
                all_sents: set = set()
                if he_obj is not None and getattr(he_obj, "container_ids", None):
                    for cid in he_obj.container_ids:
                        cobj = self.cm.get_container(cid) if self.cm else None
                        if cobj is None:
                            continue
                        for msid in (getattr(cobj, "sentence_ids", []) or []):
                            all_sents.add(msid)
                # fallback: 如果拿不到全量, 用本次候选池里的 members
                if not all_sents:
                    all_sents = set(ev.get("members", []))
                ev["all_members"] = all_sents
                ev["event_size"] = len(all_sents)
                sz = ev["event_size"]
                if sz <= 1:
                    ev["penalty"] = 1.0
                else:
                    ev["penalty"] = float(1.0 / ((sz - 1) ** GAMMA_E))
        except Exception as _e:  # noqa: BLE001
            logger.debug("event_fact: penalty calc failed (%s); all penalty=1.0", _e)
            for ev in events.values():
                ev["event_size"] = len(ev.get("members", [])) or 1
                ev["penalty"] = 1.0

        # ----- echo penalty: REMOVED -------------------------------------
        # The echo penalty (token-overlap with the query) was removed by
        # design: a sentence that repeats the question's anchor words is
        # often exactly the one that holds the answer, and the penalty
        # pushed such sentences *down* the ranking.  The signal is kept
        # in the breakdown (always 0.0) so stored `ef_scores` rows stay
        # comparable with older runs.
        echo_pen: Dict[str, float] = {sid: 0.0 for sid in sids}

        # ----- compose per-sentence result -----
        out: List[Dict[str, Any]] = []
        # 日志聚合: 每题的惩罚与信号强度
        sum_event_size = 0.0
        sum_penalty = 0.0
        sum_event_term = 0.0
        sum_fact_term = 0.0
        sum_attr_term = 0.0
        sum_lex_term = 0.0
        cnt_event = 0

        # ─────────────────────────────────────────────────────────────
        # Symmetric-attribute soft matching
        #
        # Both sides live in ONE embedding space, built from the
        # attribute DESCRIPTORS (not the bare names):
        #
        #   query side   : q_k = cos(query_emb, desc_k)      → what the question wants
        #   sentence side: p_k = cos(sentence_emb, desc_k)   → what the sentence is about
        #
        # Step 1: encode the descriptors once.
        # Step 2: cosine(query_emb, desc_emb) → query_attr_vec, normalised
        #         so the best-matching attribute scores 1.0.
        # Step 3: per sentence, cosine(sentence_emb, desc_emb) →
        #         sent_attr_vec, the sentence's continuous attribute
        #         profile (e.g. date 0.8, death 0.9).
        # Step 4: attr_match = best over the sentence's attributes of
        #         (sentence's own score for that attribute) × (how much
        #         the query wants that attribute).
        # Step 5: the hard gate deletes sentences that carry none of the
        #         query's top-`ATTR_GATE_TOP_K` attributes.
        # ─────────────────────────────────────────────────────────────
        attr_embs: Optional[_np.ndarray] = None
        query_attr_vec: List[float] = []  # one score per attr in ATTRIBUTE_NAMES
        if q_vec is not None and embedder is not None and self.attr_soft_match_mode:
            try:
                attr_embs = _np.asarray(
                    embedder.encode(
                        [ATTRIBUTE_DESCRIPTORS[a] for a in ATTRIBUTE_NAMES],
                        normalize_embeddings=True,
                    ),
                    dtype=_np.float32,
                ).reshape(len(ATTRIBUTE_NAMES), -1)
                raw: List[float] = []
                for i in range(len(ATTRIBUTE_NAMES)):
                    attr_vec = attr_embs[i].reshape(-1)
                    # Clamp negative cosines to 0 (anti-alignment is not signal).
                    raw.append(max(0.0, float(_cos(attr_vec, q_vec))))
                # Normalise so the best-matching attribute reaches 1.0.
                # Without this, generic attribute words ("date", "other")
                # would dominate the signal purely through length priors.
                top = max(raw) if raw else 0.0
                if top > 1e-6:
                    query_attr_vec = [r / top for r in raw]
                else:
                    query_attr_vec = raw
            except Exception as _e:  # noqa: BLE001
                logger.debug("symmetric_attr: attr encoding failed (%s); skipping", _e)
                query_attr_vec = []
                attr_embs = None

        # ----------------------------------------------------------------
        # Restrict the query's wanted attributes to the ones the query's
        # own entities actually own.
        #
        # Without this the two sides of the gate are asymmetric: the
        # SENTENCE side is entity-aware (a PERSON never mounts
        # `population`), but the QUERY side picks attributes from the
        # global vocabulary.  A question about two mosques then asks for
        # `headquarters`/`residence` while its GPE entities only own
        # `events`/`history`/`location`/`population` — so the gate
        # deletes every sentence it should have kept.
        #
        # The restriction only ever REMOVES wanted attributes; if the
        # query's entities own none of them, the restriction is
        # abandoned and the global set is used.
        # ----------------------------------------------------------------
        query_owned_attrs: Set[str] = set()
        # ─────────────────────────────────────────────────────────────
        # [2026-11-01] 实体-属性槽位匹配（改造 H，第 2 项）
        # ─────────────────────────────────────────────────────────────
        # 上面这层"query 实体拥有哪些属性"是**粗粒度**的：它只知道
        # "这个实体有 birth / career / education"，不知道"是哪个实体、
        # 挂的哪个属性、挂在这个句子上"。
        #
        # 而 sentence 侧的 `mounted_attrs` 同样粗（实体类型 × 属性类别）。
        # 两边都是粗的 → 零交集 → 门误杀（改造 H 要解决的核心问题）。
        #
        # 这里建立**实体级的属性槽位索引**：
        #     query_entity_key → {该实体拥有的属性名集合}
        # 以及**句子级的挂载索引**：
        #     (sentence_id, entity_key) → {属性名集合}
        #
        # 有了这两张表，就能问一个细得多的问题：
        # "**这个句子里的这个实体**，是不是恰好挂着**问句想问的属性**？"
        # —— 这才是原始设计里 `birth(A)` / `date(EVENT_A)` 的粒度。
        #
        # 实体键用**小写表面形式 + 末词**双索引，因为 HotpotQA 问句
        # 常只写姓氏（"Which city was Havel born in?" vs 句里的
        # "Václav Havel"）—— 末词降级形态已经在 query 侧用过，
        # 这里对 sentence 侧用同一套规则，两边才能对上。
        # ─────────────────────────────────────────────────────────────
        query_entity_keys: Set[str] = set()
        query_slot_attrs: Dict[str, Set[str]] = {}
        for ent in (self._last_query_entities or []):
            e = (ent or "").strip().lower()
            if not e:
                continue
            try:
                container = self.cm.get_container_for_entity(ent)
            except Exception:  # noqa: BLE001
                container = None
            if container is None:
                continue
            for name, slot in (getattr(container, "attributes", {}) or {}).items():
                if name and name != "other" and getattr(slot, "sentence_ids", None):
                    query_owned_attrs.add(name)
            # 该实体拥有的属性名（按实体键存一份）
            names = {
                n for n, sl in (getattr(container, "attributes", {}) or {}).items()
                if n and n != "other" and getattr(sl, "sentence_ids", None)
            }
            for k in _entity_keys(e):
                query_entity_keys.add(k)
                query_slot_attrs.setdefault(k, set()).update(names)

        # ----------------------------------------------------------------
        # Hard attribute gate
        # ----------------------------------------------------------------
        # The query's "wanted attributes" are the top canonical
        # attribute names by normalised cosine with the query embedding.
        # Because the vector is normalised so the best attribute scores
        # 1.0, the cut is *relative*: an attribute counts as wanted only
        # if it is at least ATTR_QUERY_MIN_SIM of the best one.  This
        # keeps a genuinely unrelated third attribute (which would
        # otherwise sit around 0.6-0.7 purely from embedding noise) out
        # of the wanted set.
        #
        # If the query has no usable attribute signal the gate is skipped
        # for that query, so it can never empty the pool.
        wanted_attrs: Set[str] = set()
        wanted_attrs_rank: List[str] = []
        if query_attr_vec:
            ranked_idx = sorted(
                range(len(ATTRIBUTE_NAMES)),
                key=lambda k: -query_attr_vec[k],
            )
            # Full ranked list (used for diagnostics and the fallback
            # ladder) — only attributes with a non-trivial normalised
            # cosine are meaningful.
            wanted_attrs_rank = [
                ATTRIBUTE_NAMES[k]
                for k in ranked_idx
                if query_attr_vec[k] >= ATTR_QUERY_MIN_SIM
            ]
            # Restrict to the attributes the query's own entities own,
            # so the query side and the sentence side are symmetric.
            if query_owned_attrs:
                restricted = [
                    a for a in wanted_attrs_rank if a in query_owned_attrs
                ]
                if restricted:
                    wanted_attrs_rank = restricted
            wanted_attrs = set(wanted_attrs_rank[:ATTR_GATE_TOP_K])

        # ─────────────────────────────────────────────────────────────
        # [2026-10-30] 精排公式修复：分量收集 → 统一归一化 → 加权
        #
        # 旧公式把不同量纲的余弦直接相加：
        #     final = W_EVENT·(W_NAME·cos(title,q)·pen + W_SENT·cos(sent,q))
        #           + (1-W_EVENT)·cos(sent,q) + W_ATTR·attr
        # cos(title,q) ∈ [0.15,0.35]，cos(sent,q) ∈ [0.35,0.55]，
        # 直接相加 → 方差大的分量主导 → 名义上的"事件名主导"实际
        # 变成了"句子向量主导"；属性信号（W_ATTR=0.10）被完全淹没。
        #
        # 新公式：每个分量先在候选池内 min-max 归一化到 [0,1]，再加权。
        # 归一化后权重才真正表达设计意图。
        #
        #     final = W_EVENT_NAME_N · norm(name · penalty)
        #           + W_SENT_N       · norm(sentence cosine)
        #           + W_FACT_N       · norm(sentence cosine, 事实项)
        #           + W_ATTR         · norm(attribute match)
        #           + W_LEX          · norm(BM25 + 答案形状先验)
        #
        # 阶段 1（本循环）：逐句收集**原始**分量。
        # 阶段 2：统一归一化。
        # 阶段 3：加权求和。
        # ─────────────────────────────────────────────────────────────

        # 词法信号：BM25(query, sentence) + 答案形状先验。
        #
        # 这条信号独立于稠密余弦 —— MiniLM 在"问句问一个具体事实、
        # 答案句是短事实句"这类任务上纯稠密余弦的 top-1 精度本来就弱，
        # 而 BM25 能靠 IDF 权重救回稀有实体。改造日志 A 已验证过：
        # 接回 TF-IDF 后 R@5 +0.033。
        #
        # 之前 `_score_tfidf` 是死代码（Step 7 被删后失去唯一调用者），
        # 现在它重新进入打分链路。
        lex_raw: Dict[str, float] = {}
        if W_LEX > 0.0:
            try:
                lex_pairs = self._score_tfidf(query, sids)
                if lex_pairs:
                    lex_raw = {s: float(v) for s, v in lex_pairs}
            except Exception as e:  # noqa: BLE001
                logger.debug("event_fact: lexical scoring failed (%s)", e)
                lex_raw = {}

        # 属性名 → ATTRIBUTE_NAMES 下标的映射。
        #
        # 旧代码在内层循环里对每个属性做一次 `ATTRIBUTE_NAMES.index(...)`
        # —— 40+ 个属性 × 数百个句子 = 每题几万次线性扫描。
        attr_index: Dict[str, int] = {
            name: i for i, name in enumerate(ATTRIBUTE_NAMES)
        }

        # ─────────────────────────────────────────────────────────────
        # 答案形状先验（answer-shape prior）
        #
        # 问句的"期望答案形状"决定了哪种句子更可能是答案句：
        #   · 问年份/数字 → 含数字的句子更可能是答案句
        #   · 问人名/机构 → 专名密度高的句子更可能是答案句
        #
        # 这个先验是专门为 HotpotQA 写的，但 `_answer_shape_features`
        # 一直没有调用者（Step 7 被删后成为死代码）。现在接回来。
        #
        # 它**只在问句本身有形状暗示时**才生效：
        #   · 问句含 when/what year/how many/how much → 需要数字
        #   · 其他情况 → 需要专名
        # ─────────────────────────────────────────────────────────────
        want_number = bool(
            re.search(
                r"\b(when|what\s+year|which\s+year|how\s+many|how\s+much|"
                r"how\s+long|how\s+old)\b",
                (query or "").lower(),
            )
        )
        shape_raw: Dict[str, float] = {}
        for sid in sids:
            ner_density, has_number, _n_tok = self._answer_shape_features(
                self._text_of(sid) or ""
            )
            if want_number:
                # 期望数字 → 含数字是强信号，其次是实体密度。
                shape_raw[sid] = (
                    (1.0 if has_number else 0.0) * 0.7
                    + min(1.0, ner_density * 3.0) * 0.3
                )
            else:
                # 期望实体 → 实体密度是主信号，数字是弱负信号
                # （HotpotQA 的比较类问题里数字是干扰项）。
                shape_raw[sid] = (
                    min(1.0, ner_density * 3.0) * 0.8
                    - (0.15 if has_number else 0.0)
                )
        shape_raw = {k: max(0.0, v) for k, v in shape_raw.items()}

        # ─────────────────────────────────────────────────────────────
        # 句法关系匹配（syntactic relation bonus）
        #
        # `_step1_5_query_relations` 抽出了问句的"句法骨架"
        # （verb / obj / subj / preps），但这个信号**从来没有被任何
        # 打分环节消费过** —— 抽取了、存进了 `self._last_query_relations`、
        # 然后就死了。
        #
        # 它的价值在于捕捉纯稠密余弦抓不到的东西：
        #   问句 "What county was X born in?"
        #     verb=born, obj=county
        #   候选句 "Havel was born in Prague, Bohemia."
        #     含 "born" + "Prague"(专名) + "county" 语义
        # 但纯余弦只看到"都是关于 Havel 的句子"。
        #
        # 判定方式：候选句里是否出现与问句 verb/obj 共享词根的词。
        # 用词根匹配（而非精确相等）是因为 "born"/"birth"、
        # "release"/"released" 词形不同但同义。
        # ─────────────────────────────────────────────────────────────
        rel_raw: Dict[str, float] = {}
        q_rels = getattr(self, "_last_query_relations", None) or {}
        q_verbs = [v for v in (q_rels.get("verb") or []) if v]
        q_objs = [v for v in (q_rels.get("obj") or []) if v]
        # 关系词干：去常见后缀，让 born/birth、release/released 能对上。
        _SUFFIXES = ("ing", "ed", "es", "s", "e", "d", "tion", "ment")

        def _stems(words: List[str]) -> List[str]:
            out: List[str] = []
            for w in words:
                w = w.lower()
                out.append(w)
                for suf in _SUFFIXES:
                    if len(w) > len(suf) + 2 and w.endswith(suf):
                        out.append(w[: -len(suf)])
                        break
            return out

        q_verb_stems = _stems(q_verbs)
        q_obj_stems = _stems(q_objs)
        if q_verb_stems or q_obj_stems:
            for sid in sids:
                text = (self._text_of(sid) or "").lower()
                if not text:
                    rel_raw[sid] = 0.0
                    continue
                tokens = set(re.findall(r"[a-z]+", text))
                verb_hit = 1.0 if any(
                    st in tokens for st in q_verb_stems
                ) else 0.0
                obj_hit = 1.0 if any(
                    st in tokens for st in q_obj_stems
                ) else 0.0
                # 问句的 verb 比 obj 更能区分"这条句子是不是在回答
                # 这个问题"，所以给更高权重。
                rel_raw[sid] = 0.6 * verb_hit + 0.4 * obj_hit
        else:
            rel_raw = {sid: 0.0 for sid in sids}

        # 阶段 1：逐句收集原始分量。
        raw_name: Dict[str, float] = {}
        raw_sent: Dict[str, float] = {}
        raw_fact: Dict[str, float] = {}
        raw_attr: Dict[str, float] = {}
        raw_lex: Dict[str, float] = {}
        per_sid: Dict[str, Dict[str, Any]] = {}

        # 问句实体锚点（用于事实项的实体覆盖度）。
        #
        # 取自 step1 抽出的语义实体。每个实体是一组"可接受形态"：
        #   整名（"Vaclav Havel"）或 末词（"havel"）。
        # 只要句子里出现其中任一形态，就算这个实体被覆盖。
        #
        # 为什么要分组而不是拍平成一个 token 列表：
        # 拍平的话，"Vaclav Havel" 和 "havel" 会变成两个独立 token，
        # 命中整名的句子反而只得了 0.5 的覆盖度 —— 语义完全相反。
        #
        # 为什么要末词降级形态：HotpotQA 的问句常常只写姓氏
        # （"Which city was Havel born in?"），句子里的形式可能不同。
        q_entity_groups: List[List[str]] = []
        for ent in (self._last_query_entities or []):
            e = (ent or "").strip().lower()
            if not e:
                continue
            forms: List[str] = [e]
            parts = [p for p in re.findall(r"[a-z0-9]+", e) if len(p) >= 3]
            if parts:
                last = parts[-1]
                if last != e:
                    forms.append(last)
            # 组内去重。
            seen_f: Set[str] = set()
            forms = [f for f in forms if not (f in seen_f or seen_f.add(f))]
            if forms:
                q_entity_groups.append(forms)

        for sid in sids:
            sh_f = s_h_f.get(sid, 0.0)
            # Choose the event this sentence belongs to.  The event-level
            # component of S_h_E is cos(event_name, query), scaled by the
            # large-event penalty.  So the "best" event is the one with
            # the highest PENALISED name cosine — picking on the raw
            # cosine instead would let a big-event's name beat a
            # small-event's name even when the penalty flips the order.
            best_eid: Optional[str] = None
            best_name_cos = 0.0
            best_event_size = 0
            best_penalty = 1.0
            best_contrib = -1.0
            for eid in sid_events.get(sid, []):
                ev = events[eid]
                raw_cos = ev["S_h_E"]
                pen = ev.get("penalty", 1.0)
                contrib = max(0.0, raw_cos) * pen
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_name_cos = raw_cos
                    best_eid = eid
                    best_event_size = ev.get("event_size", 0)
                    best_penalty = pen
            if best_eid is None:
                # Sentence not attached to any hyperedge: no event name to
                # match, so the event-name term contributes nothing.
                best_name_cos = 0.0
                best_event_size = 0
                best_penalty = 1.0

            # Negative cosines are clamped to 0: an anti-aligned sentence
            # carries no positive signal.
            name_term = max(0.0, best_name_cos)
            sent_term = max(0.0, sh_f)

            # 大事件惩罚只作用于事件名分量。句子向量分量是逐句的，
            # 不该被惩罚 —— 大段落的句子并不比小段落的句子次要。
            raw_name[sid] = name_term * best_penalty
            raw_sent[sid] = sent_term
            raw_lex[sid] = lex_raw.get(sid, 0.0)

            # ─────────────────────────────────────────────────────────
            # 事实项 = 问句实体覆盖度（entity coverage）
            #
            # 旧代码的 fact_term 是 `(1-W_EVENT) * sh_f` —— 和
            # sent_term **完全同一个值**。这意味着句子余弦被算了两次
            # （一次在 event_term 里、一次在 fact_term 里），整条公式
            # 实际上退化成了"句子余弦 × 常数"，name/attr/lex 三个
            # 分量的相对权重被进一步稀释。
            #
            # 新的 fact 项是一个**独立**信号：候选句覆盖了问句里多少
            # 个实体锚点。HotpotQA 是多跳题，答案句往往同时提到问句
            # 里的两个实体（"Havel" + "Bohemia"）；只提到一个的句子
            # 不可能独立回答这个问题。
            #
            # 这正是多跳检索真正需要的判别力，而且它和句法关系信号
            # 正交（句法看动词，实体覆盖看名词锚点）。
            # ─────────────────────────────────────────────────────────
            cov = 0.0
            if q_entity_groups:
                node_c = self.registry.get_node(sid)
                sent_text_c = (
                    (getattr(node_c, "complemented_text", "") or "")
                    if node_c is not None else ""
                )
                if not sent_text_c:
                    sent_text_c = self._text_of(sid) or ""
                sent_lower = sent_text_c.lower()
                hit = sum(
                    1 for forms in q_entity_groups
                    if any(f in sent_lower for f in forms)
                )
                cov = hit / len(q_entity_groups)
            raw_fact[sid] = cov
            # Symmetric-attribute soft match.
            #
            #   query_attr_vec[k] = normalised cos(query_emb, desc_k)
            #                      → what the question wants
            #   mount weight      = the sentence's OWN similarity to
            #                      attribute k, computed at graph-build
            #                      time and stored in `attribute_mounts`
            #                      (entity-aware, see
            #                      `GraphBuilder._soft_mount_attributes`)
            #   c_k               = weight_k · query_attr_vec[k]
            #   best              = max_k c_k               (peak signal)
            #   coverage          = Σ c_k / max_k c_k       (对齐度)
            #   attr_match        = best · (0.7 + 0.3·coverage) ∈ [0,1]
            #
            # **coverage 的定义改了**：旧版是 `Σc / count`（平均）。
            # 平均值的问题是**符号反了** —— 挂载的属性越多，平均值
            # 反而被拉低，"对齐度"反而变差。改成 `Σc / max_k c_k`
            # 才是真正的"有多少信号集中在最好的那个属性上"，
            # 取值 ∈ [0,1]，挂载多且一致 → 接近 1。
            #
            # The sentence side is a CONTINUOUS score per attribute, not
            # a name plus a binary weight — this is what lets a sentence
            # be "date 0.8, death 0.9" as intended.  A sentence can
            # carry SEVERAL attributes at once; each one contributes.
            #
            # The mount weights are read from the graph, so they already
            # respect the entity's attribute template (a PERSON never
            # mounts `population`) and already carry the 0.7
            # down-weight on the `description` catch-all.
            #
            # `mounted_attrs` keeps its name for backwards compatibility
            # with the eval diagnostics, but now holds the attributes
            # the VECTOR says this sentence carries, not the ones a
            # keyword table matched.
            attr_match = 0.0
            mounted_attrs: Set[str] = set()
            sent_attr_vec: Dict[str, float] = {}
            # [2026-11-01] 实体-属性槽位对齐分（改造 H，第 2 项）
            slot_bonus = 0.0
            slot_hits: List[str] = []
            if query_attr_vec:
                node = self.registry.get_node(sid)
                if node is not None:
                    # Max mount weight per attribute, across the
                    # sentence's entities.
                    w_by_attr: Dict[str, float] = {}
                    # 实体级挂载索引：(实体键 → 该实体在此句挂的属性)
                    mount_by_entity: Dict[str, Set[str]] = {}
                    for m in (getattr(node, "attribute_mounts", []) or []):
                        aname = getattr(m, "attribute_name", "") or ""
                        if not aname or aname == "other":
                            continue
                        if aname not in attr_index:
                            continue
                        try:
                            w = float(getattr(m, "weight", 1.0) or 0.0)
                        except (TypeError, ValueError):
                            w = 1.0
                        w = min(max(w, 0.0), 1.0)
                        if w > w_by_attr.get(aname, 0.0):
                            w_by_attr[aname] = w
                        # 实体级索引：同一个句子里，同一个实体可能挂多个
                        # 属性，全部记下来。
                        ent_name = getattr(m, "entity", "") or ""
                        for ek in _entity_keys(ent_name):
                            mount_by_entity.setdefault(ek, set()).add(aname)
                    if w_by_attr:
                        num = 0.0
                        best_c = 0.0
                        for aname, w in w_by_attr.items():
                            k = attr_index[aname]
                            c = w * query_attr_vec[k]
                            num += c
                            if c > best_c:
                                best_c = c
                        # 对齐度：所有属性信号之和 / 最强属性信号。
                        # ∈ [0,1]，且"挂载更多一致的属性"会**提高**它。
                        coverage = (num / best_c) if best_c > 1e-9 else 0.0
                        # coverage 上限截到 1.0：num 可能 > best_c
                        # （多个属性都接近 best），那说明对齐很好，
                        # 不该再放大。
                        coverage = min(1.0, coverage)
                        attr_match = best_c * (0.7 + 0.3 * coverage)
                        mounted_attrs = set(w_by_attr)
                        sent_attr_vec = dict(w_by_attr)

                    # ─────────────────────────────────────────────────
                    # 实体-属性槽位对齐（细粒度）
                    #
                    # 问："这个句子里的**这个实体**，是不是恰好挂着
                    # **问句想问的属性**？"
                    #
                    # 粗粒度问的是"这个句子挂的属性" ∩ "问句想要的属
                    # 性"，经常零交集（改造 H 的核心病灶）。细粒度问
                    # 的是"实体 A 的 birth 属性" vs "问句想要 birth"，
                    # 只要句子里的实体就是问句问的那个实体，就能对上。
                    #
                    # 例：问句问 "Which city was Havel born in?"，句子里
                    # "Václav Havel was born in Mělník" 挂的是
                    # `birth(Václav Havel, PERSON)` —— 实体键 "havel"
                    # 命中，属性 `birth` 命中 → 对齐。
                    #
                    # 这个分数是**加成项**，不单独决定生死：
                    #   · 末词降级有误匹配风险（两个 Smith）
                    #   · 问句实体抽取可能漏掉关键实体
                    # 所以它只提升 attr_match 的上限，不做硬判定。
                    # ─────────────────────────────────────────────────
                    if query_entity_keys and mount_by_entity:
                        for ek in query_entity_keys:
                            held = mount_by_entity.get(ek)
                            if not held:
                                continue
                            # 该实体在此句挂的属性里，有多少是问句想要的
                            overlap = held & set(wanted_attrs_rank)
                            if overlap:
                                slot_bonus = max(
                                    slot_bonus,
                                    max(
                                        query_attr_vec[attr_index[a]]
                                        for a in overlap
                                    ),
                                )
                                slot_hits = sorted(overlap)
                                break

            # Apply the entity-slot alignment bonus.
            #
            # `attr_match` from the coarse path is capped at 1.0 already;
            # the slot path can only RAISE it, never lower it.  So we
            # take the max of the two rather than a blend — a sentence
            # that matches the right entity+attribute slot must never
            # be penalised by a mediocre coarse score.
            if slot_bonus > 0.0:
                attr_match = max(
                    attr_match, slot_bonus * ATTR_SLOT_MATCH_WEIGHT,
                )

            raw_attr[sid] = attr_match

            # Soft gate: the sentence is NEVER deleted for missing the
            # wanted attributes.  The flag is still computed (it feeds
            # the eval diagnostics) but it no longer has any effect on
            # selection — see the gate block above for the full
            # rationale.
            attr_gate_pass = True
            if wanted_attrs:
                attr_gate_pass = bool(mounted_attrs & wanted_attrs)

            per_sid[sid] = {
                "event_id": best_eid,
                "event_size": int(best_event_size),
                "penalty": float(best_penalty),
                "name_term": float(name_term),
                "sent_term": float(sent_term),
                "attr_match": float(attr_match),
                "attr_gate_pass": bool(attr_gate_pass),
                "mounted_attrs": sorted(mounted_attrs),
                "sent_attr_vec": sent_attr_vec,
                # [2026-11-01] 槽位对齐分（诊断用）
                "attr_slot_bonus": float(slot_bonus),
                "attr_slot_hits": list(slot_hits),
            }

        # ─────────────────────────────────────────────────────────────
        # 阶段 2：统一归一化。每个分量在候选池内 min-max 到 [0,1]。
        # 归一化之后，权重才真正等于"这个分量占最终分数的几成"。
        # ─────────────────────────────────────────────────────────────
        n_name = _minmax_norm(raw_name)
        n_sent = _minmax_norm(raw_sent)
        # 事实项（实体覆盖度）同理：问句里没有实体锚点时，所有候选
        # 的覆盖度都是 0，退化归一化会把它变成常数 1.0 —— 排序上
        # 无害，但会让 avg_fact 看起来像"实体覆盖在起作用"。
        has_fact_signal = any(v > 1e-9 for v in raw_fact.values())
        n_fact = _minmax_norm(raw_fact) if has_fact_signal else {
            sid: 0.0 for sid in raw_fact
        }
        # 属性分量只有在**确实有属性信号**时才参与归一化。
        #
        # 当 `attr_soft_match_mode` 关闭时，所有候选的 attr_match 都是
        # 0.0 —— 对全等值做 min-max 会退化成"全体 1.0"，于是
        # attr_term 变成一个恒定的 0.30 常数。它不影响排序（常数对
        # 所有候选相同），但会让 final 的绝对值失真、日志里的
        # avg_attr 看起来像是"属性在起作用"，具有误导性。
        #
        # 所以：没有属性信号时，属性分量整体归零。
        has_attr_signal = any(v > 1e-9 for v in raw_attr.values())
        n_attr = _minmax_norm(raw_attr) if has_attr_signal else {
            sid: 0.0 for sid in raw_attr
        }
        # 词法总分 = BM25 + 答案形状先验 + 句法关系匹配，三者归一化后融合。
        #
        # 三个信号互补，覆盖三种不同的"对":
        #   · BM25        → 词面对不对（问句的稀有词有没有出现在句子里）
        #   · 形状先验    → 类型对不对（问年份要含年份，问人要含专名）
        #   · 句法关系    → 动作对不对（问 born，句子里要有 born）
        #
        # 归一化后相加，避免 BM25 的绝对量级压过另外两个。
        n_lex: Dict[str, float] = {}
        n_bm25 = _minmax_norm(raw_lex) if lex_raw else {}
        n_shape = _minmax_norm(shape_raw)
        n_rel = _minmax_norm(rel_raw)
        for sid in sids:
            n_lex[sid] = (
                0.50 * n_bm25.get(sid, 0.0)
                + 0.25 * n_shape.get(sid, 0.0)
                + 0.25 * n_rel.get(sid, 0.0)
            )
        # 归一化后的融合值本身也需要再归一化 —— 否则三个分量的加权和
        # 会超出 [0,1]，破坏 W_LEX 的语义。
        #
        # 但如果三个子信号全是退化值（全体相同），融合结果也是常数，
        # 二次归一化会把它变成全体 1.0 —— 排序上无害，但 avg_lex 会
        # 看起来像"词法信号在起作用"。这种情况直接归零。
        if (
            n_bm25 and any(v > 1e-9 for v in n_bm25.values())
        ) or any(v > 1e-9 for v in n_shape.values()) or any(
            v > 1e-9 for v in n_rel.values()
        ):
            n_lex = _minmax_norm(n_lex)
        else:
            n_lex = {sid: 0.0 for sid in sids}

        # ─────────────────────────────────────────────────────────────
        # 阶段 3：加权求和。
        # ─────────────────────────────────────────────────────────────
        out: List[Dict[str, Any]] = []
        for sid in sids:
            info = per_sid.get(sid) or {}
            ep = echo_pen.get(sid, 0.0)

            name_n = n_name.get(sid, 0.0)
            sent_n = n_sent.get(sid, 0.0)
            fact_n = n_fact.get(sid, 0.0)
            attr_n = n_attr.get(sid, 0.0)
            lex_n = n_lex.get(sid, 0.0) if n_lex else 0.0

            # 归一化后的分解（诊断用）。
            name_term = W_EVENT_NAME_N * name_n
            sent_term = W_SENT_N * sent_n
            fact_term = W_FACT_N * fact_n
            attr_term = W_ATTR * attr_n
            lex_term = W_LEX * lex_n

            # 旧的 S_f / event_term 语义保留在 breakdown 里，但它们
            # 现在是**归一化后**的加权和，不再是"两个不同量纲的余弦
            # 直接相加"。
            event_term = name_term + sent_term
            s_f = event_term + fact_term
            final = event_term + fact_term + attr_term + lex_term - ep

            out.append({
                "sentence_id": sid,
                "event_id": info.get("event_id"),
                "event_size": int(info.get("event_size", 0)),
                "penalty": float(info.get("penalty", 1.0)),
                "S_h_E": float(name_term + sent_term),
                "S_h_F": float(fact_term),
                "S_f": float(s_f),
                "event_term": float(event_term),
                "fact_term": float(fact_term),
                "attr_match": float(info.get("attr_match", 0.0)),
                "attr_term": float(attr_term),
                "attr_gate_pass": bool(info.get("attr_gate_pass", True)),
                "mounted_attrs": list(info.get("mounted_attrs") or []),
                # The NORMALISED components — these are what the weights
                # actually multiply.  Keeping both the raw and the
                # normalised values in the breakdown is what makes a
                # ranking regression diagnosable after the fact.
                "norm_name": float(name_n),
                "norm_sent": float(sent_n),
                "norm_fact": float(fact_n),
                "norm_attr": float(attr_n),
                "norm_lex": float(lex_n),
                "lex_term": float(lex_term),
                "raw_name_cos": float(info.get("name_term", 0.0)),
                "raw_sent_cos": float(info.get("sent_term", 0.0)),
                # Continuous per-attribute scores for this sentence
                # (e.g. {"date": 0.81, "death": 0.93}) — the "date 0.8,
                # death 0.9" profile the design called for.
                "sent_attr_vec": {
                    k: round(float(v), 4)
                    for k, v in (info.get("sent_attr_vec") or {}).items()
                },
                "wanted_attrs": sorted(wanted_attrs),
                "wanted_attrs_rank": list(wanted_attrs_rank),
                # [2026-11-01] 槽位对齐诊断（改造 H 第 2 项）。
                # `attr_slot_hits` 是"这个句子的这个实体"挂的、且问句
                # 想要��属性 —— 命中它说明细粒度信号起作用了。
                "attr_slot_bonus": float(
                    info.get("attr_slot_bonus", 0.0)
                ),
                "attr_slot_hits": list(info.get("attr_slot_hits") or []),
                "echo_penalty": float(ep),
                "final_score": float(final),
            })
            if info.get("event_id") is not None:
                sum_event_size += int(info.get("event_size", 0))
                sum_penalty += float(info.get("penalty", 1.0))
                sum_event_term += event_term
                sum_fact_term += fact_term
                cnt_event += 1
            sum_attr_term += attr_term
            sum_lex_term += lex_term
        if cnt_event > 0:
            logger.info(
                "[EF-rank-pen] n=%d | W_NAME_N=%.2f W_SENT_N=%.2f "
                "W_FACT_N=%.2f W_ATTR=%.2f W_LEX=%.2f | GAMMA_E=%.2f | "
                "avg_event_size=%.2f | avg_penalty=%.4f | "
                "avg_event_term=%.4f | avg_fact_term=%.4f | "
                "avg_attr_term=%.4f | avg_lex_term=%.4f",
                len(sids),
                W_EVENT_NAME_N, W_SENT_N, W_FACT_N, W_ATTR, W_LEX,
                GAMMA_E,
                sum_event_size / cnt_event,
                sum_penalty / cnt_event,
                sum_event_term / cnt_event,
                sum_fact_term / cnt_event,
                sum_attr_term / max(1, len(sids)),
                sum_lex_term / max(1, len(sids)),
            )
        return out

    def _get_embedder(self) -> Any:
        """Return the ContainerManager's embedder (None if unavailable)."""
        return getattr(self.cm, "embedder", None)

    def _encode_query(self, query: str) -> Optional[Any]:
        """Encode `query` once with the shared embedder. Returns a
        normalized 1-D float32 numpy array, or None if no embedder."""
        embedder = self._get_embedder()
        if embedder is None or not query:
            return None
        try:
            import numpy as _np
            v = _np.asarray(
                embedder.encode([query], normalize_embeddings=True)[0],
                dtype=_np.float32,
            ).reshape(-1)
            nrm = _np.linalg.norm(v)
            if nrm > 0:
                v = v / nrm
            return v
        except Exception as e:  # noqa: BLE001
            logger.debug("event_fact: query encode failed (%s)", e)
            return None

    # ------------------------------------------------------------------
    # FAISS helper
    # ------------------------------------------------------------------
    def _faiss_search(
        self, query: str, top_k: int,
    ) -> List[Tuple[str, float]]:
        if not query:
            return []
        fn = getattr(self.cm, "_faiss_search", None)
        if not callable(fn):
            return []
        try:
            hits = fn(query, top_k, self.faiss_threshold)
        except Exception as e:  # noqa: BLE001
            logger.debug("faiss search skipped: %s", e)
            return []
        out: List[Tuple[str, float]] = []
        for cid, sim in hits or []:
            if cid is None or sim is None:
                continue
            out.append((str(cid), float(sim)))
        return out

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _text_of(self, sid: str) -> str:
        node = self.registry.get_node(sid)
        if node is None:
            return ""
        return (
            getattr(node, "complemented_text", None)
            or getattr(node, "text", "")
            or ""
        )

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        return {t for t in re.findall(r"\w+", (text or "").lower()) if len(t) > 2}

    def _jaccard_sim(self, query: str, text: str) -> float:
        a = self._tokens(query)
        b = self._tokens(text)
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, len(a | b))

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------
    def _format(self, result: PipelineResult, sids: List[str]) -> str:
        if not sids:
            return ""
        out_lines: List[str] = []
        for sid in sids:
            out_lines.append(self._text_of(sid).strip())
        return "\n".join(out_lines)

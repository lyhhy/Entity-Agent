# StenRAG Design Document

> **Status:** Current implementation
> **Version:** 0.3.0
> **Last Updated:** 2026-10-30

This document describes the design decisions, rules, and rationale behind every non-trivial judgment made in the StenRAG codebase. It is the **source of truth** for why the system behaves the way it does.

---

## Table of Contents

1. [Intent Classification](#1-intent-classification)
2. [Core / Satellite Classification](#2-core--satellite-classification)
3. [Container Coarse Retrieval (3-tier)](#3-container-coarse-retrieval-3-tier)
4. [Evidence Chain Scoring & Deduplication](#4-evidence-chain-scoring--deduplication)
5. [Cross-Container Expansion (SAME_ENTITY 1-hop)](#5-cross-container-expansion-same_entity-1-hop)
6. [Intent-Guided Edge Selection](#6-intent-guided-edge-selection)
7. [Sentence Multi-Homing](#7-sentence-multi-homing)
8. [Container Splitting](#8-container-splitting)
9. [LLM Summary (Lightweight, Extractive)](#9-llm-summary-lightweight-extractive)
10. [Summary Vector (Semantic Retrieval)](#10-summary-vector-semantic-retrieval)
11. [Entity Resolution Strategy](#11-entity-resolution-strategy)
12. [Decision Glossary](#12-decision-glossary)
13. [Token Cost Estimation](#13-token-cost-estimation)
14. [LLM Integration (D18)](#14-llm-integration-d18)
15. [Attribute Table Rolling Optimisation Agent](#15-attribute-table-rolling-optimisation-agent-d23d28-d31d33)
16. [Rerank Scoring Formula (D34–D38)](#16-rerank-scoring-formula-d34d38)

---

## 1. Intent Classification

**File:** `src/retriever.py`
**Pattern Source:** `_extract_intent()`, `WHY_PATTERNS`, `THEN_PATTERNS`, `COMPARE_PATTERNS`, `WHAT_PATTERNS`

### Decision: Four Intent Types

We deliberately restrict intent detection to four types (WHY / THEN / COMPARE / WHAT). Adding more types (HOW, WHEN, WHO, WHERE, etc.) increases classification noise without proportional retrieval benefit in a paper-research QA setting. Future extensions should be gated by benchmark validation.

### Decision: Regex-Based, No LLM

Intent detection uses regex pattern matching on the raw query string. No LLM or embedding is used. This is because:
- Speed: regex is ~1000× faster than a single LLM call
- Determinism: same query always → same intent
- Interpretability: developers can audit exactly which keyword triggered which intent

### Decision: Tie-Breaking by Score Normalization

Pattern matching returns a **count score** for each intent group, then the highest-scoring intent wins. The exact formula (D1) is:

```python
intent_score = matched_pattern_count / total_patterns_in_group * 2
intent_score = min(intent_score, 1.0)   # cap to [0.0, 1.0]
```

Rationale for `× 2` multiplier:
- The four pattern groups have different sizes (WHY has 7, COMPARE has 10, WHAT has 5).
- Multiplying by 2 lets a single matched keyword dominate when only 1-2 patterns in its group exist, otherwise no single match would exceed 0.5.
- The cap at 1.0 prevents degenerate cases where all patterns match from inflating the score.

The winning intent is the one with the highest score. If all scores are below 0.1, the system **defaults to WHAT** (most common in open-domain QA). Final confidence reported to the caller is the raw score of the winning intent.

**Worked example:** Query `"How does the chip compare to M2?"`
| Intent | Matched | Total | Score |
|---|---|---|---|
| WHY | 0 | 7 | 0.00 |
| THEN | 0 | 8 | 0.00 |
| COMPARE | 1 (`compare`) | 10 | 0.20 |
| WHAT | 0 | 5 | 0.00 |

→ Winner: **COMPARE** (0.20). Without the `× 2` multiplier, the score would be 0.10 — exactly on the threshold and unstable.

### Pattern Rationale

| Pattern Group | Included Keywords | Excluded Keywords | Reason |
|---|---|---|---|
| `WHY` | `why`, `reason`, `cause`, `because`, `result`, `consequence`, `how come` | — | Direct causal intent |
| `THEN` | `then`, `after`, `next`, `later`, `finally`, `result`, `outcome`, `what happened` | — | Temporal sequence intent |
| `COMPARE` | `compare`, `contrast`, `difference`, `different`, `vs`, `versus`, `better`, `worse`, `similar`, `outperform`, `underperform` | — | Comparative intent |
| `WHAT` | `what`, `which`, `describe`, `explain`, `what is` | `how does` | `how does` removed — it overlaps with COMPARE and corrupts the score |

### Critical Rule: `how does` is NOT in WHAT_PATTERNS

A query like `"How does the chip compare to M2?"` contains both `"how does"` (WHAT pattern) and `"compare"` (COMPARE pattern). If both fire, COMPARE (0.18) edges out WHAT (0.0 for everything else). However, placing `how does` in WHAT_PATTERNS artificially inflated the WHAT score and displaced COMPARE as the top intent. The fix is to remove `how does` from WHAT_PATTERNS entirely — `how does` alone is insufficient to infer WHAT intent without other supporting evidence.

**Test:** `"How does the chip compare to M2?"` → COMPARE ✅

---

## 2. Core / Satellite Classification

**File:** `src/graph_builder.py`
**Pattern Source:** `ACTION_VERBS`, `STATE_VERBS`, `SUBORDINATING_CONJUNCTIONS`, `_is_passive()`, `_subject_matches_entity()`

### Decision: No RST, No LLM — SpaCy Syntactic Rules Only

We intentionally avoid both RST parsers (too heavy, RST-only) and LLM classifiers (token cost, latency, non-determinism). SpaCy dependency parsing provides sufficient syntactic signal for a 7-rule deterministic classifier.

### Decision: Rule Priority Order is Fixed (Not Learned)

The rule priority order (described in the table below) was derived from manual annotation of ~20 research paper sentences. Weights/confidences were tuned on that sample. This means:
- **Pros:** No training required, fully deterministic, interpretable
- **Cons:** May miss implicit discourse relations (sentences without explicit connectives are ambiguous)

### Decision: 8 Rules in Priority Order

| Priority | Rule | Outcome | Confidence | Rationale |
|---|---|---|---|---|
| 1 | **Subordinating conjunction at sentence start** | SATELLITE | 0.95 | `"Because..."`, `"Although..."` introduce background/cause, not main claims |
| 2 | **Question mark at end** | SATELLITE | 0.90 | Questions are typically rhetorical/background, not facts |
| 3 | **Passive voice (conservative, root-level only)** | SATELLITE | 0.80 | Passive tends to describe conditions/state; but we exclude subordinate clauses to avoid false positives |
| 4 | **STATE_VERB as root** | SATELLITE | 0.75 | `is`, `was`, `contains`, `has`, `appears`, `remains`, etc. describe properties |
| 5 | **ACTION_VERB as root** | CORE | 0.90 | `published`, `argued`, `developed`, `demonstrated`, `outperforms`, etc. report events/claims |
| 6 | **Subject matches container entity** | CORE | 0.80 | The primary entity of the container is doing the action |
| 7 | **Sentence length > 10 words** | CORE | 0.55 | Longer sentences tend to contain claims; shorter may be fragments |
| 8 | **Fallback** | CORE | 0.40 | Conservative default — better to over-include than miss facts |

### Decision: Conservative Passive Detection

The `_is_passive()` function only considers the **root verb** of the sentence. It explicitly ignores passive constructions inside subordinate clauses (e.g., `"argued that transformers are limited"` → root = `argued`, not `are`). This prevents false positives like S2 in the test suite.

**Why:** SpaCy's dependency tree on `"She argued that transformers are fundamentally limited"` marks `argued` as ROOT and `limited` as ccomp (clausal complement). A naive scan of all VBN tokens would incorrectly flag this as passive.

### Decision: `uses` is in ACTION_VERBS, Not STATE_VERBS

The spec lists `uses` as a state verb. In practice, `"The chip uses 3nm process technology"` is a **core technical claim** about what the chip does. This is a deliberate engineering deviation from the original spec to match the expected classification (CORE). If a sentence uses `uses` to describe a technical specification, it is treated as a CORE action.

### Decision: Pronoun Resolution via `resolved_subject`

When a sentence's grammatical subject is a pronoun (e.g., `"She argued..."`), the `_subject_matches_entity()` function checks whether that pronoun refers to the container entity. This requires the `resolved_subject` field (from the entity resolver) to be passed in. Without this, pronoun subjects would fail Rule 6 and potentially misclassify.

### Decision: Multi-Container Majority Vote

When a sentence belongs to multiple containers (multi-homing), we classify it with respect to **each** container's entity separately, then take a **confidence-weighted majority vote**. The label with the higher summed confidence wins. This ensures the role reflects the sentence's importance across all its topics.

**Example:** `"She argued..."` in container for `Dr. Sarah Chen` → CORE (argued action). If also in a container for `transformers` → CORE (argued action). Total: CORE × 2 → CORE.

---

## 3. Container Coarse Retrieval (3-tier)

**File:** `src/retriever.py`
**Method:** `_retrieve_containers()`

### Decision: Three Cascading Recall Strategies

Container retrieval is intentionally conservative and layered:

| Tier | Method | Recall Target |
|---|---|---|
| 1 | **Entity exact match** | Sentences directly about the queried entity |
| 2 | **Keyword match** (TF-IDF) | Sentences sharing query keywords |
| 3 | **Semantic vector similarity** (spaCy cosine ≥ 0.55) | Top 3 semantically similar containers (catches paraphrases) |

**Why tier 1 first:** Entity matches have near-perfect precision. Starting with entity matches avoids semantic noise for well-formed entity queries.

**Why tier 3 threshold = 0.55:** Empirically tuned. At 0.60, few containers pass. At 0.50, too many irrelevant containers enter the pool. 0.55 is a middle ground that catches `"AI chip"` → `"neuromorphic computing"` paraphrases without flooding with noise.

**Why top 3 for semantic:** The semantic tier is a fallback; capping at 3 prevents it from dominating the result set when entity + keyword matches are already present.

---

## 4. Evidence Chain Scoring & Deduplication

**File:** `src/retriever.py`
**Methods:** `_traverse_graph()`, `_composite_chain_score()`

### Decision: Composite Score Formula

Each evidence chain is scored by:

```
score = base_edge_score × seed_count_weight × path_length_factor

where:
  seed_count_weight = 1 + log(1 + #unique_seeds_producing_this_chain)
  path_length_factor = 0.5 + 0.5 × min(path_len / 3, 1.0)
                     # ranges from 0.5 (1-hop) to 1.0 (≥3-hop)
```

**Why log(seed_count):** Chains produced by multiple seeds are more robust. Log dampens the effect so 2 seeds don't overwhelm 1 seed (log(2) = 0.69 boost).

**Why path_length_factor ≥ 0.5:** Even a 1-hop chain is partially useful. We give it a 50% floor rather than 0, preventing 1-hop chains from being entirely discarded.

**Why `min(path_len/3, 1.0)`:** A 3-hop chain gets the full length bonus; longer chains get capped at 1.0 to prevent domination by very long chains.

### Decision: Deduplication by Sentence-ID Tuple

Chains are deduplicated by converting each chain's sentence IDs to an ordered tuple and comparing tuples. If two chains share the same sentence sequence (even if discovered via different seeds), the higher-scoring one is kept and the seed count is incremented for the survivor. This prevents the same evidence from appearing multiple times in results.

---

## 5. Cross-Container Expansion (SAME_ENTITY 1-hop)

**File:** `src/retriever.py`
**Method:** `_traverse_with_entity_edges()`

### Decision: Hard Cap at 1 SAME_ENTITY Hop

Cross-container traversal via SAME_ENTITY edges is **strictly limited to 1 hop**. After the initial jump across containers, the search continues only along **intra-container logic edges**.

**Rationale:**
- A 1-hop cap prevents the search from leaking into unrelated sub-topics across many containers
- Without this cap, SAME_ENTITY traversal could theoretically reach any sentence in the graph, defeating the purpose of container-based organization
- The cap is enforced in code (only one SAME_ENTITY neighbor is explored per seed), not just documented

**What this means for retrieval:** A query about `"Dr. Sarah Chen"` will find evidence chains within the Chen container first. If needed, it can cross to a neighboring container (e.g., `"transformer architectures"`) via ONE SAME_ENTITY edge, then continue within that container via logic edges.

---

## 6. Intent-Guided Edge Selection

**File:** `src/retriever.py`
**Method:** `_get_preferred_edge_types()`

### Decision: Per-Intent Edge Type Priority

| Intent | Preferred Edge Types (in order) |
|---|---|
| WHY | CAUSAL > ELABORATION > TEMPORAL > CONTRAST |
| THEN | TEMPORAL > CAUSAL > ELABORATION > CONTRAST |
| COMPARE | CONTRAST > ELABORATION > CAUSAL > TEMPORAL |
| WHAT (default) | ELABORATION > CAUSAL > TEMPORAL > CONTRAST |

**Rationale:**
- WHY → CAUSAL first: the query asks for causes
- THEN → TEMPORAL first: the query asks for sequence
- COMPARE → CONTRAST first: comparison queries want contrasting facts
- WHAT → ELABALATION first: descriptive queries want elaboration

This is a soft preference, not a hard filter — if a CAUSAL edge doesn't exist, the system falls through to the next type.

---

## 7. Sentence Multi-Homing

**File:** `src/graph_builder.py`, `src/container_manager.py`

### Decision: Sentences Can Belong to Multiple Containers

A sentence is added to a container for **every entity it contains**. This enables cross-container discovery: a sentence about `"Dr. Sarah Chen argued about neuromorphic computing"` lives in both the `Dr. Sarah Chen` container AND the `neuromorphic computing` container.

**Trade-off:**
- ✅ Enables cross-container SAME_ENTITY edges (semantic bridging)
- ✅ Improves recall for entity-centric queries
- ❌ Increases container index maintenance cost
- ❌ May cause duplicate evidence if not deduplicated at retrieval

The deduplication logic (Section 4) mitigates the duplicate evidence issue.

---

## 8. Container Splitting

**File:** `src/container_manager.py`
**Methods:** `_check_split_needed()`, `_cluster_sentences()`

### Decision: Split When Inter-Group Similarity < 0.4

When a container exceeds 5 sentences, it is checked for thematic sub-grouping using agglomerative clustering on spaCy sentence vectors.

**Parameters:**
| Parameter | Value | Rationale |
|---|---|---|
| `MAX_SENTENCES` | 5 | Keeps containers focused; below this threshold, splitting is not needed |
| `MAX_SENTENCES_WITH_SUBTOPICS` | 8 | Allows expansion for multi-topic entities without indefinite growth |
| `SPLIT_SIMILARITY_THRESHOLD` | 0.4 | If two sub-groups have cosine similarity ≥ 0.4, they are thematically related enough to stay together |

**Clustering method:** Agglomerative clustering with average linkage and cosine distance (`1 - cosine_similarity`).

---

## 9. LLM Summary (Lightweight, Extractive)

**File:** `src/container_manager.py`
**Methods:** `_update_with_llm_summary()`, `_update_keywords_summary()`

### Decision: Template-Based Summary Without LLM API

When `use_llm_summaries=False` (default), summaries are generated as:

```
"{entity} - relates to {keyword1}, {keyword2}, {keyword3}"
```

Example: `"Dr. Sarah Chen - relates to sarah, chief, scientist"`

This is an **extractive template** — the content comes directly from TF-IDF keywords, reformatted for readability. It is **not** a generative LLM summary.

### Decision: Optional Real LLM Hook

When `use_llm_summaries=True` and a `_llm_summarize(entity, context)` callable is registered on the ContainerManager, the system will:
1. Collect up to 5 sentences from the container as context
2. Call `llm_summarize(entity, context)` to generate a natural-language summary
3. Fall back to the template if the call fails

This allows production deployments to plug in OpenAI/Anthropic without changing the interface.

---

## 10. Summary Vector (Semantic Retrieval)

**File:** `src/container_manager.py`, `src/retriever.py`
**Methods:** `_embed_text()`, `_vector_similarity()`

### Decision: spaCy Document Vector as Summary Embedding

Each container's `summary` text is embedded using `spaCy`'s built-in document vector (`doc.vector`), which is the **mean of its token word vectors** (300-dimensional for `en_core_web_lg`).

**Why spaCy and not a separate embedding model:**
- No additional API calls or model downloads
- Already loaded for entity extraction and dependency parsing
- Sufficient for cosine similarity on short summary strings

**Why not store full sentence vectors:** Each sentence already has a representation in the sentence registry; the summary vector is specifically for **container-level coarse retrieval** — matching a query to the best container before diving into individual sentences.

---

## 11. Entity Resolution Strategy

**File:** `src/entity_resolver.py`

### Decision: Longest-Surface-Form Wins

When multiple surface forms resolve to the same entity, the **longest registered form** becomes the canonical name. Shorter forms map to it.

**Example:**
- Register `"Dr. Sarah Chen"` → canonical = `"Dr. Sarah Chen"`
- Register `"Sarah Chen"` → maps to `"Dr. Sarah Chen"`
- Register `"Dr. Chen"` → maps to `"Dr. Sarah Chen"`

**Why longest-first:** In academic writing, `"Dr. Sarah Chen"` is more specific than `"Sarah Chen"`. Using the longest form as canonical minimizes ambiguity.

**Implication for Core/Satellite:** When `"She argued..."` is complemented to `"Dr. Sarah Chen argued..."`, the `resolved_subject` field carries the canonical entity name, enabling Rule 6 (subject matches container entity) to fire even when the original text used a pronoun.

---

## 12. Decision Glossary

A quick-reference table of every non-trivial design decision in the codebase:

| ID | Category | Decision | File | Default |
|---|---|---|---|---|
| D1 | Intent | 4 intent types (WHY/THEN/COMPARE/WHAT), regex-based | `retriever.py` | WHAT fallback |
| D2 | Intent | `how does` excluded from WHAT_PATTERNS | `retriever.py` | N/A |
| D3 | Intent | COMPARE includes `outperform/underperform` variants | `retriever.py` | N/A |
| D4 | Role | 8 rules, priority order fixed (not learned) | `graph_builder.py` | CORE fallback |
| D5 | Role | Conservative passive: root-level only | `graph_builder.py` | `_is_passive()` |
| D6 | Role | `uses` in ACTION_VERBS (spec deviation) | `graph_builder.py` | N/A |
| D7 | Role | Pronoun subject matches via `resolved_subject` | `graph_builder.py` | N/A |
| D8 | Role | Multi-container: confidence-weighted majority vote | `graph_builder.py` | N/A |
| D9 | Retrieval | 3-tier container recall (entity/keyword/semantic) | `retriever.py` | Entity first |
| D10 | Retrieval | Semantic threshold = 0.55, top-3 cap | `retriever.py` | 0.55 |
| D11 | Retrieval | WHAT: CORE-only pruning; WHY/HOW: CORE-first ranking | `retriever.py` | N/A |
| D12 | Scoring | Composite score: `base × (1+log seeds) × (0.5+0.5×min(len/3,1))` | `retriever.py` | N/A |
| D13 | Scoring | Deduplication by ordered sentence-ID tuple | `retriever.py` | N/A |
| D14 | Traversal | SAME_ENTITY 1-hop hard cap | `retriever.py` | 1 hop |
| D15 | Traversal | Per-intent edge type priority table | `retriever.py` | See Section 6 |
| D16 | Multi-homing | Sentence in all entity-containers it mentions | `graph_builder.py` | All entities |
| D17 | Container | Split at similarity < 0.4, max 8 sentences | `container_manager.py` | 0.4 / 8 |
| D18 | Summary | Template: `"{entity} - relates to {kw1}, {kw2}, {kw3}"` | `container_manager.py` | Template |
| D19 | Summary | spaCy doc.vector for semantic matching | `container_manager.py` | 300-dim |
| D20 | Entity | Longest surface form → canonical | `entity_resolver.py` | Longest-first |
| D21 | Keywords | Top 3 TF-IDF keywords per container | `container_manager.py` | 3 |
| D22 | Confidence | `role_confidence = weighted_sum / total_weight` | `graph_builder.py` | N/A |
| D23 | Attribute | 属性表为运行期 schema 增量，agent 不改源码 | `attribute_schema.py` | Delta over static tables |
| D24 | Attribute | 缺口在**问句**侧检测（独立于句级词干表） | `attribute_agent.py` | `QUESTION_TRIGGERS` |
| D25 | Attribute | 候选属性/类词表**有界**，agent 不得凭空发明 | `attribute_schema.py` | Candidate lexicons |
| D26 | Attribute | 三准则把关：包容性 / 挂载效率 / 区分度 | `attribute_agent.py` | See §15 |
| D27 | Attribute | 度量必须复刻真实挂载（模板受限 + 先加权后排序） | `attribute_agent.py` | Mirrors `soft_mount` |
| D28 | Attribute | LLM 仅两处兜底：同大类多实体归属 + 缺口确认 | `attribute_agent.py` | Optional |
| D29 | Attribute | agent 增量必须被指标验证：一条命令「优化→应用→测量」，A/B 模式下先测 baseline 再应用增量 | `quick20_eval.py` | See §16 |
| D30 | LLM | `.env` 在 `make_client()` 内自动加载；优先级 显式参数 > 环境变量 > `.env`；启动时打印脱敏配置 | `llm_client.py` | See §17 |
| D31 | Attribute | 属性表可增可减：挂载效率低于门槛的条目必须被**丢弃**，用与接受门完全相同的 `_measure` 度量 | `attribute_agent.py` | See §15 |
| D32 | Attribute | 删除以**墓碑**记录（随 JSON 持久化），否则基础表会在重新加载时复活被删条目 | `attribute_schema.py` | Tombstones |
| D33 | Attribute | 剪枝范围默认只限 agent 自己新增的条目（`prune_scope="delta"`），剪基础表需显式 `--prune-scope all` | `attribute_agent.py` | See §15 |

### Decision: Rerank 公式的量纲归一化与信号接回（D34–D38）

以下五条决策是 2026-10-30 的一次精排修复，针对的实测现象是：
**启用 `--attr-soft-match` 后 R@1/R@5/R@10 全部上升，但 EM/F1 下降。**

这个组合症状本身就是诊断结论：属性信号在**召回层**是有效的，
问题出在**排序层** —— 属性把对的句子带进了候选池，却没能把它排到第一。

| 编号 | 主题 | 决策 | 文件 | 关键取值 |
|---|---|---|---|---|
| D34 | Rerank | 所有打分分量先在候选池内 **min-max 归一化**到 [0,1]，再加权求和 | `stengraph_pipeline.py` | `_minmax_norm()` |
| D35 | Rerank | **事实项改为问句实体覆盖度**，不再复用句子余弦（原公式把同一余弦算了两遍） | `stengraph_pipeline.py` | `W_FACT_N` |
| D36 | Rerank | **词法信号接回打分链路**：BM25 + 答案形状先验 + 句法关系匹配，三者归一化后融合 | `stengraph_pipeline.py` | `W_LEX` |
| D37 | Gate | 属性硬门**设保底下限**：至少保留 `ATTR_GATE_MIN_KEEP` 个候选，不足则放宽属性切面 | `stengraph_pipeline.py` | `floor = min(ATTR_GATE_MIN_KEEP, top_k)` |
| D38 | Rerank | 属性 `coverage` 定义由 `Σc/count` 改为 `Σc/max(c)`（旧定义的**符号是反的**） | `stengraph_pipeline.py` | `coverage ∈ [0,1]` |
| D39 | Metric | R@k 的 gold 判定改用数据集的 `supporting_facts`（权威），词边界包含为降级；极性题排除出分母 | `eval/recall_utils.py` | See below |
| D40 | Gate | **硬属性门降级为软门**：属性不再有删除权，只作为 `W_ATTR` 打分分量 | `stengraph_pipeline.py` | 门只观测，不删除 |
| D41 | Attr | **实体-属性槽位匹配**：`attr_match` 增加细粒度对齐项（实体对上 + 属性对上） | `stengraph_pipeline.py` | `ATTR_SLOT_MATCH_WEIGHT` |
| D42 | Gate | **保底保证**：分数最高的 N 句一定进入 LLM 上下文，任何筛选都不能踢掉 | `stengraph_pipeline.py` | `ATTR_TOP_SCORE_GUARANTEE` |
| D43 | Metric | **段落级指标**：Passage Recall@k（口径 A）与 Hit@k（口径 B），与多跳检索论文对齐 | `eval/recall_utils.py` | See below |

#### D43 段落级指标（口径 A / 口径 B）

句子级 R@k 衡量的是"gold **支撑句**有没有被检索到"，但多跳 QA 的
真正瓶颈是**证据段落** —— HotpotQA 每题有 2 个 gold 段落，检索器的
实际任务是"把这两个段落捞进上下文"。

论文里的两种标准口径（都是段落粒度）：

**口径 A —— Passage Recall@k**（MDR / Xiong 2020 最常用）

\[
\text{Recall@k} = \frac{1}{|Q|}\sum_{q}\frac{|\text{top-k} \cap \text{gold passages}_q|}{|\text{gold passages}_q|}
\]

允许只抓到一半证据就得分。top5 里抓到 2 个 gold 段落中的 1 个 →
该样本贡献 0.5。

**口径 B —— Hit@k / Pair Recall@k**（GraphRAG / Query 拆解类 RAG /
FAIR-RAG / LongRAG）

\[
\text{Recall@k} = \frac{|\{q : \text{top-k} \supseteq \text{gold passages}_q\}|}{|Q|}
\]

必须**全部** gold 段落都在 top-k 里才算成功，缺任意一个直接 0。
适合多跳 RAG —— 少任意一篇证据就无法完成推理。

HotpotQA 每题恰好 2 个 gold 段落，所以口径 A ∈ {0, 0.5, 1}，
口径 B ∈ {0, 1}。**两个口径都报**，才能和不同论文的数字对齐。

**排序转换规则**：两个指标都在段落粒度上定义，需要把"句子排序"
转换成"段落排序"。规则是 **段落的位次 = 该段落里排名最高的那个
句子的位次**（first-appearance）—— 一个段落只要有任一句进了 top-k，
它就算被命中。

**极性题不排除**：段落级指标和答案是 yes/no 无关，它只看证据段落
有没有被捞到。所以口径 A/B 的分母是**全部题数**，与句子级 R@k 的
分母（排除极性题）不同 —— 两者不可混用。

| 指标 | 口径 | 定义 |
|---|---|---|
| `R@k`（句子级） | 本项目原有 | gold 支撑句是否在 top-k |
| `passage_recall@k` | A | `\|top-k ∩ gold\| / \|gold\|`，逐题平均 |
| `hit@k` | B | 全部 gold 段落都在 top-k 才算 1 |

---

#### D39 为什么 R@k 的 gold 判定必须重写

R@k 是评估**检索质量**的指标，它的正确性完全取决于"哪一句是 gold"
的判定方式。旧判定是**子串包含**：

```python
gold_norm in _normalize_answer(node.text or "")
```

这个判定在两个方向上都是错的：

**(a) 假阴性 —— 系统性低估 recall。**
gold 是 `'yes'`（HotpotQA 的 "same X?" 类问题）时，整个 context 里
**没有任何一句包含 "yes" 这个子串**。这类题的 R@k **恒为 0**，无论
检索质量多好。在 300 题抽样上，这类题占 **2.3%** —— 它们系统性地
压低 recall，且压低幅度与检索质量无关，也就是**纯噪声**，会让 A/B
对比失去意义。

**(b) 假阳性 —— 判定过松。**
gold 是 `'Matthew Good'` 时，`'matthew good'` 是
`"…on Matthew Good's own imprint…"` 的子串 —— 但那一句讲的是
唱片的发行，不是答案句。300 题抽样里，**25.7%** 的题会匹配到 ≥3 个
句子。

新的判定规则（`eval/recall_utils.py`，唯一真相源）：

| 优先级 | 判据 | 说明 |
|---|---|---|
| 1 | `supporting_facts` 精确匹配 | 数据集自带的 gold 支撑句标注，**权威** |
| 2 | 词边界答案包含 | 降级用；`(?<!\w)gold(?!\w)`，消除 `'Matthew Good's'` 这类假阳性 |

**极性题（gold 是 `yes`/`no`）被排除在 R@k 分母之外**，而不是记 0 分。
理由：`yes`/`no` 不是句子片段，没有任何句子能"包含"它。记 0 会
系统性压低均值，记 1 会虚高 —— 正确做法是排除并在日志里报数。

**这个修复的量化价值**（seed=7，20 题，假设检索完美、gold 排第一）：

| 指标 | 旧判定 | 新判定 |
|---|---|---|
| R@1 上限 | **0.650** | **1.000** |
| R@5 上限 | **0.950** | **1.000** |
| R@10 上限 | **0.950** | **1.000** |

旧指标**即使检索完美也无法报告 R@1 = 1.0** —— 它测的不是检索质量，
而是"答案串恰好落在哪个句子"。

> **注意跨版本可比性**：修复前后 R@k 的**分母不同**（极性题被排除），
> 数字不可直接比较。summary 里新增 `recall_denominator` 字段记录
> 分母与判定方式，必须随指标一起落盘。

> **为什么单独建 `eval/recall_utils.py`**：R@k 的判定逻辑在
> `quick20_eval.py`（2 处）和 `quick20_clean.py`（1 处）各有一份拷贝，
> 而且 `quick20_eval.py` 的 DIAG 块里还有第 4 份（用于解析 gold
> 支撑句）。**同一规则四份实现，正是它们会漂移的原因** —— 改一处
> 漏三处，而且 R@k 路径与 DIAG 路径会开始对"哪句是 gold"产生分歧，
> 于是诊断数据和它要解释的指标互相矛盾。集中到一处之后，三处调用
> 点共享同一个实现。

#### D40–D42 硬属性门为什么被废除

seed=7 / 20 题的 A/B（无属性 → 有属性）：

| | 无属性 | 有属性 | Δ |
|---|---|---|---|
| EM | 0.850 | 0.750 | **-0.100** |
| F1 | 0.850 | 0.750 | -0.100 |
| R@1 | 0.684 | 0.579 | -0.105 |
| R@10 | 0.947 | 0.789 | -0.158 |
| **排序丢失**（RANK-DIAG） | 6 | **2** | **-4 ✅** |
| **门删掉**（RANK-DIAG） | 0 | **5** | **+5 ❌** |

**排序能力反而提升了**（排序丢失 6→2）。恶化 100% 来自"门删掉"。

逐题复盘那 5 道被删的题，发现两道的 gold 的 `final_score` 是**全场
第 1**（0.962 与 1.072）—— **排序器已经把 gold 排到第一，门把它删掉了**。

根因是 wanted 与 mounted **零交集**：

| 题 | gold 挂载的属性 | 门想要的 | 重叠 |
|---|---|---|---|
| 5ab9d2e2 | `achievement, career, education, event` | `location, population, residence` | **0** |
| 5a74cfed | `beliefs, events, founded, headquarters, history, location, members, origin`（8 个） | `birth, career, education` | **0** |
| 5a9094ba | `birth, career, education, leadership, origin` | `characters` | **0** |
| 5abe689a | `event, origin` | `achievement` | **0** |
| 5abaad58 | `[]`（无挂载） | `achievement` | **0** |

`wanted_attrs` 只取 query 侧 embedding 余弦的 top-3，**完全不看
sentence 侧挂了什么**；而 `mounted_attrs` 是"实体类型 × 属性类别"
的粗分类。这是**粒度问题，不是阈值问题** —— 调大调小
`ATTR_GATE_TOP_K` 都治不了"wanted 集合本身是错的"。

**D40 —— 软门。** 属性**不再有删除权**，保留为打分分量（`W_ATTR`）。
不满足的句子只是拿不到属性加权、排序时自然下沉。属性从"准入证"
退回"加分项" —— 这才是排序信号应有的位置。门仍然**观测**并记录
"如果这是硬门会删掉多少"，让误校准的属性表保持可见。

**D41 —— 实体-属性槽位匹配。** 粗粒度问的是"这个句子挂的属性 ∩
问句想要的属性"；细粒度问的是"**这个句子的这个实体**，是不是恰好
挂着**问句想要的属性**"。后者才是原始设计里 `birth(A)` /
`date(EVENT_A)` 的粒度。

实体键用**小写表面形式 + 去重音 + 末词**三套索引（末词降级是因为
HotpotQA 问句常只写姓氏："Which city was **Havel** born in?" vs
句里的 "Václav Havel"）。末词降级有误匹配风险（两个 Smith），所以
槽位分只作**加成项**（`ATTR_SLOT_MATCH_WEIGHT`），不参与删除。

**D42 —— 保底保证。** 旧门的 `ATTR_GATE_MIN_KEEP=3` 保底只在"池子
被掏空"时触发，但误校准时池子往往还有 20+ 候选，门从容删掉 5 个
gold，保底根本不启动。**旧保底保护的是"数量"，不是"质量"。**

新保底保护的是"**排序器的判断**"：分数最高的 N 句（默认 = `top_k`）
一定进入 LLM 上下文，任何后续筛选都不能踢掉它们。这是最后一道
防线，独立于属性逻辑 —— 未来即使再加别的过滤器，也绕不过它。代码里
有显式断言，违反会打 `[EF-guarantee] VIOLATION` 的 ERROR 日志。

> **`attr_gate_kept` / `attr_gate_pass` 字段语义变了**（恒为 True），
> 但**接口不变** —— eval 的 RANK-DIAG 仍然能读旧的结果文件。

#### D34 为什么必须归一化

旧公式把不同量纲的余弦直接相加：

```
final = W_EVENT·(W_NAME·cos(title,q)·pen + W_SENT·cos(sent,q))
      + (1-W_EVENT)·cos(sent,q) + W_ATTR·attr − echo
```

实测典型值：`cos(title,q) ∈ [0.15, 0.35]`，`cos(sent,q) ∈ [0.35, 0.55]`。
直接相加时，**方差大的分量主导** —— 结果是名义上"事件名主导"
（`W_EVENT_NAME=0.7`）的实现里，实际主导的是句子向量，与设计意图
完全相反。同理 `W_ATTR=0.10` 使 `attr_term ∈ [0, 0.10]`，而 final
的动态范围约 0.2–0.6 —— 属性的整个取值范围只相当于句子余弦波动的
1/4，属性表的所有改进在最终排序里几乎不可见。

归一化之后，每个分量都占满 [0,1]，**权重才真正等于"这个分量占
最终分数的几成"**。归一化是单调变换，不改变单个分量内部的排序，
只改变分量之间的相对影响力 —— 这正是需要修的东西。

#### D35 事实项为什么必须换掉

旧 `fact_term = (1-W_EVENT)·sh_f`，其中 `sh_f` 就是 `sent_term`。
**同一个值被用了两次**，等价于把句子余弦乘了个常数，而 name / attr /
lex 三个分量的相对权重被进一步稀释。

新的事实项是**问句实体覆盖度**：候选句覆盖了问句里多少个实体锚点。
HotpotQA 是多跳题，答案句往往同时提到问句里的两个实体；只提到一个的
句子不可能独立回答这个问题。这与句法关系信号正交（句法看动词，
实体覆盖看名词锚点），也与句向量余弦正交。

#### D36 三种词法信号为什么互补

| 信号 | 回答的问题 | 单独用会怎样 |
|---|---|---|
| BM25 | 词面对不对（问句的稀有词有没有出现） | 匹配到停用词/高频词的句子会挤上来 |
| 答案形状先验 | 类型对不对（问年份要含年份，问人要含专名） | 可能把"形状对但内容无关"的句子选上 |
| 句法关系匹配 | 动作对不对（问 born，句子里要有 born） | 词形差异（born/birth）会漏配 |

三者归一化后按 0.50 / 0.25 / 0.25 融合，再整体归一化一次（避免融合值
超出 [0,1] 破坏 `W_LEX` 的语义）。

**这三个信号在本次修复前全是死代码** —— Step 7 被删后
`_score_tfidf`（BM25）与 `_answer_shape_features`（形状先验）失去了
唯一调用者；`_step1_5_query_relations`（句法骨架）抽出来存进
`self._last_query_relations` 后从未被任何打分环节消费。

#### D37 属性门为什么必须设下限

旧逻辑：**只有在门把池子清空时**才放宽。一个"30 个候选只留 1 个"的
门也被当作"正常工作"。

但这正是"recall 涨、EM 跌"的直接机制：属性门决定**候选资格**，一个
校准不准的 wanted-attribute 集合会在排序器看到 gold 句**之前**就把它
删掉。活下来的句子都是 on-topic 的（recall 上升），但带着答案的那句
没了（EM 下降）。

新的保底：`floor = min(ATTR_GATE_MIN_KEEP, top_k)`，不足则按
top-3 → top-5 → top-8 → 保底分数排序 逐级放宽。

#### D38 coverage 的符号问题

旧定义 `coverage = Σc_k / count`（平均值）的**符号是反的**：挂载的属性
越多，平均值反而被拉低，"对齐度"反而变差。

新定义 `coverage = Σc_k / max_k c_k ∈ [0,1]`，含义是"有多少信号
集中在最好的那个属性上"。挂载多个且一致的属性会**提高**它。

---

## 13. Token Cost Estimation

**File:** N/A (analytical model, computed against `retriever.retrieve()` end-to-end)

### Decision: Track Per-Query Token Consumption

Retrieval is **NOT free** even though graph building uses no LLM. To compare fairly against SentGraph and other RAG baselines, we estimate the token cost of one query:

| Stage | Operation | Approximate Tokens | Notes |
|---|---|---|---|
| 1 | **Intent detection** (regex) | **0** | Pure regex on query; no API call |
| 2 | **Container retrieval** (3-tier) | **0** | spaCy vectors only, no API call |
| 3 | **Seed sentence identification** | **0** | Local in-memory iteration |
| 4 | **Graph traversal + scoring** | **0** | Local graph operations |
| 5 | **Evidence chain → LLM prompt** | **150-300** | Top 1-3 chains × ~50 tokens each (text + role + entities) |
| 6 | **LLM answer generation** | **500-800 output** | DeepSeek `deepseek-chat` typical answer length |
| 7 | **System prompt overhead** | **~50** | Instructions + intent type + container list |

### Total Cost

```
Per-query token cost = 0 + 0 + 0 + 0 + 150-300 (input) + 500-800 (output) + 50 (overhead)
                    ≈ 700-1150 tokens per query
```

### Comparison with SentGraph

| System | Avg Tokens/Query | Source |
|---|---|---|
| SentGraph (RST + LLM) | ~2000-3000 | Per their paper §5.3 |
| **StenRAG** | **~700-1150** | This estimation |
| Reduction | **~60-65%** | 2-3× fewer tokens |

The savings come from:
- Intent/role classification is local (no LLM)
- Container summary is local (no LLM)
- Only the **final answer generation** stage hits the LLM, and it's called once with a tight, evidence-bounded prompt

### Caveats

- Token counts are **prompt-side + completion-side** totals; pricing varies by provider.
- If `use_llm_summaries=True` (D18 LLM hook active), add **~200-400 tokens per container** at build time (one call per container). This is **amortized** — paid once during graph build, not per query.
- The 150-300 input tokens for the LLM prompt assume the top-3 evidence chains fit. Larger chains push the count up.

---

## 14. LLM Integration (D18)

**File:** `src/container_manager.py`, `src/llm_client.py`
**Hook:** `_update_with_llm_summary()`, callable injection via `set_llm_summarizer()`

### D18 Clarification: Template vs LLM is a Switch, Not Both

The default summary path is the **template**:
```
"{entity} - relates to {kw1}, {kw2}, {kw3}"
```

This is what runs when `_llm_summarize` is not registered on the `ContainerManager`. The LLM hook is **opt-in**: you set `use_llm_summaries=True` on the `GraphBuilder` AND register a summarizer callback. Neither alone is sufficient.

### Decision: Pluggable LLM Client (OpenAI-Compatible Interface)

We do NOT hard-code any specific provider. Instead:

```python
# 1. Create a callable with signature (entity: str, context: str) -> str
def my_summarizer(entity: str, context: str) -> str:
    return call_deepseek_or_openai(prompt=...)

# 2. Inject into ContainerManager
builder.container_manager.set_llm_summarizer(my_summarizer)

# 3. Rebuild container summaries
builder.container_manager.use_llm_summaries = True
for c in builder.container_manager.get_all_containers():
    builder.container_manager.update_container_index(c.id)
```

The hook follows the OpenAI-style Chat Completions signature, which is compatible with:
- DeepSeek (`https://api.deepseek.com/v1`)
- OpenAI (`https://api.openai.com/v1`)
- Azure OpenAI (with custom base_url)
- Local vLLM / Ollama (with OpenAI-compatible server)
- Any provider implementing the OpenAI Chat Completions schema

### Decision: Graceful Fallback

If the LLM call fails (network error, API key missing, rate limit), the system falls back to the template summary automatically. This is enforced in `_update_with_llm_summary()`:

```python
try:
    llm_summary = self._llm_summarize(container.entity, context)
except Exception:
    llm_summary = None  # fallback to template
```

**Never** crash the build pipeline because of a flaky LLM call.

### When to Use LLM Summaries

| Use Case | Recommendation |
|---|---|
| Small graph (<50 containers), offline research | **Template** (no token cost) |
| Large graph (100+ containers), high-stakes QA | **LLM** (richer semantic matching) |
| Real-time retrieval, low latency | **Template** (LLM adds 200-500ms per container at build) |
| Multi-language or specialized domain (e.g., biology) | **LLM** (better than TF-IDF for non-English stopword filtering) |

---

## 15. Attribute Table Rolling Optimisation Agent (D23–D28, D31–D33)

**File:** `src/attribute_agent.py`, `src/attribute_schema.py`
**Pattern Source:** `AttributeAgent.run()`, `AttributeAgent._gate()`, `AttributeAgent._measure()`, `AttributeAgent._prune()`

### Context: The Context-Memory Design Stage

The attribute table decides, for every sentence, WHICH property it
carries ("birth", "date", "genre", ...) and FOR WHICH entity.  A missing
attribute means the memory cannot answer a question about that property;
a missing entity class means every sentence about that kind of entity
falls back to the `description` catch-all and carries no usable signal.

The table was hard-coded in `sentence_graph.py` as module-level
constants.  The design stage requires an agent that ROLLS it: sample
questions, notice what is missing, patch the table, and re-measure.

### Decision: The Table Is a Runtime Delta, Not Source Code

`AttributeSchema` holds the agent's changes as a DELTA and applies them
to three places that must agree:

1. the module-level tables in `sentence_graph` (so registries created
   later inherit the change);
2. an already-built `AttributeRegistry` (so containers that exist right
   now see the change);
3. the container attribute slots (so a new attribute can be mounted
   immediately).

**Why not rewrite the source file?**  The constants are interleaved with
the legacy regex tables, `ATTRIBUTE_NAMES` must stay in sync with
`ATTRIBUTE_DESCRIPTORS` (both define the attribute embedding matrix
layout), and an agent that rewrites source files on every iteration is
unreviewable.  The delta is persisted as JSON; `export_python()` emits a
pasteable snippet, and a human decides whether to make it permanent.

**Critical implementation note:** the module tables are mutated **IN
PLACE**.  `graph_builder` and `stengraph_pipeline` import
`ATTRIBUTE_NAMES` and `ATTRIBUTE_DESCRIPTORS` by name at import time;
rebinding the module attribute would leave them pointing at the old
list while the agent believed it had applied the change.

### Decision: Gaps Are Detected On The Question Side

The tables in `sentence_graph` carry SENTENCE-level stems ("born",
"died").  Gap detection works on the QUESTION side: "When was X born"
asks for `birth` even though the question never contains "born" on its
own.  `QUESTION_TRIGGERS` is therefore a separate table, keyed by
attribute, holding the phrasings a question uses.

### Decision: The Candidate Lexicons Are Bounded

The agent may only propose attributes and classes that appear in
`ATTRIBUTE_CANDIDATE_LEXICON` / `CLASS_CANDIDATE_LEXICON`.  An
unbounded vocabulary would let the agent invent an attribute per
sentence — the exact failure the "inclusivity" criterion exists to
prevent.  Every proposal is still subject to the three gates.

### Decision: Three Gates, And What Each One Buys

| Gate | Rule | What it prevents |
|---|---|---|
| **Inclusivity** | ≥2 triggers or stems; descriptor not a near-duplicate of an existing attribute (cos ≥ 0.80) | One-off details; duplicate attributes that add a slot but no recall |
| **Mount efficiency** | `mount_count ≥ min_mount_count` AND `global_rate ≥ min_global_mount_rate` | Attributes that occupy a slot in the embedding matrix and in every entity template while answering almost nothing |
| **Distinctiveness** | The attribute's top-K hit rate for its own class must exceed the best rate among the other classes that could plausibly own it, by `min_distinctiveness_margin` | Attributes that rank equally for several classes and therefore discriminate nothing |

**Why the margin is measured against "classes that could plausibly own
it"** (the lexicon's `types` list) rather than against every class:
most classes never mount a `religion` slot at all, so they contribute a
rate of 0.0 and would inflate the margin into meaninglessness.

**Why a duplicate is rejected rather than merged:** a duplicate
descriptor means the existing attribute already covers those sentences;
adding it buys no recall and costs a slot in the matrix that every
entity template pays for.

### Decision: The Owner Class Is Measured, Not Assumed

The gap is DETECTED on the class the question anchors on, but the
attribute is not necessarily OWNED by that class.  `religion` fits
PERSON and NORP equally well; the sentences that carry the fact may
belong to either.

The measurement therefore decides the owner: among the classes the
candidate lexicon allows (`ATTRIBUTE_CANDIDATE_LEXICON[...]["types"]`),
the one on which the attribute wins the most top-K slots becomes its
owner, and the attribute is declared by THAT class.  This is what makes
the "属性挂载句子以后要有区分度是谁的属性" requirement hold at the
class level rather than merely at the sentence level.

**Tie-break:** when two allowed classes win the SAME number of slots,
the class the question anchored on wins.  Without this the tie would be
broken alphabetically, which is arbitrary and would put `religion` on
NORP for a question about a person — even though the question's own
anchor says PERSON.  The anchor is a *preference*, not a guarantee: if
the other class genuinely wins more slots, it takes ownership.

### Decision: The Measurement Mirrors The Real Mount

This is the single most important implementation constraint.  A
candidate that would never win a mount slot in production must not be
accepted, so `_measure()` reproduces `AttributeRegistry.soft_mount()`:

* the ranking set for a sentence of class X is **X's own template**
  (the one it WOULD have, including the candidate) — the real mount
  restricts itself to the attributes the entity declares, so a PERSON
  never mounts `population`;
* the per-attribute weight is applied **BEFORE** the ranking, so the
  `description` catch-all (weight 0.7) cannot occupy a top-K slot;
* only the top `ATTR_MOUNT_TOP_K` attributes count as mounted.

**Why this matters:** judging a candidate by the raw cosine between
the sentence and the descriptor systematically over-estimates it.  The
raw cosine ignores both the template restriction and the catch-all
down-weight, and both change top-K membership.  An earlier version of
the agent used the raw cosine and accepted candidates that the real
mount would never have mounted.

### Decision: The LLM Is A Fallback In Exactly Two Places

Both are optional (`use_llm=False` disables them) and both are bounded
by `max_llm_calls`:

1. **Same-class multi-entity ownership.**  A sentence carrying two
   entities of the SAME class ("Washington and Lincoln") mounts under
   the same attribute for both, because the vector profile carries no
   entity signal.  The LLM decides which entity the attribute belongs
   to.  This is the case the design calls "如果句子包含多实体且在同
   一个大类则调用 llm 判断兜底".
2. **Gap confirmation.**  The deterministic detector fires on question
   trigger vocabulary, which can produce false positives ("when was"
   matches the `founded` trigger for an ORG too).  The LLM confirms
   before the gap is accepted, so the table does not accumulate junk.

Everything else — sampling, gap detection, measurement, gating — is
deterministic.  Sampling derives a per-round seed from `config.seed`,
so a run is reproducible but each round sees different questions.

### Decision: Sentence IDs Are Namespaced Per Question

Sentence IDs restart at `s_0001` in every graph.  Accumulating profiles
across 200 draws without namespacing would count one graph's statistics
200 times, and every corpus-level statistic (mount rate, class hit rate)
would be wrong.  Profiles are keyed `"{qid}::{sid}"`.

### Decision: The Agent Applies The Delta And Re-Mounts

After accepting a gap, the agent applies the delta to the module tables
and re-mounts the last graph's sentences, so the new attributes take
effect immediately rather than only on the next build.  `GraphBuilder`
also accepts an `attribute_schema` argument and applies it **before the
first container is created**, so a persisted delta is part of every
template the builder hands out.

### Decision: The Table Can Also SHRINK — The Prune Pass (D31)

The original requirement is explicit: *"如果这个属性后面挂载句子效率低就
丢弃这个属性或者类"* — an attribute (or a whole entity class) whose mount
efficiency is low must be **DISCARDED**, not merely rejected on arrival.

An agent that only adds is not a rolling optimisation, it is a
monotonically growing table.  Every stale attribute keeps occupying a
slot in the attribute embedding matrix, and every stale class keeps
occupying a template slot, and both cost recall forever.

**The prune pass (pass 5) uses the SAME `_measure()` as the acceptance
gate.**  An entry is pruned exactly when it would have been rejected on
arrival.  That symmetry is the whole point: *what the agent proposes, it
also audits*.  Without it, the agent could accept an attribute whose
mount efficiency depends on a rare corpus feature, and never notice
when that feature stopped being rare.

**Ordering:** the prune pass runs on a **COPY** of the schema, after the
accepted proposals have been added to that copy but before anything is
applied to the live tables.  So:

* a newly accepted attribute is audited in the same run it was accepted —
  if it immediately fails the efficiency bar it is dropped before it can
  occupy a slot in the embedding matrix;
* a dry run (`apply=False`) never mutates the live delta;
* a pruned attribute that was accepted in the same run is NOT resurrected
  by the apply pass — the tombstone wins.

**Scope (`prune_scope`):**

| Scope | What it may prune | Default |
|---|---|---|
| `delta` | only the entries **this agent added** (the schema delta) | ✅ |
| `all` | every entry in the table, including the hand-designed base ones | — |

`delta` is the default because deleting a hand-designed base attribute
is a decision that should be explicit.  `all` is available for the case
where the base table itself has decayed.

**Deletion is a tombstone, not a `del`.**  The delta is persisted to
JSON and reloaded on the next run.  A tombstone (`removed_attributes` /
`removed_classes`) survives that round trip; a deleted key does not —
`sentence_graph`'s base table would resurrect the entry.  The tombstones
are applied:

* in `effective_templates()` — after the base table and the delta are
  merged, so a tombstone also removes an attribute the BASE declared;
* in `apply_to_modules()` — the entry is actually popped from
  `ATTRIBUTE_DESCRIPTORS` / `ATTRIBUTE_NAMES` / `_ATTRIBUTE_STEMS` /
  `_ATTRIBUTE_PATTERNS` / `_ATTRIBUTE_TEMPLATES`;
* in `apply_to_registry()` — the live registry stops mounting it;
* in `to_dict()` / `from_dict()` — so it persists;
* in `export_python()` — so a human pasting the snippet also drops it.

**A class carries its attributes with it.**  When a class is pruned, the
attributes that ONLY it declared are pruned too — they would otherwise
be orphans with no owner.  An attribute another class also declares
survives; it still has a home.

---

## 16. The Agent Is Measured, Not Assumed (D29)
**File:** `eval/attribute_agent_eval.py`, `eval/quick20_eval.py`
**Pattern Source:** `run_attribute_agent()`, `_evaluate_sample()`

### Context

D23–D28 define HOW the agent decides.  They do not answer the question
that actually matters: **does adding an attribute help or hurt
retrieval?**  Before this section, the agent produced a JSON delta and
the evaluation harness produced EM/F1, and the two were never connected.
A delta could be accepted by all three gates and still make retrieval
worse — e.g. by adding a slot that occupies the top-K of every sentence
and displaces the attribute the question actually asked for.

### Decision: One Command Runs Optimise → Apply → Measure

`eval/quick20_eval.py` gained the `--attr-agent` flag.  It runs the agent
on the sampled questions, applies the accepted delta to the module
tables, and then evaluates — so the graphs that are measured are graphs
built from the *new* attribute table.

The reusable entry point is `run_attribute_agent()` in
`eval/attribute_agent_eval.py` (the standalone CLI's `main()` is now a
thin wrapper around it).  It accepts either raw HotpotQA records or
already-loaded question dicts, and takes an optional pre-loaded spaCy
pipeline so the evaluation harness does not load `en_core_web_lg` twice.

### Decision: A/B Mode Runs The Evaluation Twice, In A Fixed Order

`--attr-agent-ab` measures the delta directly:

1. **baseline** — evaluate the sample with the CURRENT attribute table;
2. **agent** — run the agent, apply the accepted delta;
3. **treatment** — evaluate the same sample again;
4. print the per-metric delta and write `<out>_ab.json`.

**The order is a hard constraint.**  Containers draw their attribute
slots from the module-level templates in `sentence_graph`, so the
"currently effective attribute table" is the only thing that differs
between the two runs.  Running the agent first would make both sides
use the new table and the delta would be identically zero.

The evaluation loop was extracted into `_evaluate_sample()` for this —
returning `(summary, records)` instead of writing files — so both sides
of the comparison share exactly the same code path and the same sample.

### Decision: The Delta Stays A Delta

The agent still does not rewrite source files.  The accepted delta is
written to JSON (`--attr-agent-out`) with a pasteable Python snippet
beside it; whether to make it permanent stays a human decision.  A/B
mode exists to give that human the evidence.

---

## 17. LLM Configuration Is Auto-Loaded (D30)

**File:** `src/llm_client.py`
**Pattern Source:** `load_env_file()`, `make_client()`, `describe_llm_config()`

### Context

The evaluation scripts call `make_client()` with no arguments.  Before
this section, `make_client()` read `os.environ` only — the repo-root
`.env` was never loaded by any code path.  The README told the user to
call `load_dotenv()` manually, but no script did it, so the observable
result was that `make_client()` returned a **disabled** client and every
LLM-dependent feature silently degraded to a heuristic fallback.

That failure mode is invisible in the logs (the client is disabled
rather than erroring) and it makes an A/B measurement meaningless: the
"treatment" run would look identical to the "baseline" run because
neither actually used an LLM.

### Decision: `.env` Is Loaded Inside `make_client()`

`_ensure_env_loaded()` reads the repo-root `.env` exactly once per
process, on the first `make_client()` / `DeepSeekClient()` call.

**Precedence: explicit argument > real environment variable > `.env`.**
An already-exported environment variable is never overwritten by the
file, so a single-run override still works:

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."; python eval/quick20_eval.py
```

`python-dotenv` is used when installed; otherwise a ~30-line built-in
parser (`_parse_dotenv`) handles the subset of the format that is
actually needed (`KEY=value`, `export KEY=value`, quotes, comments).
The fallback exists because the project rule forbids new heavy
dependencies — but a *missing optional* package must not become a
missing feature.

### Decision: The Configuration Is Logged, Secret-Free

`describe_llm_config()` returns a one-line summary of WHICH endpoint
and model will be used and whether a key is present — never the key
itself.  Both eval scripts log it at startup, so a run's log states
whether the LLM path was live.  `llm_available()` and
`local_llm_configured()` give the same signal programmatically.

**Why this matters for the measurements:** a run whose LLM was silently
disabled looks exactly like a run whose LLM had no effect.  Making the
configuration explicit in the log is what allows the A/B numbers in
§16 to be trusted.

---

## 16. Rerank Scoring Formula (D34–D38)

**File:** `src/stengraph_pipeline.py`
**Pattern Source:** `_event_fact_score()`, `_step10_final_rerank()`, `_minmax_norm()`

### Context: The Symptom That Forced This Rewrite

The measured behaviour with `--attr-soft-match` enabled was:

| Metric | Direction |
|---|---|
| R@1 / R@5 / R@10 | **up** |
| EM / F1 | **down** |

That combination is itself the diagnosis.  If recall were up *and* EM
were up, the attribute table would be doing its job end to end.
Recall up + EM down means: **the attribute table successfully brought
the right sentence into the candidate pool, and then the reranker
failed to put it at the top.**

So the fix had to be in the ranking, not in the attributes.

### Decision: The Formula Is Normalised-Then-Weighted

The new formula, computed per query over the candidate pool:

```
final(F) = W_EVENT_NAME_N · norm( cos(event_name, q) · penalty(E) )
         + W_SENT_N       · norm( cos(sentence, q) )
         + W_FACT_N       · norm( entity_coverage(F, q) )
         + W_ATTR         · norm( attribute_match(F, q) )
         + W_LEX          · norm( bm25 ⊕ shape ⊕ relation )
         − echo_penalty(F)
```

Where `norm(·)` is min-max normalisation **over the candidate pool**
(§D34), and `⊕` is the three-way lexical fusion described below.

### Decision: The Five Components Are Deliberately Orthogonal

Each component answers a different question, and none of them can be
substituted by another:

| Component | Question it answers | Signal type |
|---|---|---|
| `norm_name` | Is this the right *paragraph*? | dense (title vs query) |
| `norm_sent` | Is this sentence *about* the query topic? | dense (sentence vs query) |
| `norm_fact` | Does this sentence carry *all* the query's entity anchors? | lexical / structural |
| `norm_attr` | Does this sentence carry the *property* the query asks for? | attribute table (vector) |
| `norm_lex` | Is the *wording / type / action* right? | BM25 + shape + syntax |

The old formula had only two effective components (`name`, `sent`),
and `sent` was counted twice.  Everything the system had learned about
attributes, BM25, answer shape, and query syntax was either drowned
out or not wired in at all.

### Decision: Default Weights

| Weight | Default | Env override | Rationale |
|---|---|---|---|
| `W_EVENT_NAME_N` | 0.35 | `STENRAG_W_EVENT_NAME` | Highest single weight — the paragraph is the primary structural signal |
| `W_SENT_N` | 0.20 | `STENRAG_W_SENT` | Topic relevance, weaker than structure |
| `W_FACT_N` | 0.15 | `STENRAG_W_FACT` | Entity coverage breaks ties in multi-hop questions |
| `W_ATTR` | 0.30 | `STENRAG_W_ATTR` | The attribute table is the point of the whole exercise — it must be able to move the ranking |
| `W_LEX` | 0.30 | `STENRAG_W_LEX` | Three independent signals fused; the largest single source of signal diversity |

The weights sum to 1.30, which is intentional and harmless: `final` is
used only for *ordering*, never compared against an absolute threshold.
What matters is the **ratio** between components, and after
normalisation each component occupies exactly [0, 1].

**Why `W_ATTR` went from 0.10 to 0.30:** under the old un-normalised
formula `attr_term ∈ [0, 0.10]` while `final` ranged over ~0.2–0.6.
The attribute signal's *entire* dynamic range was a quarter of the
sentence cosine's swing — no amount of attribute-table improvement
could reliably change the top-1.  After normalisation, `attr_term`
spans [0, 0.30].

### Decision: The Lexical Fusion

Three signals, each normalised, then mixed 0.50 / 0.25 / 0.25, then
the mixture is normalised once more (so it stays in [0, 1] and
`W_LEX` keeps its meaning):

| Sub-signal | Weight | Source |
|---|---|---|
| BM25 | 0.50 | `_score_tfidf()` — was dead code |
| Answer-shape prior | 0.25 | `_answer_shape_features()` — was dead code |
| Syntactic relation match | 0.25 | `_last_query_relations` — was extracted but never consumed |

**The answer-shape prior** asks what shape of answer the question
wants.  If the question contains `when / what year / how many / how
much / how long / how old`, the answer must contain a number, so
sentences containing a year score higher.  Otherwise the answer is
expected to be an entity, so proper-noun density is the signal.

**The syntactic relation match** compares the question's parsed
skeleton (verb / object) against each candidate sentence's tokens,
using **stem matching** so that `born` matches `birth` and `release`
matches `released`.  The verb is weighted higher than the object
because the verb distinguishes "is this sentence answering THIS
question", while the object is often a common noun that appears
everywhere.

### Decision: The Attribute Gate Has A Floor (D37)

The attribute gate is a **deletion**, and it is the most dangerous
thing in the pipeline — it decides eligibility *before* the reranker
gets a vote.

The old rule was "relax only if the gate empties the pool".  Under
that rule a gate keeping 1 candidate out of 30 was considered healthy.
This is precisely the mechanism behind "recall up, EM down": a
mis-calibrated wanted-attribute set removes the gold sentence before
the reranker can rank it, leaving a pool that is uniformly on-topic
(higher recall) but missing the answer (lower EM).

The new rule:

```
floor = min(ATTR_GATE_MIN_KEEP, top_k)
for level in (3, 5, 8):
    passed = candidates whose mounted_attrs ∩ wanted_attrs[:level] ≠ ∅
    if len(passed) >= floor:
        keep passed; break
else:
    keep top-N by final_score   # N = max(floor, ATTR_GATE_MIN_KEEP)
```

The level actually used is recorded in `attr_gate_level`, so a
mis-calibrated gate is visible in the logs rather than silent.

### Decision: Diagnostics Carry Both Raw And Normalised Values

Every scored row keeps `norm_name`, `norm_sent`, `norm_fact`,
`norm_attr`, `norm_lex` **and** `raw_name_cos`, `raw_sent_cos`.  The
weights multiply the normalised values; keeping the raw ones makes a
weight regression diagnosable after the fact instead of guessable.

The `[EF-rank]` log line was renamed accordingly — it used to print
`W_EVENT` and `avg_(1-wE)·S_h_F`, which described the old formula.
A log line that names the wrong weights is worse than no log line,
because it confidently misreports the ranking behaviour.

### Decision: `_minmax_norm` Degenerate Case

When every candidate has the same value for a component, min-max
normalisation would map them all to 0.0 — destroying any
discrimination the component had (and making the whole formula
degenerate).  The helper returns 1.0 for all candidates in that case,
preserving the "no signal either way" semantics.

---

*This document is maintained alongside the code. When a design decision changes, this document must be updated in the same commit.*

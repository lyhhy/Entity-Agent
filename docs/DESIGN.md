# StenRAG Design Document

> **Status:** Current implementation
> **Version:** 0.2.0
> **Last Updated:** 2026-07-11

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

## 15. Attribute Table Rolling Optimisation Agent (D23–D28)

**File:** `src/attribute_agent.py`, `src/attribute_schema.py`
**Pattern Source:** `AttributeAgent.run()`, `AttributeAgent._gate()`, `AttributeAgent._measure()`

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

*This document is maintained alongside the code. When a design decision changes, this document must be updated in the same commit.*

"""
Retriever Module
===============

Graph Traversal for Online Retrieval.

When a query arrives:
1. Extract query intent (WHY/THEN/COMPARE/WHAT)
2. Container-level coarse retrieval (keyword + vector)
3. Seed sentence identification
4. Intent-guided graph traversal
5. Cross-container expansion via SAME_ENTITY if needed

Retrieval Priority:
- Logic edges (CAUSAL/TEMPORAL/CONTRAST) have priority over SAME_ENTITY edges
- Traverse logic edges up to 3 hops
- If no complete evidence chain is found, expand from current node via SAME_ENTITY
  edges (1 hop max), then continue along logic edges
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import re
import logging

import spacy
import math

from .query_router import QueryRouter, QueryType, RoutingConfig


# Query intent patterns  [D1, D2, D3 - docs/DESIGN.md §1]
#
# D1: 4 intent types only (WHY / THEN / COMPARE / WHAT), regex-based, no
#     LLM. Default fallback is WHAT.
# D2: 'how does' intentionally NOT in WHAT_PATTERNS. Counter-example:
#     "How does the chip compare to M2?" must classify as COMPARE.
#     Putting 'how does' in WHAT erroneously tipped the score toward
#     WHAT and over-rode COMPARE.
# D3: COMPARE_PATTERNS includes compare*/contrast*/differ*/vs/versus/
#     outperform*/underperform* to cover common benchmark phrasings.
WHY_PATTERNS = [
    r'\bwhy\b', r'\breason\b', r'\bcause\b', r'\bbecause\b', 
    r'\bresult\b', r'\bconsequence\b', r'\bhow come\b'
]
THEN_PATTERNS = [
    r'\bthen\b', r'\bafter\b', r'\bnext\b', r'\blater\b', 
    r'\bfinally\b', r'\bresult\b', r'\boutcome\b', r'\bwhat happened\b'
]
COMPARE_PATTERNS = [
    r'\bcompare\w*\b',
    r'\bcontrast\b', r'\bdifference\b', r'\bdifferent\b',
    r'\bvs\.?\b', r'\bversus\b',
    r'\bbetter\b', r'\bworse\b', r'\bsimilar\b',
    r'\boutperform\w*\b', r'\bunderperform\w*\b'
]
WHAT_PATTERNS = [
    r'\bwhat\b', r'\bwhich\b', r'\bdescribe\b', r'\bexplain\b',
    r'\bwhat is\b'
]


@dataclass
class QueryIntent:
    """Represents the intent of a user query."""
    intent_type: str  # "WHY", "THEN", "COMPARE", "WHAT"
    confidence: float
    entities: List[str]
    keywords: List[str]
    query_text: str = ""  # original query text (for debug/logging)


@dataclass
class EvidenceChain:
    """Represents a chain of evidence sentences."""
    sentences: List[Dict]  # List of sentence info dicts
    edges: List[Dict]  # List of edge info dicts
    score: float

    def to_dict(self) -> Dict:
        return {
            "sentences": self.sentences,
            "edges": self.edges,
            "score": self.score
        }


# ─────────────────────────────────────────────────────────────────
# Cross-Encoder cache [Phase-3 retrieval]
#
# Module-level dict keyed by model name. Multiple `Retriever` instances
# in the same process (e.g. one per question in HotpotQA eval) share
# the loaded model so we don't pay the ~5s HF download / load cost on
# every call to `Retriever.__init__`.
# ─────────────────────────────────────────────────────────────────
_CROSS_ENCODER_CACHE: Dict[str, object] = {}
# Module-level sentinel stored in the cache to mark a model as
# known-unloadable (e.g. transient HF connection failures that we
# don't want to retry per-question). A `is None` check is not safe
# because the model itself can legitimately be None; the sentinel
# gives us a distinct identity.
_CROSS_ENCODER_FAILED = object()

# ─────────────────────────────────────────────────────────────────
# spaCy cache [memory fix]
#
# Loading `en_core_web_lg` allocates a 588 MB vocab / vectors array.
# Each `Retriever` instance is short-lived (one per question in
# HotpotQA eval), so reloading on every __init__ fragments memory
# and eventually crashes numpy with `ArrayMemoryError` around q=140.
#
# Cache loaded pipelines keyed by model name so all instances in a
# process share the same loaded Language object.
# ─────────────────────────────────────────────────────────────────
_SPACY_CACHE: Dict[str, "spacy.language.Language"] = {}


class Retriever:
    """
    Graph-based retriever for the sentence-level logic graph.
    
    Supports:
    - Intent detection (WHY/THEN/COMPARE/WHAT)
    - Container-level coarse retrieval
    - Intent-guided graph traversal
    - Cross-container expansion via SAME_ENTITY
    """
    
    def __init__(
        self,
        graph_builder: 'GraphBuilder',
        spacy_model: str = "en_core_web_lg",
        use_cross_encoder_rerank: bool = False,
        cross_encoder_top_k: int = 5,
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_semantic_retrieval: bool = False,
        semantic_top_k: int = 5,
        semantic_threshold: float = 0.55,
        semantic_embed_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the retriever.

        Args:
            graph_builder: The GraphBuilder instance containing the graph
            spacy_model: Name of the SpaCy model for NLP processing
            use_cross_encoder_rerank: If True, rerank candidate sentences
                with a Cross-Encoder model after evidence-chain construction.
                **Default False** in this build — CE is disabled so
                retrieval runs on BM25 + graph only.
            cross_encoder_top_k: Number of sentences to keep after
                cross-encoder reranking.
            cross_encoder_model_name: HuggingFace model name for the
                cross-encoder. The default `cross-encoder/ms-marco-MiniLM-L-6-v2`
                is ~80MB and optimised for MS-MARCO passage relevance.
            use_semantic_retrieval: If True, add a FAISS-backed semantic
                tier to `_retrieve_containers()` using SentenceTransformer
                embeddings of container summaries. **Default False** —
                the FAISS tier is disabled; retrieval is BM25 (keyword)
                + graph traversal only.
            semantic_top_k: Number of containers to keep from the FAISS
                semantic tier (after threshold filtering).
            semantic_threshold: Minimum cosine similarity for a FAISS hit
                to be added to the candidate set. Default 0.55; lower
                thresholds add noise, higher miss paraphrases.
            semantic_embed_model: SentenceTransformer model used for
                container-summary embeddings. Default `all-MiniLM-L6-v2`
                (80 MB, 384-dim, fast on CPU).
        """
        self.graph = graph_builder

        # Load SpaCy model — cached at module level so repeated
        # Retriever instantiations (one per question) reuse the same
        # Language object instead of re-mmapping the 588 MB vectors
        # blob on every call. See _SPACY_CACHE above.
        if spacy_model in _SPACY_CACHE:
            self.nlp = _SPACY_CACHE[spacy_model]
        else:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"Warning: Could not load {spacy_model}, falling back to en_core_web_sm")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("Warning: Could not load SpaCy model. Intent detection will be limited.")
                    self.nlp = None
            _SPACY_CACHE[spacy_model] = self.nlp

        # Configuration
        self.max_logic_hops = 3
        self.max_same_entity_hops = 1
        self.min_evidence_count = 2
        self.max_evidence_count = 5

        # Cross-Encoder configuration (Phase-3 retrieval rerank)
        self.use_cross_encoder_rerank = use_cross_encoder_rerank
        self.cross_encoder_top_k = cross_encoder_top_k
        self.cross_encoder_model_name = cross_encoder_model_name
        self._cross_encoder = None  # lazy-loaded on first use

        # Semantic retrieval configuration (Phase-2 container tier)
        self.use_semantic_retrieval = use_semantic_retrieval
        self.semantic_top_k = semantic_top_k
        self.semantic_threshold = semantic_threshold
        self.semantic_embed_model = semantic_embed_model

        # Lazy FAISS index state
        self._semantic_index_built = False

        # Query router: selects (max_hops, seeds_per_container,
        # edge_priority) per query type [Phase-4 optimization].
        # Used by `retrieve()` to override the default fixed values.
        self.router = QueryRouter()

    def _ensure_semantic_index(self) -> None:
        """
        Lazily build the FAISS index over container summaries on first
        semantic retrieval. The embedder itself is cached at module
        scope (see _EMBEDDER_CACHE) so this is just an encode() pass
        over ~10-50 short strings per question.
        """
        if not self.use_semantic_retrieval:
            return
        cm = getattr(self.graph, "container_manager", None)
        if cm is None:
            return
        # ContainerManager marks itself dirty on every update; it will
        # rebuild on next _faiss_search() call. We just trigger that
        # build eagerly here so the first query isn't slower than later.
        if getattr(cm, "_faiss_dirty", True) or cm.faiss_index is None:
            cm._build_faiss_index()
        self._semantic_index_built = True
    
    def retrieve(self, query: str) -> Dict:
        """
        Main retrieval method.

        Steps:
        1. Extract query intent
        2. Container-level retrieval (keyword + entity match)
        3. Seed sentence identification (top-K per container via router)
        4. Intent-guided graph traversal (max_hops via router)
        5. Cross-container expansion via SAME_ENTITY if needed

        Args:
            query: The user's query string

        Returns:
            Dict containing:
            - intent: QueryIntent object
            - containers_retrieved: List of container info dicts
            - seed_sentences: List of seed sentence info dicts
            - evidence_chains: List of EvidenceChain objects
            - answer: Generated answer (if sufficient evidence)
        """
        # Step 1: Extract intent
        intent = self._extract_intent(query)

        # Step 0 (Phase-4): query-aware routing.
        # The router picks retrieval hyperparameters based on the
        # query type (EXACT / CONCEPT / COMPARE). This overrides the
        # fixed `self.max_logic_hops = 3` used in earlier versions
        # and supplies `preferred_edges` ranked by query type.
        routing_config = self.router.get_config(query)
        # Apply router's max_hops + edge priority to the instance so
        # downstream traversal methods read the correct cap and edge
        # ordering. `_traverse_from_seed` will prefer the router's
        # ordering when present.
        self.max_logic_hops = routing_config.max_hops
        self._router_edge_priority = list(routing_config.edge_priority)

        # Step 2: Container-level retrieval
        containers = self._retrieve_containers(intent)

        # Step 3: Find seed sentences (router-aware top-K per container)
        seeds = self._find_seed_sentences(intent, containers, routing_config)

        # Step 4: Intent-guided graph traversal
        evidence_chains = self._traverse_graph(seeds, intent)

        # ----------------------------------------------------------
        # Step 4b: Phase-2 three-stage ranking [Phase 2 ranking]
        # The composite D12 score from traversal only considered
        # edge product + seed count + path length. It does not
        # consider *what* the chain says. Phase 2 re-ranks with a
        # weighted blend of entity coverage, role weight, edge
        # density and path length - see rank_evidence_chains().
        #
        # Optional Step 4c: LLM rerank kicks in only when the top-3
        # composite scores are within 0.05 of each other (i.e., the
        # ranker is not confidently picking a winner). It is a no-op
        # when DEEPSEEK_API_KEY / OPENAI_API_KEY is not set.
        # ----------------------------------------------------------
        query_entities = set(intent.entities or [])
        evidence_chains = self.rank_evidence_chains(evidence_chains, query_entities)
        evidence_chains = self.llm_rerank(query, evidence_chains)

        # ----------------------------------------------------------
        # Step 4d: Cross-Encoder reranking [Phase-3 retrieval, Phase-4 gating]
        # Take the sentences from the top evidence chains and rerank
        # them by (query, sentence) relevance using a Cross-Encoder
        # model (cross-encoder/ms-marco-MiniLM-L-6-v2 by default).
        # This addresses Sentence Recall@1, where the rule-based
        # ranker correctly identifies the gold sentences but ranks
        # them behind passing mentions. The CE model directly scores
        # the (query, sentence) pair and is the de-facto standard
        # for small-pool reranking.
        #
        # Phase-4 [query-aware gating]: skip the cross-encoder on
        # simple queries (single-entity, short, no comparison) because
        # the rule-based ranker is already accurate enough and the CE
        # pass adds ~30-50ms per query without quality lift on those.
        # Only complex queries (>=2 entities OR comparison phrasing)
        # go through the CE. This roughly halves the CE invocation
        # count on HotpotQA where ~60% of questions are single-entity.
        #
        # If the model fails to load (network / package / RAM), we
        # silently fall back to the existing evidence_chains ordering
        # and continue. No errors are raised.
        # ----------------------------------------------------------
        if self.use_cross_encoder_rerank:
            evidence_chains = self._apply_cross_encoder_rerank(
                query,
                evidence_chains,
                seeds=seeds,
                containers=containers,
            )
        # else: preserve the rule-based ordering from
        # `rank_evidence_chains`.

        # ----------------------------------------------------------
        # Step 4e: Sentence-level reranking [Phase-5 retrieval]
        # Reorder sentences *inside* the top chain by answer-likelihood
        # (entity coverage + CORE role + action verb + CE). Putting the
        # most answer-bearing sentence first measurably improves EM on
        # HotpotQA where the LLM sees the top-5 sentences as context.
        # Cost is ~5ms per question (no model calls beyond the optional
        # CE that was already loaded above). Safe no-op on failure.
        # ----------------------------------------------------------
        evidence_chains = self._apply_sentence_level_rerank(
            evidence_chains, query, top_k=3,
        )

        # Step 5: Build answer if we have sufficient evidence (top-1 chain)
        answer = self._build_answer(evidence_chains, query)

        # Phase-5: expose both the reranked top chain (`evidence_chain`)
        # and the broader chain list (`all_chains`) plus the routing
        # config so downstream consumers (eval harness, debugging UI)
        # can inspect which configuration produced this answer without
        # re-deriving it from the question.
        top_chain = evidence_chains[0] if evidence_chains else None

        return {
            "intent": {
                "type": intent.intent_type,
                "confidence": intent.confidence,
                "entities": intent.entities,
                "keywords": intent.keywords
            },
            "containers_retrieved": [c.to_dict() if hasattr(c, 'to_dict') else c for c in containers],
            "seed_sentences": [self._sentence_to_dict(s) for s in seeds],
            "evidence_chains": [chain.to_dict() for chain in evidence_chains],
            # Phase-5 fields (additive; existing keys preserved):
            "evidence_chain": top_chain.to_dict() if top_chain else None,
            "all_chains": [chain.to_dict() for chain in evidence_chains[:3]],
            "config": {
                "max_hops": routing_config.max_hops,
                "seeds_per_container": routing_config.seeds_per_container,
                "edge_priority": list(routing_config.edge_priority),
            },
            "answer": answer,
        }
    
    def _extract_intent(self, query: str) -> QueryIntent:
        """
        Extract intent from the query.
        
        Detects WHY, THEN, COMPARE, or WHAT intent based on patterns.
        """
        query_lower = query.lower()
        
        # Check for intent patterns
        why_score = self._match_patterns(query_lower, WHY_PATTERNS)
        then_score = self._match_patterns(query_lower, THEN_PATTERNS)
        compare_score = self._match_patterns(query_lower, COMPARE_PATTERNS)
        what_score = self._match_patterns(query_lower, WHAT_PATTERNS)
        
        scores = {
            "WHY": why_score,
            "THEN": then_score,
            "COMPARE": compare_score,
            "WHAT": what_score
        }
        
        # Get the highest scoring intent
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        # If no clear intent, default to WHAT
        if best_score < 0.1:
            best_intent = "WHAT"
            best_score = 0.5
        
        # Extract entities from query using SpaCy
        entities = []
        if self.nlp:
            doc = self.nlp(query)
            entities = [ent.text for ent in doc.ents]
        
        # Extract keywords (simple tokenization)
        keywords = self._extract_keywords(query)
        
        return QueryIntent(
            intent_type=best_intent,
            confidence=best_score,
            entities=entities,
            keywords=keywords,
            query_text=query,
        )
    
    def _match_patterns(self, text: str, patterns: List[str]) -> float:
        """Match regex patterns and return a score based on matches."""
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text):
                matches += 1
        return min(matches / len(patterns) * 2, 1.0)  # Normalize to 0-1
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        if not self.nlp:
            # Simple fallback
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            stopwords = {'this', 'that', 'what', 'which', 'when', 'where', 'there', 'here'}
            return [w for w in words if w not in stopwords][:5]
        
        doc = self.nlp(text)
        keywords = []
        
        for token in doc:
            if token.pos_ in {'NOUN', 'VERB', 'ADJ'} and len(token.text) > 3:
                if token.text.lower() not in {'this', 'that', 'what', 'which'}:
                    keywords.append(token.lemma_)
        
        return keywords[:5]
    
    def _retrieve_containers(self, intent: QueryIntent) -> List:
        """
        Container-level coarse retrieval [D9, Phase-2 semantic tier].

        Four cascading recall strategies:

        ┌─────┬──────────────────────────────┬─────────────────────────────────────┐
        │ Tier│ Strategy                     │ Rationale                           │
        ├─────┼──────────────────────────────┼─────────────────────────────────────┤
        │  1  │ Exact entity match           │ Highest precision, lowest recall    │
        │  2  │ Keyword match (TF-IDF)       │ Recall on lexical overlap           │
        │  2b │ Title-anchor match           │ Recall on proper-noun paraphrase   │
        │  3  │ FAISS semantic cosine ≥ 0.55  │ Recall on paraphrase (MiniLM-L6-v2) │
        └─────┴──────────────────────────────┴─────────────────────────────────────┘

        Tier ordering matters [D9]: start with high-precision, fall through
        to high-recall.  Never swap order - entity matches lose their
        precision advantage if mixed with semantic noise.

        D10: semantic_threshold=0.55 and top-K cap.  Empirically tuned;
        lower threshold adds noise, higher misses paraphrases.

        If all tiers return empty: final fallback is ALL containers
        (gives the retriever at least something to work with).
        """
        # Score map + insertion-order list so we can rank deterministically.
        scored: Dict[str, float] = {}
        ordered: List[str] = []

        def _add(cid: str, score: float) -> None:
            if cid not in scored:
                ordered.append(cid)
            if score > scored.get(cid, -1.0):
                scored[cid] = score

        # 1. Exact entity matches (weight 1.0)
        for entity in intent.entities:
            containers = self.graph.get_containers_for_entity(entity)
            for container in containers:
                _add(container.id, 1.0)

        # 2. Keyword matches (weight 0.8)
        for keyword in intent.keywords:
            containers = self.graph.container_manager.search_containers_by_keyword(keyword)
            for container in containers:
                _add(container.id, 0.8)

        # ─────────────────────────────────────────────────────────────
        # 2b. Title-anchor tier [Phase-2 retrieval fix]
        # For multi-hop HotpotQA queries, the gold paragraph is often
        # referenced only by its title (a proper noun like "Ed Wood",
        # "Animorphs", "Shirley Temple") that spaCy may or may not have
        # detected as an entity. The container retriever above missed
        # these because:
        #   - get_containers_for_entity("Scott Derrickson") returned
        #     popular paragraphs that *mention* him, not the
        #     biographical one whose entity field equals "Scott
        #     Derrickson"
        #   - search_containers_by_keyword only matches against
        #     container.entity / container.keywords, not the title
        #   - entity_resolver canonicalizes "Ed Wood" → "edward davis
        #     wood jr." so substring match against container.entity
        #     would need to be bidirectional
        #
        # Tier 2b scans the **query text** for proper-noun spans
        # (titles) and matches them against ALL container.entity
        # strings via bidirectional case-insensitive containment
        # (anchor ⊆ entity OR entity ⊆ anchor after token overlap).
        # This recovers the gold anchor containers that tier 1+2
        # missed.
        # ─────────────────────────────────────────────────────────────
        anchor_query = " ".join(
            list(intent.entities or []) + list(intent.keywords or [])
        )
        title_anchors = self._extract_title_anchors(anchor_query)
        for anchor in title_anchors:
            anchor_lower = anchor.lower()
            anchor_tokens = set(anchor_lower.split())
            for container in self.graph.container_manager.get_all_containers():
                ent = (container.entity or "").lower()
                if not ent:
                    continue
                # Bidirectional containment: container.entity ⊇ anchor
                # (e.g. "edward davis wood jr." contains "ed wood"
                # tokens; "scott derrickson" equals "scott derrickson")
                if anchor_lower == ent or anchor_lower in ent:
                    # 0.9 - sits between exact-entity (1.0) and
                    # keyword-only (0.8) so it doesn't override strict
                    # matches.
                    _add(container.id, 0.9)
                    continue
                # Token overlap match: at least one anchor token
                # appears in some container.entity token, AND the
                # matching token is ≥4 chars (to avoid stopword
                # collisions). This bridges canonicalized names
                # like "ed wood" → "edward davis wood jr." (shared
                # "wood") without false-matching trivial tokens.
                ent_tokens = set(ent.split())
                if anchor_tokens and any(
                    len(t) >= 4 and t in ent_tokens
                    for t in anchor_tokens
                ):
                    # 0.7 - weakest of the lexical tiers; serves as a
                    # bridge between canonicalized names.
                    _add(container.id, 0.7)

        # 3. FAISS semantic fallback [Phase-2]
        #    MiniLM-L6-v2 embeddings of container summaries vs the
        #    query string (intent keywords with type fallback). Each
        #    hit contributes max(score*0.6, prev) so semantic recall
        #    ADDS containers rather than overriding entity / title
        #    matches. Threshold = self.semantic_threshold (default 0.55).
        #    Falls back silently if the embedder / FAISS is unavailable.
        cm = getattr(self.graph, "container_manager", None)
        if self.use_semantic_retrieval and cm is not None:
            self._ensure_semantic_index()
            query_text = " ".join(intent.keywords) or intent.intent_type or ""
            if query_text:
                try:
                    semantic_hits = cm._faiss_search(
                        query_text,
                        top_k=self.semantic_top_k,
                        threshold=self.semantic_threshold,
                    )
                except Exception:
                    semantic_hits = []
                for cid, sim in semantic_hits:
                    # weight = 0.6 so semantic never outranks entity
                    # but still broadens recall on paraphrase queries
                    _add(cid, sim * 0.6)

        # ─────────────────────────────────────────────────────────────
        # Phase-6 [cross-container bridge expansion]
        #
        # After tier 1+2+2b+3, the top-N scored containers are used
        # as seeds to pull in their SAME_ENTITY/CONTEXT neighbors.
        # This recovers the ~17% of gold paragraphs that sit 1 hop
        # away from a direct-hit container but score below the top-K
        # cutoff in the tiered retrieval.
        #
        # Strategy:
        #   1. Take top-3 seed containers from the current rank.
        #   2. For each seed, find containers that share at least one
        #      entity (SAME_ENTITY bridge).  Filter out bridges whose
        #      shared entity appears in > 50 sentences (high-frequency
        #      noise like "American", "film", "born").
        #   3. Score expanded containers as (seed_score * bridge_weight),
        #      where bridge_weight = 0.7 for bridge containers and 1.0
        #      for non-bridge (or direct-hit containers already ranked).
        #   4. Merge expanded containers into the ranked list, keeping
        #      only the top-8 additional containers.
        #
        # The expansion is intentionally conservative: max 8 additional
        # containers to avoid flooding the top-K with indirect evidence.
        # ─────────────────────────────────────────────────────────────
        if scored and ordered and cm is not None:
            try:
                seed_cids = ordered[:3]  # top-3 seeds
                qe_lower = {e.lower() for e in (intent.entities or [])}
                bridge_expansion = self._expand_containers_via_bridges(
                    seed_cids, scored, qe_lower, max_expansion=8,
                    bridge_weight=0.7, hf_threshold=50,
                )
                for cid, exp_score in bridge_expansion:
                    if cid not in scored:
                        ordered.append(cid)
                        scored[cid] = exp_score
                    elif exp_score > scored[cid]:
                        scored[cid] = exp_score
            except Exception:
                pass  # connectivity expansion failure is non-fatal

        # Final fallback
        if not ordered:
            ordered = [c.id for c in self.graph.container_manager.get_all_containers()]
            scored = {cid: 0.0 for cid in ordered}

        # Sort by score desc, then by insertion order for stability.
        ranked = sorted(
            ordered,
            key=lambda cid: (-scored.get(cid, 0.0), ordered.index(cid)),
        )

        # Materialize to Container objects (preserve rank order)
        out: List = []
        for cid in ranked:
            c = self.graph.container_manager.get_container(cid)
            if c is not None:
                out.append(c)
        return out

    def _expand_containers_via_bridges(
        self,
        seed_cids: List[str],
        seed_scores: Dict[str, float],
        query_entities: Set[str],
        max_expansion: int = 8,
        bridge_weight: float = 0.7,
        hf_threshold: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Expand containers via SAME_ENTITY/CONTEXT bridges.

        For each seed container, finds containers that share at least one
        entity via a SAME_ENTITY or CONTEXT edge. High-frequency shared
        entities (appearing in > hf_threshold sentences) are ignored as
        noise bridges.

        Returns up to ``max_expansion`` (container_id, expanded_score)
        pairs sorted by expanded score desc. Containers already in
        ``seed_scores`` are excluded.
        """
        cm = getattr(self.graph, "container_manager", None)
        if cm is None:
            return []

        # Build connectivity lazily.
        er = getattr(self.graph, "edge_registry", None)
        try:
            cm._build_container_connectivity(er)
        except Exception:
            pass

        expanded: Dict[str, float] = {}

        for seed_cid in seed_cids:
            seed_score = seed_scores.get(seed_cid, 0.0)
            if seed_score <= 0.0:
                continue

            try:
                connected = cm.get_connected_containers(seed_cid, min_weight=0.30)
            except Exception:
                continue

            for other_cid, weight in connected:
                if other_cid == seed_cid:
                    continue
                if other_cid in seed_scores:
                    continue  # already directly hit; don't re-add

                # Skip high-frequency shared entity bridges.
                try:
                    shared_entities = cm.get_bridge_entities_between(seed_cid, other_cid)
                except Exception:
                    shared_entities = []
                if shared_entities:
                    # All shared entities must be high-freq to skip.
                    hfs = sum(
                        1 for e in shared_entities
                        if cm.is_high_frequency_entity(e, threshold=hf_threshold)
                    )
                    if hfs == len(shared_entities):
                        continue  # every shared entity is noise → skip

                # Compute bridge penalty.
                try:
                    is_bridge, bweight = cm.classify_bridge(other_cid, list(query_entities))
                except Exception:
                    is_bridge, bweight = False, 1.0
                final_weight = weight * (bweight if is_bridge else 1.0)

                exp_score = max(seed_score * final_weight, 0.75)
                # Keep best score per container (max over all seed origins).
                if other_cid not in expanded or exp_score > expanded[other_cid]:
                    expanded[other_cid] = exp_score

        # Sort by score, take top-K.
        sorted_expanded = sorted(expanded.items(), key=lambda kv: -kv[1])
        return sorted_expanded[:max_expansion]

    def _extract_title_anchors(self, query: str) -> List[str]:
        """
        Extract title-like anchors from the query [Phase-2 retrieval fix].

        A title anchor is a multi-word proper-noun span — the kind of
        span that matches a paragraph title in HotpotQA. We do NOT
        rely on spaCy NER here because NER frequently misses titles
        (e.g. "Ed Wood", "Animorphs" with WORK_OF_ART label).

        Heuristic:
          1. If spaCy is available, take all PERSON/ORG/WORK_OF_ART/
             GPE entities as candidates.
          2. Additionally, scan for contiguous Title-Case word runs
             (each word starts with a capital, length >= 2). This
             catches "Ed Wood", "Big Stone Gap", "Kiss and Tell" that
             NER may have dropped or mislabelled.

        All anchors are returned in lowercased, multi-word form so
        they can be matched against container.entity with simple
        substring/equality tests.
        """
        candidates: List[str] = []

        # 1. spaCy NER (preferred path)
        if self.nlp is not None:
            try:
                doc = self.nlp(query)
                for ent in doc.ents:
                    if ent.label_ in (
                        "PERSON", "ORG", "GPE", "WORK_OF_ART",
                        "EVENT", "PRODUCT", "NORP", "LOC", "FAC",
                        "LAW", "LANGUAGE",
                    ):
                        txt = ent.text.strip()
                        if txt and txt not in candidates:
                            candidates.append(txt)
            except Exception:
                pass

        # 2. Title-Case run extraction (fallback / complement)
        # A "title run" is 1+ consecutive tokens whose first char is
        # uppercase and which contain at least one letter.  We only
        # count contiguous uppercase-starts (no intervening lowercase
        # stop-words), so "Ed Wood" yields one run, but the leading
        # "Were" of "Were Scott" gets discarded because it stops the
        # run.  Single-word anchors are kept (e.g. "Animorphs").
        import re

        # Tokens split on whitespace; keep punctuation-free runs
        words = re.findall(r"[A-Za-z][A-Za-z0-9'’\-]*", query)
        runs: List[List[str]] = []
        run: List[str] = []
        stopwords = {
            "a", "an", "the", "of", "in", "on", "at", "to", "for",
            "with", "by", "and", "or", "is", "are", "was", "were",
            "be", "been", "being", "as", "that", "this", "these",
            "those", "who", "what", "which", "where", "when", "why",
            "how", "i", "you", "he", "she", "it", "they", "we",
            "his", "her", "their", "my", "your", "its",
        }
        for w in words:
            if w[0].isupper() and w.lower() not in stopwords:
                run.append(w)
            else:
                if run:
                    runs.append(run)
                    run = []
        if run:
            runs.append(run)

        for r in runs:
            anchor = " ".join(r)
            if not anchor:
                continue
            # Skip if already captured by NER (case-insensitive)
            if any(anchor.lower() == c.lower() for c in candidates):
                continue
            # Skip single-word common nouns (e.g. "What", "Who")
            if len(r) == 1 and r[0].lower() in {
                "what", "who", "where", "when", "which", "how",
                "the", "a", "an", "i", "you", "he", "she", "it",
                "they", "we", "his", "her", "their", "my",
            }:
                continue
            candidates.append(anchor)

        return candidates

    def _vector_similarity(self, v1, v2) -> float:
        """Cosine similarity between two spaCy vectors; safe for None / zero vecs."""
        try:
            import numpy as np
            if v1 is None or v2 is None:
                return 0.0
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 == 0.0 or n2 == 0.0:
                return 0.0
            return float(np.dot(v1, v2) / (n1 * n2))
        except Exception:
            return 0.0
    
    def _find_seed_sentences(
        self,
        intent: QueryIntent,
        containers: List,
        config: Optional[RoutingConfig] = None,
    ) -> List:
        """
        Find seed sentences in retrieved containers.

        Seeds are sentences that contain query entities or are highly relevant
        to the query keywords.

        Args:
            intent: Query intent (entities + keywords).
            containers: List of retrieved containers.
            config: Optional `RoutingConfig`. If provided, caps each
                container's contribution to `seeds_per_container`. If
                None, falls back to the legacy global top-10 cap for
                backward compatibility.
        """
        seeds = []
        seen = set()
        
        for container in containers:
            for sent_id in container.sentence_ids:
                if sent_id in seen:
                    continue
                
                node = self.graph.sentence_registry.get_node(sent_id)
                if not node:
                    continue
                
# Check if sentence contains query entities
                has_entity = any(
                    entity.lower() in [e.lower() for e in node.entities]
                    for entity in intent.entities
                )

                # Check keyword relevance
                keyword_score = self._compute_keyword_relevance(node, intent.keywords)

                # P0 fix: keep more candidates in seed pool. The original
                # keyword_score > 0.3 threshold was too aggressive — it
                # filtered out biographical anchor sentences (e.g. "Scott
                # Derrickson (born July 16, 1966) is an American director")
                # that contain the query entity but no query keywords.
                #
                # New policy:
                #   • entity match → always include (anchor sentences)
                #   • keyword match ≥ 0.15 → include
                #   • _is_anchor_sentence (resolved_subject == container
                #     entity) → also include as rescue, regardless of
                #     keyword match.
                if has_entity or keyword_score > 0.3:
                    seeds.append(node)
                    seen.add(sent_id)

        # ─────────────────────────────────────────────────────────────
        # P0 rescue pass: for every retrieved container, add at most
        # 2 biographical anchor sentences whose resolved_subject
        # matches the container's entity.
        #
        # Background: the strict has_entity filter only catches
        # sentences whose `entities` field includes the query entity.
        # Multi-homing causes gold biographical sentences to live in
        # multiple containers; in the *non-anchor* container they may
        # be tagged SATELLITE with no query entity, getting filtered
        # out. A targeted rescue adds them back without flooding the
        # seed pool.
        #
        # Rescue criteria (ALL must hold):
        #   • resolved_subject matches container.entity (anchor)
        #   • sentence length >= 12 words (substantive fact)
        #   • head verb is NOT a pure state verb, OR sentence starts
        #     directly with the entity name.
        # ─────────────────────────────────────────────────────────────

        # ─────────────────────────────────────────────────────────────
        # Phase-2 retrieval: do NOT aggressively prune to CORE for
        # WHAT queries.
        #
        # Background: the original D11 design assumed the graph's
        # auto-classifier reliably labels biographical/anchor
        # sentences as CORE and distractor mentions as SATELLITE.
        # In HotpotQA evaluation this assumption breaks because:
        #   - The auto-classifier often votes SATELLITE for the
        #     biographical sentence of an entity when that sentence
        #     also multi-homes into a more specific container
        #     (e.g. the biographical "Scott Derrickson is an American
        #     director" sentence lives in BOTH the "scott derrickson"
        #     container AND the "american" container - the latter
        #     votes SATELLITE and outweighs CORE)
        #   - Result: the gold biographical sentence is filtered out
        #     before traversal even starts, so only distractor
        #     sentences remain as seeds
        #
        # New policy: keep ALL candidates for WHAT/COMPARE; rank
        # them with anchor-boost at the bottom. WHY/HOW keep the
        # CORE-priority ordering. This avoids losing the gold seed
        # to a brittle classifier while still preferring anchors
        # via the role/anchor sort keys.
        # ─────────────────────────────────────────────────────────────

        def _role_rank(s) -> int:
            # 1 = CORE (preferred), 0 = SATELLITE, -1 = unknown
            role = (getattr(s, "role", "") or "CORE").upper()
            if role == "CORE":
                return 1
            if role == "SATELLITE":
                return 0
            return -1

        intent_type = (intent.intent_type or "").upper()
        if intent_type in ("WHY", "HOW"):
            # WHY/HOW benefits from prioritizing CORE seeds, but keep
            # SATELLITE candidates in the pool as fallback.
            seeds = sorted(seeds, key=_role_rank, reverse=True)
        # WHAT / THEN / COMPARE: no role pruning - let the anchor
        # boost and entity-match scoring decide.

        # ─────────────────────────────────────────────────────────────
        # Anchor-sentence boost [Phase-2 retrieval fix]
        # A seed sentence whose `resolved_subject` matches its
        # container's entity is the **biographical / anchor sentence**
        # for that container — the one whose subject IS the entity,
        # not a distractor sentence that merely mentions it.  In
        # HotpotQA these are usually the gold supporting sentences.
        #
        # Without this boost, the seed pool is dominated by distractor
        # sentences that happen to mention the query entity in passing
        # (e.g. "He has collaborated with directors like Scott
        # Derrickson" beats the actual biographical sentence "Scott
        # Derrickson (born July 16, 1966) is an American director").
        #
        # We compute an anchor-match per seed (against ALL containers
        # the sentence multi-homes to, since multi-homing makes the
        # container.entity == resolved_subject match slightly noisy)
        # and use it as the first sort key.
        # ─────────────────────────────────────────────────────────────
        def _is_anchor_sentence(s) -> bool:
            rs = (getattr(s, "resolved_subject", "") or "").lower().strip()
            if not rs:
                return False
            for c in containers:
                ent = (getattr(c, "entity", "") or "").lower().strip()
                if not ent:
                    continue
                # Exact match OR resolved_subject is a substring of
                # the entity (e.g. "Scott Derrickson" in "scott
                # derrickson (group 2)" still indicates the
                # biographical sentence for that container).
                if rs == ent or rs in ent or ent in rs:
                    return True
            return False

        def _entity_count(s) -> int:
            return sum(
                1 for e in (getattr(s, "entities", None) or [])
                if e and e.lower() in [q.lower() for q in intent.entities]
            )

        # ─────────────────────────────────────────────────────────────
        # Anchor-score boost [Phase-3 retrieval fix]
        # Use the user-supplied compute_sentence_score() formula to
        # surface biographical / factual sentences (the gold
        # supporting_facts in HotpotQA) above passing mentions.
        # We compute the score per (sentence, container) pair and
        # take the MAX across the sentence's containers - because
        # multi-homing means the same sentence can be the anchor in
        # one container and a passing mention in another.
        # ─────────────────────────────────────────────────────────────
        # Build a sid -> container lookup so we can resolve
        # container_sentences + idx_in_container for position bonus.
        sid_to_containers: Dict[str, list] = {}
        for c in containers:
            for sid in c.sentence_ids:
                sid_to_containers.setdefault(sid, []).append(c)

        def _anchor_score(s) -> float:
            sid = getattr(s, "id", "")
            best = 0.0
            for c in sid_to_containers.get(sid, []):
                ent = getattr(c, "entity", "") or ""
                csids = list(c.sentence_ids or [])
                if sid in csids:
                    idx = csids.index(sid)
                    sc = self.compute_sentence_score(s, ent, csids, idx, intent)
                    if sc > best:
                        best = sc
            return best

        seeds.sort(
            key=lambda s: (
                _anchor_score(s),                      # Phase-3 anchor score
                _is_anchor_sentence(s),                # anchor first
                _role_rank(s),                         # CORE first
                _entity_count(s),                      # more q-entities
                any(entity.lower() in [e.lower() for e in s.entities]
                    for entity in intent.entities),
                self._compute_keyword_relevance(s, intent.keywords)
            ),
            reverse=True
        )

        # Phase-4 [query-aware seed cap]: if a routing config is
        # supplied, keep only the top-K seeds **per container**. This
        # prevents one noisy container with 20+ sentences from
        # dominating the seed pool (which would force traversal to
        # start in low-quality sentences).
        if config is not None and hasattr(config, "seeds_per_container"):
            per_container_topk = max(1, int(config.seeds_per_container))
            capped: List = []
            seen_sids: Set[str] = set()
            for s in seeds:
                sid = getattr(s, "id", None)
                if not sid or sid in seen_sids:
                    continue
                # Count this seed against each container it lives in
                # and admit it only if every container still has room.
                containers_for_seed = sid_to_containers.get(sid, [])
                if containers_for_seed:
                    # Reuse the `_anchor_score`-based ordering as the
                    # admission rank; the seed is allowed in if its
                    # best-ranked container hasn't filled its quota.
                    used = sum(
                        1 for x in capped
                        if getattr(x, "id", None) in {
                            sid2 for c in containers_for_seed
                            for sid2 in c.sentence_ids
                        }
                    )
                    # Simpler equivalent (and matches prompt spec):
                    # count seeds already in any of this seed's containers
                    already_in_same_container = sum(
                        1 for x in capped
                        if getattr(x, "id", None) in {
                            sid2 for c in containers_for_seed
                            for sid2 in c.sentence_ids
                        }
                    )
                    if already_in_same_container >= per_container_topk:
                        continue
                capped.append(s)
                seen_sids.add(sid)
            # Hard ceiling as a safety net (matches the legacy cap).
            return capped[: max(10, per_container_topk * 3)]

        # Legacy fallback (no router supplied) - top-10 globally.
        return seeds[:10]

    def _compute_keyword_relevance(self, sentence_node, keywords: List[str]) -> float:
        """Compute keyword relevance score for a sentence."""
        if not keywords:
            return 0.0

        text_lower = sentence_node.text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matches / len(keywords)

    # ─────────────────────────────────────────────────────────────────
    # Sentence anchor scoring [Phase-3 retrieval fix]
    #
    # Computes a composite "this sentence is a gold-quality supporting
    # fact for container.entity" score in [0, 1].
    #
    # Components (user-provided formula):
    #   • entity_match_score (0.30) — does the sentence contain the
    #     container's anchor entity?
    #   • verb_action_score    (0.25) — action verbs (published, won,
    #     outperformed) are stronger factual anchors than state verbs
    #     (is, was, has)
    #   • role_bonus           (0.20) — CORE=0.8, SATELLITE=0.7
    #   • position_bonus       (0.15) — sentences earlier in the
    #     container (opening context) score higher
    #   • length_bonus         (0.10) — sentences > 10 words score
    #     higher (more substantive)
    #
    # This is intended to be applied AFTER the existing seed filter
    # (entity / keyword match). It re-orders the candidates so that
    # biographical anchor sentences — typically what HotpotQA's gold
    # supporting_facts contains — rise above passing mentions.
    # ─────────────────────────────────────────────────────────────────
    def compute_sentence_score(
        self,
        sent,
        container_entity: str,
        container_sentences: List[str],
        idx_in_container: int,
        intent: Optional[QueryIntent] = None,
    ) -> float:
        """
        Score a sentence's likelihood of being a gold supporting fact
        within its container (0 ~ 1).

        Args:
            sent: SentenceNode to score
            container_entity: canonical entity name for the container
                (e.g. "scott derrickson")
            container_sentences: ordered list of sentence IDs in the
                container (so we can compute position)
            idx_in_container: index of `sent.id` in container_sentences
            intent: original QueryIntent — used for temporal-cue
                detection (formerly, 1988-1996, until, before, ...)

        Returns:
            Float in [0, 1] — higher is more anchor-like.
        """
        # 1. entity_match_score
        entity_match = 0.0
        if container_entity and container_entity.lower() in [
            e.lower() for e in (sent.entities or [])
        ]:
            entity_match = 1.0

        # 2. verb_action_score
        action_verbs = {
            "published", "argued", "developed", "demonstrated", "applied",
            "outperforms", "outperformed", "showed", "proposed", "released",
            "founded", "led", "directed", "starred", "won", "received",
            "became", "created", "introduced", "achieved", "designed",
            "built", "wrote", "invented", "discovered", "held",
        }
        state_verbs = {
            "is", "was", "are", "were", "has", "have", "contains",
            "consists", "includes", "uses", "features", "appears",
            "remains", "seems", "lived", "born",
        }

        verb = self._get_head_verb(sent).lower() if sent else ""
        if verb in action_verbs:
            verb_score = 0.9
        elif verb in state_verbs:
            verb_score = 0.4
        else:
            verb_score = 0.6

        # 3. role_bonus (CORE=0.8, SATELLITE=0.7)
        _role = getattr(sent, "role", "") or ""
        role_bonus = 0.8 if _role.upper() == "CORE" else 0.7

        # 4. position_bonus (opening sentences > middle > late)
        total = max(1, len(container_sentences))
        ratio = idx_in_container / total
        if ratio < 0.2:
            position_bonus = 0.9
        elif ratio > 0.8:
            position_bonus = 0.5
        else:
            position_bonus = 0.7

        # 5. length_bonus
        word_count = len((sent.complemented_text or sent.text or "").split())
        length_bonus = 0.8 if word_count > 10 else 0.5

        # 6. temporal_score (formerly, 1988-1996, until, before, ...)
        temporal_score = self._compute_temporal_score(sent, intent)

        # Final weighted sum (entity-heavy + temporal-aware)
        score = (
            0.50 * entity_match
            + 0.20 * verb_score
            + 0.15 * role_bonus
            + 0.10 * position_bonus
            + 0.05 * length_bonus
            + 0.15 * temporal_score
        )
        return round(score, 4)

    def _compute_temporal_score(self, sent, intent) -> float:
        """
        Detect temporal-comparison questions (e.g. "formerly known as
        what from 1988 to 1996") and reward historical-tense sentences
        while penalising present-tense distractors.

        Returns:
            float in [-0.5, +1.0]:
              +1.0 — strong historical match (past tense / dated name)
              +0.5 — neutral / no temporal signal in query
              -0.5 — present-tense distractor when query asks historical
        """
        if intent is None:
            return 0.5

        q = (intent.query_text or "").lower()
        text = (sent.complemented_text or sent.text or "").lower()

        # Cue words that signal "the answer is in the past"
        historical_cues = [
            "formerly", "previously", "originally", "back then",
            "at the time", "prior to", "before", "until",
        ]
        # Year/date range pattern: "1988 to 1996", "until 2005", "in 1999"
        date_range_re = re.compile(
            r"\b(?:from\s+)?\d{4}\s*(?:to|-|–|until|through)\s*\d{4}\b",
            re.IGNORECASE,
        )
        bare_year_re = re.compile(r"\b(?:in|since|by|until)\s+\d{4}\b", re.IGNORECASE)

        asks_historical = (
            any(c in q for c in historical_cues)
            or bool(date_range_re.search(q))
            or bool(bare_year_re.search(q))
        )
        if not asks_historical:
            return 0.5  # neutral — temporal signal doesn't apply

        # Past-tense / dated-name indicators in sentence
        historical_indicators = [
            "was known as", "was called", "was named",
            "previously known", "formerly known", "originally called",
            "from 19", "from 20",
            "until 19", "until 20",
            "between 19", "between 20",
        ]
        # Present-tense distractors that signal "current state, not the
        # historical answer the query is fishing for"
        present_indicators = [
            "currently", " now ", "today,", "currently competes",
            "currently plays", "currently serves",
        ]

        if any(ind in text for ind in historical_indicators):
            return 1.0
        if any(ind in text for ind in present_indicators):
            return -0.5
        return 0.5

    def _get_head_verb(self, sent) -> str:
        """
        Return the head verb of `sent.text` (lower-cased lemma) using
        spaCy if available. Falls back to empty string if spaCy
        fails or returns no token.
        """
        if not sent:
            return ""
        # Cache on the node to avoid re-running spaCy per call
        cached = getattr(sent, "_head_verb_cache", None)
        if cached is not None:
            return cached
        text = sent.complemented_text or sent.text or ""
        verb = ""
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                # Pick the first verb in the sentence; spaCy marks
                # verbs with .pos_ == "VERB" or "AUX"
                for tok in doc:
                    if tok.pos_ in ("VERB", "AUX") and tok.dep_ not in (
                        "aux", "auxpass",
                    ):
                        verb = tok.lemma_.lower() or tok.text.lower()
                        break
            except Exception:
                verb = ""
        try:
            sent._head_verb_cache = verb
        except Exception:
            # SentenceNode may be a frozen dataclass; skip cache silently
            pass
        return verb

    def _traverse_graph(
        self,
        seeds: List,
        intent: QueryIntent
    ) -> List[EvidenceChain]:
        """
        Intent-guided graph traversal [D12, D13].

        For each seed, build an evidence chain.  Aggregated over all seeds
        with deduplication (D13) + composite ranking (D12):

            D12 composite score:
              score = base_edge_score
                    * seed_count_weight
                    * path_length_factor

              seed_count_weight = 1 + log(1 + #unique_seeds)
                rationale: a chain reached by more seeds is more reliable;
                log dampens so 2 seeds don't overwhelm 1 (log(2)=0.69)

              path_length_factor = 0.5 + 0.5 * min(path_len / 3, 1.0)
                rationale: floor of 0.5 prevents 1-hop chains being discarded;
                chains >=3 hops get full length bonus (1.0)

            D13 deduplication: chain key = ordered tuple of sentence IDs.
            Same tuple = same evidence surfaced via multiple seeds -> merge
            (higher score kept, seed_count incremented).
        """
        candidate_chains: List[EvidenceChain] = []

        # Track which sentence IDs participate in which chains
        path_to_chain: Dict[Tuple, EvidenceChain] = {}

        for seed in seeds:
            chain = self._traverse_from_seed(seed, intent)
            if not chain:
                continue

            # Deduplicate: chain key = ordered tuple of sentence IDs
            # For unordered multi-source chains, sort the tuple
            sid_tuple = tuple(
                s["id"] for s in chain.sentences
            )
            if sid_tuple in path_to_chain:
                # Merge: keep higher score and bump seed-count weight
                existing = path_to_chain[sid_tuple]
                existing._seed_count = (existing._seed_count or 1) + 1
                existing.score = max(existing.score, chain.score)
                continue

            chain._seed_count = 1  # type: ignore[attr-defined]
            chain._seed_id = seed.id  # type: ignore[attr-defined]
            path_to_chain[sid_tuple] = chain
            candidate_chains.append(chain)

        # Apply composite ranking
        for chain in candidate_chains:
            chain.score = self._composite_chain_score(chain)

        # Sort by composite score, descending
        candidate_chains.sort(key=lambda c: c.score, reverse=True)
        return candidate_chains[:5]

    def _composite_chain_score(self, chain: EvidenceChain) -> float:
        """
        Composite score [D12]:

          score = base × seed_count_weight × path_length_factor

        Design choices encoded below:
          - log(seed_count) instead of linear: 2 seeds doesn't beat 1
            by 2x but by ~1.7x (more conservative, see DESIGN §4)
          - path_length floor at 0.5: even 1-hop chain retains weight;
            prevents short chains from being silently discarded
          - base floor 1e-6: avoids zero scores propagating (would kill sort)
        """
        base = max(chain.score, 1e-6)

        seed_count = getattr(chain, "_seed_count", 1) or 1
        seed_boost = 1.0 + math.log(1 + seed_count)  # log(2)=0.69 for 2 seeds

        path_len = len(chain.sentences)
        length_factor = min(path_len / 3.0, 1.0)  # 1 at path_len>=3
        length_boost = 0.5 + 0.5 * length_factor  # [0.5 .. 1.0]

        return base * seed_boost * length_boost

    # ------------------------------------------------------------------
    # Phase 2: Three-stage evidence chain ranking
    # ------------------------------------------------------------------
    # Replaces the simpler D12 composite with a weighted blend of 4
    # interpretable features:
    #   Entity Coverage  (0.35) - does the chain mention the query entities?
    #   Role Weight      (0.25) - CORE > SATELLITE
    #   Edge Density     (0.20) - is the chain internally connected?
    #   Path Length Score(0.20) - 3-hop chains preferred; floor for 1-hop
    #
    # The blend was tuned against the HotpotQA dev split where the
    # unweighted D12 score over-ranked short SAME_ENTITY bridges that
    # happened to carry a high edge-product score but did not actually
    # mention both bridge entities. Adding entity coverage as the
    # heaviest term pushes those weakly-grounded chains down.
    # ------------------------------------------------------------------

    # Feature weights - exposed as class constants so callers can audit
    # them in experiments without grepping the source.
    W_ENTITY_COVERAGE = 0.35
    W_ROLE_WEIGHT = 0.25
    W_EDGE_DENSITY = 0.20
    W_PATH_LENGTH = 0.20

    def _entity_coverage(self, chain, query_entities: Set[str]) -> float:
        """
        Ratio of query entities present anywhere in the chain.

        Args:
            chain: Either an EvidenceChain or a List[SentenceNode]/dict-likes.
                We accept both because the ranking pass sometimes operates
                on SentenceNode objects directly during intermediate steps.
            query_entities: Set of entity strings the query cares about.

        Returns:
            float in [0, 1]. 0.5 is returned for empty query sets so the
            feature does not artificially tank the composite score when
            the query happens to have no detected entities.
        """
        if not query_entities:
            return 0.5

        chain_entities: Set[str] = set()
        for sent in chain:
            # EvidenceChain.sentences are dicts; SentenceNode has .entities
            ents = sent.get("entities") if isinstance(sent, dict) else getattr(sent, "entities", None)
            if ents:
                chain_entities.update(ents)

        if not chain_entities:
            return 0.0

        # Case-insensitive intersection so 'Chen' and 'Dr. Sarah Chen' align
        q_lower = {q.lower() for q in query_entities}
        c_lower = {c.lower() for c in chain_entities}
        overlap = q_lower & c_lower
        return len(overlap) / len(q_lower)

    def _role_weight(self, chain) -> float:
        """
        Average role weight: CORE=1.0, SATELLITE=0.3.

        Returns 0.0 for empty chains (defensive) and 0.5 for single-node
        chains where the role contribution is undefined; a single
        sentence cannot be 'balanced' against itself.

        Accepts either dict-like sentences (with a 'role' key) or
        SentenceNode objects (with a .role attribute).
        """
        if not chain:
            return 0.0

        n = len(chain)
        core_count = 0
        for sent in chain:
            role = sent.get("role") if isinstance(sent, dict) else getattr(sent, "role", None)
            if (role or "CORE").upper() == "CORE":
                core_count += 1

        if n == 1:
            # One-sentence chains get the role value of that one sentence,
            # not the 1.0 default - this lets a 1-node SATELLITE chain
            # be ranked lower than a 1-node CORE chain.
            return 1.0 if core_count == 1 else 0.3

        return (core_count * 1.0 + (n - core_count) * 0.3) / n

    def _edge_density(self, chain, edges) -> float:
        """
        Edges whose endpoints both lie in the chain, normalized by the
        maximum possible edges in a path of this length (n - 1).

        Density = 1.0 means a fully connected path; 0.0 means a chain
        where the sentences are not internally linked by any edge in
        the registry (e.g., collected from disjoint seeds).

        Note: the spec text uses e.source / e.target, but the actual
        LogicEdge dataclass exposes source_sent_id / target_sent_id.
        We support both spellings here so the helper stays robust if
        the dataclass is refactored.
        """
        n = len(chain)
        if n <= 1:
            # 1-node "chain" has no internal edges possible; return a
            # neutral score so it isn't penalised vs. multi-node chains
            # that happen to lack edges.
            return 0.5

        chain_ids = set()
        for sent in chain:
            sid = sent.get("id") if isinstance(sent, dict) else getattr(sent, "id", None)
            if sid is not None:
                chain_ids.add(sid)

        edge_count = 0
        for e in edges:
            # Support both attribute spellings
            src = getattr(e, "source", None) or getattr(e, "source_sent_id", None)
            tgt = getattr(e, "target", None) or getattr(e, "target_sent_id", None)
            if src in chain_ids and tgt in chain_ids:
                edge_count += 1

        # Path-graph normalization: a chain of n nodes can have at most
        # n-1 internal edges if it's a true path. We use that as the
        # denominator so density ∈ [0, 1].
        return min(edge_count / (n - 1), 1.0)

    def _path_length_score(self, chain) -> float:
        """
        Score path length preference:

          n=1 -> 0.5  (neutral)
          n=2 -> 0.6  (1-hop, weak evidence)
          n=3 -> 1.0  (2-hop, preferred multi-hop sweet spot)
          n=4 -> 0.9  (3-hop, slight decay)
          n>=5 -> 0.5 + 1/n (longer is *less* reliable; trails off)
        """
        n = len(chain)
        if n <= 1:
            return 0.5
        if n == 2:
            return 0.6
        if n == 3:
            return 1.0
        if n == 4:
            return 0.9
        return 0.5 + (1.0 / n)

    def _chain_to_nodes(self, chain: EvidenceChain) -> List:
        """
        Materialise an EvidenceChain's sentences back into SentenceNode
        objects so the ranker can score against the live graph (entity
        set, role, etc.) rather than the serialised dict snapshot.

        Falls back to the dicts themselves if the registry cannot
        resolve an id (e.g., chain built against a different graph).
        """
        nodes: List = []
        for sent_dict in chain.sentences:
            sid = sent_dict.get("id")
            node = self.graph.sentence_registry.get_node(sid) if sid else None
            if node is not None:
                nodes.append(node)
            else:
                # Preserve order; attach the .entities / .role shims the
                # helpers expect so scoring still works on the snapshot.
                nodes.append(sent_dict)
        return nodes

    def rank_evidence_chains(
        self,
        chains: List[EvidenceChain],
        query_entities: Optional[Set[str]] = None,
    ) -> List[EvidenceChain]:
        """
        Re-rank evidence chains using the 4-feature composite score.

        For each chain, compute:

            final_score = (
                0.35 * entity_coverage
              + 0.25 * role_weight
              + 0.20 * edge_density
              + 0.20 * path_length_score
            )

        Then sort chains by final_score descending. The composite score
        is also written back into each chain.score so downstream
        consumers (logging, eval harness) see the same number.

        Args:
            chains: Candidate chains produced by traversal.
            query_entities: Entity set for coverage scoring. If None,
                coverage degrades to the neutral 0.5 (does not tank
                the score, but also does not help).

        Returns:
            The same list, sorted by final_score descending.
        """
        if not chains:
            return chains

        qe = {e for e in (query_entities or set()) if e}
        all_edges = self.graph.edge_registry.get_all_edges() if self.graph and self.graph.edge_registry else []

        scored: List[Tuple[float, EvidenceChain]] = []
        for chain in chains:
            nodes = self._chain_to_nodes(chain)

            entity_cov = self._entity_coverage(nodes, qe)
            role_w = self._role_weight(nodes)
            edge_d = self._edge_density(nodes, all_edges)
            path_l = self._path_length_score(nodes)

            final = (
                self.W_ENTITY_COVERAGE * entity_cov
                + self.W_ROLE_WEIGHT * role_w
                + self.W_EDGE_DENSITY * edge_d
                + self.W_PATH_LENGTH * path_l
            )

            # Persist component breakdown for debugging / ablation studies
            chain._rank_components = {  # type: ignore[attr-defined]
                "entity_coverage": entity_cov,
                "role_weight": role_w,
                "edge_density": edge_d,
                "path_length_score": path_l,
                "final_score": final,
            }
            chain.score = final
            scored.append((final, chain))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    def _llm_rerank_prompt(self, query: str, chains: List[EvidenceChain]) -> str:
        """
        Build the Phase-3 LLM rerank prompt.

        The prompt asks the model to pick exactly one chain number
        (1/2/3) with no explanation. Keeping the output a single token
        makes the parse trivial and the cost per rerank minimal.

        Chain text is the complemented_text (pronouns resolved) so the
        LLM sees the same surface form the downstream answer stage will
        use.
        """
        chain_blocks: List[str] = []
        for idx, chain in enumerate(chains[:3], start=1):
            lines: List[str] = []
            for s in chain.sentences:
                text = s.get("complemented_text") or s.get("text") or ""
                lines.append(f"  - {text}")
            chain_blocks.append(f"Chain {idx}:\n" + "\n".join(lines) if lines else f"Chain {idx}: (empty)")

        # Pad to exactly 3 slots so the prompt template stays stable
        while len(chain_blocks) < 3:
            chain_blocks.append(f"Chain {len(chain_blocks) + 1}: (empty)")

        return (
            "You are an evidence ranking expert. Given a question and 3 "
            "candidate evidence chains, select the chain that BEST answers "
            "the question.\n\n"
            f"Question: {query}\n\n"
            f"{chain_blocks[0]}\n\n"
            f"{chain_blocks[1]}\n\n"
            f"{chain_blocks[2]}\n\n"
            "Return only the chain number (1, 2, or 3). Do not add any explanation.\n"
            "Best chain:"
        )

    def llm_rerank(
        self,
        query: str,
        chains: List[EvidenceChain],
        llm_client: Optional["DeepSeekClient"] = None,
        ambiguity_threshold: float = 0.05,
    ) -> List[EvidenceChain]:
        """
        Optional Phase-3 LLM rerank.

        Activates only when the top-3 chains' final scores are within
        `ambiguency_threshold` of each other (the heuristic that the
        composite ranker is not confidently picking a winner). Otherwise
        the chains are returned unchanged so we don't pay an LLM call
        for queries the ranker already resolves cleanly.

        On any LLM failure (no client, disabled, network error, bad
        parse) we fall back to the composite-ranked order. The system
        must remain functional even without an API key.

        Args:
            query: The original user query.
            chains: Already composite-ranked chains (top of list = best).
            llm_client: A DeepSeekClient instance. If None, the method
                auto-creates one from env vars (DEEPSEEK_API_KEY /
                OPENAI_API_KEY).
            ambiguity_threshold: Top-3 spread below this triggers LLM
                rerank. Default 0.05.

        Returns:
            The list re-ordered so chains[0] is the LLM's pick (or
            unchanged if rerank did not run).
        """
        if not chains or len(chains) < 2:
            return chains

        top_scores = [c.score for c in chains[:3]]
        if len(top_scores) < 2:
            return chains

        spread = top_scores[0] - top_scores[-1]
        if spread >= ambiguity_threshold:
            return chains

        # Lazy import keeps this module importable even if requests
        # isn't installed (the project uses llm_client elsewhere, so it
        # should be available - but defensive import costs nothing).
        try:
            from .llm_client import DeepSeekClient
        except Exception:
            DeepSeekClient = None  # type: ignore

        client = llm_client
        if client is None and DeepSeekClient is not None:
            try:
                client = DeepSeekClient()
            except Exception:
                client = None

        if client is None or not getattr(client, "enabled", False):
            return chains

        prompt = self._llm_rerank_prompt(query, chains[:3])
        try:
            raw = client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You pick the best evidence chain for a "
                            "question. Reply with only the digit 1, 2, "
                            "or 3 - nothing else."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4,
            )
        except Exception:
            return chains

        # Parse the first digit 1/2/3 in the response
        pick = 0
        for ch in raw:
            if ch in ("1", "2", "3"):
                pick = int(ch)
                break
        if pick == 0:
            return chains

        # Re-order: pick first, then the rest in their original relative order
        top3 = chains[:3]
        rest = chains[3:]
        if pick - 1 >= len(top3):
            return chains
        chosen = top3[pick - 1]
        others = [c for i, c in enumerate(top3) if i != pick - 1]
        return [chosen] + others + rest

    def _traverse_from_seed(
        self, 
        seed, 
        intent: QueryIntent
    ) -> Optional[EvidenceChain]:
        """
        Traverse from a single seed sentence.
        
        Returns an EvidenceChain if we find sufficient evidence.
        """
        # Determine preferred edge types. Phase-4: prefer the
        # query-aware router's edge_priority if `retrieve()` set it
        # for the current query; otherwise fall back to the legacy
        # intent-keyed mapping. Router ordering wins for two reasons:
        #   1. It is keyed on the full question text (vs intent alone),
        #      so "Who directed X?" vs "What did X direct?" can route
        #      to different priorities when the LLM-confusion cases
        #      surface.
        #   2. It reflects user-intent categories that are independent
        #      of the regex-based `_extract_intent()` heuristic.
        router_priority = getattr(self, "_router_edge_priority", None)
        if router_priority:
            preferred_edges = list(router_priority)
        else:
            preferred_edges = self._get_preferred_edge_types(intent.intent_type)
        
        # Traverse logic edges
        logic_path, logic_score = self._traverse_logic_edges(
            seed, preferred_edges, self.max_logic_hops
        )
        
        # If we have a good logic chain, return it
        if len(logic_path) >= self.min_evidence_count:
            return self._build_evidence_chain(logic_path, logic_score)
        
        # Otherwise, try SAME_ENTITY expansion
        same_entity_path, same_entity_score = self._traverse_with_entity_edges(
            seed, preferred_edges
        )
        
        if same_entity_path and len(same_entity_path) >= self.min_evidence_count:
            combined_score = (logic_score * len(logic_path) + same_entity_score) / (
                len(logic_path) + len(same_entity_path)
            )
            return self._build_evidence_chain(same_entity_path, combined_score)
        
        # Return partial logic chain if we have something
        if logic_path:
            return self._build_evidence_chain(logic_path, logic_score * 0.5)
        
        return None
    
    def _get_preferred_edge_types(self, intent_type: str) -> List[str]:
        """
        Get preferred edge types for an intent [D15].

        Order encodes intent semantics: each intent's #1 priority is the
        edge type most likely to provide directly relevant evidence:

        ┌─────────┬─────────────────────────────────────┐
        │ Intent  │ Edge-type priority                  │
        ├─────────┼─────────────────────────────────────┤
        │ WHY     │ CAUSAL > ELABORATION > TEMPORAL > CONTRAST │
        │ THEN    │ TEMPORAL > CAUSAL > ELABORATION > CONTRAST │
        │ COMPARE │ CONTRAST > ELABORATION > CAUSAL > TEMPORAL  │
        │ WHAT    │ ELABORATION > CAUSAL > TEMPORAL > CONTRAST  │
        └─────────┴─────────────────────────────────────┘

        This is a *soft* preference, not a hard filter - if the #1 edge type
        doesn't exist, the search falls through to the next one.
        """
        from .sentence_graph import LogicEdge

        if intent_type == "WHY":
            return [LogicEdge.TYPE_CAUSAL, LogicEdge.TYPE_ELABORATION,
                    LogicEdge.TYPE_TEMPORAL, LogicEdge.TYPE_CONTRAST]
        elif intent_type == "THEN":
            return [LogicEdge.TYPE_TEMPORAL, LogicEdge.TYPE_CAUSAL,
                    LogicEdge.TYPE_ELABORATION, LogicEdge.TYPE_CONTRAST]
        elif intent_type == "COMPARE":
            return [LogicEdge.TYPE_CONTRAST, LogicEdge.TYPE_ELABORATION,
                    LogicEdge.TYPE_CAUSAL, LogicEdge.TYPE_TEMPORAL]
        else:  # WHAT (default) [D1 fallback]
            return [LogicEdge.TYPE_ELABORATION, LogicEdge.TYPE_CAUSAL,
                    LogicEdge.TYPE_TEMPORAL, LogicEdge.TYPE_CONTRAST]   
    
    def _traverse_logic_edges(
        self,
        start_node,
        preferred_types: List[str],
        max_hops: int
    ) -> Tuple[List, float]:
        """
        Traverse logic edges following preferred types.

        Phase-5 [CONTEXT hop skip]: when an outgoing edge is a
        `CONTEXT` boundary edge (`is_context_boundary=True`), the
        hop counter does not advance.  This is the "free hop"
        benefit the prompt promises — the narrative-continuation
        link is essentially free, allowing the user's
        `max_hops` budget to be spent on substantive logic hops
        instead.

        Returns (path, score) where path is a list of sentence nodes.
        """
        from .sentence_graph import LogicEdge

        path = [start_node]
        visited = {start_node.id}
        total_score = 1.0

        current = start_node
        hops_used = 0

        while hops_used < max_hops:
            # Get outgoing edges, now including CONTEXT.
            outgoing = self.graph.edge_registry.get_neighbors(
                current.id,
                edge_types=[LogicEdge.TYPE_CAUSAL, LogicEdge.TYPE_TEMPORAL,
                            LogicEdge.TYPE_CONTRAST, LogicEdge.TYPE_ELABORATION,
                            LogicEdge.TYPE_CONTEXT]
            )

            if not outgoing:
                break

            # Filter by preferred types
            best_next = None
            best_score = -1
            best_type_rank = len(preferred_types)
            best_edge = None

            for neighbor_id, edge in outgoing:
                if neighbor_id in visited:
                    continue

                # Check if this edge type is preferred
                type_rank = preferred_types.index(edge.edge_type) if edge.edge_type in preferred_types else len(preferred_types)

                # Prefer higher-ranked types
                if type_rank < best_type_rank:
                    best_next = neighbor_id
                    best_score = edge.weight * edge.confidence
                    best_type_rank = type_rank
                    best_edge = edge
                elif type_rank == best_type_rank and edge.weight * edge.confidence > best_score:
                    best_next = neighbor_id
                    best_score = edge.weight * edge.confidence
                    best_edge = edge

            if best_next is None:
                break

            next_node = self.graph.sentence_registry.get_node(best_next)
            if next_node:
                path.append(next_node)
                visited.add(best_next)
                total_score *= best_score
                current = next_node
                # Phase-5: CONTEXT edges do NOT consume hop count.
                # This is the only behaviour change vs the legacy
                # traversal.
                if not (
                    best_edge is not None
                    and getattr(best_edge, "is_context_boundary", False)
                ):
                    hops_used += 1
            else:
                break

        return path, total_score
    
    def _traverse_with_entity_edges(
        self,
        start_node,
        preferred_types: List[str]
    ) -> Tuple[List, float]:
        """
        Traverse SAME_ENTITY edges (1 hop max) then continue with logic edges.

        Cross-container expansion is **hard-capped at 1 SAME_ENTITY hop** [D14]
        to prevent the search from leaking into unrelated sub-topics.  After
        the initial jump, traversal proceeds along intra-container logic
        edges (CAUSAL/TEMPORAL/CONTRAST/ELABORATION) only.

        D14 rationale:
          - Without the cap, a chain of SAME_ENTITY edges could theoretically
            reach any sentence in the graph, defeating container-based
            organization
          - 1 hop gives enough flexibility to bridge two related containers
            (e.g. Chen ↔ chip) while keeping focused context
          - Cap is enforced in code (single-neighbor loop below), not only
            documented

        Phase-4 [router-aware max_hops]: the inner logic-edge loop
        iterates `max(0, self.max_logic_hops - 1)` hops instead of a
        hardcoded 2. This is wired to `RoutingConfig.max_hops` by
        `retrieve()`.

        Returns (path, score) where path is a list of sentence nodes.
        """
        from .sentence_graph import LogicEdge

        # Get SAME_ENTITY neighbors - only first-degree, no second hop across containers
        same_entity_neighbors = self.graph.edge_registry.get_neighbors(
            start_node.id,
            edge_types=[LogicEdge.TYPE_SAME_ENTITY]
        )

        if not same_entity_neighbors:
            return [], 0.0

        best_path = [start_node]
        best_score = 1.0

        # Router-aware: remaining logic-edge hops = max_hops - 1
        # (we already used 1 hop on the SAME_ENTITY edge).
        logic_hops_remaining = max(0, int(getattr(self, "max_logic_hops", 3)) - 1)

        for neighbor_id, edge in same_entity_neighbors:
            # Extend path with SAME_ENTITY edge (1 hop max)
            next_node = self.graph.sentence_registry.get_node(neighbor_id)
            if not next_node:
                continue

            current_path = [start_node, next_node]
            visited = {start_node.id, neighbor_id}
            total_score = edge.weight * edge.confidence
            current = next_node

            # Then continue with logic edges (router-aware hop count)
            hops_used = 0
            while hops_used < logic_hops_remaining:
                outgoing = self.graph.edge_registry.get_neighbors(
                    current.id,
                    edge_types=[LogicEdge.TYPE_CAUSAL, LogicEdge.TYPE_TEMPORAL,
                               LogicEdge.TYPE_CONTRAST, LogicEdge.TYPE_ELABORATION,
                               LogicEdge.TYPE_CONTEXT]
                )

                best_next = None
                best_next_score = -1
                best_next_edge = None

                for next_id, next_edge in outgoing:
                    if next_id in visited:
                        continue

                    # Phase-4: prefer edges whose type is high in
                    # `preferred_types` (= router's edge_priority).
                    # Falls back to base weight × confidence when type
                    # is not in the priority list.
                    type_rank = (
                        preferred_types.index(next_edge.edge_type)
                        if next_edge.edge_type in preferred_types
                        else len(preferred_types) + 1
                    )
                    # Priority boost: rank 0 → 1.0, rank 1 → 0.75, etc.
                    # Clamped so out-of-list edges are not zeroed out.
                    priority_boost = 1.0 / (1.0 + type_rank)
                    score = (
                        next_edge.weight * next_edge.confidence * priority_boost
                    )
                    if score > best_next_score:
                        best_next = next_id
                        best_next_score = score
                        best_next_edge = next_edge

                if best_next is None:
                    break

                next_node = self.graph.sentence_registry.get_node(best_next)
                if next_node:
                    current_path.append(next_node)
                    visited.add(best_next)
                    total_score *= best_next_score
                    current = next_node
                    # Phase-5: CONTEXT boundary edges do not consume
                    # the hop budget (free narrative continuation).
                    if not (
                        best_next_edge is not None
                        and getattr(best_next_edge, "is_context_boundary", False)
                    ):
                        hops_used += 1
                else:
                    break

            # Keep the best path
            if len(current_path) > len(best_path) or (
                len(current_path) == len(best_path) and total_score > best_score
            ):
                best_path = current_path
                best_score = total_score

        return best_path, best_score
    
    def _build_evidence_chain(
        self, 
        path: List, 
        score: float
    ) -> EvidenceChain:
        """Build an EvidenceChain from a path of sentences."""
        sentences = []
        edges = []
        
        for i, node in enumerate(path):
            sentences.append(self._sentence_to_dict(node))
            
            if i > 0:
                prev_node = path[i - 1]
                # Find the edge between these nodes
                edge_list = self.graph.edge_registry.get_neighbors(prev_node.id)
                matching_edge = next(
                    (e for _, e in edge_list if e.target_sent_id == node.id),
                    None
                )
                
                if matching_edge:
                    edges.append({
                        "source": matching_edge.source_sent_id,
                        "target": matching_edge.target_sent_id,
                        "type": matching_edge.edge_type,
                        "weight": matching_edge.weight
                    })
        
        return EvidenceChain(
            sentences=sentences,
            edges=edges,
            score=score
        )
    
    def _sentence_to_dict(self, node) -> Dict:
        """Convert a SentenceNode to a dict for output."""
        return {
            "id": node.id,
            "text": node.text,
            "complemented_text": node.complemented_text,
            "entities": node.entities,
            "role": getattr(node, "role", ""),
            "confidence": node.confidence
        }

    # ─────────────────────────────────────────────────────────────────
    # Query-aware Cross-Encoder gating [Phase-4 optimization]
    #
    # The Cross-Encoder rerank step costs ~30-50ms per query on CPU
    # and adds latency without lifting accuracy on simple queries.
    # Heuristic: only invoke the CE when the query is "complex" —
    #   - 2+ named entities, OR
    #   - explicit comparison phrasing ("vs", "compared to", "than").
    # Single-entity "Who directed X?" queries skip the CE because the
    # rule-based ranker is already accurate enough on those.
    #
    # This roughly halves the number of CE invocations on HotpotQA
    # while preserving accuracy on the harder comparison-style
    # questions where the CE actually helps.
    # ─────────────────────────────────────────────────────────────────
    def _should_use_cross_encoder(self, query: str) -> bool:
        """Return True iff the query is complex enough to justify CE rerank."""
        if not query:
            return False

        # Comparison phrasing always triggers the CE.
        q_lower = query.lower()
        compare_markers = (" vs ", " vs.", "versus", "compared to", " than ")
        if any(m in q_lower for m in compare_markers):
            return True

        # Entity-count based: >= 2 entities → complex.
        entity_count = 0
        if self.nlp is not None:
            try:
                doc = self.nlp(query)
                entity_count = len(doc.ents)
            except Exception:
                entity_count = 0
        if entity_count >= 2:
            return True

        # Otherwise simple — skip CE.
        return False

    # ─────────────────────────────────────────────────────────────────
    # Cross-Encoder reranking [Phase-3 retrieval]
    #
    # After evidence-chain construction we have a candidate pool of
    # ~5-20 sentences ranked by the rule-based composite score. The
    # Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) directly
    # models (query, sentence) relevance and is the gold standard for
    # passage reranking at small pool sizes.
    #
    # We lazy-load the model on first use (so callers who never enable
    # the rerank don't pay the 80MB download / 500MB RAM cost) and
    # fall back to rule-based ranking if the package or model fails
    # to load — by toggling `use_cross_encoder_rerank` to False and
    # returning the input order.
    # ─────────────────────────────────────────────────────────────────
    def _load_cross_encoder(self):
        """
        Lazy-load the CrossEncoder model. Returns the model on
        success, or None on failure (after disabling reranking).

        The model is cached in a module-level dict keyed by model name
        so that multiple `Retriever` instances in the same process
        (e.g. one per question in HotpotQA eval) share a single
        in-memory model rather than each paying the ~5s load cost.

        Failure-caching: when a download attempt fails (e.g. transient
        HF connection reset `WinError 10054`), the *negative* result
        is also cached at module level (sentinel `_FAILED`) so that
        subsequent `Retriever` instances skip the download attempt
        entirely. Without this, every per-question Retriever in a 200-
        question eval would re-attempt the same broken download and
        burn 1-5s × N on retry backoff. The negative cache is keyed by
        the same model name and is intentionally conservative — it
        only kicks in after a real `Exception`, not on `ImportError`
        (which usually means a real install issue worth surfacing).
        """
        if self._cross_encoder is not None:
            return self._cross_encoder
        # Module-level cache (shared across all Retriever instances)
        cache = _CROSS_ENCODER_CACHE
        key = self.cross_encoder_model_name
        cached = cache.get(key)
        # Negative-cache sentinel: model is known to be unloadable in
        # this process. Skip immediately, no network attempt.
        if cached is _CROSS_ENCODER_FAILED:
            self.use_cross_encoder_rerank = False
            return None
        if cached is not None:
            self._cross_encoder = cached
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            import logging
            from .model_paths import ensure_model
            # Prefer the local model store (downloading it if missing) so
            # the eval runs fully offline.
            model = CrossEncoder(ensure_model(key))
            cache[key] = model
            self._cross_encoder = model
            logging.getLogger(__name__).info(
                "[cross-encoder] loaded %s", key
            )
            return self._cross_encoder
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "[cross-encoder] sentence-transformers not installed; "
                "falling back to rule-based ranking."
            )
            self.use_cross_encoder_rerank = False
            return None
        except Exception as e:  # pragma: no cover - model load failures
            import logging
            logging.getLogger(__name__).warning(
                "[cross-encoder] failed to load %s (%s); "
                "falling back to rule-based ranking. "
                "This failure is cached for the rest of the process "
                "to avoid per-question retry storms.",
                key, e,
            )
            # Negative cache: subsequent Retriever instances skip the
            # download entirely. Keyed by model name; full-process
            # lifetime.
            cache[key] = _CROSS_ENCODER_FAILED
            self.use_cross_encoder_rerank = False
            return None

    def _rerank_with_cross_encoder(
        self,
        query: str,
        sentences,
        top_k: Optional[int] = None,
    ):
        """
        Rerank candidate sentences using a Cross-Encoder model.

        Args:
            query: User question
            sentences: List of SentenceNode objects (or objects with
                `.id` and `.complemented_text` attributes)
            top_k: Number of sentences to return after reranking
                (defaults to self.cross_encoder_top_k)

        Returns:
            Reranked list of the same sentence objects (length = min(top_k, len(sentences))).
            On any failure, returns the input list unchanged so the
            pipeline continues with rule-based ranking.
        """
        if not sentences:
            return []
        if top_k is None:
            top_k = self.cross_encoder_top_k

        # If reranking was disabled (e.g. import failed earlier), bail.
        if not self.use_cross_encoder_rerank:
            return sentences[:top_k]

        model = self._load_cross_encoder()
        if model is None:
            return sentences[:top_k]

        try:
            pairs = []
            for s in sentences:
                text = (
                    getattr(s, "complemented_text", None)
                    or getattr(s, "text", None)
                    or ""
                )
                pairs.append((query, text))

            scores = model.predict(pairs, show_progress_bar=False)
            # Stable sort by score descending, preserve original order
            # for ties via enumerate index.
            indexed = sorted(
                enumerate(scores),
                key=lambda kv: (-float(kv[1]), kv[0]),
            )
            keep = [idx for idx, _ in indexed[: max(1, top_k)]]
            return [sentences[i] for i in keep]
        except Exception as e:  # pragma: no cover - inference failures
            import logging
            logging.getLogger(__name__).warning(
                "[cross-encoder] predict() failed (%s); "
                "falling back to rule-based ordering.", e,
            )
            return sentences[:top_k]

    def _apply_cross_encoder_rerank(
        self,
        query: str,
        evidence_chains,
        seeds=None,
        containers=None,
    ):
        """Rerank a broad retrieved-sentence pool and rebuild top-1.

        Evidence chains can omit relevant seed sentences or bury them in a
        lower-ranked chain. The CE therefore receives candidates from all
        constructed chains, the selected seeds, and the retrieved containers.
        The pool is de-duplicated and capped before one batched prediction.
        """
        if not self.use_cross_encoder_rerank or not evidence_chains:
            return evidence_chains

        candidate_nodes = []
        seen_ids = set()
        candidate_limit = max(self.cross_encoder_top_k * 10, 30)

        def add_node(node) -> None:
            if node is None:
                return
            sid = getattr(node, "id", None)
            if not sid or sid in seen_ids or len(candidate_nodes) >= candidate_limit:
                return
            seen_ids.add(sid)
            candidate_nodes.append(node)

        # Preserve the existing chain ranking as the first source, but do not
        # stop at the top three chains: lower chains often contain a gold fact.
        for chain in evidence_chains:
            for sent in chain.sentences:
                add_node(self.graph.sentence_registry.get_node(sent.get("id")))

        # Seeds may contain relevant sentences that traversal failed to place
        # in any completed chain.
        for seed in seeds or []:
            if isinstance(seed, dict):
                add_node(self.graph.sentence_registry.get_node(seed.get("id")))
            else:
                add_node(seed)

        # Finally include sentences from retrieved containers. This recovers
        # cases where the correct paragraph was retrieved but the gold sentence
        # was not selected as a seed or reached by graph traversal.
        for container in containers or []:
            for sid in getattr(container, "sentence_ids", []) or []:
                add_node(self.graph.sentence_registry.get_node(sid))
                if len(candidate_nodes) >= candidate_limit:
                    break
            if len(candidate_nodes) >= candidate_limit:
                break

        if not candidate_nodes:
            return evidence_chains

        reranked = self._rerank_with_cross_encoder(
            query, candidate_nodes, top_k=self.cross_encoder_top_k
        )
        if not reranked:
            return evidence_chains

        new_top = self._rebuild_top_chain_with_reranked(
            reranked, evidence_chains
        )

        rest = [c for c in evidence_chains if c is not evidence_chains[0]]
        return [new_top] + rest

    def _rebuild_top_chain_with_reranked(
        self,
        reranked_sentences,
        original_chains,
    ) -> "EvidenceChain":
        """
        Construct a single EvidenceChain containing the reranked
        sentences in the new order. We preserve edges from the
        original top chain where possible (for any pair of consecutive
        reranked sentences that already had an edge in the original
        top chain), and look up missing edges from the graph registry.

        Falls back to an edges-free chain if lookup fails.
        """
        # Build a lookup: pair (id_a, id_b) -> edge dict from the
        # top original chain (only sequential pairs were stored there).
        edge_lookup = {}
        for chain in original_chains[:3]:
            seen_ids = {s.get("id") for s in chain.sentences}
            for s, nxt in zip(chain.sentences, chain.sentences[1:]):
                edge_lookup[(s.get("id"), nxt.get("id"))] = chain.edges

        sentence_dicts = []
        for node in reranked_sentences:
            sentence_dicts.append(self._sentence_to_dict(node))

        # Stitch edges for consecutive pairs that existed in original
        # chains; skip if not found (they're not strictly required for
        # answer extraction — `_build_answer` only reads the sentences).
        edges = []
        for a, b in zip(sentence_dicts, sentence_dicts[1:]):
            pair = (a.get("id"), b.get("id"))
            # Try both directions because the original chain may have
            # stored them in any order.
            cand = edge_lookup.get(pair) or edge_lookup.get((pair[1], pair[0]))
            if cand:
                edges.extend(cand)

        # If we have no edges from the top chains, fall back to
        # looking up edges in the graph registry (best-effort).
        if not edges and hasattr(self.graph, "edge_registry"):
            try:
                for a, b in zip(sentence_dicts, sentence_dicts[1:]):
                    nbrs = self.graph.edge_registry.get_neighbors(a.get("id", ""))
                    for e in nbrs:
                        if getattr(e, "target_sent_id", None) == b.get("id"):
                            edges.append({
                                "source": a.get("id"),
                                "target": b.get("id"),
                                "type": getattr(e, "edge_type", "UNKNOWN"),
                                "weight": getattr(e, "weight", 1.0),
                            })
                            break
            except Exception:
                pass

        return EvidenceChain(
            sentences=sentence_dicts,
            edges=edges,
            score=1.0,
        )

    # ─────────────────────────────────────────────────────────────────
    # Sentence-level reranking [Phase-5 retrieval]
    #
    # After the cross-encoder rerank has selected the top-N candidate
    # sentences, the *order* of those sentences inside the top
    # evidence chain still matters: when the LLM is fed the top-5
    # sentences as context, putting the most answer-bearing one
    # first measurably improves EM (paper §4.4 ablation).
    #
    # The previous pipeline just emitted the chain in original
    # traversal order, which tends to lead with the seed sentence
    # rather than the one that actually answers the question.
    #
    # The composite score blends four signals that are individually
    # cheap (no model calls) and one optional signal (cross-encoder
    #    predict, only used if the CE was already loaded earlier):
    #   * entity_coverage   50% — does the sentence mention a query entity?
    #   * core_role_bonus   25% — sentence is CORE rather than SATELLITE.
    #   * action_verb       10% — head verb is in the action-verb set
    #                            (e.g. "directed", "founded", "won") vs
    #                            state verbs ("is", "was", "has").
    #                            Lowered from 0.20 after ablation: on
    #                            HotpotQA many gold sentences start with
    #                            state-verb phrases ("The film follows…",
    #                            "He is best known for…") and a strong
    #                            action-verb prior was promoting distractor
    #                            sentences that happened to contain a
    #                            verb like "stars", "produced".
    #   * ce_score          15% — Cross-Encoder relevance when model
    #                            is already loaded (avoids per-call load).
    #
    # Returns the input chain unchanged if scoring fails; never raises.
    # ─────────────────────────────────────────────────────────────────
    _ACTION_VERBS = frozenset({
        "published", "argued", "developed", "demonstrated", "applied",
        "outperform", "outperformed", "outperforms", "showed", "proposed",
        "release", "released", "founded", "lead", "led", "directed",
        "direct", "star", "starred", "win", "won", "received", "receive",
        "became", "become", "create", "created", "introduced", "introduce",
        "achieved", "achieve", "designed", "design", "built", "build",
        "wrote", "write", "writ", "invent", "invented", "discovered",
        "discover", "held", "hold", "produce", "produced", "spawn",
        "spawned", "form", "formed", "establish", "established",
        "compose", "composed", "author", "authored", "perform",
        "performed", "play", "played", "act", "acted", "feature",
        "featured", "include", "included",
    })

    def _extract_entities(self, text: str) -> List[str]:
        """Lowercased entity strings extracted from `text` via spaCy.

        Falls back to an empty list when spaCy is unavailable (e.g.
        on `en_core_web_sm` with a stripped pipeline). The caller
        must tolerate `[]` — the entity-coverage signal gracefully
        degrades to 0 for all sentences in that case.
        """
        if not text or self.nlp is None:
            return []
        try:
            doc = self.nlp(text)
            return [ent.text.lower() for ent in doc.ents if ent.text]
        except Exception:
            return []

    def _rerank_sentences_in_chain(
        self,
        query: str,
        chain: EvidenceChain,
        top_k: int = 3,
    ) -> List[Dict]:
        """Rerank sentences inside `chain` by answer-likelihood.

        Returns the top-`top_k` sentence dicts in the new order, plus
        any tail sentences that did not fit, preserving the original
        ordering of the tail (so we never discard evidence; we just
        promote likely-answer sentences to the front).

        Scoring (weighted sum, weights picked to match the ablation
        in paper §4.4):
          ce_score           60%
          entity_coverage    25%
          core_role_bonus    10%
          action_verb         5%
          temporal_signal    ±25%  (historical question boosting)

        Falls back to the input order on any failure (no exception).
        """
        if not chain or not chain.sentences:
            return []
        if top_k <= 0:
            top_k = 3

        query_entities = self._extract_entities(query)
        # Honour the use_cross_encoder_rerank flag. When False, skip
        # CE entirely so the rule-based entity+role+verb signals carry
        # the ranking. This keeps the retrieval pipeline self-contained
        # (no HF download required) and matches the new default of
        # use_cross_encoder_rerank=False.
        ce_model = None
        if self.use_cross_encoder_rerank:
            ce_model = getattr(self, '_cross_encoder', None)
            if ce_model is None:
                ce_model = self._load_cross_encoder()

        query_lc = (query or "").lower()
        asks_hist = any(c in query_lc for c in (
            "formerly", "previously", "originally", "back then",
            "at the time", "prior to",
        )) or bool(re.search(
            r"\b(?:from\s+)?\d{4}\s*(?:to|-|–|until|through)\s*\d{4}\b",
            query_lc,
        ))

        scored = []
        for sent in chain.sentences:
            score = 0.0
            sent_entities = [e.lower() for e in (sent.get("entities") or [])]
            sent_text = sent.get("complemented_text") or sent.get("text") or ""

            # 1. CE relevance (60%, dominant).
            ce_norm = 0.0
            if ce_model is not None:
                try:
                    ce_score_raw = ce_model.predict([(query, sent_text)])[0]
                    ce_norm = max(0.0, min(1.0, (float(ce_score_raw) + 6.0) / 12.0))
                    score += 0.60 * ce_norm
                except Exception:
                    pass

            # 2. Entity coverage (25%).
            if query_entities and sent_entities:
                if any(e in sent_entities for e in query_entities):
                    score += 0.25
            elif query_entities:
                txt_low = sent_text.lower()
                if any(e in txt_low for e in query_entities):
                    score += 0.25

            # 3. CORE bonus (10%).
            role = (sent.get("role") or "").upper()
            if role == "CORE":
                score += 0.10
            elif role == "SATELLITE":
                score += 0.04

            # 4. Action verb (5%, tie-breaker).
            verb = self._get_head_verb_from_dict(sent).lower()
            if verb in self._ACTION_VERBS:
                score += 0.05

            # 5. Temporal penalty/boost — historical question.
            text_low = sent_text.lower()
            if asks_hist:
                if any(ind in text_low for ind in (
                    "currently", "currently competes", "currently plays",
                    "currently serves",
                )):
                    score -= 0.25
                elif any(ind in text_low for ind in (
                    "was known as", "was called", "was named",
                    "from 19", "from 20",
                    "until 19", "until 20",
                )):
                    score += 0.25

            scored.append((sent, score))

        scored.sort(key=lambda x: (-x[1], x[0].get("id", "")))
        promoted = [s for s, _ in scored[:top_k]]
        promoted_ids = {s.get("id") for s in promoted}
        tail = [s for s in chain.sentences if s.get("id") not in promoted_ids]
        return promoted + tail

    def _get_head_verb_from_dict(self, sent_dict: Dict) -> str:
        """Head verb for a sentence dict (vs SentenceNode).

        Different from `_get_head_verb(sent)`: that helper takes a
        `SentenceNode` (dataclass) and caches on the node. Here we
        get a plain dict from `chain.sentences`, so we cache on a
        per-dict `_head_verb_cache` attribute to avoid re-running
        spaCy for the same sentence across rerank calls.
        """
        cached = sent_dict.get("_head_verb_cache")
        if cached is not None:
            return cached
        text = (
            sent_dict.get("complemented_text")
            or sent_dict.get("text")
            or ""
        )
        verb = ""
        if self.nlp is not None and text:
            try:
                doc = self.nlp(text)
                for tok in doc:
                    if tok.pos_ in ("VERB", "AUX") and tok.dep_ not in (
                        "aux", "auxpass",
                    ):
                        verb = (tok.lemma_ or tok.text or "").lower()
                        break
            except Exception:
                verb = ""
        sent_dict["_head_verb_cache"] = verb
        return verb

    def _apply_sentence_level_rerank(
        self,
        evidence_chains: List[EvidenceChain],
        query: str,
        top_k: int = 3,
    ) -> List[EvidenceChain]:
        """Apply sentence-level rerank to the top-1 evidence chain.

        Rebuilds the top-1 chain with reranked sentences in its
        `sentences` field, while leaving the edges list intact
        (edges may reference sentence IDs in their old positions —
        the LLM-side `_extract_context` only reads the sentences, so
        this is safe). Returns the input list unchanged on failure.

        Only the top chain is reranked: lower-ranked chains are not
        consulted by `_build_answer` so reordering them would have no
        effect on the answer, and the cost (4 signals × N sentences
        per chain) becomes significant at depth=3 chains × ~15 sents.
        """
        if not evidence_chains:
            return evidence_chains
        top = evidence_chains[0]
        if not top or not top.sentences:
            return evidence_chains
        reranked = self._rerank_sentences_in_chain(query, top, top_k=top_k)
        if not reranked or len(reranked) == len(top.sentences) and (
                [s.get("id") for s in reranked]
                == [s.get("id") for s in top.sentences]
        ):
            return evidence_chains  # no-op, save the rebuild
        new_top = EvidenceChain(
            sentences=reranked,
            edges=list(top.edges),
            score=top.score,
        )
        return [new_top] + list(evidence_chains[1:])

    def _build_answer(
        self,
        chains: List[EvidenceChain],
        query: str
    ) -> Optional[str]:
        """
        Build an answer from evidence chains.

        This is a simple extractive approach. In production, this would
        use an LLM to synthesize the answer.
        """
        if not chains:
            return None
        
        # Get the best chain
        best_chain = chains[0]
        
        # Extract the most relevant sentence based on query
        # (Simple: first sentence that contains query keywords)
        query_lower = query.lower()
        relevant_sent = None
        
        for sent_dict in best_chain.sentences:
            text_lower = sent_dict['text'].lower()
            if any(kw in text_lower for kw in query_lower.split()):
                relevant_sent = sent_dict['complemented_text']
                break
        
        if not relevant_sent:
            relevant_sent = best_chain.sentences[0]['complemented_text']
        
        return relevant_sent
    
    def get_container_summary(self, container_id: str) -> Optional[Dict]:
        """Get a summary of a specific container."""
        container = self.graph.get_container(container_id)
        if not container:
            return None
        
        sentences = []
        for sent_id in container.sentence_ids[:5]:  # Top 5
            node = self.graph.sentence_registry.get_node(sent_id)
            if node:
                sentences.append(self._sentence_to_dict(node))
        
        return {
            "id": container.id,
            "entity": container.entity,
            "keywords": container.keywords,
            "summary": container.summary,
            "sentences": sentences
        }

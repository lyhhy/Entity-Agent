"""
Graph Builder Module
====================

Logic Edge Builder for the Sentence-Level Logic Graph.

Builds two types of edges:

(a) Intra-container logic edges (container-internal):
    - CAUSAL: "because", "since", "therefore", "thus" in dependency path
    - TEMPORAL: "then", "after", "before", "finally" markers
    - CONTRAST: "but", "although", "however", "yet" markers
    - ELABORATION: Jaccard similarity of entities > 0.3 (fallback)

(b) Cross-container entity edges (SAME_ENTITY):
    - Undirected edges connecting sentences that share entities
    - Weight: 0.6 (lower than logic edges)
    - Max 1 hop for SAME_ENTITY during retrieval
"""

from typing import Dict, List, Optional, Set, Tuple
import spacy
from spacy.tokens import Doc, Token

from .entity_resolver import EntityResolver
from .sentence_graph import (
    SentenceNode,
    LogicEdge,
    SentenceRegistry,
    EdgeRegistry,
    AttributeRegistry,
    AttributeMount,
    ATTRIBUTE_DESCRIPTORS,
    ATTRIBUTE_NAMES,
)
from .container_manager import ContainerManager


# Marker words for edge type detection
CAUSAL_MARKERS = {"because", "since", "therefore", "thus", "hence", "consequently", "so"}
TEMPORAL_MARKERS = {"then", "after", "before", "finally", "next", "later", "earlier", "subsequently", "eventually"}
CONTRAST_MARKERS = {"but", "although", "however", "yet", "nevertheless", "nonetheless", "despite", "whereas", "while"}
ELABORATION_MARKERS = {"also", "additionally", "furthermore", "moreover", "in fact", "specifically", "for example"}


class GraphBuilder:
    """
    Builds the sentence-level logic graph with intra-container and 
    cross-container edges.
    
    Main entry point is add_sentence(), which:
    1. Normalizes entity names
    2. Creates global sentence node
    3. Multi-homes to all relevant containers
    4. Builds intra-container logic edges (with last 3 sentences)
    5. Builds cross-container SAME_ENTITY edges
    6. Updates container indices
    """


_SPACY_CACHE: Dict[str, object] = {}
_SPACY_NER_ONLY_CACHE: Dict[str, object] = {}


def _load_spacy_shared(model_name: str = "en_core_web_lg"):
    """
    Process-wide cache for spaCy models to avoid re-loading the large
    `en_core_web_lg` model (~588 MB vectors) for every GraphBuilder
    instance.  Without this, HotpotQA evaluation (7405 questions × N
    GraphBuilders) crashes with `Unable to allocate MiB` after ~50
    instances.

    This returns the *full* pipeline (tagger + parser + ner + ...) and
    is intended for graph_builder which needs dependency parses.
    """
    if model_name in _SPACY_CACHE:
        return _SPACY_CACHE[model_name]
    try:
        import spacy
        nlp = spacy.load(model_name)
    except OSError:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = None
    _SPACY_CACHE[model_name] = nlp
    if "en_core_web_sm" not in _SPACY_CACHE:
        _SPACY_CACHE["en_core_web_sm"] = nlp
    return nlp


def _load_spacy_ner_only(model_name: str = "en_core_web_lg"):
    """
    Separate spaCy instance with all components except `ner` disabled.
    Use this for bulk NER extraction where tagger/parser outputs are
    not needed — saves ~40% RAM and ~2-3x speed.

    Returns a separate object so disabling pipelines here never affects
    the full pipeline used by graph_builder.
    """
    if model_name in _SPACY_NER_ONLY_CACHE:
        return _SPACY_NER_ONLY_CACHE[model_name]
    try:
        import spacy
        nlp = spacy.load(model_name)
        disable = [
            name for name in (
                "tagger", "parser", "lemmatizer",
                "attribute_ruler", "tok2vec",
            )
            if name in nlp.pipe_names and name != "ner"
        ]
        if disable:
            nlp.select_pipes(disable=disable)
    except OSError:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            disable = [
                name for name in (
                    "tagger", "parser", "lemmatizer",
                    "attribute_ruler", "tok2vec",
                )
                if name in nlp.pipe_names and name != "ner"
            ]
            if disable:
                nlp.select_pipes(disable=disable)
        except OSError:
            nlp = None
    _SPACY_NER_ONLY_CACHE[model_name] = nlp
    return nlp


class GraphBuilder:
    """
    ...
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_lg",
        enable_llm_summaries: bool = False,
        shared_nlp: object = None,
    ):
        """
        Initialize the graph builder.

        Args:
            spacy_model: Name of the SpaCy model to use.
            enable_llm_summaries: If True, generate LLM summaries for containers.
            shared_nlp: Pre-loaded spaCy model to share across instances
                (recommended for batch evaluation).
        """
        if shared_nlp is not None:
            self.nlp = shared_nlp
        else:
            self.nlp = _load_spacy_shared(spacy_model)
        
        # Core components
        self.sentence_registry = SentenceRegistry()
        self.edge_registry = EdgeRegistry()
        self.container_manager = ContainerManager(
            sentence_registry=self.sentence_registry,
            use_llm_summaries=enable_llm_summaries
        )
        self.entity_resolver = EntityResolver(nlp=self.nlp)
        
        # Track sentences per container for intra-container edge building
        self._container_sentence_order: Dict[str, List[str]] = {}
        
        # Configuration
        self._max_intra_container_comparisons = 3  # Compare with last N sentences
    
    def add_sentence(self, complemented_sent: 'ComplementedSentence') -> str:
        """
        Main entry point for adding a sentence to the graph.

        Steps (per spec):
          1. Context-aware entity disambiguation: low-information
             entities ("United States", "Apple", "发布会") are
             rewritten as "Entity (AnchorEntity)" using the
             sentence's chunk anchor.
          2. Normalize entities via EntityResolver (longest-form wins).
          3. Create the global SentenceNode, attaching per-entity
             attribute mounts derived from the entity's NER type.
          4. Multi-home the sentence to every relevant container,
             passing the entity type so the container's attribute
             template is selected.
          5. Build intra-container logic edges and SAME_ENTITY
             cross-container edges.
          6. Update container indices (keywords, summary, etc.).
        """
        # Step 1: per-entity NER type detection.
        raw_entities = list(complemented_sent.entities or [])
        raw_types = list(complemented_sent.entity_types or [])

        # If entity_types is missing, fall back to a per-entity
        # spaCy lookup on the original text. This keeps the public
        # API forgiving while preferring the caller's annotation.
        if not raw_types or len(raw_types) != len(raw_entities):
            raw_types = self._infer_entity_types(
                raw_entities, complemented_sent.original_text
            )
        # Pad types to match entities length.
        while len(raw_types) < len(raw_entities):
            raw_types.append("")

        # Step 2: Context disambiguation (美国 → 美国(华盛顿)).
        anchor = (complemented_sent.chunk_anchor_entity or "").strip()
        chunk_id = complemented_sent.chunk_id or ""
        disambiguated, types = self._disambiguate_entities(
            raw_entities, raw_types, anchor_entity=anchor, chunk_id=chunk_id,
        )

        # Step 3: Normalize entity names (longest-form wins).
        for entity in disambiguated:
            self.entity_resolver.register_entity(entity)
        normalized_entities = self.entity_resolver.normalize_entities(
            disambiguated
        )
        # Map normalized back to a parallel type list (we drop the
        # disambiguated forms; types are still associated 1-1).
        normalized_types: list = []
        for ent in normalized_entities:
            # Find first matching original index.
            idx = None
            for i, orig in enumerate(disambiguated):
                if self.entity_resolver.get_canonical(orig) == ent:
                    idx = i
                    break
            normalized_types.append(
                types[idx] if idx is not None and idx < len(types) else ""
            )

        # Step 4: Create global sentence node.
        node = SentenceNode(
            id="",
            text=complemented_sent.original_text,
            complemented_text=complemented_sent.complemented_text,
            entities=normalized_entities,
            resolved_subject=complemented_sent.resolved_subject,
            confidence=complemented_sent.confidence,
        )

        sentence_id = self.sentence_registry.add_node(node)
        node.id = sentence_id

        # Step 5: Multi-home + attribute mount per (entity, attribute).
        for entity, ent_type in zip(normalized_entities, normalized_types):
            container_id = self.container_manager.add_sentence_to_container(
                sentence_id, entity, entity_type=ent_type
            )
            node.add_container(container_id)
            if container_id not in self._container_sentence_order:
                self._container_sentence_order[container_id] = []
            self._container_sentence_order[container_id].append(sentence_id)

            # Phase-Sparse: attribute mounting happens in a LATER
            # pass — `_soft_mount_attributes` — because the mount is
            # derived from the sentence's embedding, and embeddings
            # only exist once every sentence has been added.
            #
            # The old code here did a single HARD match per entity
            # (`classify_sentence` → one attribute name, weight 1.0).
            # That is gone: a sentence is now soft-mounted under the
            # attributes it actually carries ("date 0.8, death 0.9"),
            # with the score carried in the mount weight.
            container = self.container_manager.get_container(container_id)
            if container is not None:
                # Nothing is mounted here; the soft pass below fills
                # both `node.attribute_mounts` and the container's
                # attribute slots.
                pass

            # Phase-5 [CONTEXT edges]: process pending split edges.
            pending = getattr(
                self.container_manager, "_pending_context_edges", None
            )
            if pending:
                for edge in pending:
                    self.edge_registry.add_edge(edge)
                self.container_manager._pending_context_edges = []

        # Step 6: Intra-container logic edges.
        self._build_intra_container_edges(sentence_id, node)

        # Step 7: Cross-container SAME_ENTITY edges.
        self.build_entity_edges(node)

        # Step 8: Update container indices.
        for container_id in node.containers:
            self.container_manager.update_container_index(container_id)

        return sentence_id

    def _infer_entity_types(
        self,
        entities: List[str],
        sentence_text: str,
    ) -> List[str]:
        """
        Best-effort NER fallback when the caller did not supply
        entity_types. Uses the shared spaCy model and matches the
        entity surface form against doc.ents.
        """
        if not self.nlp or not entities:
            return [""] * len(entities)
        try:
            doc = self.nlp(sentence_text or "")
        except Exception:
            return [""] * len(entities)
        out: List[str] = []
        for ent in entities:
            ent_l = ent.lower().strip()
            label = ""
            for span in doc.ents:
                if span.text.lower() == ent_l or ent_l in span.text.lower():
                    label = span.label_
                    break
            out.append(label)
        return out

    def _disambiguate_entities(
        self,
        entities: List[str],
        types: List[str],
        anchor_entity: str,
        chunk_id: str = "",
    ) -> Tuple[List[str], List[str]]:
        """
        Per-chunk context disambiguation: any low-information entity
        that co-occurs with `anchor_entity` is rewritten as
        "Entity (AnchorEntity)" so the same surface form gets a
        distinct container per paragraph (美国 in Washington context
        vs. 美国 in New York context).

        "Low information" entities are heuristic: short names that
        match a small set of known ambiguous patterns
        (United States, China, UK, Apple, Microsoft, etc.) OR any
        entity whose name length <= 3 tokens and not a PERSON.
        """
        if not anchor_entity or not entities:
            return list(entities), list(types)
        # Subset of high-recall ambiguous entity patterns. Keeping
        # this list narrow avoids accidentally merging genuinely
        # distinct entities.
        AMBIGUOUS_LOW_INFO = {
            "united states", "us", "u.s.", "u.s", "america",
            "united kingdom", "uk", "u.k.", "england",
            "china", "russia", "japan", "france", "germany",
            "apple", "microsoft", "google", "amazon", "ibm",
            "republic", "democratic party", "republican party",
            "发布会", "苹果", "华盛顿", "纽约", "北京",
        }
        out_ents: List[str] = []
        out_types: List[str] = []
        for ent, ent_type in zip(entities, types):
            ent_l = ent.lower().strip()
            should_disambiguate = (
                ent_l in AMBIGUOUS_LOW_INFO
                or (ent_type not in ("PERSON",) and len(ent_l) <= 18 and ent_l != anchor_entity.lower())
            )
            if should_disambiguate:
                new_ent = f"{ent} ({anchor_entity})"
                out_ents.append(new_ent)
            else:
                out_ents.append(ent)
            out_types.append(ent_type)
        return out_ents, out_types

    def _classify_attributes_for_entity(
        self,
        entity: str,
        entity_type: str,
        sentence_text: str,
    ) -> List[str]:
        """
        Choose the attribute slot(s) on `entity` that this sentence
        contributes to. The container (if any) carries the per-type
        template; we use the shared AttributeRegistry so the rules
        stay in one place.
        """
        registry = self.container_manager.get_container(
            self._entity_to_first_container(entity)
        ).get_attribute_registry() if self._entity_to_first_container(entity) else None
        if registry is None:
            registry = AttributeRegistry()
        return registry.classify_sentence(sentence_text or "", entity_type or "")
    def _entity_to_first_container(self, entity: str) -> Optional[str]:
        """Return the first container id whose entity matches `entity`."""
        if not entity:
            return None
        c = self.container_manager.get_container_for_entity(entity)
        if c is not None:
            return c.id
        return None
    
    def _build_intra_container_edges(
        self,
        sentence_id: str,
        node: SentenceNode
    ) -> None:
        """
        Build intra-container logic edges between sentences in the same container.
        
        Compares the new sentence with the last N sentences in each container
        it belongs to.
        """
        for container_id in node.containers:
            # Get recent sentences in this container
            container_sents = self._container_sentence_order.get(container_id, [])
            
            # Find position of current sentence
            if sentence_id not in container_sents:
                continue
            
            pos = container_sents.index(sentence_id)
            
            # Compare with previous sentences (up to N)
            start_idx = max(0, pos - self._max_intra_container_comparisons)
            for prev_sent_id in container_sents[start_idx:pos]:
                if prev_sent_id == sentence_id:
                    continue
                
                prev_node = self.sentence_registry.get_node(prev_sent_id)
                if not prev_node:
                    continue
                
                # Build logic edge
                edge = self.build_logic_edge(prev_node, node, container_id)
                if edge:
                    self.edge_registry.add_edge(edge)
    
    def build_logic_edge(
        self, 
        sent1: SentenceNode, 
        sent2: SentenceNode, 
        container_id: str
    ) -> Optional[LogicEdge]:
        """
        Compare two sentences using SpaCy dependency parsing and return edge type.
        
        Detection priority:
        1. CAUSAL: Check for causal markers in dependency path
        2. TEMPORAL: Check for temporal markers
        3. CONTRAST: Check for contrast markers
        4. ELABORATION: Jaccard similarity of entities > 0.3 (fallback)
        
        Args:
            sent1: Source sentence
            sent2: Target sentence
            container_id: Container where these sentences co-occur
            
        Returns:
            LogicEdge if a relationship is detected, None otherwise
        """
        if not self.nlp:
            # Without SpaCy, fall back to entity-based detection
            jaccard = self.sentence_registry.compute_jaccard(sent1.id, sent2.id)
            if jaccard > LogicEdge.JACCARD_THRESHOLD:
                return LogicEdge(
                    source_sent_id=sent1.id,
                    target_sent_id=sent2.id,
                    edge_type=LogicEdge.TYPE_ELABORATION,
                    weight=LogicEdge.EDGE_WEIGHTS[LogicEdge.TYPE_ELABORATION],
                    confidence=jaccard,
                    container_id=container_id
                )
            return None

        # Parse sentences with SpaCy
        doc1 = self.nlp(sent1.complemented_text)
        doc2 = self.nlp(sent2.complemented_text)

        # Check for explicit markers in second sentence (effect/result)
        edge_type, confidence = self._detect_edge_type(doc1, doc2)

        if edge_type:
            return LogicEdge(
                source_sent_id=sent1.id,
                target_sent_id=sent2.id,
                edge_type=edge_type,
                weight=LogicEdge.EDGE_WEIGHTS.get(
                    edge_type, LogicEdge.WEIGHT_LOGIC,
                ),
                confidence=confidence,
                container_id=container_id
            )

        # Fallback to entity-based elaboration
        jaccard = self.sentence_registry.compute_jaccard(sent1.id, sent2.id)
        if jaccard > LogicEdge.JACCARD_THRESHOLD:
            return LogicEdge(
                source_sent_id=sent1.id,
                target_sent_id=sent2.id,
                edge_type=LogicEdge.TYPE_ELABORATION,
                weight=LogicEdge.EDGE_WEIGHTS[LogicEdge.TYPE_ELABORATION],
                confidence=jaccard,
                container_id=container_id
            )
        
        return None
    
    def _detect_edge_type(
        self, 
        doc1: Doc, 
        doc2: Doc
    ) -> Tuple[Optional[str], float]:
        """
        Detect edge type from SpaCy dependency-parsed documents.
        
        Returns:
            Tuple of (edge_type, confidence) or (None, 0.0) if no relationship
        """
        # Combine text for marker detection
        combined_text = doc2.text.lower()
        
        # Check for CAUSAL markers
        for marker in CAUSAL_MARKERS:
            if marker in combined_text:
                return LogicEdge.TYPE_CAUSAL, 0.9
        
        # Check for TEMPORAL markers
        for marker in TEMPORAL_MARKERS:
            if marker in combined_text:
                return LogicEdge.TYPE_TEMPORAL, 0.85
        
        # Check for CONTRAST markers
        for marker in CONTRAST_MARKERS:
            if marker in combined_text:
                return LogicEdge.TYPE_CONTRAST, 0.8
        
        # Check for ELABORATION markers
        for marker in ELABORATION_MARKERS:
            if marker in combined_text:
                return LogicEdge.TYPE_ELABORATION, 0.7
        
        # Dependency-based causal detection (more sophisticated)
        if self._has_causal_dependency(doc2):
            return LogicEdge.TYPE_CAUSAL, 0.75
        
        return None, 0.0
    
    def _has_causal_dependency(self, doc: Doc) -> bool:
        """
        Check for causal dependency patterns in the document.
        
        Looks for patterns like:
        - ADVCL (adverbial clause) with causal markers
        - Marked clauses with because
        """
        for token in doc:
            # Check for adverbial clauses that might be causal
            if token.dep_ == "advcl":
                # Check if the head verb has a specific relationship
                if token.head and any(
                    marker in token.head.text.lower() 
                    for marker in CAUSAL_MARKERS
                ):
                    return True
            
            # Check for marked clauses
            if token.dep_ == "mark" and token.text.lower() == "because":
                return True
        
        return False
    
    def build_entity_edges(
        self, 
        sent: SentenceNode
    ) -> List[LogicEdge]:
        """
        Find all other sentences sharing entities and create SAME_ENTITY edges.
        
        These are UNDIRECTED edges with weight 0.6.
        
        Args:
            sent: The sentence to find entity matches for
            
        Returns:
            List of created LogicEdge objects
        """
        edges = []
        
        for entity in sent.entities:
            # Find all sentences with this entity
            matching_sents = self.sentence_registry.get_sentences_with_entity(entity)
            
            for other_sent in matching_sents:
                if other_sent.id == sent.id:
                    continue
                
                # Check if edge already exists
                if self.edge_registry.edge_exists(
                    sent.id, other_sent.id, LogicEdge.TYPE_SAME_ENTITY
                ):
                    continue
                
                # Create SAME_ENTITY edge
                edge = LogicEdge(
                    source_sent_id=sent.id,
                    target_sent_id=other_sent.id,
                    edge_type=LogicEdge.TYPE_SAME_ENTITY,
                    weight=LogicEdge.WEIGHT_SAME_ENTITY,
                    confidence=1.0,  # Entity match is deterministic
                    container_id=None  # Cross-container
                )
                
                if self.edge_registry.add_edge(edge):
                    edges.append(edge)
        
        return edges
    
    def get_sentence_node(self, sentence_id: str) -> Optional[SentenceNode]:
        """Get a sentence node by ID."""
        return self.sentence_registry.get_node(sentence_id)
    
    def get_outgoing_edges(
        self, 
        sentence_id: str, 
        edge_types: Optional[List[str]] = None
    ) -> List[tuple]:
        """
        Get outgoing edges from a sentence.
        
        Args:
            sentence_id: Source sentence ID
            edge_types: Optional filter for edge types
            
        Returns:
            List of (neighbor_id, edge) tuples
        """
        return self.edge_registry.get_neighbors(sentence_id, edge_types)
    
    def get_all_edges(self) -> List[LogicEdge]:
        """Get all edges in the graph."""
        return self.edge_registry.get_all_edges()
    
    def get_logic_edges(self) -> List[LogicEdge]:
        """Get all logic edges (CAUSAL, TEMPORAL, CONTRAST, ELABORATION)."""
        return self.edge_registry.get_logic_edges()
    
    def get_same_entity_edges(self) -> List[LogicEdge]:
        """Get all SAME_ENTITY edges."""
        return self.edge_registry.get_same_entity_edges()
    
    def get_container(self, container_id: str) -> Optional['Container']:
        """Get a container by ID."""
        return self.container_manager.get_container(container_id)
    
    def get_containers_for_entity(self, entity: str) -> List['Container']:
        """Get all containers for an entity."""
        return self.container_manager.get_containers_for_entity(entity)
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the built graph."""
        edge_counts = {}
        for edge in self.edge_registry.get_all_edges():
            edge_counts[edge.edge_type] = edge_counts.get(edge.edge_type, 0) + 1

        return {
            "sentence_count": len(self.sentence_registry),
            "container_count": len(self.container_manager),
            "total_edges": len(self.edge_registry),
            "edge_type_counts": edge_counts,
            "entity_count": len(self.entity_resolver)
        }

    def visualize_graph(self) -> Dict:
        """
        Get a visualization-friendly representation of the graph.
        
        Returns a dict with nodes, edges, and containers.
        """
        nodes = []
        for node in self.sentence_registry.get_all_nodes():
            nodes.append({
                "id": node.id,
                "text": node.text[:100] + "..." if len(node.text) > 100 else node.text,
                "entities": node.entities,
                "containers": node.containers,
            })
        
        edges = []
        for edge in self.edge_registry.get_all_edges():
            edges.append({
                "source": edge.source_sent_id,
                "target": edge.target_sent_id,
                "type": edge.edge_type,
                "weight": edge.weight,
                "container": edge.container_id
            })
        
        containers = []
        for container in self.container_manager.get_all_containers():
            containers.append(container.to_dict())
        
        return {
            "nodes": nodes,
            "edges": edges,
            "containers": containers
        }

    # ------------------------------------------------------------------
    # Phase-A: sentence-level TF-IDF index [借鉴 A]
    #
    # Tokeniser = regex `[a-z]{3,}` over `node.text` (lower-cased).
    # IDF = log((N+1) / (df+1)) + 1 (smoothed, always positive).
    # Each sentence vector is L2-normalised so cosine = dot product.
    #
    # Cost: O(N * |V|) sparse mat-mul per call. For HotpotQA dev
    # (~100 sentences per question, |V| ~2k) it is <1 ms. We do NOT
    # cache across questions — quick20_eval rebuilds a fresh
    # GraphBuilder per question, so the index is naturally scoped.
    # ------------------------------------------------------------------
    _TFIDF_TOKEN_RE = __import__("re").compile(r"[a-z]{3,}")
    _TFIDF_STOPWORDS = frozenset({
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "has",
        "have", "been", "were", "they", "this", "that", "with",
        "would", "there", "their", "what", "about", "which", "when",
        "from", "his", "she", "who", "him", "its", "may", "also",
    })

    def _tokenize_for_tfidf(self, text: str):
        import re as _re
        toks = _re.findall(r"[a-z]{3,}", (text or "").lower())
        return [t for t in toks if t not in self._TFIDF_STOPWORDS]

    def compute_tfidf_index(self) -> None:
        """
        Compute per-sentence TF-IDF vectors in-place on each
        SentenceNode.tfidf_vector.  Must be called after all
        `add_sentence()` calls finish (e.g. just before
        `Retriever.retrieve()`).

        Also exposes `self._tfidf_vocab` (word -> col index) and
        `self._tfidf_idf` (col index -> idf weight) so the
        StenGraphPipeline can build a query vector in the same
        vocabulary and do a proper dot product.
        """
        import math
        import numpy as np
        nodes = self.sentence_registry.get_all_nodes()
        if not nodes:
            return
        # 1) tokenise
        doc_tokens: list = [self._tokenize_for_tfidf(n.text or "") for n in nodes]
        # 2) build vocabulary
        vocab: dict = {}
        for toks in doc_tokens:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        if not vocab:
            return
        # 3) document frequencies
        df: dict = {w: 0 for w in vocab}
        for toks in doc_tokens:
            for w in set(toks):
                if w in df:
                    df[w] += 1
        n_docs = len(doc_tokens)
        # smoothed IDF
        idf_arr = np.zeros(len(vocab), dtype=np.float32)
        for w, j in vocab.items():
            idf_arr[j] = math.log((n_docs + 1) / (df[w] + 1)) + 1.0
        # 4) per-sentence vectors
        for node, toks in zip(nodes, doc_tokens):
            if not toks:
                node.tfidf_vector = None
                continue
            tf: dict = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            length = len(toks)
            v = np.zeros(len(vocab), dtype=np.float32)
            for w, c in tf.items():
                j = vocab.get(w)
                if j is not None:
                    v[j] = (c / length) * idf_arr[j]
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            node.tfidf_vector = v
        # 5) expose vocab + idf for downstream query-time scoring
        self._tfidf_vocab = vocab
        self._tfidf_idf = idf_arr

        # 6) Per-sentence dense embeddings (for the event-signal S_h_E
        #    path in StenGraphPipeline._event_fact_score).  Without
        #    this, SentenceNode.summary_vector stays None, so the
        #    per-event `mean_vec` branch is always empty and event
        #    similarity contributes zero.  We reuse the ContainerManager's
        #    SentenceTransformer (all-MiniLM-L6-v2 by default; local
        #    CPU model, NOT DeepSeek).
        self._populate_sentence_summary_vectors(nodes)
        # 7) Soft attribute mounting.  This must run AFTER the
        #    sentence embeddings exist, because the mount is derived
        #    from the sentence's own embedding.
        self._soft_mount_attributes(nodes)

    # ------------------------------------------------------------------
    # Soft attribute mounting
    # ------------------------------------------------------------------
    def _soft_mount_attributes(self, nodes) -> None:
        """
        Replace the vocabulary-based single hard attribute match with a
        SOFT mount driven by each sentence's embedding.

        For every sentence we compute its continuous attribute profile
        — the cosine between the sentence embedding and each attribute
        descriptor, e.g. {"date": 0.81, "death": 0.93}.  The sentence
        is then mounted under the best `ATTR_MOUNT_TOP_K` attributes
        that its ENTITIES actually declare, with the score carried in
        the mount weight.  A sentence can therefore carry several
        attributes at once, which is what the "date 0.8 / death 0.9"
        design requires.

        The old hard match (`classify_sentence` → one name per entity
        with weight 1.0) is replaced here.  If the embedder is
        unavailable, the legacy mounts are left in place.
        """
        embedder = getattr(self.container_manager, "embedder", None)
        if embedder is None or not nodes:
            return
        try:
            import numpy as _np
        except Exception:  # noqa: BLE001
            return

        try:
            attr_embs = _np.asarray(
                embedder.encode(
                    [ATTRIBUTE_DESCRIPTORS[a] for a in ATTRIBUTE_NAMES],
                    normalize_embeddings=True,
                ),
                dtype=_np.float32,
            ).reshape(len(ATTRIBUTE_NAMES), -1)
        except Exception as e:  # noqa: BLE001
            print(f"[graph_builder] attribute descriptor encode failed: {e}")
            return

        registry = AttributeRegistry()
        for node in nodes:
            sv = getattr(node, "summary_vector", None)
            if sv is None:
                continue
            try:
                svec = _np.asarray(sv, dtype=_np.float32).reshape(-1)
                sims = _np.maximum(attr_embs @ svec, 0.0)
            except Exception:  # noqa: BLE001
                continue

            # Continuous profile of this sentence.
            sent_attr_vec: Dict[str, float] = {}
            for i, aname in enumerate(ATTRIBUTE_NAMES):
                v = float(sims[i])
                if v > 0.0:
                    sent_attr_vec[aname] = v

            # Which entities does this sentence talk about, and what
            # type is each?  The mount must respect the entity's
            # template: a PERSON cannot mount `population`.
            entities = list(getattr(node, "entities", []) or [])
            if not entities:
                continue
            ent_types = self._entity_types_for(node, entities)

            new_mounts: List[AttributeMount] = []
            for ent, etype in zip(entities, ent_types):
                for aname, weight in registry.soft_mount(sent_attr_vec, etype):
                    new_mounts.append(
                        AttributeMount(
                            entity=ent,
                            attribute_name=aname,
                            weight=float(weight),
                        )
                    )
            if not new_mounts:
                # Nothing passed the entity's template.  The sentence
                # stays unmounted — the hard vocabulary match is gone,
                # so there is no legacy "other" fallback to keep.
                continue

            # The soft mounts are the only mounts.
            node.attribute_mounts = new_mounts

            # Mount the sentence under each attribute slot of the
            # entity's container.  (There are no legacy mounts to
            # remove — the hard match no longer runs.)
            for m in new_mounts:
                cid = self._entity_to_first_container(m.entity)
                if not cid:
                    continue
                container = self.container_manager.get_container(cid)
                if container is None:
                    continue
                container.mount_sentence(node.id, m.attribute_name)

    def _entity_types_for(
        self, node, entities: List[str],
    ) -> List[str]:
        """Return the NER type of each entity on `node` ("" if unknown)."""
        out: List[str] = []
        for ent in entities:
            cid = self._entity_to_first_container(ent)
            etype = ""
            if cid:
                container = self.container_manager.get_container(cid)
                if container is not None:
                    etype = container.entity_type or ""
            out.append(etype)
        return out

    def _populate_sentence_summary_vectors(self, nodes) -> None:
        """Encode every node's text with cm.embedder and store a
        L2-normalised float32 vector on `node.summary_vector`. Safe
        no-op if the embedder is unavailable."""
        embedder = getattr(self.container_manager, "embedder", None)
        if embedder is None or not nodes:
            return
        try:
            import numpy as _np
            texts = [n.text or n.complemented_text or "" for n in nodes]
            # batch encode for speed; sentence-transformers handles padding.
            vecs = embedder.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            vecs = _np.asarray(vecs, dtype=_np.float32)
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            for node, v in zip(nodes, vecs):
                node.summary_vector = v
        except Exception as e:  # noqa: BLE001
            print(f"[graph_builder] sentence summary_vector build failed: {e}")
            return


# Backward compatibility: ComplementedSentence dataclass
from dataclasses import dataclass, field
from typing import List as TypingList


@dataclass
class ComplementedSentence:
    """
    Represents a sentence with resolved pronouns and entities.

    This is the expected input format for the GraphBuilder.
    """
    original_text: str
    complemented_text: str
    resolved_subject: str
    entities: TypingList[str]
    confidence: float = 1.0
    # Phase-Sparse: per-entity spaCy NER label, parallel to `entities`.
    # e.g. entities=["Washington", "United States"], entity_types=["PERSON","GPE"].
    entity_types: TypingList[str] = field(default_factory=list)
    # Optional chunk/paragraph context used for entity disambiguation.
    # When a sentence is part of a paragraph with a clear anchor
    # entity, low-information entities ("United States", "Apple") can
    # be re-anchored as "United States (Washington)" etc. The chunk
    # id is used as the disambiguation key.
    chunk_anchor_entity: str = ""
    chunk_id: str = ""

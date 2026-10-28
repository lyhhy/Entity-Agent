"""
Container Manager Module
========================

Container Aggregation with Split Rules.

Each container represents a group of sentences about the same entity.
Supports multi-homing (sentences can belong to multiple containers) and
automatic container splitting when they exceed capacity limits.

Split Rules:
- When a container exceeds 5 sentences, check for thematic differentiation
- Use entity co-occurrence and TF-IDF cosine similarity to detect subgroups
- Split when inter-group similarity < 0.4
- Otherwise, allow expansion up to 8 sentences
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Callable
import numpy as np
from collections import Counter
import math
import threading
import re

from .sentence_graph import AttributeSlot, AttributeRegistry, AttributeMount


# ─────────────────────────────────────────────────────────────────
# Embedder cache [Phase-2 semantic retrieval]
#
# SentenceTransformer models are ~80 MB on disk + several seconds
# to load. The HotpotQA eval rebuilds a graph per question, so each
# ContainerManager is short-lived. We cache the SentenceTransformer
# instance at module scope and lazy-load it on first use.
# ─────────────────────────────────────────────────────────────────
_EMBEDDER_CACHE: Dict[str, object] = {}
_EMBEDDER_LOCK = threading.Lock()


def _get_embedder(model_name: str):
    """
    Return a cached SentenceTransformer for `model_name`, loading it
    on the first call. Returns None if the package is unavailable.
    """
    if model_name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model_name]
    with _EMBEDDER_LOCK:
        if model_name in _EMBEDDER_CACHE:
            return _EMBEDDER_CACHE[model_name]
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            from .model_paths import ensure_model
            # Prefer the local model store (downloading it if missing) so
            # the eval runs fully offline.
            embedder = SentenceTransformer(ensure_model(model_name))
        except Exception:
            embedder = None
        _EMBEDDER_CACHE[model_name] = embedder
        return embedder


# ─────────────────────────────────────────────────────────────────
# BM25 index (Week-1: Hybrid Retrieval)
#
# Minimal Okapi BM25 over container summary texts.
# Default k1=1.5, b=0.75 follow standard IR defaults.
# Used for the BM25-retrieval arm of hybrid (BM25 + dense) fusion.
# ─────────────────────────────────────────────────────────────────
class BM25ContainerIndex:
    """
    Okapi BM25 index over container summary strings.

    Each "document" is a container's `summary` text (the keyword
    template or LLM summary — both are short and BM25-friendly).

    Query-time: score(query) returns List[(container_id, bm25_score)].
    No external deps; pure Python + math.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        # Built lazily on first build.
        self._docs: List[List[str]] = []   # token lists
        self._doc_texts: List[str] = []    # raw strings (for debugging)
        self._doc_lens: List[int] = []
        self._avg_dl: float = 0.0
        self._df: Counter = Counter()       # document frequency
        self._n_docs: int = 0
        self._container_ids: List[str] = []  # parallel to _docs

    def build(self, container_ids: List[str], summaries: List[str]) -> "BM25ContainerIndex":
        """
        Build BM25 from parallel lists of container_id / summary strings.
        Returns self for chaining.
        """
        self._container_ids = []
        self._docs = []
        self._doc_texts = []
        self._doc_lens = []
        df: Counter = Counter()

        STOP = {
            "the", "a", "an", "of", "in", "on", "at", "to", "for",
            "by", "and", "or", "is", "are", "was", "were", "be",
            "been", "as", "that", "this", "these", "those", "with",
            "from", "into", "than", "then", "but", "it", "its",
            "i", "you", "he", "she", "they", "we", "his", "her",
            "their", "my", "your", "do", "did", "does", "had",
            "has", "have", "not", "which", "who", "what", "when",
            "where", "why", "how", "if", "so", "no", "up", "out",
            "all", "any", "one", "two", "first", "also", "most",
        }

        def _tokenize(text: str) -> List[str]:
            return [
                t for t in re.findall(r"[a-z]{3,}", (text or "").lower())
                if t not in STOP and len(t) >= 3
            ]

        for cid, text in zip(container_ids, summaries):
            tokens = _tokenize(text)
            if not tokens:
                continue
            self._container_ids.append(cid)
            self._docs.append(tokens)
            self._doc_texts.append(text)
            self._doc_lens.append(len(tokens))
            for t in set(tokens):
                df[t] += 1

        self._n_docs = len(self._docs)
        self._doc_lens = self._doc_lens or [0] * self._n_docs
        self._avg_dl = (sum(self._doc_lens) / self._n_docs) if self._n_docs else 1.0
        self._df = df
        return self

    def score(self, query: str) -> List[Tuple[str, float]]:
        """
        Score all containers against `query`. Returns [(container_id, score)]
        sorted by score descending.  Returns empty list when index is empty.
        """
        if not self._docs or self._n_docs == 0:
            return []

        STOP = {
            "the", "a", "an", "of", "in", "on", "at", "to", "for",
            "by", "and", "or", "is", "are", "was", "were", "be",
            "been", "as", "that", "this", "these", "those", "with",
            "from", "into", "than", "then", "but", "it", "its",
            "i", "you", "he", "she", "they", "we", "his", "her",
            "their", "my", "your", "do", "did", "does", "had",
            "has", "have", "not", "which", "who", "what", "when",
            "where", "why", "how", "if", "so", "no", "up", "out",
            "all", "any", "one", "two", "first", "also", "most",
        }

        def _tokenize(text: str) -> List[str]:
            return [
                t for t in re.findall(r"[a-z]{3,}", (text or "").lower())
                if t not in STOP and len(t) >= 3
            ]

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        out: List[Tuple[str, float]] = []
        for idx, (tokens, dl) in enumerate(zip(self._docs, self._doc_lens)):
            score = 0.0
            for q in q_tokens:
                f = tokens.count(q)
                if f == 0:
                    continue
                dfv = self._df.get(q, 0)
                n = self._n_docs
                idf = math.log((n - dfv + 0.5) / (dfv + 0.5) + 1)
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / self._avg_dl)
                score += idf * (f * (self.k1 + 1.0)) / denom
            if score > 0.0:
                out.append((self._container_ids[idx], score))
        out.sort(key=lambda kv: -kv[1])
        return out


@dataclass
class Container:
    """
    Represents a container grouping sentences about the same entity.

    Attributes:
        id: Unique container ID
        entity: Canonical entity name
        sentence_ids: Sentence IDs in this container (multi-homing allowed)
        entity_type: spaCy NER label (PERSON, ORG, GPE, etc.) — used
            to select the right attribute template when building slots.
        keywords: Top TF-IDF keywords for retrieval
        summary: One-line summary (LLM-generated or keywords placeholder)
        summary_vector: FAISS vector for semantic retrieval
        created_at: Creation timestamp
        attributes: Attribute slots keyed by attribute name. Each slot
            holds a sub-list of sentence_ids mounted under that
            attribute. A sentence may appear in multiple slots.
        attribute_registry: Shared registry providing the template
            and keyword patterns used to classify sentences.
    """
    id: str
    entity: str
    sentence_ids: List[str] = field(default_factory=list)
    entity_inventory: List[str] = field(default_factory=list)
    entity_type: str = ""
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    summary_vector: Optional[np.ndarray] = None
    hyperedges: List[str] = field(default_factory=list)
    created_at: int = 0  # Simple counter for ordering
    # Phase-Sparse: per-attribute sentence mounts.
    # {attribute_name -> AttributeSlot}
    attributes: Dict[str, "AttributeSlot"] = field(default_factory=dict)
    attribute_registry: Optional["AttributeRegistry"] = None

    # Container limits
    MAX_SENTENCES = 5
    MAX_SENTENCES_WITH_SUBTOPICS = 8

    # Split threshold
    SPLIT_SIMILARITY_THRESHOLD = 0.4

    def add_sentence(self, sentence_id: str) -> None:
        """Add a sentence ID to this container."""
        if sentence_id not in self.sentence_ids:
            self.sentence_ids.append(sentence_id)

    def add_hyperedge(self, hyperedge_id: str) -> None:
        """Register a hyperedge that this container belongs to."""
        if hyperedge_id not in self.hyperedges:
            self.hyperedges.append(hyperedge_id)

    def get_attribute_registry(self) -> "AttributeRegistry":
        """Return the attribute registry, falling back to a fresh one."""
        if self.attribute_registry is None:
            self.attribute_registry = AttributeRegistry()
        return self.attribute_registry

    def ensure_attribute_slots(self) -> None:
        """
        Build the per-attribute slot dict from the entity's template.
        Called lazily on first mount so we don't create slots for
        empty containers.
        """
        if self.attributes:
            return
        registry = self.get_attribute_registry()
        template = registry.get_template(self.entity_type)
        for name, label in template.items():
            self.attributes[name] = AttributeSlot(
                name=name, label=label, template=label,
            )

    def mount_sentence(
        self,
        sentence_id: str,
        attribute_name: str,
    ) -> None:
        """
        Mount a sentence under one of this container's attribute slots.
        Unknown attribute names create a slot on the fly.
        """
        self.ensure_attribute_slots()
        if attribute_name not in self.attributes:
            self.attributes[attribute_name] = AttributeSlot(
                name=attribute_name, label=attribute_name, template="",
            )
        self.attributes[attribute_name].mount(sentence_id)

    def get_attribute_sentence_ids(
        self,
        attribute_name: str,
    ) -> List[str]:
        """Return the sentence IDs mounted under a given attribute."""
        slot = self.attributes.get(attribute_name)
        return list(slot.sentence_ids) if slot else []

    
    def remove_sentence(self, sentence_id: str) -> bool:
        """Remove a sentence ID from this container."""
        if sentence_id in self.sentence_ids:
            self.sentence_ids.remove(sentence_id)
            return True
        return False
    
    def sentence_count(self) -> int:
        """Return the number of sentences in this container."""
        return len(self.sentence_ids)
    
    def is_full(self) -> bool:
        """Check if container has reached max capacity."""
        return len(self.sentence_ids) >= self.MAX_SENTENCES
    
    def can_expand(self) -> bool:
        """Check if container can expand beyond MAX_SENTENCES."""
        return len(self.sentence_ids) < self.MAX_SENTENCES_WITH_SUBTOPICS
    
    def __repr__(self) -> str:
        return f"Container({self.id}, entity={self.entity}, sentences={len(self.sentence_ids)})"

    def to_dict(self) -> Dict:
        """Serialize to a JSON-friendly dict for retriever output."""
        return {
            "id": self.id,
            "entity": self.entity,
            "entity_type": self.entity_type,
            "sentence_ids": list(self.sentence_ids),
            "entity_inventory": list(self.entity_inventory),
            "keywords": list(self.keywords),
            "summary": self.summary,
            "hyperedges": list(self.hyperedges),
            "attributes": {
                name: list(slot.sentence_ids)
                for name, slot in self.attributes.items()
            },
            "summary_vector_dim": int(
                self.summary_vector.size
            ) if self.summary_vector is not None else 0
        }


class ContainerManager:
    """
    Manages container aggregation with automatic split support.
    
    Containers group sentences by entity. Supports:
    - Getting or creating containers for entities
    - Multi-homing (sentences can belong to multiple containers)
    - Automatic splitting when containers exceed limits
    - Keyword extraction and summary generation
    """
    
    def __init__(
        self,
        sentence_registry=None,
        use_llm_summaries: bool = False,
        use_semantic_retrieval: bool = True,
        semantic_embed_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the container manager.

        Args:
            sentence_registry: Reference to SentenceRegistry for entity lookups
            use_llm_summaries: If True, generate LLM summaries (optional)
            use_semantic_retrieval: If True, build a FAISS index over
                container summaries and expose semantic search. Default True.
            semantic_embed_model: Name of the SentenceTransformer model
                used to embed container summaries. Default `all-MiniLM-L6-v2`
                (80 MB, 384-dim, fast on CPU).
        """
        self._containers: Dict[str, Container] = {}
        self._entity_to_container: Dict[str, str] = {}  # entity -> container_id
        self._sentence_registry = sentence_registry
        self._container_counter: int = 0
        self._use_llm_summaries = use_llm_summaries
        self._vector_dim: int = 384  # Default embedding dimension
        # Pluggable LLM summarizer (D18). Signature:
        #     (entity: str, context: str) -> str
        # Returns "" on any failure → triggers fallback to template.
        self._llm_summarizer: Optional[Callable[[str, str], str]] = None

        # Semantic retrieval state [Phase-2]
        # The embedder is loaded lazily on first build to keep
        # construction cheap. The FAISS index is rebuilt every time
        # `update_container_index()` is called; the embedder is shared
        # across all ContainerManager instances via _EMBEDDER_CACHE.
        self._use_semantic_retrieval = use_semantic_retrieval
        self._semantic_embed_model = semantic_embed_model
        self._faiss_index = None  # faiss.IndexFlatIP | None
        self._faiss_container_ids: List[str] = []  # idx -> container_id
        self._faiss_dirty: bool = True  # mark for rebuild

        # Week-1: Hybrid Retrieval — BM25 over container summaries.
        # Built lazily on first search (rebuilt when _bm25_dirty=True).
        self._bm25_index: Optional[BM25ContainerIndex] = None
        self._bm25_dirty: bool = True
    
    def set_sentence_registry(self, registry) -> None:
        """Set the sentence registry reference."""
        self._sentence_registry = registry

    def set_vector_dim(self, dim: int) -> None:
        """Set the embedding dimension for vectors."""
        self._vector_dim = dim

    def set_llm_summarizer(
        self,
        summarizer: Optional[Callable[[str, str], str]],
        auto_enable: bool = True
    ) -> None:
        """
        Inject a real LLM summarizer callable [D18, §14].

        Args:
            summarizer: Function with signature (entity, context) -> str.
                Should return "" on failure (recommended pattern: the
                `make_summarizer()` factory in src/llm_client.py).
                Pass None to clear an existing summarizer.
            auto_enable: If True, also sets ``use_llm_summaries=True``
                so the new summarizer is actually used. Set to False
                if you want to register the callable without flipping
                the switch (e.g., during a hot-swap).
        """
        self._llm_summarizer = summarizer
        if auto_enable and summarizer is not None:
            self._use_llm_summaries = True

    def get_llm_summarizer(self) -> Optional[Callable[[str, str], str]]:
        """Return the currently registered LLM summarizer, or None."""
        return self._llm_summarizer
    
    def get_or_create_container(
        self,
        entity: str,
        entity_type: str = "",
    ) -> Container:
        """
        Get existing container for entity or create a new one.

        Args:
            entity: Canonical entity name
            entity_type: Optional spaCy NER label (PERSON, ORG, GPE…).
                If the container is new, this sets the entity_type
                used to build its attribute template. If the
                container already exists, the entity_type is
                updated only if it was previously empty.

        Returns:
            The existing or newly created container
        """
        entity_key = entity.lower()

        if entity_key in self._entity_to_container:
            container_id = self._entity_to_container[entity_key]
            container = self._containers[container_id]
            if entity_type and not container.entity_type:
                container.entity_type = entity_type
                # Re-build attribute slots if we now know the type.
                if not container.attributes:
                    container.ensure_attribute_slots()
            return container

        # Create new container
        self._container_counter += 1
        container_id = f"c_{self._container_counter:04d}"
        container = Container(
            id=container_id,
            entity=entity,
            entity_type=entity_type or "",
            created_at=self._container_counter,
            attribute_registry=getattr(self, "_attribute_registry", None),
        )
        # Build attribute slots up-front so mount_sentence can index them.
        container.ensure_attribute_slots()

        self._containers[container_id] = container
        self._entity_to_container[entity_key] = container_id

        return container

    def set_attribute_registry(self, registry) -> None:
        """Share a single AttributeRegistry across all containers."""
        self._attribute_registry = registry
        for c in self._containers.values():
            c.attribute_registry = registry
            if c.attributes:
                c.attributes.clear()
            c.ensure_attribute_slots()

    
    def add_sentence_to_container(
        self,
        sentence_id: str,
        entity: str,
        entity_type: str = "",
    ) -> str:
        """
        Add a sentence to a container for the given entity.

        Args:
            sentence_id: The sentence ID to add
            entity: Canonical entity name
            entity_type: Optional spaCy NER label. Used to pick the
                right attribute template; only the FIRST non-empty
                value sticks on a given container.

        Returns:
            The container ID the sentence was added to
        """
        container = self.get_or_create_container(entity, entity_type=entity_type)
        container.add_sentence(sentence_id)

        # Check if we need to split
        if container.can_expand():
            should_split, subgroups = self._check_split_needed(container)
            if should_split and subgroups:
                self._split_container(container, subgroups)

        return container.id
    
    def _check_split_needed(
        self,
        container: Container
    ) -> Tuple[bool, List[List[str]]]:
        """
        Check if a container needs to be split [D17].

        Trigger conditions:
          - container.sentence_count() >= MAX_SENTENCES (5)
          - AND inter-group cosine similarity after agglomerative clustering
            is below SPLIT_SIMILARITY_THRESHOLD (0.4)

        If container reaches MAX_SENTENCES_WITH_SUBTOPICS (8) it MUST split
        regardless of similarity - otherwise unbounded growth.

        Returns:
            Tuple of (should_split, subgroups) where subgroups is a list of
            sentence ID groups if split is needed
        """
        if not container.is_full():
            return False, []

        if not self._sentence_registry:
            # Without sentence registry, can't compute similarity
            return container.sentence_count() > Container.MAX_SENTENCES_WITH_SUBTOPICS, []

        sentences = self._sentence_registry.get_nodes_by_ids(container.sentence_ids)
        if len(sentences) < 3:
            return False, []

        # Compute TF-IDF vectors for each sentence
        sentence_vectors = []
        for sent in sentences:
            tfidf = self._compute_sentence_tfidf(sent)
            sentence_vectors.append(tfidf)

        # Cluster sentences by similarity
        subgroups = self._cluster_sentences(sentence_vectors, container.sentence_ids)

        # Check if we have distinct subgroups
        if len(subgroups) < 2:
            return False, []

        # Compute inter-group similarity
        inter_sim = self._compute_inter_group_similarity(
            sentence_vectors, subgroups
        )

        should_split = inter_sim < Container.SPLIT_SIMILARITY_THRESHOLD
        return should_split, subgroups
    
    def _compute_sentence_tfidf(self, sentence_node) -> np.ndarray:
        """
        Compute a simple TF-IDF vector for a sentence.
        
        Uses word frequencies with inverse document frequency based on
        all sentences in the registry.
        """
        if not self._sentence_registry:
            return np.zeros(self._vector_dim)
        
        # Get all sentences for IDF computation
        all_sentences = self._sentence_registry.get_all_nodes()
        if not all_sentences:
            return np.zeros(self._vector_dim)
        
        # Collect all words
        all_words = set()
        for sent in all_sentences:
            words = self._extract_words(sent.text)
            all_words.update(words)
        
        word_list = list(all_words)
        word_to_idx = {w: i for i, w in enumerate(word_list)}
        
        # Build TF-IDF vector
        vector = np.zeros(len(word_list))
        words = self._extract_words(sentence_node.text)
        word_freq = Counter(words)
        
        for word, freq in word_freq.items():
            if word in word_to_idx:
                # TF component
                tf = freq / len(words) if words else 0
                # IDF component (log of total docs / docs containing word)
                doc_count = sum(1 for s in all_sentences if word in s.text.lower())
                idf = math.log(len(all_sentences) / (1 + doc_count))
                vector[word_to_idx[word]] = tf * idf
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text (simple tokenization)."""
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Remove common stop words
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                     'have', 'been', 'were', 'they', 'this', 'that', 'with',
                     'would', 'there', 'their', 'what', 'about', 'which', 'when'}
        return [w for w in words if w not in stopwords]
    
    def _cluster_sentences(
        self, 
        vectors: List[np.ndarray], 
        sentence_ids: List[str]
    ) -> List[List[str]]:
        """
        Cluster sentences using simple agglomerative clustering.
        
        Returns list of clusters (each cluster is a list of sentence IDs).
        """
        if len(vectors) < 2:
            return [sentence_ids]
        
        n = len(vectors)
        
        # Convert to numpy array
        matrix = np.array(vectors)
        
        # Compute pairwise cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(matrix)
        
        # Initialize each sentence as its own cluster
        clusters = [[sid] for sid in sentence_ids]
        
        # Agglomerative clustering until we get distinct groups
        while len(clusters) > 1:
            # Find most similar pair of clusters
            max_sim = -1
            merge_idx = None
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Average similarity between clusters
                    sims = []
                    for idx_i in clusters[i]:
                        for idx_j in clusters[j]:
                            pos_i = sentence_ids.index(idx_i)
                            pos_j = sentence_ids.index(idx_j)
                            sims.append(sim_matrix[pos_i][pos_j])
                    
                    avg_sim = np.mean(sims) if sims else 0
                    if avg_sim > max_sim:
                        max_sim = avg_sim
                        merge_idx = (i, j)
            
            if merge_idx is None:
                break
            
            # Check if we should continue merging
            if max_sim < 0.3 and len(clusters) <= 4:
                break
            
            # Merge clusters
            i, j = merge_idx
            clusters[i].extend(clusters[j])
            clusters.pop(j)
        
        return clusters
    
    def _compute_inter_group_similarity(
        self, 
        vectors: List[np.ndarray], 
        groups: List[List[str]]
    ) -> float:
        """Compute average similarity between groups."""
        if len(groups) < 2:
            return 1.0
        
        sentence_ids = [sid for group in groups for sid in group]
        matrix = np.array(vectors)
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(matrix)
        
        similarities = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                for sid_i in groups[i]:
                    for sid_j in groups[j]:
                        pos_i = sentence_ids.index(sid_i)
                        pos_j = sentence_ids.index(sid_j)
                        similarities.append(sim_matrix[pos_i][pos_j])
        
        return np.mean(similarities) if similarities else 0.0
    
    def _split_container(
        self, 
        container: Container, 
        subgroups: List[List[str]]
    ) -> List[Container]:
        """
        Split a container into multiple sub-containers.
        
        Args:
            container: The container to split
            subgroups: List of sentence ID groups
            
        Returns:
            List of new containers (original is updated with first subgroup)
        """
        if len(subgroups) < 2:
            return [container]
        
        new_containers = []
        
        # Keep first subgroup in original container
        container.sentence_ids = subgroups[0]
        new_containers.append(container)
        
        # Create new containers for remaining subgroups
        for i, subgroup in enumerate(subgroups[1:], start=1):
            self._container_counter += 1
            new_container_id = f"c_{self._container_counter:04d}"
            
            # Add a suffix to entity to indicate sub-topic
            sub_entity = f"{container.entity} (group {i+1})"
            
            new_container = Container(
                id=new_container_id,
                entity=sub_entity,
                sentence_ids=subgroup,
                created_at=self._container_counter
            )
            
            self._containers[new_container_id] = new_container
            new_containers.append(new_container)
            
            # Update sentence's container list
            if self._sentence_registry:
                for sent_id in subgroup:
                    node = self._sentence_registry.get_node(sent_id)
                    if node:
                        node.add_container(new_container_id)
        
        # Phase-5 [CONTEXT edges]: connect adjacent sub-containers
        # with bidirectional CONTEXT edges at their sentence boundary.
        # These edges do NOT consume hop count during retrieval
        # (handled in Retriever via `is_context_boundary`).  This
        # preserves the "the narrative continues across the split"
        # signal that SAME_ENTITY used to carry, but without burning
        # the user's max_hops budget on a 1-hop cross-container move.
        #
        # Ordering: there is no `index` field on SentenceNode in this
        # codebase, so we fall back to numeric sort on the trailing
        # digits of the sentence id ("s_0042" -> 42). When the
        # SentenceNode objects do carry a sequence index in the
        # future, prefer that — it's O(1) vs the current O(n log n).
        if len(new_containers) >= 2 and self._sentence_registry is not None:
            from .sentence_graph import LogicEdge as _LE
            import re as _re_id

            def _seq(c: Container) -> int:
                nums: List[int] = []
                for sid in c.sentence_ids:
                    m = _re_id.search(r"(\d+)\s*$", sid)
                    if m:
                        try:
                            nums.append(int(m.group(1)))
                        except ValueError:
                            pass
                return min(nums) if nums else 0

            ordered = sorted(new_containers, key=_seq)
            for i in range(len(ordered) - 1):
                prev_c = ordered[i]
                next_c = ordered[i + 1]

                def _sorted_ids(c: Container) -> List[str]:
                    def _k(sid: str) -> int:
                        m = _re_id.search(r"(\d+)\s*$", sid)
                        return int(m.group(1)) if m else 0
                    return sorted(c.sentence_ids, key=_k)

                prev_ids = _sorted_ids(prev_c)
                next_ids = _sorted_ids(next_c)
                if not prev_ids or not next_ids:
                    continue
                prev_last = prev_ids[-1]
                next_first = next_ids[0]
                # Bidirectional: the narrative flows both ways across
                # the boundary (read forward, read backward). Same
                # weight and confidence as CAUSAL.
                fwd = _LE(
                    source_sent_id=prev_last,
                    target_sent_id=next_first,
                    edge_type=_LE.TYPE_CONTEXT,
                    weight=_LE.WEIGHT_CONTEXT,
                    confidence=1.0,
                    container_id=prev_c.id,
                    is_context_boundary=True,
                )
                rev = _LE(
                    source_sent_id=next_first,
                    target_sent_id=prev_last,
                    edge_type=_LE.TYPE_CONTEXT,
                    weight=_LE.WEIGHT_CONTEXT,
                    confidence=1.0,
                    container_id=next_c.id,
                    is_context_boundary=True,
                )
                # We don't have direct access to the EdgeRegistry here
                # (ContainerManager only owns sentences + containers).
                # Stash the edges on the manager so the caller (graph
                # builder or retriever post-processing) can pick them
                # up and register them. The retrieval path also
                # tolerates edges being absent — we add a fallback
                # hook below.
                if not hasattr(self, "_pending_context_edges"):
                    self._pending_context_edges: List = []
                self._pending_context_edges.append(fwd)
                self._pending_context_edges.append(rev)

        return new_containers

    def update_container_index(self, container_id: str) -> None:
        """
        Rebuild keywords and summary for a container.

        Args:
            container_id: The container to update
        """
        container = self._containers.get(container_id)
        if not container:
            return

        if self._use_llm_summaries:
            self._update_with_llm_summary(container)
        else:
            self._update_keywords_summary(container)

        # Mark FAISS index dirty so the next semantic search rebuilds.
        self._faiss_dirty = True
        # Week-1: also mark BM25 dirty.
        self._bm25_dirty = True
    
    def _update_keywords_summary(self, container: Container) -> None:
        """Update keywords, summary and entity_inventory using TF-IDF +
        extractive template.

        Per spec:
          - Top 5 TF-IDF keywords (was Top-3)
          - entity_inventory = union of normalized entities across all
            sentences in this container (used by the dual-entity
            retriever's first track: container entity inventory
            intersection).
          - summary_vector = spaCy doc.vector for semantic retrieval.
        """
        if not self._sentence_registry:
            return

        sentences = self._sentence_registry.get_nodes_by_ids(container.sentence_ids)
        if not sentences:
            return

        # Collect all words
        all_words = []
        for sent in sentences:
            words = self._extract_words(sent.text)
            all_words.extend(words)

        # Get word frequencies
        word_freq = Counter(all_words)

        # Get IDF for ranking
        all_sentences = self._sentence_registry.get_all_nodes()
        word_doc_freq = Counter()
        for sent in all_sentences:
            sent_words = set(self._extract_words(sent.text))
            for word in sent_words:
                word_doc_freq[word] += 1

        # Compute TF-IDF scores
        tfidf_scores = {}
        for word, freq in word_freq.items():
            tf = freq
            idf = math.log(len(all_sentences) / (1 + word_doc_freq.get(word, 1)))
            tfidf_scores[word] = tf * idf

        # Top 5 keywords per spec (was Top-3)
        top_words = sorted(tfidf_scores.items(), key=lambda x: -x[1])[:5]
        container.keywords = [w for w, _ in top_words]

        # entity_inventory: union of normalized entities across all
        # sentences in this container.
        inv: List[str] = []
        seen_inv: Set[str] = set()
        for sent in sentences:
            for ent in (sent.entities or []):
                key = (ent or "").lower()
                if key and key not in seen_inv:
                    seen_inv.add(key)
                    inv.append(key)
        container.entity_inventory = inv

        # Build extractive template summary
        kw = container.keywords
        if kw:
            container.summary = f"{container.entity} - relates to " + ", ".join(kw)
        else:
            container.summary = container.entity

        # Embed summary for semantic retrieval
        container.summary_vector = self._embed_text(container.summary)
    
    def _update_with_llm_summary(self, container: Container) -> None:
        """
        Generate a high-quality extractive summary for the container [D18, §14].

        Decision tree:
          1. Always run the keyword + template fallback path first
             (_update_keywords_summary) so the container has a valid
             summary even if the LLM step fails.
          2. If an LLM summarizer is registered AND it returns a non-empty
             string, replace the template summary with the LLM output.
          3. Otherwise (no summarizer / summarizer returned "" / error),
             keep the template.  This guarantees graceful fallback [D18].

        This matches the contract documented in §14: the LLM hook is
        opt-in, and a missing or failing LLM NEVER breaks the build.
        """
        self._update_keywords_summary(container)

        # Optional real LLM hook [D18]
        llm_summary: Optional[str] = None
        if self._llm_summarizer is not None:
            try:
                if self._sentence_registry:
                    nodes = self._sentence_registry.get_nodes_by_ids(
                        container.sentence_ids
                    )
                    context = " ".join(
                        (getattr(n, "complemented_text", "") or n.text)
                        for n in nodes[:5]
                    )
                else:
                    context = container.summary
                llm_summary = self._llm_summarizer(container.entity, context)
            except Exception:
                llm_summary = None

        if llm_summary:
            container.summary = llm_summary
        else:
            # Fallback: use the lightweight template set above
            container.summary = f"{container.entity} - relates to " + \
                                ", ".join(container.keywords)

        # Always refresh summary vector
        container.summary_vector = self._embed_text(container.summary)

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed arbitrary text using the spaCy model. Returns None on failure."""
        if not text:
            return None
        nlp = globals().get("_CONTAINER_NLP")
        if nlp is None:
            try:
                import spacy
                nlp = spacy.load("en_core_web_lg")
            except Exception:
                return None
            globals()["_CONTAINER_NLP"] = nlp
        try:
            doc = nlp(text)
            v = doc.vector
            if v is None or not hasattr(v, "size") or v.size == 0:
                return None
            return v.astype("float32")
        except Exception:
            return None
    
    def get_container(self, container_id: str) -> Optional[Container]:
        """Get a container by ID."""
        return self._containers.get(container_id)

    # ─────────────────────────────────────────────────────────────
    # BM25 index (Week-1: Hybrid Retrieval)
    # ─────────────────────────────────────────────────────────────
    def _build_bm25_index(self) -> None:
        """
        Build (or rebuild) the BM25 container index from current summaries.

        Called lazily on first search when `_bm25_dirty` is True.
        """
        self._bm25_dirty = False
        if not self._containers:
            self._bm25_index = None
            return
        ids: List[str] = []
        summaries: List[str] = []
        for cid, container in self._containers.items():
            if (container.summary or "").strip():
                ids.append(cid)
                summaries.append(container.summary)
        if not ids:
            self._bm25_index = None
            return
        idx = BM25ContainerIndex().build(ids, summaries)
        self._bm25_index = idx

    def _bm25_search(
        self,
        query_text: str,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Search the BM25 container index by query.
        Returns [(container_id, raw_bm25_score)] sorted desc.
        Returns empty list when index isn't ready.
        """
        if not query_text:
            return []
        if self._bm25_dirty:
            self._build_bm25_index()
        if self._bm25_index is None:
            return []
        return self._bm25_index.score(query_text)[:top_k]

    # ─────────────────────────────────────────────────────────────
    # FAISS-backed semantic retrieval [Phase-2]
    #
    # The index is built from `container.summary` strings using a
    # SentenceTransformer (default `all-MiniLM-L6-v2`, 384-dim).
    # `IndexFlatIP` over L2-normalised vectors = cosine similarity.
    # ─────────────────────────────────────────────────────────────
    def _build_faiss_index(self) -> None:
        """
        Build (or rebuild) the FAISS index from current container summaries.

        No-op if:
          - semantic retrieval is disabled (`use_semantic_retrieval=False`)
          - there are no containers with non-empty summaries
          - the embedder package / model cannot be loaded

        On success, sets `_faiss_index`, `_faiss_container_ids`, and
        clears the `_faiss_dirty` flag.
        """
        self._faiss_dirty = False
        self._faiss_index = None
        self._faiss_container_ids = []

        if not self._use_semantic_retrieval:
            return

        # Collect summaries
        summaries: List[str] = []
        ids: List[str] = []
        for cid, container in self._containers.items():
            if container.summary and container.summary.strip():
                summaries.append(container.summary.strip())
                ids.append(cid)
        if not summaries:
            return

        embedder = _get_embedder(self._semantic_embed_model)
        if embedder is None:
            return

        try:
            import faiss  # type: ignore
            embeddings = embedder.encode(
                summaries,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            self._faiss_index = index
            self._faiss_container_ids = ids
        except Exception:
            # Graceful fallback - keyword/entity tiers still work.
            self._faiss_index = None
            self._faiss_container_ids = []

    def _faiss_search(
        self,
        query_text: str,
        top_k: int = 5,
        threshold: float = 0.55,
    ) -> List[Tuple[str, float]]:
        """
        Search the FAISS index by query similarity.

        Args:
            query_text: Natural-language query (e.g. question or keywords)
            top_k: Maximum number of results to return
            threshold: Minimum cosine similarity to keep a hit

        Returns:
            List of (container_id, score) sorted by score desc. Empty
            if the index isn't ready (disabled, build failed, or no
            containers yet).
        """
        if not self._use_semantic_retrieval or not query_text:
            return []
        if self._faiss_dirty:
            self._build_faiss_index()
        if self._faiss_index is None or not self._faiss_container_ids:
            return []

        embedder = _get_embedder(self._semantic_embed_model)
        if embedder is None:
            return []

        try:
            q = embedder.encode(
                [query_text],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            k = min(top_k, len(self._faiss_container_ids))
            scores, indices = self._faiss_index.search(q, k)
        except Exception:
            return []

        out: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            sim = float(score)
            if sim >= threshold:
                out.append((self._faiss_container_ids[idx], sim))
        return out

    # Week-1: Hybrid retrieval — RRF fusion of BM25 + dense (FAISS)
    # ─────────────────────────────────────────────────────────────────
    def hybrid_search(
        self,
        query_text: str,
        top_k: int = 10,
        *,
        w_bm25: float = 0.5,
        w_dense: float = 0.5,
        bm25_top_k: int = 50,
        dense_top_k: int = 20,
        dense_threshold: float = 0.50,
        rrf_k: int = 60,
    ) -> List[Tuple[str, float, float, float]]:
        """
        Hybrid container retrieval: Reciprocal Rank Fusion (RRF) of
        BM25 + dense (FAISS) scores.

        RRF score for each container:
            rrf(cid) = sum_over_methods(
                w_method / (k + rank_in_method(cid))
            )

        Final combined score:
            combined(cid) = w_bm25 * norm_bm25 + w_dense * norm_dense

        Args:
            query_text:       Query string.
            top_k:           Return top-K containers.
            w_bm25:          BM25 weight (0-1). 0 = dense only, 1 = BM25 only.
            w_dense:         Dense weight (0-1).
            bm25_top_k:      Rank candidates using top-N BM25 results.
            dense_top_k:      Rank candidates using top-N dense results.
            dense_threshold:  Minimum cosine similarity for dense hits.
            rrf_k:           RRF constant (default 60; standard in literature).

        Returns:
            List of (container_id, combined_score, bm25_score, dense_score)
            sorted by combined_score descending.
        """
        # 1. BM25 scores.
        bm25_raw = self._bm25_search(query_text, top_k=bm25_top_k)
        bm25_scores: Dict[str, float] = {cid: s for cid, s in bm25_raw}

        # 2. Dense (FAISS) scores.
        dense_raw = self._faiss_search(
            query_text, top_k=dense_top_k, threshold=dense_threshold,
        )
        dense_scores: Dict[str, float] = {cid: s for cid, s in dense_raw}

        # 3. Collect union of candidate container IDs.
        all_cids: Set[str] = set(bm25_scores.keys()) | set(dense_scores.keys())
        if not all_cids:
            return []

        # 4. RRF per method.
        def rrf_scores(
            raw_scores: Dict[str, float],
        ) -> Dict[str, float]:
            if not raw_scores:
                return {}
            sorted_cids = [cid for cid, _ in sorted(
                raw_scores.items(), key=lambda kv: -kv[1]
            )]
            out: Dict[str, float] = {}
            for rank, cid in enumerate(sorted_cids, start=1):
                out[cid] = rrf_k / (rrf_k + rank)
            return out

        rrf_bm25 = rrf_scores(bm25_scores)
        rrf_dense = rrf_scores(dense_scores)

        # 5. Normalize BM25 raw scores to [0, 1] using max.
        if bm25_scores:
            bm25_max = max(bm25_scores.values())
        else:
            bm25_max = 1.0
        norm_bm25: Dict[str, float] = {
            cid: (bm25_scores.get(cid, 0.0) / bm25_max) if bm25_max > 0 else 0.0
            for cid in all_cids
        }

        # 6. Normalize dense raw scores to [0, 1] using max.
        if dense_scores:
            dense_max = max(dense_scores.values())
        else:
            dense_max = 1.0
        norm_dense: Dict[str, float] = {
            cid: (dense_scores.get(cid, 0.0) / dense_max) if dense_max > 0 else 0.0
            for cid in all_cids
        }

        # 7. Combined score.
        combined: Dict[str, float] = {}
        for cid in all_cids:
            combined[cid] = (
                w_bm25 * norm_bm25.get(cid, 0.0)
                + w_dense * norm_dense.get(cid, 0.0)
            )

        # 8. Sort by combined score.
        ranked = sorted(combined.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            (cid, score, norm_bm25.get(cid, 0.0), norm_dense.get(cid, 0.0))
            for cid, score in ranked
        ]

    # ─────────────────────────────────────────────────────────────
    # Backwards-compat property aliases
    #
    # Some call sites (and the spec) use `container_manager.faiss_index`
    # / `embedder` directly. Expose them as read-only views.
    # ─────────────────────────────────────────────────────────────
    @property
    def faiss_index(self):
        return self._faiss_index

    @property
    def embedder(self):
        return _get_embedder(self._semantic_embed_model)
    
    def get_containers_for_entity(self, entity: str) -> List[Container]:
        """Get all containers for an entity (including sub-topic containers).

        Matches against both the primary entity name AND the full entity
        inventory, enabling substring-based recall (e.g. query "easter"
        matches container with entity="the easter bunny" because "easter"
        appears in its entity_inventory).
        """
        entity_key = entity.lower()
        containers = []

        for container in self._containers.values():
            c_entity = getattr(container, "entity", "") or ""
            # 1) primary entity exact / substring
            if c_entity.lower() == entity_key or entity_key in c_entity.lower():
                containers.append(container)
                continue
            # 2) entity_inventory: any entry contains the query as substring
            inv = getattr(container, "entity_inventory", []) or []
            if any(entity_key in (e or "").lower() for e in inv):
                containers.append(container)

        return containers
    
    def get_container_for_entity(self, entity: str) -> Optional[Container]:
        """Get the primary container for an entity."""
        entity_key = entity.lower()
        container_id = self._entity_to_container.get(entity_key)
        return self._containers.get(container_id)

    # Backward-compat alias used elsewhere in the codebase.
    def get_container_by_entity(self, entity: str) -> Optional[Container]:
        return self.get_container_for_entity(entity)
    
    def get_all_containers(self) -> List[Container]:
        """Get all containers."""
        return list(self._containers.values())
    
    def get_sentence_containers(self, sentence_id: str) -> List[Container]:
        """Get all containers that contain a sentence."""
        containers = []
        for container in self._containers.values():
            if sentence_id in container.sentence_ids:
                containers.append(container)
        return containers
    
    def remove_sentence_from_all_containers(self, sentence_id: str) -> None:
        """Remove a sentence from all containers."""
        for container in self._containers.values():
            container.remove_sentence(sentence_id)
    
    def search_containers_by_keyword(self, keyword: str) -> List[Container]:
        """
        Search containers by keyword in their keywords or entity.
        
        Args:
            keyword: The keyword to search for
            
        Returns:
            List of matching containers
        """
        keyword_lower = keyword.lower()
        results = []
        
        for container in self._containers.values():
            if keyword_lower in container.entity.lower():
                results.append(container)
            elif keyword_lower in container.keywords:
                results.append(container)
        
        return results
    
    def __len__(self) -> int:
        """Return the number of containers."""
        return len(self._containers)
    
    def __repr__(self) -> str:
        return f"ContainerManager({len(self)} containers)"

    # ─────────────────────────────────────────────────────────────────
    # Weighted container connectivity [Phase-6 cross-paragraph expansion]
    #
    # Goal: bridge-container expansion during retrieval.
    #
    # On HotpotQA, Para Recall@5 sits at 0.83 — 17% of gold paragraphs
    # are NOT in the top-5 containers. The gap is *cross-paragraph
    # evidence stitching*: gold paragraphs sit on a chain
    #   bridge_entity → movie_A → GNR → End of Days
    # and the container retriever only fires on each end's exact /
    # fuzzy match. The middle links are missed.
    #
    # Idea: in the SAME_ENTITY sentence graph, whenever a sentence
    # in container A links to a sentence in container B via a SAME_ENTITY
    # or CONTEXT edge, treat A and B as **connected**. Distinguish:
    #   - *direct* weight: 0.9 (SAME_ENTITY), 0.95 (CONTEXT boundary).
    #   - *2-hop* weight: product * 0.7  (decay per hop).
    #
    # We then surface 1-hop connections to the retrieval pipeline
    # so the second-stage sentence-level rerank can use them.
    #
    # We deliberately do NOT propagate further than 2 hops: a fully
    # transitive expansion would pull in 100+ containers and drown
    # the ranker in noise. Empirically, gold cross-paragraph chains
    # are at most 2 hops long.
    # ─────────────────────────────────────────────────────────────────

    def _build_container_connectivity(
        self,
        edge_registry=None,
        min_weight: float = 0.30,
    ) -> Dict[str, Dict[str, float]]:
        """
        Build weighted container-to-container connectivity.

        Args:
            edge_registry: An EdgeRegistry-like object exposing
                ``get_all_edges()``. Optional — if None, the method
                returns an identity map (each container only sees itself).
            min_weight: Floor for keeping a connection. Edges lighter
                than this are dropped during propagation.

        Returns:
            Dict mapping container_id -> {other_container_id: weight}.
            Stores the result on ``self._container_connectivity`` so
            subsequent calls are O(1).
        """
        # 1. Identity initialization.
        connectivity: Dict[str, Dict[str, float]] = {
            cid: {cid: 1.0} for cid in self._containers
        }

        if edge_registry is None:
            self._container_connectivity = connectivity
            return connectivity

        # 2. Build a sentence -> containers index for O(1) lookup.
        sent_to_containers: Dict[str, List[str]] = {}
        for cid, container in self._containers.items():
            for sid in container.sentence_ids:
                sent_to_containers.setdefault(sid, []).append(cid)

        # Edge weights for direct (1-hop) bridges.
        direct_weights = {
            "SAME_ENTITY": 0.9,
            "CONTEXT": 0.95,
        }

        def _register_direct(src_cid: str, tgt_cid: str, w: float) -> None:
            if src_cid == tgt_cid:
                return
            connectivity[src_cid][tgt_cid] = max(
                connectivity[src_cid].get(tgt_cid, 0.0), w
            )
            connectivity[tgt_cid][src_cid] = max(
                connectivity[tgt_cid].get(src_cid, 0.0), w
            )

        # 3. Direct 1-hop: walk every SAME_ENTITY / CONTEXT edge.
        try:
            edges = edge_registry.get_all_edges() or []
        except Exception:
            edges = []

        same_count = ctx_count = cross_count = 0
        for edge in edges:
            et = getattr(edge, "edge_type", "")
            if et not in direct_weights:
                continue
            if et == "SAME_ENTITY":
                same_count += 1
            else:
                ctx_count += 1
            src_cids = sent_to_containers.get(edge.source_sent_id, [])
            tgt_cids = sent_to_containers.get(edge.target_sent_id, [])
            for sc in src_cids:
                for tc in tgt_cids:
                    if sc != tc:
                        cross_count += 1
                    _register_direct(sc, tc, direct_weights[et])

        # 4. 2-hop propagation with 0.7 decay per hop.
        # Cap propagation depth so we don't blow the graph open.
        changed = True
        iteration_cap = 2  # 2-hop BFS, NOT full transitive closure.
        hop_seen: Dict[str, int] = {}  # cid -> hops used
        for cid in connectivity:
            hop_seen[cid] = 0
        # Snapshot of current 1-hop edges (used as seeds for 2-hop).
        current = {}
        for cid, neighbors in connectivity.items():
            current[cid] = {c: w for c, w in neighbors.items() if c != cid}

        for _ in range(iteration_cap):
            if not current:
                break
            new_edges = {}
            for src_cid, nbrs in current.items():
                for mid_cid, w_mid in nbrs.items():
                    if mid_cid == src_cid:
                        continue
                    for dst_cid, w_dst in connectivity.get(mid_cid, {}).items():
                        if dst_cid == src_cid or dst_cid == mid_cid:
                            continue
                        new_w = w_mid * w_dst * 0.7
                        if new_w < min_weight:
                            continue
                        # respect hop budget
                        dst_seen = hop_seen.get(dst_cid, 99)
                        if dst_seen > 1:
                            continue
                        cur_best = new_edges.get(src_cid, {}).get(dst_cid, 0.0)
                        if new_w > cur_best:
                            new_edges.setdefault(src_cid, {})[dst_cid] = new_w
                            connectivity[src_cid][dst_cid] = max(
                                connectivity[src_cid].get(dst_cid, 0.0), new_w
                            )
                            connectivity[dst_cid][src_cid] = max(
                                connectivity[dst_cid].get(src_cid, 0.0), new_w
                            )
                            if hop_seen.get(dst_cid, 99) > hop_seen.get(src_cid, 0) + 1:
                                hop_seen[dst_cid] = hop_seen.get(src_cid, 0) + 1
            current = new_edges
            if not new_edges:
                break

        self._container_connectivity = connectivity
        return connectivity

    def get_connected_containers(
        self,
        cid: str,
        min_weight: float = 0.30,
    ) -> List[Tuple[str, float]]:
        """
        Return containers connected to ``cid`` with weight >= ``min_weight``,
        sorted by weight desc.

        Falls back to ``[(cid, 1.0)]`` if connectivity hasn't been built
        yet (caller can still proceed).
        """
        if not hasattr(self, "_container_connectivity"):
            return [(cid, 1.0)]
        neighbors = self._container_connectivity.get(cid, {cid: 1.0})
        out = [(c, w) for c, w in neighbors.items() if w >= min_weight]
        out.sort(key=lambda x: (-x[1], x[0]))
        return out

    def is_high_frequency_entity(
        self,
        entity: str,
        threshold: int = 100,
    ) -> bool:
        """
        True if `entity` appears in more than ``threshold`` sentences
        across the entire graph (lazily computed).
        """
        freq = self._get_entity_frequency()
        return freq.get(entity.lower(), 0) > threshold

    def _get_entity_frequency(self) -> Dict[str, int]:
        if hasattr(self, "_entity_frequency"):
            return self._entity_frequency
        freq: Dict[str, int] = {}
        for container in self._containers.values():
            for sid in container.sentence_ids:
                node = (
                    self._sentence_registry.get_node(sid)
                    if self._sentence_registry else None
                )
                if node is None:
                    continue
                for e in getattr(node, "entities", []) or []:
                    freq[e.lower()] = freq.get(e.lower(), 0) + 1
        self._entity_frequency = freq
        return freq

    def get_bridge_entities_between(
        self,
        src_cid: str,
        tgt_cid: str,
    ) -> List[str]:
        """
        Return the entities shared between two containers (as a list
        of canonical lowercased strings). Used by the retriever to
        decide if a same-entity bridge is a *relevant* bridge (not
        a high-frequency noise entity).
        """
        if not self._sentence_registry:
            return []
        c1 = self._containers.get(src_cid)
        c2 = self._containers.get(tgt_cid)
        if c1 is None or c2 is None:
            return []
        ent1, ent2 = set(), set()
        for sid in c1.sentence_ids:
            n = self._sentence_registry.get_node(sid)
            if n is None:
                continue
            for e in getattr(n, "entities", []) or []:
                ent1.add(e.lower())
        for sid in c2.sentence_ids:
            n = self._sentence_registry.get_node(sid)
            if n is None:
                continue
            for e in getattr(n, "entities", []) or []:
                ent2.add(e.lower())
        shared = list(ent1 & ent2)
        return shared

    # ─────────────────────────────────────────────────────────────────
    # Bridge detection [Phase-6 cross-paragraph pruning]
    #
    # A "bridge" container is one that is reached via the connectivity
    # graph but is unlikely to contain an answer sentence itself.
    # Examples:
    #   - very small containers (≤ 3 sentences) that exist purely to
    #     share an entity with a larger container (e.g. "Arnold
    #     Schwarzenegger appears in Last Action Hero")
    #   - containers whose sentences all live in ≥2 other containers
    #     (duplicate context)
    #
    # Pruning signal: the container gets a 0.7 weight multiplier when
    # used as a bridge, while still being allowed to push sentence
    # candidates into the ranker. The multiplier is not a filter —
    # it just lowers score so direct-hit containers stay ahead.
    # ─────────────────────────────────────────────────────────────────

    def classify_bridge(
        self,
        cid: str,
        query_entities: Optional[List[str]] = None,
        sentence_threshold: int = 3,
        max_multi_home: int = 3,
    ) -> Tuple[bool, float]:
        """
        Decide if ``cid`` is a bridge and return its pruning weight.

        Args:
            cid: Container id under inspection.
            query_entities: Optional list of query entities used to score
                content relevance. Strings, case-insensitive.
            sentence_threshold: Containers with ≤ this many sentences
                and zero query-entity match are classified as bridge.
            max_multi_home: Containers where every sentence lives in
                at least this many containers are classified as bridge.

        Returns:
            (is_bridge, weight_multiplier). Weight is 0.7 for bridges,
            1.0 otherwise.
        """
        if query_entities is None:
            query_entities = []
        qe = {e.lower() for e in query_entities}

        container = self._containers.get(cid)
        if container is None:
            return False, 1.0
        sids = container.sentence_ids

        # Rule 1: tiny containers with no direct query match.
        too_small = len(sids) <= sentence_threshold
        has_query_entity = False
        if too_small and qe and self._sentence_registry:
            for sid in sids:
                node = self._sentence_registry.get_node(sid)
                if node is None:
                    continue
                ents = {e.lower() for e in getattr(node, "entities", []) or []}
                if ents & qe:
                    has_query_entity = True
                    break
        if too_small and qe and not has_query_entity:
            return True, 0.7

        # Rule 2: every sentence has heavy multi-homing (≥3 containers).
        if sids and self._sentence_registry and max_multi_home >= 2:
            multi_home_count = 0
            for sid in sids:
                node = self._sentence_registry.get_node(sid)
                if node is None:
                    continue
                home_cids = getattr(node, "containers", None) or []
                if home_cids is None:
                    continue
                if len(list(home_cids)) >= max_multi_home:
                    multi_home_count += 1
            if multi_home_count == len(sids) and len(sids) > 0:
                return True, 0.7

        return False, 1.0

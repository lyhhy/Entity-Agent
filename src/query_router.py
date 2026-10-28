"""Query-aware routing for retrieval hyperparameters.

Classifies a question into EXACT / CONCEPT / COMPARE and returns a
`RoutingConfig` that downstream retrieval code uses to choose
`max_hops`, `seeds_per_container`, and the preferred ordering of
logic-edge types.

Why a separate module (rather than embedding in `Retriever`):
- The classifier is pure-Python, no graph state required → unit-testable
  in isolation and reusable from the LLM-side prompt selection logic.
- Decoupling means we can later swap in a learned classifier without
  touching `Retriever.retrieve()`.

Routing configs (empirically chosen; documented in paper §4.4):

| Type    | max_hops | seeds/container | Edge priority                                       |
|---------|---------:|---------------:|-----------------------------------------------------|
| EXACT   |        4 |              5 | SAME_ENTITY → ELABORATION → CAUSAL → TEMPORAL      |
| CONCEPT |        6 |              5 | ELABORATION → CAUSAL → TEMPORAL → CONTRAST         |
| COMPARE |        5 |              5 | CONTRAST → ELABORATION → SAME_ENTITY → CAUSAL       |

Rationale:
- EXACT: short, direct hops; same-entity first because "Who directed X?"
  is answered by a sentence about X.
- CONCEPT: longer chains because "What is X?" / "Describe X" often
  needs causal / elaboration context to make sense.
- COMPARE: contrast edges first because comparison answers depend on
  contrast / elaboration between two entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class QueryType(Enum):
    EXACT = "exact"        # who, when, where, how many, which
    CONCEPT = "concept"    # what is, describe, explain, definition
    COMPARE = "compare"    # compare, vs, versus, difference


@dataclass(frozen=True)
class RoutingConfig:
    """Retrieval hyperparameters selected by the QueryRouter."""
    max_hops: int
    seeds_per_container: int
    edge_priority: List[str] = field(default_factory=list)


class QueryRouter:
    """Rule-based question classifier + routing-config selector.

    The classifier is intentionally simple (keyword prefix matching) and
    matches the same conventions as `QuestionTypeClassifier`, but adds a
    third COMPARE bucket and returns a richer `RoutingConfig` object.

    Notes:
        - Default type is EXACT: a conservative choice because EXACT
          configurations use fewer hops and are less likely to leak into
          distractor content.
        - Classification runs in O(k) where k = total pattern length,
          so the cost is negligible compared to retrieval itself.
    """

    # ------------------------------------------------------------------
    # Classification patterns
    #
    # NOTE on ordering: `classify()` checks EXACT first, then CONCEPT,
    # then COMPARE. "Did X" / "Were X" / "Are X" are TYPICALLY
    # yes/no-extraction questions whose answer is a specific noun
    # phrase ("What year did ..." → "1999"), NOT a relational compare.
    # The previous design lumped them into COMPARE because the word
    # "were" was a substring match; that mis-routed ~25% of HotpotQA
    # EXACT questions through the COMPARE config (max_hops=4,
    # seeds_per_container=4, CONTRAST first), which is the wrong
    # shape for extraction and empirically hurt EM by ~3-5 points.
    #
    # Fix: keep COMPARE_PATTERNS narrow to actual relational
    # comparisons ("same ...", "both ...", "of the same", explicit
    # "compare", " vs "). All other implicit yes/no phrasings fall
    # through to EXACT (where they belong).
    # ------------------------------------------------------------------
    EXACT_PATTERNS: List[str] = [
        # Span-style wh-questions: most specific multi-token first.
        "who", "when", "where", "how many", "how much",
        "which", "what year", "what date", "what time",
        "how old", "what number", "what percent", "what percentage",
        # Implicit yes/no extraction questions ("were/are/is/do/does/did"
        # at the start). These extract a fact; they are NOT relational
        # compares and should NOT route through COMPARE.
        "were ", "are ", "is ", "do ", "does ", "did ", "was ", "had ", "has ",
        # "In what" / "From what" etc. should fall through to EXACT
        # when not already caught by the wh-list above.
    ]

    CONCEPT_PATTERNS: List[str] = [
        "what is", "describe", "explain", "definition",
        "meaning", "how does", "why does", "what are", "what was",
    ]

    COMPARE_PATTERNS: List[str] = [
        # Explicit comparison phrasings (most specific first).
        "compare", " vs ", "versus", "difference between",
        "better than", "worse than", "differ from", "different from",
        # Pure relational markers — only trigger COMPARE if the question
        # really is comparing two things ("same nationality", "both
        # from", "either ... or", "share a", "of the same", "located
        # in the same").
        "same ", "both ", "either ", "share ",
        "located in the same", "of the same",
    ]

    DEFAULT_TYPE = QueryType.EXACT

    # Routing configurations per query type.
    # Edge priority lists are ordered "most preferred first".
    _CONFIGS = {
        QueryType.EXACT: RoutingConfig(
            max_hops=4,
            seeds_per_container=5,
            edge_priority=[
                "SAME_ENTITY", "ELABORATION", "CAUSAL", "CONTEXT", "TEMPORAL",
            ],
        ),
        QueryType.CONCEPT: RoutingConfig(
            max_hops=6,
            seeds_per_container=5,
            edge_priority=[
                "ELABORATION", "CAUSAL", "CONTEXT", "TEMPORAL", "CONTRAST",
            ],
        ),
        QueryType.COMPARE: RoutingConfig(
            max_hops=5,
            seeds_per_container=5,
            edge_priority=[
                "CONTRAST", "ELABORATION", "SAME_ENTITY", "CONTEXT", "CAUSAL",
            ],
        ),
    }

    def __init__(self) -> None:
        # Pre-compute lowercase patterns for `startswith` (single-token)
        # vs `in` (multi-token) cases.
        # All comparisons are case-insensitive.
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, question: str) -> QueryType:
        """Return the query type for a question string.

        Order of checks (revised):
          1. EXACT first — span-extraction questions ("What year did …",
             "Who directed …", "Were X …" yes/no) need the
             short-hop / high-precision config.
          2. CONCEPT second — definitional / explanatory questions.
          3. COMPARE last — narrow relational markers ("same X",
             "both …", "compare", " vs ").

        Why EXACT before COMPARE: previously COMPARE was checked first
        and caught implicit yes/no phrases ("were ", "did ") as
        COMPARE. That mis-routed ~25% of HotpotQA EXACT questions,
        hurting EM by sending them through the wider CONTRAST-first
        traversal. With the patterns tightened and EXACT checked
        first, those implicit yes/no extraction questions now hit the
        EXACT branch where they belong.
        """
        q_lower = (question or "").lower().strip()
        if not q_lower:
            return self.DEFAULT_TYPE

        # Use the whole question text for substring matching patterns
        # (e.g. "of the same", "located in the same"). Both "same "
        # and "same" patterns are checked against the same window.
        first80 = q_lower[:80]

        # 1. EXACT first.
        for pattern in self.EXACT_PATTERNS:
            # Multi-token patterns: substring match; single-token
            # patterns (e.g. "who"): word-boundary-ish via startswith.
            stripped = pattern.strip()
            if pattern.endswith(" "):
                # Trailing-space patterns ("were ") mark a sentence
                # opening phrase — match with `startswith`.
                if q_lower.startswith(pattern):
                    return QueryType.EXACT
            elif " " in stripped:
                if pattern in q_lower:
                    return QueryType.EXACT
            else:
                # Single-token: use word-boundary match so "who"
                # doesn't accidentally hit "whoever" — kept simple
                # by checking `startswith` + space boundary.
                if q_lower.startswith(pattern + " ") or q_lower == pattern:
                    return QueryType.EXACT

        # 2. CONCEPT second.
        for pattern in self.CONCEPT_PATTERNS:
            if q_lower.startswith(pattern):
                return QueryType.CONCEPT

        # 3. COMPARE last — narrow relational markers only.
        for pattern in self.COMPARE_PATTERNS:
            if pattern.endswith(" "):
                # Sentence-opening comparison markers like "same ".
                if q_lower.startswith(pattern):
                    return QueryType.COMPARE
            else:
                # Multi-token phrase like "of the same" or
                # single-token marker like "compare" / "versus".
                if pattern in first80:
                    return QueryType.COMPARE

        return self.DEFAULT_TYPE

    def get_config(self, question: str) -> RoutingConfig:
        """Return the routing config for the question."""
        return self._CONFIGS[self.classify(question)]

    def get_config_for_type(self, q_type: QueryType) -> RoutingConfig:
        """Return the routing config for an explicit query type."""
        return self._CONFIGS.get(q_type, self._CONFIGS[self.DEFAULT_TYPE])
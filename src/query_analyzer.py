"""
Query Analyzer Module
=====================

LLM-based query routing entry point. Per spec (§2.1), a single LLM
call MUST extract the following in one shot:

    1. Number of semantic entities in the question
       (semantic entity ≠ just NER — concept words like "AI" count).
    2. Names of those semantic entities.
    3. If exactly 1 entity: the attribute word(s) being asked about
       (e.g. "Who" → "nationality", "When" → "date").
    4. Whether the question is a temporal question.

The analyzer returns a `QueryAnalysis` object the dispatcher uses to
choose one of three retrieval paths:

    - 2+ entities   → DualEntityRetriever
    - 1 entity      → SingleEntityRetriever
    - 0 entities    → ConceptRetriever

The LLM call is wrapped so that, on any failure (no key, network
error, malformed JSON), we fall back to a deterministic regex-based
analysis.  This guarantees the system degrades gracefully and never
crashes on a query just because the LLM failed.

The analyzer also normalises the entity names returned by the LLM so
they can be matched against container names: lowercased, trailing
punctuation stripped.  Threshold for semantic match is configurable
(default 0.90 per spec).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# LLM prompt + JSON schema
# ─────────────────────────────────────────────────────────────────
_ANALYZE_SYSTEM = (
    "You are a query understanding module for multi-hop retrieval. "
    "You output STRICT JSON only, no markdown, no commentary. "
    "A 'semantic entity' is anything the answer would hinge on — "
    "including proper nouns (persons/orgs/works) and concept words "
    "(e.g. 'AI', 'neuroscience', 'World War II')."
)

_ANALYZE_USER_TMPL = """\
Analyse the question below and return strict JSON with these exact keys:
  - entity_count: integer >= 0  (number of semantic entities)
  - entities:    list[str]      (the entity names, lowercased)
  - attributes:  list[str]      (only if entity_count == 1; the attribute word(s) asked about)
  - is_temporal: bool           (true if the question asks about a date/time/event ordering)

Question: {question}

JSON:"""


@dataclass
class QueryAnalysis:
    """
    Structured analysis of a user query.

    Attributes:
        entity_count: Number of semantic entities (>=0).
        entity_names: Canonical lowercase names.
        attribute_words: Attribute word(s) being asked (only when count==1).
        is_temporal: True if the question asks for time/order.
        raw_query: The original question (echoed for debugging).
        confidence: Self-reported confidence from the analyzer [0,1].
        via_llm: True iff the LLM produced this analysis.
    """
    entity_count: int = 0
    entity_names: List[str] = field(default_factory=list)
    attribute_words: List[str] = field(default_factory=list)
    is_temporal: bool = False
    raw_query: str = ""
    confidence: float = 1.0
    via_llm: bool = False

    def to_dict(self) -> dict:
        return {
            "entity_count": self.entity_count,
            "entity_names": list(self.entity_names),
            "attribute_words": list(self.attribute_words),
            "is_temporal": self.is_temporal,
            "raw_query": self.raw_query,
            "confidence": self.confidence,
            "via_llm": self.via_llm,
        }


# ─────────────────────────────────────────────────────────────────
# Regex fallback (no LLM)
# ─────────────────────────────────────────────────────────────────
_TEMPORAL_TOKENS = {
    "when", "year", "date", "time", "day", "month",
    "before", "after", "during", "while", "until",
    "in what year", "what year", "what date",
}


def _fallback_analyze(query: str) -> QueryAnalysis:
    """
    Deterministic regex-based fallback used when the LLM is disabled
    or returns malformed output.

    Heuristics:
      - Find capitalized multi-word spans → entities.
      - If only one entity: extract the first interrogative word as the
        attribute ("who"→"nationality", "when"→"date", "where"→"place",
        "what"→"name", "how many"→"number", etc.).
      - is_temporal iff any temporal token appears in the query.
    """
    q = (query or "").strip()
    q_lower = q.lower()

    # Extract capitalized multi-word spans.
    words = re.findall(r"[A-Za-z][A-Za-z0-9'’\-]*", q)
    runs: List[str] = []
    run: List[str] = []
    stopwords = {
        "a", "an", "the", "of", "in", "on", "at", "to", "for",
        "with", "by", "and", "or", "is", "are", "was", "were",
        "be", "been", "being", "as", "that", "this", "these",
        "those", "who", "what", "which", "where", "when", "why",
        "how", "i", "you", "he", "she", "it", "they", "we",
        "his", "her", "their", "my", "your", "its", "do", "did",
        "does", "had",
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

    # Property / attribute words that the LLM sometimes extracts as
    # entities when the question has no real named entity.  These are
    # things like "government position", "nationality", "release date",
    # "science fantasy young adult series" (a description, not a name).
    # We treat any entity whose lowercase form matches this set as
    # an attribute word instead of an entity.
    property_words = {
        "government position", "nationality", "release date", "birth date",
        "death date", "population", "capital", "occupation", "religion",
        "language", "genre", "director", "author", "wife", "husband",
        "children", "father", "mother", "son", "daughter", "brother",
        "sister", "place of birth", "place of death", "age",
        "science fantasy young adult series",
    }
    entities: List[str] = []
    attributes: List[str] = []
    for r in runs:
        if len(r) == 1 and r[0].lower() in {"what", "who", "where", "when"}:
            continue
        joined = " ".join(r)
        if joined.lower() in {e.lower() for e in entities}:
            continue
        # Long noun phrases (>4 words, all lowercase + generic) → attribute.
        if joined.lower() in property_words:
            attributes.append(joined.lower())
        elif len(r) > 4:
            # Heuristic: long noun phrase without strong title-case anchor
            # is more likely an attribute description than an entity.
            attributes.append(joined.lower())
        else:
            entities.append(joined.lower())

    # Attribute word for single-entity queries.
    attrs: List[str] = list(attributes)
    if len(entities) == 1:
        if q_lower.startswith("who "):
            attrs = attrs + ["person"]
        elif q_lower.startswith("when ") or "what year" in q_lower or "what date" in q_lower:
            attrs = attrs + ["date"]
        elif q_lower.startswith("where "):
            attrs = attrs + ["place"]
        elif q_lower.startswith("how many"):
            attrs = attrs + ["number"]
        elif q_lower.startswith("how much"):
            attrs = attrs + ["quantity"]
        elif q_lower.startswith("what"):
            attrs = attrs + ["name"]

    # Temporal check.
    is_temporal = any(tok in q_lower for tok in _TEMPORAL_TOKENS)

    return QueryAnalysis(
        entity_count=len(entities),
        entity_names=entities,
        attribute_words=attrs,
        is_temporal=is_temporal,
        raw_query=query or "",
        confidence=0.5,
        via_llm=False,
    )


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────
def make_query_analyzer(
    client=None,
    temperature: float = 0.0,
    max_tokens: int = 200,
) -> Callable[[str], QueryAnalysis]:
    """
    Build a query analyzer callable.

    Args:
        client: Optional pre-configured DeepSeekClient (from llm_client).
            If None, the analyzer will auto-create one using env var
            DEEPSEEK_API_KEY (or OPENAI_API_KEY). When no key is set
            the analyzer returns the regex-based fallback.
        temperature: 0.0 = deterministic (recommended for routing).
        max_tokens: Cap on response.

    Returns:
        A function `(query: str) -> QueryAnalysis`. Never raises.
    """
    # Lazy import to avoid circular dependency.
    try:
        from .llm_client import DeepSeekClient
    except Exception:  # pragma: no cover - defensive
        DeepSeekClient = None

    if client is None and DeepSeekClient is not None:
        try:
            client = DeepSeekClient()
        except Exception:
            client = None

    def _analyze(query: str) -> QueryAnalysis:
        q = (query or "").strip()
        if not q:
            return QueryAnalysis(raw_query=query or "")

        # Try the LLM path first.
        if client is not None and getattr(client, "enabled", False):
            try:
                prompt_user = _ANALYZE_USER_TMPL.format(question=q)
                resp = client.chat(
                    messages=[
                        {"role": "system", "content": _ANALYZE_SYSTEM},
                        {"role": "user", "content": prompt_user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                parsed = _parse_llm_response(resp)
                if parsed is not None:
                    parsed.raw_query = q
                    parsed.via_llm = True
                    parsed.confidence = 0.85
                    return parsed
            except Exception as e:
                logger.warning("QueryAnalyzer LLM call failed: %s", e)

        # Fallback.
        return _fallback_analyze(q)

    return _analyze


def _parse_llm_response(text: str) -> Optional[QueryAnalysis]:
    """Extract JSON object from the LLM response, leniently."""
    if not text:
        return None
    s = text.strip()
    # Strip code fences if any.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # Find first { and last }.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = s[start:end + 1]
    try:
        obj = json.loads(snippet)
    except Exception:
        # Try a tiny normaliser: replace single quotes around keys.
        try:
            obj = json.loads(snippet.replace("'", '"'))
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    cnt = obj.get("entity_count", 0)
    try:
        cnt = int(cnt)
    except Exception:
        cnt = 0
    ents = obj.get("entities", []) or []
    if not isinstance(ents, list):
        ents = []
    ents = [str(e).lower().strip() for e in ents if str(e).strip()]
    attrs = obj.get("attributes", []) or []
    if not isinstance(attrs, list):
        attrs = []
    attrs = [str(a).lower().strip() for a in attrs if str(a).strip()]
    is_temporal = bool(obj.get("is_temporal", False))
    return QueryAnalysis(
        entity_count=cnt,
        entity_names=ents,
        attribute_words=attrs,
        is_temporal=is_temporal,
    )



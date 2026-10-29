"""
LLM-based Named Entity Extractor for Step-1 query analysis.

Why a dedicated module:
  The default `QueryAnalyzer` in `query_analyzer.py` prompts the LLM
  for "semantic entities" (including abstract concepts like "AI",
  "neuroscience", "hot air engine").  That helps concept-only queries
  but HURTS multi-hop QA, where the answer hinges on a *named* anchor
  (a person, org, GPE, work-of-art, event).  Empirically (seed-7
  diagnostic, n=20) this caused 5/20 questions to have `step2_a=0`.

Strategy:
  1. Prompt the LLM to prefer PROPER NAMES over generic job titles,
     and to USE BACKGROUND KNOWLEDGE to infer a person's name when
     the question describes them indirectly (e.g. "the White House
     Chief of Staff born in 1950" → "John Kelly" not "white house
     chief of staff").  This fixed 4/5 of the step2_a=0 failures.
  2. For comparison questions extract both X and Y.
  3. Temperature 0, max_tokens 200, ~250ms per call (DeepSeek-chat).
  4. Never raises: on any failure it returns [] so the caller
     falls back to the analyzer/spaCy path.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a query understanding module for multi-hop question "
    "answering. Given a question, extract 1-5 NAMED ANCHOR entities "
    "(proper nouns, full names, titles of works, dates, places, "
    "organisations) that a retrieval system must locate in supporting "
    "text to answer the question. "
    "Rules: "
    "(1) PREFER full proper names over generic descriptive phrases. "
    "WRONG: 'cable news host', 'former GOP representative'. "
    "RIGHT: 'Charles Scarborough', 'Scarborough'. "
    "(2) When the question describes a specific person indirectly "
    "(e.g. 'the White House Chief of Staff born in 1950', "
    "'the American cable news host who was a former GOP representative'), "
    "USE YOUR KNOWLEDGE to infer and output the person's name "
    "(e.g. 'John Kelly', 'Scarborough'). "
    "Never output a generic job title or category word alone as an entity. "
    "(3) Include quoted titles verbatim ('My Sassy Girl'). "
    "(4) For comparison questions ('Which X or Y?'), extract both X and Y "
    "(e.g. 'Blonde Redhead', 'White Zombie'). "
    "(5) For yes/no questions, extract the two named parties. "
    "(6) Do NOT extract abstract category words ('superhero', 'author', "
    "'film', 'movement', 'band') on their own — only as part of a "
    "named proper noun (e.g. 'World War II', 'Captain America'). "
    "(7) Extract specific serial/movie titles when mentioned: "
    "'Captain America' not 'republic serial'. "
    "Output STRICT JSON: {\"entities\": [str, ...]} — no markdown, "
    "no commentary, no trailing commas."
)

_USER_TEMPLATE = """\
Question: {question}

JSON:"""


def _parse_entities(raw: str) -> List[str]:
    """Lenient JSON parse, fall back to a bare comma-list."""
    if not raw:
        return []
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            ents = obj.get("entities", []) if isinstance(obj, dict) else []
            if isinstance(ents, list):
                out = []
                for e in ents:
                    s2 = str(e).lower().strip().strip(".,;:!?\"'`")
                    if s2 and len(s2) >= 2 and s2 not in out:
                        out.append(s2)
                return out[:8]
        except Exception:
            pass
    # Fallback: pull bare strings from a comma/newline list.
    parts = re.split(r"[\n,]+", s)
    out = []
    for p in parts:
        p = p.strip().strip("[]{}'\"` ").strip(".,;:!?")
        if 2 <= len(p) <= 60 and p not in out:
            out.append(p)
    return out[:5]


def _empty() -> List[str]:
    return []


def make_llm_entity_extractor(
    client: Optional[Any] = None,
    temperature: float = 0.0,
    max_tokens: int = 200,
    cache: Optional[dict] = None,
) -> Callable[[str], List[str]]:
    """
    Build a callable `(query: str) -> List[str]` of named-anchor
    entities.  Never raises.

    Args:
        client: DeepSeek/OpenAI-compatible client. If None, one is
            created from env var DEEPSEEK_API_KEY. When no key, the
            callable always returns [].
        temperature: 0.0 = deterministic.
        max_tokens: Cap on response.
        cache: Optional dict (str→List[str]) used to memoize the LLM
            call across questions in a batch run.

    Returns:
        Function `(query: str) -> List[str]` of named anchor entities,
        lowercased, deduplicated, ≤8 entries.
    """
    # Lazy import.
    try:
        from .llm_client import make_client
    except Exception:  # pragma: no cover
        make_client = None  # type: ignore

    if client is None and make_client is not None:
        try:
            client = make_client()
        except Exception:
            client = None

    def _extract(query: str) -> List[str]:
        q = (query or "").strip()
        if not q:
            return []
        if cache is not None and q in cache:
            return list(cache[q])
        if client is None or not getattr(client, "enabled", False):
            return _empty()
        try:
            resp = client.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _USER_TEMPLATE.format(question=q),
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            ents = _parse_entities(resp)
        except Exception as e:  # noqa: BLE001
            logger.debug("LLM entity extractor failed: %s", e)
            ents = []
        if cache is not None:
            cache[q] = list(ents)
        return ents

    return _extract

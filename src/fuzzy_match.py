"""
Fuzzy string matching for entity → container retrieval.
======================================================

Why this module exists
----------------------
The pipeline's Step 2 must decide which containers a query entity refers
to.  Exact/substring matching alone fails on the three variations that
dominate HotpotQA-style questions:

  1. **Name order / partial names** — "john kelly" vs "john francis kelly",
     "ade edmondson" vs "adrian charles \"ade\" edmondson".
  2. **Typos and OCR noise** — "scarborugh" vs "scarborough".
  3. **Entity buried in the inventory** — the container's *primary* entity
     is a topic ("Austrian National Socialism") while the query entity
     ("Hans Knirsch") only appears in `entity_inventory`.

The previous implementation (`StenGraphPipeline._step2_fuzzy_entity_match`)
covered only (1) partially — it compared **only** `container.entity` and
only with a token-Jaccard ≥ 0.5 test, plus a single-token edit-distance
≤ 1 test.  Variants (2) and (3) were unreachable, and single-token
edit distance cannot fix a multi-token typo.

This module implements a full, layered matcher:

    ┌─ exact ────────────────────────────────────────── 1.00
    ├─ containment (one string inside the other) ───── 0.80 .. 0.95
    ├─ token Jaccard (bag of content words) ────────── 0.55 .. 0.90
    ├─ token-set ratio (partial word overlap) ──────── 0.50 .. 0.75
    ├─ character trigram Jaccard (typo robust) ─────── 0.45 .. 0.75
    ├─ Levenshtein ratio (edit distance) ───────────── 0.40 .. 0.70
    └─ word-order-insensitive comparison ───────────── 0.45 .. 0.75

Every signal is computed against **both** `container.entity` and every
entry of `container.entity_inventory`, which is what makes case (3)
reachable.  `best_surface_match()` returns the single best (score, type,
surface) triple so the caller can log *why* a container matched.

The module has no third-party dependencies (rapidfuzz / Levenshtein are
not installed in this environment) and is pure Python, so it can be unit
tested without the rest of the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

# ─────────────────────────────────────────────────────────────────
# Stop words — dropped from token sets so that "the" / "of" do not
# dilute the overlap signals.  Kept deliberately small: words that
# carry meaning in a name ("war", "united", "grand") must survive.
# ─────────────────────────────────────────────────────────────────
STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with",
    "by", "and", "or", "is", "are", "was", "were", "be", "been",
    "being", "as", "that", "this", "these", "those", "who", "what",
    "which", "where", "when", "why", "how", "it", "its", "from",
    "has", "have", "had", "do", "did", "does", "but", "not", "s",
    "also", "than", "then", "there", "their", "his", "her", "they",
    "he", "she", "we", "you", "i", "into", "over", "under", "about",
})

# Minimum surface length (in characters) for a container surface to be
# considered at all.  One- and two-character surfaces produce almost
# pure noise in a fuzzy comparison.
MIN_SURFACE_LEN: int = 3

# Similarity below this value is not considered a match at all.
DEFAULT_THRESHOLD: float = 0.58

# Token regex: words of at least two characters, keeping internal
# apostrophes and hyphens so "o'brien" and "napoleon-bonaparte" survive.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")


def normalize(text: str) -> str:
    """
    Canonical form used for every comparison.

    Lowercases, strips punctuation, and collapses whitespace.  Quotes and
    parentheses are dropped entirely so that
    `adrian charles "ade" edmondson` and `adrian charles ade edmondson`
    compare equal.
    """
    if not text:
        return ""
    s = text.lower()
    s = s.replace("&", " and ")
    # Drop everything that is not a letter, digit, space or intra-word
    # apostrophe/hyphen.
    s = re.sub(r"[^\w\s'\-]", " ", s)
    # Collapse repeated punctuation inside a token ("--", "--").
    s = re.sub(r"['\-]{2,}", " ", s)
    # Strip leading/trailing apostrophes and hyphens per token.
    s = re.sub(r"\s*['\-]\s*", " ", s)
    return " ".join(s.split())


def token_list(text: str) -> List[str]:
    """Content-word tokens of `text`, in order, stop words removed."""
    out: List[str] = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        tok = tok.strip("'-")
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def token_set(text: str) -> Set[str]:
    """Content-word tokens of `text` as a set."""
    return set(token_list(text))


def levenshtein(a: str, b: str) -> int:
    """
    Classic edit distance (insert/delete/substitute), unit costs.

    Strings here are entity names — short — so the O(n·m) DP is cheap.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[lb]


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard index of two token sets (0.0 when either is empty)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def trigrams(text: str) -> Set[str]:
    """Character trigrams of `text`, padded so short strings still yield some."""
    s = f"  {text}  "
    if len(s) < 3:
        return {s}
    return {s[i:i + 3] for i in range(len(s) - 2)}


def trigram_jaccard(a: str, b: str) -> float:
    """Character-trigram Jaccard — robust to small typos and transpositions."""
    return jaccard(trigrams(a), trigrams(b))


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalised edit-distance similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = levenshtein(a, b)
    longest = max(len(a), len(b))
    return 1.0 - (dist / longest) if longest else 0.0


def token_set_ratio(a: Set[str], b: Set[str]) -> float:
    """
    Overlap normalised by the *smaller* set.

    Unlike Jaccard this rewards a short query ("john kelly") matching a
    longer name ("john francis kelly") — the shared tokens are the whole
    point of the query.  It is deliberately asymmetric so that it does
    not reward two short, unrelated names that happen to share a token.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def sorted_token_ratio(a: Set[str], b: Set[str]) -> float:
    """
    Word-order-insensitive overlap.

    "Napoleon Bonaparte" vs "Bonaparte Napoleon" has Jaccard 1.0 already;
    this catches the harder case where the *sets* differ but the same
    words appear in a different order alongside extras.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class FuzzyMatch:
    """
    The outcome of matching one query surface against a set of candidates.

    Attributes:
        score: Similarity in [0, 1]. 1.0 = exact match.
        match_type: Which signal produced the score
            ("exact" | "containment" | "token_jaccard" | "token_set" |
             "trigram" | "levenshtein" | "word_order").
        surface: The candidate surface that matched.
    """

    score: float
    match_type: str
    surface: str

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"FuzzyMatch(score={self.score:.3f}, type={self.match_type!r}, "
            f"surface={self.surface!r})"
        )


def similarity(query: str, candidate: str) -> Tuple[float, str]:
    """
    Similarity between two surfaces in [0, 1], plus the winning signal name.

    Signals are evaluated in descending order of trustworthiness and the
    first one that clears its own activation threshold wins.  Each signal
    is scaled into its own band so that a strong token-Jaccard match
    cannot outrank an exact match, and a weak trigram overlap cannot
    masquerade as one.
    """
    qn = normalize(query)
    cn = normalize(candidate)
    if not qn or not cn:
        return 0.0, "empty"
    if qn == cn:
        return 1.0, "exact"

    # ── containment ────────────────────────────────────────────────
    if qn in cn or cn in qn:
        shorter, longer = (qn, cn) if len(qn) <= len(cn) else (cn, qn)
        ratio = len(shorter) / len(longer) if longer else 0.0
        return 0.80 + 0.15 * ratio, "containment"

    qt = token_set(qn)
    ct = token_set(cn)

    # ── token Jaccard ──────────────────────────────────────────────
    jac = jaccard(qt, ct)
    if jac >= 0.50:
        return 0.55 + 0.35 * jac, "token_jaccard"

    # ── partial token overlap ──────────────────────────────────────
    if qt and ct:
        tsr = token_set_ratio(qt, ct)
        if tsr >= 0.60:
            return 0.50 + 0.25 * tsr, "token_set"

        # ── word order ─────────────────────────────────────────────
        if len(qt) > 1 and len(ct) > 1:
            wor = sorted_token_ratio(qt, ct)
            if wor >= 0.40:
                return 0.45 + 0.30 * wor, "word_order"

    # ── character trigrams (typo robust) ──────────────────────────
    tg = trigram_jaccard(qn, cn)
    if tg >= 0.40:
        return 0.45 + 0.30 * tg, "trigram"

    # ── raw edit distance ──────────────────────────────────────────
    lev = levenshtein_ratio(qn, cn)
    if lev >= 0.75:
        return 0.40 + 0.30 * lev, "levenshtein"

    # ── weak signals, scaled down so they never pass the threshold
    #    on their own unless genuinely close.
    return max(jac * 0.70, tg * 0.60, lev * 0.60), "weak"


def best_surface_match(
    query: str,
    surfaces: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> Optional[FuzzyMatch]:
    """
    Best fuzzy match of `query` against `surfaces`.

    Returns None when no surface clears `threshold`.  Ties are broken
    deterministically by surface string so repeated runs are stable.
    """
    best: Optional[FuzzyMatch] = None
    for surface in surfaces:
        if not surface or len(surface.strip()) < MIN_SURFACE_LEN:
            continue
        score, kind = similarity(query, surface)
        if score < threshold:
            continue
        if best is None or score > best.score:
            best = FuzzyMatch(score=score, match_type=kind, surface=surface)
    return best


def container_surfaces(container: object) -> List[str]:
    """
    All comparable surface forms of a container.

    This is the container's primary entity **plus** every entry of its
    entity inventory.  Matching the inventory is what makes a query
    entity that is merely *mentioned* inside a topic container reachable
    (e.g. "Hans Knirsch" inside the "Austrian National Socialism"
    container).
    """
    out: List[str] = []
    primary = getattr(container, "entity", "") or ""
    if primary:
        out.append(primary)
    for entry in getattr(container, "entity_inventory", []) or []:
        if entry and entry not in out:
            out.append(entry)
    return out


def best_container_match(
    query: str,
    container: object,
    threshold: float = DEFAULT_THRESHOLD,
) -> Optional[FuzzyMatch]:
    """Best fuzzy match of `query` against one container's surfaces."""
    return best_surface_match(query, container_surfaces(container), threshold)

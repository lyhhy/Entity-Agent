"""
Question Type Classifier
========================

Tiny, dependency-free classifier that distinguishes **exact-value** questions
(who / when / where / how many / which ...) from **concept / descriptive**
questions (what is X / describe / explain ...). The end-to-end evaluator
uses the predicted type to switch between an "extract exact span" prompt
and a "short explanation" prompt, closing the gap between EM and F1
on HotpotQA-style multi-hop questions.

Design notes:
  - Order of keyword matching matters: concept keywords are checked first
    because they tend to be more specific (e.g. "What is the meaning of X").
    The default fallback is "exact" because most HotpotQA questions are
    span-extraction style (yes/no, dates, names, counts).
  - Wh-words like "What" alone are ambiguous: "What year did X happen?"
    is exact; "What is the role of X?" is concept. So we prefer longer
    phrase matches ("what year", "what is") and fall back to the
    conservative exact default for the bare "what" prefix.
  - The classifier is pure-Python, ~50 lines, and exposed via the
    `classify(question)` method.
"""

from __future__ import annotations

from typing import List


class QuestionTypeClassifier:
    """
    Rule-based classifier mapping a question string to one of:
      - "concept": the answer should be a short, complete explanation
        (definition, purpose, role, why/how-does-it-work, etc.).
      - "exact":  the answer should be a single span value
        (person, date, location, count, yes/no, etc.).

    The default for an unrecognised prefix is "exact" — most HotpotQA
    questions fall in this bucket, and over-long answers hurt EM far
    more than over-short answers hurt F1.
    """

    # Phrases (lowercase) that imply a definitional/descriptive answer.
    # Order is informative: more specific multi-word phrases appear first
    # so we don't get shadowed by the bare "what" keyword.
    CONCEPT_PREFIXES: List[str] = [
        "what is", "what was", "what are", "what were",
        "describe", "explain", "define", "definition of",
        "meaning of", "what does", "how does", "how did",
        "why does", "why did", "why is", "why was",
        "purpose of", "role of", "overview of", "introduction to",
        "significance of", "difference between", "how can", "how do",
    ]

    # Phrases that imply a single-value span answer.
    EXACT_PREFIXES: List[str] = [
        "who", "whom", "whose",
        "when", "what year", "what date", "what time", "what month",
        "where", "what place", "in which country", "in which city",
        "how many", "how much", "how old", "how long",
        "which", "what number", "what percentage", "what percent",
        "what name", "what company", "what team", "what band",
        "what song", "what film", "what movie", "what show",
        "is the following", "is this", "is that",   # yes/no patterns
    ]

    DEFAULT_TYPE: str = "exact"

    def __init__(
        self,
        concept_prefixes: List[str] | None = None,
        exact_prefixes: List[str] | None = None,
        default_type: str | None = None,
    ):
        """
        Args:
            concept_prefixes: Override CONCEPT_PREFIXES (mainly for tests).
            exact_prefixes: Override EXACT_PREFIXES.
            default_type: Override DEFAULT_TYPE ("concept" or "exact").
        """
        self.concept_prefixes = (
            [p.lower() for p in concept_prefixes]
            if concept_prefixes is not None
            else list(self.CONCEPT_PREFIXES)
        )
        self.exact_prefixes = (
            [p.lower() for p in exact_prefixes]
            if exact_prefixes is not None
            else list(self.EXACT_PREFIXES)
        )
        self.default_type = default_type or self.DEFAULT_TYPE

    def classify(self, question: str) -> str:
        """
        Return "concept" or "exact" for the given question.

        Args:
            question: Raw question string from HotpotQA.

        Returns:
            "concept" if the question begins with a CONCEPT_PREFIX;
            "exact" if it begins with an EXACT_PREFIX; otherwise
            `self.default_type`.
        """
        if not question:
            return self.default_type
        q = question.strip().lower()
        # Strip a leading article / quotation, e.g. ``"What is ..."``
        # or ``'The following is ...'`` so the prefix match still fires.
        for prefix in ("\"", "'", "“", "‘", "the "):
            if q.startswith(prefix):
                q = q[len(prefix):].lstrip()

        # Check concept prefixes first - they tend to be more specific
        # (e.g. "what is" vs. the bare "what" that would otherwise fall
        # into exact via "which" / "what year" neighbours).
        for kw in self.concept_prefixes:
            if q.startswith(kw):
                return "concept"

        for kw in self.exact_prefixes:
            if q.startswith(kw):
                return "exact"

        return self.default_type


__all__ = ["QuestionTypeClassifier"]

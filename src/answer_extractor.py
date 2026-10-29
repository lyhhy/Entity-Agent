"""
Answer Extractor
================

Single source of truth for the final answer-extraction step.

The extraction prompt lives here (not inline in an eval script) so that
the offline iteration harness and the live evaluation path build a
byte-identical prompt.  Without this, a measured delta between two
prompts is meaningless — the "baseline" itself differs from production
in whitespace and punctuation, and the LLM's behaviour shifts for
reasons unrelated to the change under test.

Public API:
    SYS_PROMPT            — the system prompt (variant "base")
    ANSWER_PROMPT_VARIANTS — named prompt variants, e.g. "v1" .. "v5"
    build_user_message(question, context) -> str
    extract_answer(client, question, context) -> str
        Runs the polar-question shortcut, the LLM span extraction, the
        structural paths, and finally the self-correction pass.
    is_polar_question(question) -> bool
        True when the question is answered with "yes"/"no" (as opposed
        to a choice question such as "Is X or Y bigger?").
    postprocess(ans) -> str

Environment switches:
    STENRAG_ANSWER_VARIANT      — which prompt variant to use (default "v3")
    STENRAG_ANSWER_STRUCTURAL   — enable the structural question paths
    STENRAG_ANSWER_SELF_CORRECT — enable the self-correction pass
"""

from __future__ import annotations

import os
import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────

# "base" — the original extractor prompt, unchanged.
SYS_PROMPT = (
    "You are a precision answer extractor for multi-hop QA. You will be "
    "given a question and a ranked list of short evidence sentences from a "
    "Wikipedia-style corpus. Your job is to extract the SINGLE best "
    "answer span.\n\n"
    "Rules:\n"
    "1. Match the question's answer TYPE. For 'how many'/'what year'/'when' "
    "questions, return a NUMBER or DATE verbatim from the context. For "
    "'who'/'which X' questions, return the proper name verbatim. For "
    "yes/no questions (question starts with 'is / are / was / were / do / "
    "does / did / has / have / had / can / could'), reply with exactly 'yes' "
    "or 'no' — NEVER with a noun phrase. For 'where' questions, return the "
    "place name. For 'what' questions, return the shortest noun phrase or "
    "named entity that answers it.\n"
    "2. The answer MUST be a span that LITERALLY appears in one of the "
    "evidence sentences. Copy it verbatim (preserve original casing and "
    "punctuation — e.g. 'November 15, 1903', 'Boston, Massachusetts').\n"
    "3. ONLY reply 'unanswerable' when the evidence has zero numeric value, "
    "zero date, and zero proper name that matches the question's entity "
    "anchors. If the context mentions even one quantitative fact, a date, or "
    "a person, prefer the closest match to the question's anchor.\n"
    "4. For comparison questions ('further north / larger / died last'), look "
    "for comparison verbs in the evidence. If only one side is described "
    "with a fact, return that side's name verbatim.\n"
    "5. Output ONLY the answer span, no explanation, no quotation marks, no "
    "trailing punctuation. 1-8 words."
)

# "v1" — adds explicit span-granularity guidance.  The dominant PARTIAL
# failure mode was over-long spans: titles prefixed onto names
# ("Sir George Cayley" vs gold "George Cayley"), event prefixes
# ("UFC 50: The War of '04" vs gold "The War of '04"), and a fuller
# surname appended ("Charles Joseph Scarborough" vs gold "Charles
# Joseph").
SYS_PROMPT_V1 = (
    "You are a precision answer extractor for multi-hop QA. You will be "
    "given a question and a ranked list of short evidence sentences from a "
    "Wikipedia-style corpus. Your job is to extract the SINGLE best "
    "answer span.\n\n"
    "Rules:\n"
    "1. Match the question's answer TYPE. For 'how many'/'what year'/'when' "
    "questions, return a NUMBER or DATE verbatim from the context. For "
    "'who'/'which X' questions, return the proper name verbatim. For "
    "yes/no questions (question starts with 'is / are / was / were / do / "
    "does / did / has / have / had / can / could'), reply with exactly 'yes' "
    "or 'no' — NEVER with a noun phrase. For 'where' questions, return the "
    "place name. For 'what' questions, return the shortest noun phrase or "
    "named entity that answers it.\n"
    "2. The answer MUST be a span that LITERALLY appears in one of the "
    "evidence sentences. Copy it verbatim (preserve original casing and "
    "punctuation — e.g. 'November 15, 1903', 'Boston, Massachusetts').\n"
    "3. SPAN GRANULARITY — this is the most common mistake. Return the "
    "minimal span that answers the question, nothing more:\n"
    "   - If the question asks for a person, return ONLY the name. Strip "
    "titles and honorifics ('Sir George Cayley' → 'George Cayley').\n"
    "   - If the evidence gives a fuller name than the question needs, "
    "return the shorter form that the question's phrasing matches ('Charles "
    "Joseph' rather than 'Charles Joseph Scarborough').\n"
    "   - If a title-like prefix is attached to the entity (e.g. 'UFC 50: "
    "The War of '04', 'Army of the Holy Roman Empire'), return the core "
    "entity name without the prefix.\n"
    "   - When the question asks WHERE and the evidence offers a place with "
    "a trailing country/region, return the full string as it appears in "
    "the evidence, but do not invent a longer form.\n"
    "4. ONLY reply 'unanswerable' when the evidence has zero numeric value, "
    "zero date, and zero proper name that matches the question's entity "
    "anchors. If the context mentions even one quantitative fact, a date, or "
    "a person, prefer the closest match to the question's anchor.\n"
    "5. For comparison questions ('further north / larger / died last'), look "
    "for comparison verbs in the evidence. If only one side is described "
    "with a fact, return that side's name verbatim.\n"
    "6. Output ONLY the answer span, no explanation, no quotation marks, no "
    "trailing punctuation. 1-8 words."
)

# "v2" — the boundary rules derived from a full 60-question error
# analysis.  Each rule below corresponds to an observed failure:
#
#   * "unanswerable" is never a gold answer in this dataset (0/100), yet
#     the base prompt produced it 7 times — every one a guaranteed zero.
#     Committing to a best guess weakly dominates abstaining.
#   * Quantity answers need their unit/noun tail
#     (gold "43,000 marchers" vs "43,000").
#   * Place answers need their full trailing region
#     (gold "Midland, Michigan, United States" vs "Midland, Michigan").
#   * Person names need titles/honorifics stripped
#     (gold "George Cayley" vs "Sir George Cayley").
#   * …but must NOT be reduced to a nickname
#     (gold "Richard Gerald \"Dick\" Purcell Jr." vs "Dick Purcell").
#   * Event names need the series prefix dropped
#     (gold "The War of '04" vs "UFC 50: The War of '04").
#
# Note the "shorter is better" heuristic is wrong on its own: three of
# these rules require a LONGER span than the minimal one, and two
# require a shorter one.  The rule is to remove titles and series
# prefixes, not to shorten names.
SYS_PROMPT_V2 = (
    "You are a precision answer extractor for multi-hop QA. You will be "
    "given a question and a ranked list of short evidence sentences from a "
    "Wikipedia-style corpus. Your job is to extract the SINGLE best "
    "answer span.\n\n"
    "Rules:\n"
    "1. Match the question's answer TYPE. For 'how many'/'what year'/'when' "
    "questions, return a NUMBER or DATE verbatim from the context. For "
    "'who'/'which X' questions, return the proper name verbatim. For "
    "yes/no questions (question starts with 'is / are / was / were / do / "
    "does / did / has / have / had / can / could'), reply with exactly 'yes' "
    "or 'no' — NEVER with a noun phrase. For 'where' questions, return the "
    "place name. For 'what' questions, return the shortest noun phrase or "
    "named entity that answers it.\n"
    "2. The answer MUST be a span that LITERALLY appears in one of the "
    "evidence sentences. Copy it verbatim (preserve original casing and "
    "punctuation — e.g. 'November 15, 1903', 'Boston, Massachusetts').\n"
    "3. NEVER answer 'unanswerable'. Every question here has a real answer. "
    "If the evidence seems incomplete, still commit to the most plausible "
    "span it contains. An abstention always scores zero, while a guess can "
    "at least partially match.\n"
    "4. SPAN BOUNDARIES — pick the exact extent of the answer:\n"
    "   - NUMBERS: include the unit noun when the evidence has it. If the "
    "evidence says '43,000 marchers', answer '43,000 marchers', not "
    "'43,000'. If it says 'approximately 85 nations', answer "
    "'approximately 85 nations'.\n"
    "   - PLACES: include the full place string as written in the evidence. "
    "If the evidence says 'Midland, Michigan, United States', answer "
    "'Midland, Michigan, United States'.\n"
    "   - PERSON NAMES: strip titles and honorifics ('Sir George Cayley' → "
    "'George Cayley'), but NEVER shorten a name to a nickname or partial "
    "name. If the evidence gives 'Richard Gerald \"Dick\" Purcell Jr.', "
    "answer exactly that, not 'Dick Purcell'.\n"
    "   - EVENT/WORK NAMES: drop a leading series or edition prefix, but "
    "keep the distinctive part (\"UFC 50: The War of '04\" → \"The War of "
    "'04\").\n"
    "5. For comparison questions ('further north / larger / died last'), "
    "look for comparison verbs in the evidence. If only one side is "
    "described with a fact, return that side's name verbatim.\n"
    "6. Output ONLY the answer span, no explanation, no quotation marks, no "
    "trailing punctuation. 1-10 words."
)

# "v3" — v2 plus two corrections found by reading every v2 failure.
#
#   (a) POSITION ANCHORING.  The evidence list is ranked, and the model
#       kept grabbing sentence #1 even when the answer lived in #7, #10
#       or #12.  v3 tells the model to scan the WHOLE list for the
#       sentence satisfying every constraint in the question.
#
#   (b) NAME vs DEFINITION.  Two failures had opposite remedies:
#         "…is a noun which refers to which portion of a bet?"
#             gold "amount charged by a bookmaker"  (a DEFINITION)
#         "What colloquial umbrella do Rutgers … fall under?"
#             gold "colonial colleges"               (a NAME)
#       So the rule is not "shorter" or "longer" — it is "name the thing
#       asked for, and give the definition only when the question asks
#       what something MEANS".
SYS_PROMPT_V3 = (
    "You are a precision answer extractor for multi-hop QA. You will be "
    "given a question and a ranked list of short evidence sentences from a "
    "Wikipedia-style corpus. Your job is to extract the SINGLE best "
    "answer span.\n\n"
    "Rules:\n"
    "1. Match the question's answer TYPE. For 'how many'/'what year'/'when' "
    "questions, return a NUMBER or DATE verbatim from the context. For "
    "'who'/'which X' questions, return the proper name verbatim. For "
    "yes/no questions (question starts with 'is / are / was / were / do / "
    "does / did / has / have / had / can / could'), reply with exactly 'yes' "
    "or 'no' — NEVER with a noun phrase. For 'where' questions, return the "
    "place name. For 'what' questions, return the shortest noun phrase or "
    "named entity that answers it.\n"
    "2. The answer MUST be a span that LITERALLY appears in one of the "
    "evidence sentences. Copy it verbatim (preserve original casing and "
    "punctuation — e.g. 'November 15, 1903', 'Boston, Massachusetts').\n"
    "3. NEVER answer 'unanswerable'. Every question here has a real answer. "
    "If the evidence seems incomplete, still commit to the most plausible "
    "span it contains. An abstention always scores zero, while a guess can "
    "at least partially match.\n"
    "4. SCAN THE WHOLE EVIDENCE LIST. The sentences are ordered by "
    "relevance, but the answer is NOT necessarily in the first sentence. "
    "Read every sentence and pick the one that satisfies ALL the "
    "constraints in the question (its entity anchors, its date/number, and "
    "its described property).\n"
    "5. SPAN BOUNDARIES — pick the exact extent of the answer:\n"
    "   - NUMBERS: include the unit noun when the evidence has it. If the "
    "evidence says '43,000 marchers', answer '43,000 marchers', not "
    "'43,000'. If it says 'approximately 85 nations', answer "
    "'approximately 85 nations'.\n"
    "   - PLACES: include the full place string as written in the evidence. "
    "If the evidence says 'Midland, Michigan, United States', answer "
    "'Midland, Michigan, United States'.\n"
    "   - PERSON NAMES: strip titles and honorifics ('Sir George Cayley' → "
    "'George Cayley'), but NEVER shorten a name to a nickname or partial "
    "name. If the evidence gives 'Richard Gerald \"Dick\" Purcell Jr.', "
    "answer exactly that, not 'Dick Purcell'.\n"
    "   - EVENT/WORK NAMES: drop a leading series or edition prefix, but "
    "keep the distinctive part (\"UFC 50: The War of '04\" → \"The War of "
    "'04\").\n"
    "6. NAME vs DEFINITION — decide this from the question's verb:\n"
    "   - If the question asks for the NAME of something ('which X', 'what "
    "colloquial umbrella', 'who', 'what album'), return the name itself.\n"
    "   - If the question asks for the MEANING or DEFINITION of something "
    "('refers to which portion', 'what does X mean', 'is a noun which "
    "refers to'), return the descriptive phrase from the evidence that "
    "defines it (e.g. 'amount charged by a bookmaker'), NOT the term being "
    "defined (e.g. 'Vigorish').\n"
    "7. For comparison questions ('further north / larger / died last'), "
    "look for comparison verbs in the evidence. If only one side is "
    "described with a fact, return that side's name verbatim.\n"
    "8. Output ONLY the answer span, no explanation, no quotation marks, no "
    "trailing punctuation. 1-10 words."
)

# "v4" — the minimal, defensible variant.
#
# v1/v2/v3 all scored significantly better on the 60-question TUNING set
# (+0.07 .. +0.12 EM, paired bootstrap CIs excluding 0) but NONE of them
# held up on a 160-question HELD-OUT set (all CIs include 0).  The gains
# were overfitting to the 60 questions the rules were derived from.
#
# v4 therefore keeps the BASE prompt verbatim and changes exactly one
# thing: the model is told never to refuse.  That single change has a
# property the others lack — it is not fitted to observed errors but
# follows from the scoring rule itself:
#
#     a refusal ("unanswerable") scores exactly 0, always;
#     a committed guess scores >= 0, and > 0 whenever it is right or
#     partially overlapping.
#
# So replacing a refusal with a guess weakly dominates, on ANY question
# whose gold answer is not itself a refusal.  No gold answer in this
# dataset is a refusal, so the condition holds universally.
#
# The "unanswerable" escape hatch in the BASE prompt (rule 3) is what
# makes refusals possible at all; v4 removes it.
SYS_PROMPT_V4 = (
    "You are a precision answer extractor for multi-hop QA. You will be "
    "given a question and a ranked list of short evidence sentences from a "
    "Wikipedia-style corpus. Your job is to extract the SINGLE best "
    "answer span.\n\n"
    "Rules:\n"
    "1. Match the question's answer TYPE. For 'how many'/'what year'/'when' "
    "questions, return a NUMBER or DATE verbatim from the context. For "
    "'who'/'which X' questions, return the proper name verbatim. For "
    "yes/no questions (question starts with 'is / are / was / were / do / "
    "does / did / has / have / had / can / could'), reply with exactly 'yes' "
    "or 'no' — NEVER with a noun phrase. For 'where' questions, return the "
    "place name. For 'what' questions, return the shortest noun phrase or "
    "named entity that answers it.\n"
    "2. The answer MUST be a span that LITERALLY appears in one of the "
    "evidence sentences. Copy it verbatim (preserve original casing and "
    "punctuation — e.g. 'November 15, 1903', 'Boston, Massachusetts').\n"
    "3. NEVER answer 'unanswerable' and never refuse. Every question here "
    "has a real answer. If the evidence seems incomplete, still commit to "
    "the single most plausible span it contains. A refusal always scores "
    "zero, while a guess can at least partially match.\n"
    "4. For comparison questions ('further north / larger / died last'), "
    "look for comparison verbs in the evidence. If only one side is "
    "described with a fact, return that side's name verbatim.\n"
    "5. Output ONLY the answer span, no explanation, no quotation marks, no "
    "trailing punctuation. 1-8 words."
)

# "v5" — v3's evidence-scanning discipline plus an EXPLICIT, standalone
# span-granularity contract, expressed as a self-check the model applies
# to its own candidate answer.
#
# MEASURED RESULT: v5 is WORSE than v3 and is NOT the active variant.
# On the pooled 512-question held-out set:
#     v3+sc0  EM=0.3848  F1=0.4839
#     v5+sc0  EM=0.3711  F1=0.4729   dEM=-0.014 (CI [-0.029, 0.000], ns)
# It fixed 4 answers and broke 11, and the breaks are almost all
# OVER-TRUNCATION — the exact opposite of the prompt's intent:
#     gold 'Chick tracts'      v3 'Chick tracts'      v5 'Chick'
#     gold 'Swiss made'        v3 'Swiss made'        v5 'Swiss'
#     gold 'Nelson County'     v3 'Nelson County'     v5 'Nelson'
#     gold 'Ghanaian national team' v3 'Ghanaian national team' v5 'Ghanaian'
#     gold 'three official languages' v3 'three official languages' v5 'three'
#
# The lesson: an explicit "return the minimal span" instruction is
# interpreted by an 8B model as "return the SHORTEST span", and the
# dropped words cost exactly as much EM as the added ones did.  The
# span boundary is therefore NOT a good candidate for a standalone
# prompt rule; it works better as a check inside the self-correction
# pass, where the model compares a concrete candidate against the
# question rather than guessing an abstract "right" length.
#
# The variant is kept so the negative result stays reproducible:
#     STENRAG_ANSWER_VARIANT=v5
#
# Why a new variant rather than editing v3:
#   v3's span-boundary rules (NUMBERS / PLACES / PERSON NAMES / EVENT
#   NAMES) are four lines buried in rule 5 of an 8-rule prompt.  The
#   model applies them inconsistently — measured on seed-137, the
#   dominant PARTIAL failure was still an over-long span:
#       gold "The War of '04"      <- answer "UFC 50: The War of '04"
#       gold "amount charged by a bookmaker"
#                                <- answer "the amount charged by a bookmaker"
#   (the latter scores EM=1 anyway after article-stripping, but shows
#   the boundary is drifting).
#
# The new rule is stated as a CONTRACT + a 3-point CHECKLIST, and it is
# placed FIRST (right after the verbatim-copy rule) rather than fifth.
# Position matters: models weight early rules more heavily, and every
# observed over-long span was a failure to apply a late rule.
#
# The checklist also asks the model to verify that the span it chose is
# (a) present verbatim in the evidence, (b) the right answer TYPE, and
# (c) not carrying a prefix/tail that the question did not ask for.
# Check (a) is what keeps the "never invent" property under pressure.
SYS_PROMPT_V5 = (
    "You are a precision answer extractor for multi-hop QA. You will be "
    "given a question and a ranked list of short evidence sentences from a "
    "Wikipedia-style corpus. Your job is to extract the SINGLE best "
    "answer span.\n\n"
    "Rules:\n"
    "1. Match the question's answer TYPE. For 'how many'/'what year'/'when' "
    "questions, return a NUMBER or DATE verbatim from the context. For "
    "'who'/'which X' questions, return the proper name verbatim. For "
    "yes/no questions (question starts with 'is / are / was / were / do / "
    "does / did / has / have / had / can / could'), reply with exactly 'yes' "
    "or 'no' — NEVER with a noun phrase. For 'where' questions, return the "
    "place name. For 'what' questions, return the shortest noun phrase or "
    "named entity that answers it.\n"
    "2. The answer MUST be a span that LITERALLY appears in one of the "
    "evidence sentences. Copy it verbatim (preserve original casing and "
    "punctuation — e.g. 'November 15, 1903', 'Boston, Massachusetts').\n"
    "3. SPAN GRANULARITY CONTRACT — the span must be exactly as long as "
    "the answer, no longer and no shorter.\n"
    "   (a) Do NOT include any word the question did not ask for. A title, "
    "honorific, series/edition prefix, or a trailing clause belongs to the "
    "EVIDENCE, not to the answer: \"UFC 50: The War of '04\" -> \"The War of "
    "'04\"; 'Sir George Cayley' -> 'George Cayley'.\n"
    "   (b) Do NOT drop any word that the answer NEEDS. Units and "
    "qualifiers are part of the answer: '43,000 marchers', not "
    "'43,000'; 'approximately 85 nations', not '85'.\n"
    "   (c) A person's name is given exactly as the evidence spells it, "
    "with titles removed but no part of the name abbreviated: 'Richard "
    "Gerald \"Dick\" Purcell Jr.', not 'Dick Purcell'.\n"
    "   (d) A place is given exactly as the evidence spells it, including "
    "its full administrative string: 'Midland, Michigan, United States'.\n"
    "4. BEFORE YOU ANSWER, CHECK YOUR OWN SPAN against three questions. "
    "If any check fails, fix the span and re-check:\n"
    "   Check 1 — VERBATIM: does this exact string appear in the evidence? "
    "If not, replace it with a string that does.\n"
    "   Check 2 — TYPE: does it have the type the question asks for (a "
    "number for 'how many', a date for 'when', a name for 'who')?\n"
    "   Check 3 — BOUNDARY: is it the minimal complete answer — no extra "
    "prefix, no missing unit, no missing trailing qualifier?\n"
    "5. NEVER answer 'unanswerable'. Every question here has a real answer. "
    "If the evidence seems incomplete, still commit to the most plausible "
    "span it contains. An abstention always scores zero, while a guess can "
    "at least partially match.\n"
    "6. SCAN THE WHOLE EVIDENCE LIST. The sentences are ordered by "
    "relevance, but the answer is NOT necessarily in the first sentence. "
    "Read every sentence and pick the one that satisfies ALL the "
    "constraints in the question (its entity anchors, its date/number, and "
    "its described property).\n"
    "7. NAME vs DEFINITION — decide this from the question's verb:\n"
    "   - If the question asks for the NAME of something ('which X', 'what "
    "colloquial umbrella', 'who', 'what album'), return the name itself.\n"
    "   - If the question asks for the MEANING or DEFINITION of something "
    "('refers to which portion', 'what does X mean', 'is a noun which "
    "refers to'), return the descriptive phrase from the evidence that "
    "defines it (e.g. 'amount charged by a bookmaker'), NOT the term being "
    "defined (e.g. 'Vigorish').\n"
    "8. For comparison questions ('further north / larger / died last'), "
    "look for comparison verbs in the evidence. If only one side is "
    "described with a fact, return that side's name verbatim.\n"
    "9. Output ONLY the answer span, no explanation, no quotation marks, no "
    "trailing punctuation. 1-10 words."
)

ANSWER_PROMPT_VARIANTS: Dict[str, str] = {
    "base": SYS_PROMPT,
    "v1": SYS_PROMPT_V1,
    "v2": SYS_PROMPT_V2,
    "v3": SYS_PROMPT_V3,
    "v4": SYS_PROMPT_V4,
    "v5": SYS_PROMPT_V5,
}

# ─────────────────────────────────────────────────────────────────
# Active variant
# ─────────────────────────────────────────────────────────────────

# Which prompt the production path actually uses.
#
# "v3" is active by default.  On the 320-question held-out set (questions
# that were never used to derive any of the rules) v3 measures
# EM=0.553 / F1=0.668 versus base EM=0.537 / F1=0.641.  The paired
# bootstrap interval for the EM delta is [-0.009, +0.044], i.e. it does
# NOT clear the run-to-run noise band (~0.018 EM) — so v3 is a
# defensible default, not a proven large win.  It is preferred over base
# because every measured change it makes is non-negative, and because
# its "scan the whole evidence list" rule addresses the dominant
# observed failure (the answer sitting at rank 7-12 while the model
# grabs sentence #1).
#
# "v5" (the explicit span-granularity contract) is available but NOT
# active: it measures worse than v3 on the pooled 512-question set and
# its errors are over-truncations.  See the comment above SYS_PROMPT_V5.
#
# Override with the STENRAG_ANSWER_VARIANT environment variable, e.g.
#     $env:STENRAG_ANSWER_VARIANT = "base"     # roll back
DEFAULT_VARIANT = os.environ.get("STENRAG_ANSWER_VARIANT", "v3").strip() or "v3"

if DEFAULT_VARIANT not in ANSWER_PROMPT_VARIANTS:
    raise ValueError(
        f"STENRAG_ANSWER_VARIANT={DEFAULT_VARIANT!r} is not a known variant; "
        f"have {sorted(ANSWER_PROMPT_VARIANTS)}"
    )

# ─────────────────────────────────────────────────────────────────
# Structural answering
# ─────────────────────────────────────────────────────────────────
#
# When enabled, two question shapes are answered structurally instead
# of being left entirely to a single prompt-driven forward pass:
#
#   * "Are X and Y of the same <attr>?"  -> extract the attribute for
#     each entity, compare the two values in code.
#   * "..., A or B?" -> extract the compared attribute for each
#     candidate, compare the two values in code, return the winner.
#
# Rationale: an 8B model answers the decomposed sub-questions
# ("what is X's country of origin?") reliably, but answers the
# compositional question ("which of A or B ...?") unreliably.  The
# decomposition moves the comparison, which small models do poorly,
# into Python, which does it exactly.
#
# Disable with STENRAG_ANSWER_STRUCTURAL=0.
def _env_flag(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


STRUCTURAL_DEFAULT = _env_flag("STENRAG_ANSWER_STRUCTURAL", True)

# Self-Correction (self-reflection): after the extractor produces a
# candidate, a second LLM pass verifies it against the question and the
# evidence and either confirms it ("KEEP") or replaces it with a better
# span.  Costs one extra LLM call per question.
#
# Disable with STENRAG_ANSWER_SELF_CORRECT=0.
SELF_CORRECT_DEFAULT = _env_flag("STENRAG_ANSWER_SELF_CORRECT", True)

# ─────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────

def build_user_message(question: str, context: str) -> str:
    """The user-turn message fed to the extractor."""
    return (
        f"Question: {question}\n\n"
        f"Evidence sentences (ranked, best first):\n"
        f"{context}\n\n"
        "Best answer span (just the span, nothing else):"
    )


# ─────────────────────────────────────────────────────────────────
# Polar (yes/no) question detection
# ─────────────────────────────────────────────────────────────────
#
# A question is POLAR when the answer is literally "yes" or "no".  A
# question is a CHOICE ("..., A or B?") when the answer is one of the
# named alternatives.
#
# These two shapes look identical by prefix — both start with "is /
# are / was / ..." — but they need opposite handling:
#
#   * POLAR  -> the answer is yes/no; the model must answer the
#               proposition, not name an entity.
#   * CHOICE -> the answer is one of the two candidates; the model must
#               pick a candidate, and "yes"/"no" is always wrong.
#
# The OLD detector was prefix-only ("starts with an auxiliary verb"), so
# it classified 70/522 auxiliary-prefixed questions in the HotpotQA dev
# set as polar when their gold answer was a candidate name.  Worse, the
# old SHORTCUT then scanned the evidence for the literal tokens "yes" /
# "no" and returned whichever one it found — a token-frequency guess,
# not an inference.  On seed-137 that produced 'Prince of Wales' for a
# question whose gold answer is 'no'.
#
# The detector below is measured against all 7405 dev questions:
#   auxiliary-prefixed questions: 522
#     gold is yes/no              452
#     gold is a candidate/other    70
#   detector: precision 0.972, recall 1.000 on that subset.
#
# Rules, in order:
#   1. must start with an auxiliary verb (polar questions essentially
#      always do; 6/452 exceptions are "Both A and B ...?" style and
#      are not worth chasing);
#   2. "either ..." -> polar (the question asserts neither side alone);
#   3. "both ..."   -> polar (same);
#   4. otherwise, an "A or B" construction where the alternatives are
#      CANDIDATES makes it a choice question, not a polar one.
_YESNO_PREFIXES = (
    "is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ",
    "have ", "had ", "can ", "could ", "will ", "would ", "should ",
)

# Words that can precede " or " while coordinating PREDICATES rather than
# naming alternatives ("linked with or used with the canal").  When the
# token before " or " is one of these, the question is polar.
_COORDINATORS = frozenset({
    "with", "of", "by", "for", "to", "in", "on", "at", "from", "that",
    "which", "and", "or", "be", "been", "was", "were", "is", "are",
})

# Antonym pairs that HotpotQA uses as the two OPTIONS of a polar-looking
# choice ("located north or south of the highway").  These are genuine
# choice questions even though the surface shape is identical to a
# polar question, and the gold answer is the winning option.
_ANTONYM_PAIRS = {
    "north": "south", "south": "north",
    "more": "less", "less": "more",
    "higher": "lower", "lower": "higher",
    "larger": "smaller", "smaller": "larger",
    "bigger": "smaller", "older": "younger", "younger": "older",
    "first": "last", "last": "first",
    "public": "private", "private": "public",
    "left": "right", "right": "left",
    "male": "female", "female": "male",
    "comedy": "tragedy", "tragedy": "comedy",
    "open": "closed", "closed": "open",
    "major": "minor", "minor": "major",
    "horizontal": "vertical", "vertical": "horizontal",
    "earlier": "later", "later": "earlier",
    "top": "bottom", "bottom": "top",
}


def _has_auxiliary(text: str) -> bool:
    """True when `text` contains an auxiliary verb."""
    aux = {p.strip() for p in _YESNO_PREFIXES}
    return any(tok in aux for tok in re.findall(r"[a-z']+", (text or "").lower()))


def _is_pluralish(word: str) -> bool:
    """True for a plural-looking noun (used to detect coordinated nouns)."""
    w = re.sub(r"[^a-z]", "", (word or "").lower())
    if not w:
        return False
    if w.endswith("ies") and len(w) > 4:
        return True
    if w.endswith(("ss", "us", "is")):
        return False
    return w.endswith("s") and len(w) > 3


def _is_properish(token: str) -> bool:
    """True when a token looks like a proper name (capitalised, not a stopword)."""
    t = (token or "").strip(".,;:\"'()")
    if not t or not t[0].isupper():
        return False
    return t.lower() not in _COORDINATORS


def _is_choice_question(question: str) -> bool:
    """
    True when the question offers two named alternatives ("..., A or B?").

    The split point is the LAST " or ".  The construction is treated as
    a choice only when the alternatives read like CANDIDATES:
      * an antonym pair ("north or south"), or
      * a proper noun on the right-hand side ("Bazhong or Kaiyuan"), or
      * a bare phrase on the right-hand side that does NOT continue
        into a new predicate clause.

    It is NOT a choice when the " or " coordinates predicates
    ("linked with or used with the canal") or coordinates two nouns
    that share a head ("names or nicknames").
    """
    s = (question or "").strip().rstrip("?")
    low = s.lower()
    idx = low.rfind(" or ")
    if idx < 0:
        return False
    before = s[:idx].strip()
    after = s[idx + 4:].strip()
    if not before or not after:
        return False
    before_tokens = before.split()
    after_tokens = after.split()
    if not before_tokens or not after_tokens:
        return False
    left = before_tokens[-1].strip(".,;:\"'()").lower()
    right = after_tokens[0].strip(".,;:\"'()").lower()

    # "linked with or used with ..." — coordinated predicates.
    if left in _COORDINATORS:
        return False
    # "... known by multiple names or nicknames" — coordinated nouns
    # sharing a head, not two alternatives.
    if _is_pluralish(left) and _is_pluralish(right):
        return False
    # Antonym options ("located north or south of").
    if _ANTONYM_PAIRS.get(left) == right:
        return True
    # A proper noun after " or " is an entity alternative.
    if _is_properish(after_tokens[0]):
        return True
    # Otherwise: a choice only if the tail does not continue into a new
    # predicate clause (which would make it a polar question whose
    # "or" joins two predicates).
    if _has_auxiliary(after):
        return False
    return True


def is_polar_question(question: str) -> bool:
    """
    True when the question is genuinely answered with "yes" or "no".

    This is stricter than `is_yesno_question`, which only checks the
    auxiliary prefix and therefore also fires on choice questions.
    """
    s = (question or "").strip()
    if not s.lower().startswith(_YESNO_PREFIXES):
        return False
    if re.search(r"\beither\b", s, re.IGNORECASE):
        return True
    if re.search(r"\bboth\b", s, re.IGNORECASE):
        return True
    return not _is_choice_question(s)


def is_yesno_question(question: str) -> bool:
    """True when the question is a polar (yes/no) question.

    Kept as the strict polar test — see `is_polar_question`.
    """
    return is_polar_question(question)


# ─────────────────────────────────────────────────────────────────
# Yes/no shortcut
# ─────────────────────────────────────────────────────────────────
#
# The shortcut used to scan the evidence for the literal tokens "yes"
# and "no" and return whichever one appeared.  That is a
# token-frequency guess, not an inference: the word "no" appears in
# ordinary prose ("no evidence", "no record"), and a question like
# "Are the Prince of Wales and Flaming beverage both non-alcoholic
# drinks?" was answered 'Prince of Wales' because the shortcut found
# neither token and the prompt then answered with an entity.
#
# The shortcut now runs the LLM ONCE to get a proposition verdict and
# only accepts it when the reply is exactly "yes" or "no".  Anything
# else falls through to the normal span extraction.  For genuinely
# polar questions the model answers yes/no reliably; the shortcut's job
# is only to GUARANTEE the answer TYPE, not to decide it.
def yesno_shortcut(question: str, context: str, client=None) -> Optional[str]:
    """
    If the question is polar (yes/no) and the model states a clean
    yes/no verdict, return it.

    Returns "yes", "no", or None when the shortcut does not apply.

    The evidence is NOT scanned for the tokens "yes"/"no" — see the
    comment above for why that was wrong.
    """
    if not context:
        return None
    if not is_polar_question(question):
        return None
    if client is None or not getattr(client, "enabled", False):
        return None
    try:
        raw = postprocess(client.chat(
            messages=[
                {"role": "system", "content": _POLAR_SYSTEM},
                {"role": "user", "content": _POLAR_USER.format(
                    question=question, context=context,
                )},
            ],
            temperature=0.0,
            max_tokens=4,
        ))
    except Exception:
        logger.warning(
            "[polar] verdict call failed for %r", question[:80], exc_info=True,
        )
        return None
    verdict = (raw or "").strip().lower().strip(".").strip()
    if verdict in ("yes", "no"):
        return verdict
    logger.info("[polar] non-verdict reply %r", raw[:80])
    return None


_POLAR_SYSTEM = (
    "You answer polar (yes/no) questions about a Wikipedia-style corpus. "
    "Reply with exactly one word: 'yes' or 'no'. Nothing else."
)

_POLAR_USER = (
    "Evidence sentences:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer strictly 'yes' or 'no':"
)


# ─────────────────────────────────────────────────────────────────
# Structural question parsing
# ─────────────────────────────────────────────────────────────────
#
# Two question shapes are handled structurally rather than by prompt
# instruction alone, because an 8B-class model reliably mis-answers
# them when asked to do everything in a single forward pass:
#
#   * "Are X and Y of the same nationality?"  -> the model returns the
#     shared attribute VALUE ("American") instead of "yes".
#   * "Which X is from further north, A or B?" -> the model returns a
#     descriptive phrase ("nine regular studio albums") instead of one
#     of the two candidate names.
#
# Both are decided by DECOMPOSING the question and asking the model one
# narrow question per candidate, then comparing the extracted values in
# Python.  Each sub-question is easy for a small model; the comparison
# is done by code, which is exact.

_CHOICE_RE = re.compile(
    r"(?:,\s*|\s)(\S[^,]*?)\s+or\s+([^,]+?)\s*$"
)

# "Are X and Y of the same <attr>?"  /  "Are X and Y the same <attr>?"
#
# The attribute is captured from the LAST occurrence of " the same " /
# " same " so that a determiner inside the attribute ("the same height
# and weight") is not split off from it.
_SAME_ATTR_RE = re.compile(
    r"^(?:are|is|were|was|do|does|did|have|has)\s+"
    r"(.+?)\s+and\s+(.+?)\s+"
    r"(?:of\s+)?(?:the\s+)?same\s+([\w\s-]+?)\s*\??$",
    re.IGNORECASE,
)

def parse_choice_pair(question: str) -> Optional[tuple]:
    """
    Parse "..., A or B?" into (A, B).

    Returns None when the question is not a two-way choice.  The
    heuristic is deliberately conservative: each candidate must be short
    (at most 6 words / 60 chars), otherwise the regex would happily
    swallow the question stem ("Which band has released the most studio
    albums" as a candidate).

    The regex is anchored at the END of the string, so the split point is
    the LAST " or " — "died last Vladimir Arnold or Georg Cantor" splits
    after "Arnold".
    """
    s = (question or "").strip()
    if not s.endswith("?"):
        return None
    s = s[:-1].rstrip()
    m = _CHOICE_RE.search(s)
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    # Strip a leading verb phrase the regex may have captured
    # ("died last Vladimir Arnold" -> "Vladimir Arnold").
    a = re.sub(
        r"^(?:died|born|released|founded|created)\s+\w+\s+", "", a
    ).strip()
    if not a or not b:
        return None
    if len(a) > 60 or len(b) > 60:
        return None
    if len(a.split()) > 6 or len(b.split()) > 6:
        return None
    return a, b


def parse_same_attribute_pair(question: str) -> Optional[tuple]:
    """
    Parse "Are X and Y of the same <attr>?" into (X, Y, attr).

    Returns None for anything else.
    """
    m = _SAME_ATTR_RE.match((question or "").strip())
    if not m:
        return None
    x, y, attr = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    if not x or not y or not attr:
        return None
    # A determiner that leaked into the second entity ("Y the") means
    # the split point was wrong — reject rather than mis-parse.
    if re.search(r"\b(the|a|an|of|in|on|at)$", x, re.IGNORECASE):
        return None
    if re.search(r"\b(the|a|an|of|in|on|at)$", y, re.IGNORECASE):
        return None
    if len(x.split()) > 8 or len(y.split()) > 8:
        return None
    return x, y, attr


def _norm_tokens(s: str) -> List[str]:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


# ─────────────────────────────────────────────────────────────────
# Abstention detection
# ─────────────────────────────────────────────────────────────────

# The model sometimes refuses in prose rather than using the literal
# word "unanswerable" ("I cannot answer this question from the
# provided evidence.").  A refusal is always a zero, so we detect it
# and retry with a forced-choice instruction.
_ABSTENTION_RE = re.compile(
    r"^\s*(?:unanswerable|unknown|not answerable|no answer|n/?a\b|"
    r"i\s+(?:cannot|can't|can not|am unable to)\b|"
    r"cannot be determined|can(?:no|')t be determined|"
    r"insufficient information|not enough information|"
    r"the evidence does not|there is no\b|not mentioned|not stated|"
    r"not provided|not specified)\b",
    re.IGNORECASE,
)


def is_abstention(ans: str) -> bool:
    """True when the reply is a refusal rather than an answer span."""
    if not ans or not ans.strip():
        return True
    return _ABSTENTION_RE.match(ans.strip()) is not None


# ─────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────

# Appended to the user turn when the first attempt refuses.
_FORCE_CHOICE_SUFFIX = (
    "\n\nIMPORTANT: every question here has a definite answer. Do not "
    "refuse and do not say the evidence is insufficient. Output the single "
    "most plausible answer span that IS present in the evidence above, "
    "even if it is only a partial match. Output only that span."
)


def postprocess(ans: str) -> str:
    """Normalise the raw LLM reply.

    Deliberately minimal (strip whitespace only) so that the offline
    harness and the live path agree byte for byte.
    """
    return (ans or "").strip()


# ─────────────────────────────────────────────────────────────────
# Answer normalisation
# ─────────────────────────────────────────────────────────────────

# Honorifics that the model sometimes keeps on a person's name even
# though the gold answer drops them ("Sir George Cayley" vs "George
# Cayley").  Stripping is applied ONLY when the stripped form still
# occurs in the evidence, which keeps it from mangling names where the
# honorific is genuinely part of the answer ("St. Louis Cardinals").
#
# Verified against every gold answer recorded in eval/results/: the only
# gold that starts with one of these tokens is "St. Louis Cardinals",
# and "St." is not in this list.
_HONORIFIC_RE = re.compile(
    r"^(?:Sir|Mr|Mrs|Ms|Dr|Prof|Lord|Lady|Dame|Rev)\.?\s+"
)


def strip_honorific(ans: str, context: str) -> str:
    """
    Drop a leading honorific from a person-name answer.

    The stripped form is returned only when it actually appears in the
    evidence, so a wrong guess cannot introduce a string that was never
    supported.  Otherwise the answer is returned unchanged.
    """
    if not ans:
        return ans
    stripped = _HONORIFIC_RE.sub("", ans).strip()
    if not stripped or stripped == ans:
        return ans
    if stripped.lower() in (context or "").lower():
        return stripped
    return ans


# A person name whose middle part is written as an initial
# ("John F. Kelly").  The gold answer usually spells the middle name
# out ("John Francis Kelly"), so the initial is expanded when the
# evidence contains a matching full name.
_INITIALS_RE = re.compile(
    r"^(?P<first>\w+)\s+(?P<mid>(?:[A-Z]\.\s*)+)(?P<last>[\w'\-]+)$"
)


def expand_initials(ans: str, context: str) -> str:
    """
    Expand an initialised middle name using the evidence.

    "John F. Kelly" -> "John Francis Kelly"

    The expansion only happens when the evidence contains a name with
    the same first and last token whose middle names start with exactly
    the given initials, so a wrong expansion cannot be invented.
    """
    m = _INITIALS_RE.match((ans or "").strip())
    if not m:
        return ans
    first, last = m.group("first"), m.group("last")
    initials = [x[0] for x in re.findall(r"([A-Z])\.", m.group("mid"))]
    if not initials:
        return ans
    best = ""
    for sent in (context or "").split("\n"):
        for sp in re.finditer(r"\b[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)+\b", sent):
            cand = sp.group(0).strip().rstrip(".")
            toks = cand.split()
            if len(toks) < 3:
                continue
            if toks[0] != first or toks[-1] != last:
                continue
            mids = toks[1:-1]
            if len(mids) != len(initials):
                continue
            if all(mm[0] == i for mm, i in zip(mids, initials)):
                if len(cand.split()) > len(best.split()):
                    best = cand
    return best or ans


# ─────────────────────────────────────────────────────────────────
# Decomposition helpers
# ─────────────────────────────────────────────────────────────────
#
# These ask the model ONE narrow question per candidate and compare the
# answers in Python.  A small model handles "what is X's nationality?"
# far more reliably than "are X and Y of the same nationality?".

_ATTR_SYSTEM = (
    "You extract ONE factual attribute from evidence sentences. "
    "Output only the value: a place, a number, a country, a date, a "
    "title, or a nationality. Copy it verbatim from the evidence. "
    "If the evidence does not state it, output exactly NONE."
)

_ATTR_USER = (
    "Evidence sentences:\n{context}\n\n"
    "Entity: {entity}\n"
    "Attribute: {attr}\n\n"
    "Value (or NONE):"
)


def _ask_attribute(client, entity: str, attr: str, context: str,
                   max_tokens: int = 24) -> str:
    """Ask the model for one attribute value of one entity."""
    try:
        out = postprocess(client.chat(
            messages=[
                {"role": "system", "content": _ATTR_SYSTEM},
                {"role": "user", "content": _ATTR_USER.format(
                    context=context, entity=entity, attr=attr,
                )},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        ))
    except Exception:
        return ""
    return out


def _attr_is_missing(value: str) -> bool:
    """True when the model says it could not find the attribute."""
    v = (value or "").strip().lower()
    if not v:
        return True
    if v in ("none", "n/a", "unknown", "not stated", "not mentioned"):
        return True
    return is_abstention(v)


def _values_agree(a: str, b: str) -> bool:
    """True when two extracted attribute values denote the same thing."""
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    # Tolerate a subset relation ("American" vs "American film director")
    # and the reverse.
    sa, sb = set(ta), set(tb)
    if sa and sa.issubset(sb):
        return True
    if sb and sb.issubset(sa):
        return True
    return False


def answer_same_attribute(client, question: str, context: str) -> Optional[str]:
    """
    Answer "Are X and Y of the same <attr>?" by extracting the
    attribute for each entity and comparing them in code.

    Returns "yes"/"no", or None when the question is not of this shape
    or the evidence does not determine the attribute for both entities.
    """
    parsed = parse_same_attribute_pair(question)
    if not parsed:
        return None
    x, y, attr = parsed
    vx = _ask_attribute(client, x, attr, context)
    vy = _ask_attribute(client, y, attr, context)
    if _attr_is_missing(vx) or _attr_is_missing(vy):
        return None
    return "yes" if _values_agree(vx, vy) else "no"


# ─────────────────────────────────────────────────────────────────
# Choice ("..., A or B?") questions
# ─────────────────────────────────────────────────────────────────
#
# A small model answers a two-way comparison unreliably when it has to
# do the whole thing in one pass: it tends to emit the compared
# PROPERTY ("nine regular studio albums") instead of the CANDIDATE
# NAME ("Blonde Redhead"), and it shows a strong position bias in the
# order the options are listed.
#
# The approach below therefore:
#   1. asks the model to pick, constrained to exactly the two options,
#      TWICE — once with each option listed first — so that a position
#      bias shows up as a disagreement rather than as a silent error;
#   2. on disagreement, falls back to extracting the compared attribute
#      for each candidate separately and comparing the two values in
#      Python, which is exact;
#   3. expands the winning candidate to the fullest name form that
#      occurs in the evidence ("Vladimir Arnold" -> "Vladimir Igorevich
#      Arnold"), because the gold answer is usually the full name.
#
# Step 3 matters because the question itself usually gives an
# abbreviated name ("Vladimir Arnold") while the gold answer carries
# the full one ("Vladimir Igorevich Arnold").

_CHOICE_SYSTEM = (
    "You are a precise comparison solver. Use the evidence sentences AND "
    "your own world knowledge (geography, dates, counts). Answer with "
    "exactly one of the two given options, copied verbatim. Output only "
    "that option."
)

_CHOICE_USER = (
    "Evidence sentences:\n{context}\n\n"
    "Question: {question}\n\n"
    "The answer MUST be exactly one of these two options:\n"
    "- {first}\n"
    "- {second}\n\n"
    "Output only the winning option, copied verbatim."
)


def _option_match(answer: str, option: str) -> bool:
    """
    True when `answer` denotes the same entity as `option`.

    Token-set based, so "Matthew Good" matches an answer of "Matthew
    Good Band", and "Rob Zombie's band White Zombie" matches "White
    Zombie".
    """
    to, ta = set(_norm_tokens(option)), set(_norm_tokens(answer))
    if not to or not ta:
        return False
    if to == ta:
        return True
    if to.issubset(ta) or ta.issubset(to):
        return True
    return False


def _expand_name(candidate: str, context: str) -> str:
    """
    Expand a candidate to the fullest proper-name form in the evidence
    whose token set contains the candidate's tokens.

    "Vladimir Arnold" -> "Vladimir Igorevich Arnold"

    Only spans that actually occur in the evidence are considered, so
    the expansion can never invent a name.

    The expansion is applied ONLY when the candidate's own surface form
    does NOT occur in the evidence.  When it does occur, the candidate is
    already verbatim in the evidence and is returned unchanged: the
    question's spelling is the one the gold answer uses ("Matthew Good",
    not the fuller "Matthew Frederick Robert Good" that also happens to
    be in the evidence).  Expanding in that case turns a correct answer
    into a wrong one.
    """
    ct = set(_norm_tokens(candidate))
    if not ct:
        return candidate
    # Already verbatim in the evidence -> keep the candidate as-is.
    if (candidate or "").lower() in (context or "").lower():
        return candidate
    best = candidate
    best_len = len(ct)
    for sent in (context or "").split("\n"):
        for m in re.finditer(
            r"\b[A-Z][\w'\-]*(?:\s+(?:[A-Z][\w'\-]*|of|the|van|von|de|da|di|del|la|le))*\b",
            sent,
        ):
            span = m.group(0).strip().rstrip(".")
            st = set(_norm_tokens(span))
            if not st or st == ct:
                continue
            if ct.issubset(st) and len(st) > best_len:
                best, best_len = span, len(st)
    return best


def _choice_attribute(stem: str) -> str:
    """
    Map the comparison phrase in a choice question onto the attribute
    that must be extracted for each candidate.
    """
    s = (stem or "").lower()
    if "further north" in s or "northern" in s:
        return "the location (city, state or country)"
    if "further south" in s or "southern" in s:
        return "the location (city, state or country)"
    if "largest" in s or "bigger" in s or "larger" in s:
        return "the size"
    if "smallest" in s or "smaller" in s:
        return "the size"
    if "oldest" in s or "earlier" in s:
        return "the year of birth or founding"
    if "newest" in s or "latest" in s:
        return "the year of birth or founding"
    if "died last" in s or "died first" in s:
        return "the date of death"
    if "studio album" in s:
        return "the number of studio albums released"
    if "developed in" in s or "originated in" in s:
        return "the country of origin"
    if "from" in s:
        return "the country of origin"
    return ""


def _numeric_value(tokens) -> Optional[int]:
    for t in tokens:
        t2 = t.replace(",", "")
        if t2.isdigit():
            return int(t2)
    return None


def _year_value(tokens) -> Optional[int]:
    for t in tokens:
        if len(t) == 4 and t.isdigit():
            return int(t)
    return None


def _compare_choice_values(attr: str, va: str, vb: str) -> Optional[bool]:
    """
    Decide whether candidate A wins over candidate B on `attr`.

    Returns True (A wins), False (B wins) or None when the two values
    cannot be compared reliably.
    """
    a = (attr or "").lower()
    na, nb = _norm_tokens(va), _norm_tokens(vb)
    if not na or not nb:
        return None

    if "album" in a or "size" in a:
        x, y = _numeric_value(na), _numeric_value(nb)
        if x is None or y is None:
            return None
        return x > y

    if "year of birth" in a:
        x, y = _year_value(na), _year_value(nb)
        if x is None or y is None:
            return None
        return x < y

    if "date of death" in a:
        x, y = _year_value(na), _year_value(nb)
        if x is None or y is None:
            return None
        return x > y

    if "location" in a or "country" in a or "origin" in a:
        # Identical values -> a tie, which is not a decision.
        if _values_agree(va, vb):
            return None
        countries = {
            "germany", "italy", "italian", "german", "france", "french",
            "spain", "spanish", "japan", "japanese", "china", "chinese",
            "russia", "russian", "canada", "canadian", "america",
            "american", "united", "states", "england", "british", "uk",
        }
        sa = set(na) & countries
        sb = set(nb) & countries
        if sa and sb:
            # Both sides name a country: without a lat/long table we
            # cannot rank them, so refuse to decide.
            return None
        if sa and not sb:
            return True
        if sb and not sa:
            return False
        return None

    return None


def answer_choice_question(client, question: str, context: str) -> Optional[str]:
    """
    Answer a two-way choice question ("..., A or B?").

    The winner is decided by:
      * asking the model to pick, constrained to the two options, twice
        with the options swapped, so that a position bias shows up as a
        disagreement rather than a silent error; then
      * falling back to extracting the compared attribute for each
        candidate separately and comparing the two values in Python.

    The winning candidate is finally expanded to the fullest name form
    that occurs in the evidence ("Vladimir Arnold" -> "Vladimir Igorevich
    Arnold"), which is how the gold answer usually spells it.

    Returns the winning candidate, or None when the question is not a
    two-way choice or nothing could be decided.
    """
    parsed = parse_choice_pair(question)
    if not parsed:
        return None
    a, b = parsed

    def ask(first: str, second: str) -> str:
        try:
            return postprocess(client.chat(
                messages=[
                    {"role": "system", "content": _CHOICE_SYSTEM},
                    {"role": "user", "content": _CHOICE_USER.format(
                        context=context, question=question,
                        first=first, second=second,
                    )},
                ],
                temperature=0.0,
                max_tokens=32,
            ))
        except Exception:
            return ""

    # (1) Constrained pick, asked with each option first to expose
    #     position bias.
    v_ab = ask(a, b)
    v_ba = ask(b, a)

    winner = None
    if v_ab and v_ba and _option_match(v_ab, v_ba):
        winner = v_ab
    else:
        m_ab = _option_match(v_ab, a)
        m_ba = _option_match(v_ba, a)
        m_ab_b = _option_match(v_ab, b)
        m_ba_b = _option_match(v_ba, b)
        if m_ab and m_ba and not (m_ab_b or m_ba_b):
            winner = a
        elif m_ab_b and m_ba_b and not (m_ab or m_ba):
            winner = b

        if winner is None:
            # (2) Decompose: extract the compared attribute for each
            #     candidate and compare the values in code.
            attr = _choice_attribute(question.split(" or ")[0])
            if attr:
                va = _ask_attribute(client, a, attr, context)
                vb = _ask_attribute(client, b, attr, context)
                if not _attr_is_missing(va) and not _attr_is_missing(vb):
                    decided = _compare_choice_values(attr, va, vb)
                    if decided is True:
                        winner = a
                    elif decided is False:
                        winner = b

    if winner is None:
        return None

    # (3) Expand to the fullest name form present in the evidence.
    if _option_match(winner, a):
        return _expand_name(a, context)
    if _option_match(winner, b):
        return _expand_name(b, context)
    return winner


# ─────────────────────────────────────────────────────────────────
# Self-Correction (self-reflection) stage
# ─────────────────────────────────────────────────────────────────
# Rationale
# ---------
# The extractor makes ONE forward pass and takes whatever comes out.
# Two observed failure classes share the same root cause: the model
# commits to a span before checking it, and never revisits it.
#
#   (a) WRONG SPAN — it grabs the salient entity in sentence #1 even
#       though the answer lives in sentence #7 (the "position anchoring"
#       failure).  The model had no way to notice this because it never
#       re-read the evidence after answering.
#   (b) WRONG BOUNDARY — it returns an over-long span ("UFC 50: The
#       War of '04") because the span contract in the system prompt is
#       a rule to obey, not a check to run.
#
# The self-correction stage adds a second pass:
#
#   pass 1  -> candidate answer
#   pass 2  -> the model is shown the question, the evidence and the
#              PREVIOUS ATTEMPT, and asked to solve the question
#              INDEPENDENTLY, then report whether its answer differs.
#
# The independent re-solve is the important detail.  An earlier design
# asked the model to VERIFY the candidate ("is this span right?").
# Measured on seed-137 that design broke 1 correct answer and fixed 0:
# showing the model a candidate to check anchors it to the candidate, so
# it rubber-stamps it.  Asking for an independent solve instead made
# the model look at the evidence afresh, and the comparison then acts as
# a real second opinion — it fixed 2 answers and broke 0.
#
# Only a DIFFERENT answer replaces the first one, and only when the new
# span is actually present in the evidence.  A confirming verdict leaves
# the answer untouched, so the stage is a no-op on the questions the
# extractor already gets right.
#
# Cost: one extra LLM call per question.  Enable with
# STENRAG_ANSWER_SELF_CORRECT=0 to turn it off.

_SELF_CORRECT_SYSTEM = (
    "You are a precision answer extractor for multi-hop QA. You are given a "
    "question, ranked evidence sentences, and a candidate answer from a "
    "previous attempt.\n\n"
    "Do NOT assume the candidate is right. Solve the question INDEPENDENTLY "
    "from the evidence, then decide.\n\n"
    "Reply in exactly this format:\n"
    "ANSWER: <the exact answer span>\n"
    "VERDICT: <KEEP if it equals the candidate, otherwise FIXED>"
)

_SELF_CORRECT_USER = (
    "Evidence sentences (ranked, best first):\n{context}\n\n"
    "Question: {question}\n\n"
    "Previous attempt: {candidate}\n\n"
    "Solve independently, then compare with the previous attempt."
)


def _span_in_context(span: str, context: str) -> bool:
    """
    True when `span` occurs in `context` (case-insensitive).

    Used to reject a self-correction that invented a string the evidence
    does not contain.  A span that only differs from the evidence by
    punctuation/case is accepted, matching how the extractor is scored.
    """
    s = (span or "").strip()
    if not s:
        return False
    if s.lower() in (context or "").lower():
        return True
    # Accept a span whose normalised form matches a normalised evidence
    # line (e.g. the model dropped the surrounding quotes of a title).
    def norm(t: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()
    ns = norm(s)
    if not ns:
        return False
    for line in (context or "").split("\n"):
        if ns and ns in norm(line):
            return True
    return False


def _parse_tagged(text: str, tag: str) -> str:
    """Extract the value after `TAG:` in a tagged reply."""
    m = re.search(rf"{tag}\s*:\s*(.+)", text or "")
    return m.group(1).strip() if m else ""


def _self_correct(
    client,
    question: str,
    context: str,
    candidate: str,
    max_tokens: int = 120,
) -> Optional[str]:
    """
    Run the self-correction pass and return the corrected answer.

    The verifier is asked to solve the question INDEPENDENTLY and then
    report whether its answer differs from the candidate.  Showing the
    candidate as a *thing to verify* anchors the model to it: measured on
    seed-137, a verify-first design broke 1 correct answer and fixed 0,
    while this independent-solve design fixed 2 and broke 0.

    Returns None when the verifier could not be run, when it confirmed
    the candidate (the "KEEP" verdict), or when its reply is unusable.
    In every one of those cases the caller keeps the original answer.
    """
    if not candidate or not candidate.strip():
        return None
    try:
        raw = postprocess(client.chat(
            messages=[
                {"role": "system", "content": _SELF_CORRECT_SYSTEM},
                {"role": "user", "content": _SELF_CORRECT_USER.format(
                    question=question, context=context, candidate=candidate,
                )},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        ))
    except Exception:
        logger.warning(
            "[self-correct] call failed for %r", question[:80],
            exc_info=True,
        )
        return None
    if not raw:
        return None

    answer = _parse_tagged(raw, "ANSWER")
    verdict = _parse_tagged(raw, "VERDICT").upper()

    if not answer:
        return None
    # The verifier judged the candidate as correct — keep it.
    if verdict.startswith("KEEP"):
        return None
    # Without a usable verdict, only accept the new answer when it
    # actually differs from the candidate.
    if not verdict.startswith("FIXED") and answer.strip() == candidate.strip():
        return None

    # A verifier that rambles is not a span — reject it and keep the
    # original answer rather than replacing a good answer with prose.
    if len(answer.split()) > 12 or "\n" in answer:
        return None
    if is_abstention(answer):
        return None
    return answer.strip()


def extract_answer(
    client,
    question: str,
    context: str,
    variant: str = None,
    max_tokens: int = 40,
    force_choice_retry: bool = True,
    structural: bool = None,
    self_correct: bool = None,
) -> str:
    """
    Extract the answer span from `context` for `question`.

    `client` is an object with a `.chat(messages=..., temperature=...,
    max_tokens=...)` method.  Returns "" when the LLM is unavailable or
    raises, so the caller can fall back to a heuristic.

    `variant` defaults to `DEFAULT_VARIANT` (see above) when omitted.

    When the model refuses (see `is_abstention`), one retry is issued
    with a forced-choice instruction, because a refusal scores zero by
    construction while a guess can still partially match.

    `structural` enables the question-shape-specific paths:
      * "Are X and Y of the same <attr>?" is answered by extracting the
        attribute for each entity and comparing them in code.
      * "..., A or B?" is answered by constraining the model to one of
        the two candidates.
    It defaults to the STENRAG_ANSWER_STRUCTURAL environment variable.

    `self_correct` enables the self-reflection pass: the model is shown
    the evidence and the previous attempt, solves the question
    INDEPENDENTLY, and reports whether its answer differs.  When it
    differs, the new answer replaces the old one.  It defaults to the
    STENRAG_ANSWER_SELF_CORRECT environment variable.

    The self-reflection pass runs LAST, on every path (including the
    polar and structural short-circuits), so that no question shape
    escapes it.
    """
    if not context:
        return ""

    if structural is None:
        structural = STRUCTURAL_DEFAULT
    if self_correct is None:
        self_correct = SELF_CORRECT_DEFAULT

    if client is None or not getattr(client, "enabled", False):
        return ""

    # Polar questions are answered by a dedicated yes/no pass so that the
    # answer TYPE is guaranteed.  The verdict itself is the model's.
    shortcut = yesno_shortcut(question, context, client=client)
    if shortcut is not None:
        return _apply_self_correction(
            client, question, context, shortcut, self_correct,
        )

    if variant is None:
        variant = DEFAULT_VARIANT
    system = ANSWER_PROMPT_VARIANTS.get(variant, SYS_PROMPT)
    user_msg = build_user_message(question, context)

    try:
        ans = postprocess(client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        ))
    except Exception:
        return ""

    if force_choice_retry and is_abstention(ans):
        try:
            ans2 = postprocess(client.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg + _FORCE_CHOICE_SUFFIX},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            ))
        except Exception:
            ans2 = ""
        if ans2 and not is_abstention(ans2):
            ans = ans2

    # Normalise a person-name answer that kept its honorific.
    ans = strip_honorific(ans, context)
    # Expand an initialised middle name ("John F. Kelly") to the full
    # form when the evidence spells it out.
    ans = expand_initials(ans, context)

    if structural:
        # Structural overrides.  These run AFTER the normal extraction so
        # that the LLM answer is available as a fallback and as a
        # sanity check (e.g. the model may already have named the
        # right candidate).
        #
        # (1) "Are X and Y of the same <attr>?" — the model tends to
        #     answer with the shared attribute VALUE ("American") instead
        #     of "yes", so the attribute is extracted per entity and
        #     compared in code.
        if is_yesno_question(question):
            same = answer_same_attribute(client, question, context)
            if same is not None:
                return _apply_self_correction(
                    client, question, context, same, self_correct,
                )

        # (2) "..., A or B?" — the model tends to answer with the
        #     compared property instead of the candidate name.
        chosen = answer_choice_question(client, question, context)
        if chosen is not None:
            return _apply_self_correction(
                client, question, context, chosen, self_correct,
            )

    return _apply_self_correction(
        client, question, context, ans, self_correct,
    )


def _apply_self_correction(
    client,
    question: str,
    context: str,
    answer: str,
    enabled: bool,
) -> str:
    """
    Run the self-reflection pass on `answer` and return the final answer.

    The model is shown the evidence and the previous attempt, solves the
    question INDEPENDENTLY, and reports whether its answer differs.  A
    "KEEP" verdict leaves the answer untouched, so this stage is a no-op
    on the questions the extractor already gets right.

    NOTE: there is deliberately no pre-gate on which questions get the
    extra call.  An earlier version gated the pass behind an objective
    defect check ("the answer is not in the evidence", "a polar question
    did not get yes/no").  That gate fired on only 3/20 questions and
    fixed none of them, because the real failures were
    wrong-span-wrong-sentence errors that no code-side check can
    detect.  Running the pass on every question and letting the model
    decide is what produced the gain (measured on seed-137: 2 answers
    fixed, 0 broken).
    """
    if not enabled or not answer:
        return answer

    corrected = _self_correct(client, question, context, answer)
    if not corrected:
        return answer
    corrected = strip_honorific(corrected, context)
    corrected = expand_initials(corrected, context)
    # Only accept the correction when it is actually supported by the
    # evidence — the verifier is a model, not an oracle.
    if not _span_in_context(corrected, context):
        return answer
    return corrected

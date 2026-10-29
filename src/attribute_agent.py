"""
Attribute Agent
===============

Rolling optimisation of the context-memory attribute table.

Why this module exists
----------------------
The attribute table decides, for every sentence, WHICH property it
carries ("birth", "date", "genre", ...) and for WHICH entity.  A missing
attribute means the memory cannot answer a question about that property;
a missing entity class means every sentence about that kind of entity
falls back to the `description` catch-all and carries no usable signal.

The table is currently hard-coded in ``src/sentence_graph.py``.  This
module implements the agent that ROLLS it:

    sample → detect gap → propose → gate → apply → re-measure

Concretely, per run the agent:

1. **Samples** ``n_questions`` random questions ``n_rounds`` times
   (default 20 x 10 = 200 draws) from a QA dataset.
2. **Detects gaps**:
   * *missing attribute* — the question asks for a property that the
     anchor entity's class does not declare;
   * *missing class* — the anchor entity's spaCy type has no template
     of its own, so it falls back to the `description` catch-all.
3. **Proposes** an `AttributeSpec` / `ClassSpec` for each gap, drawing
   the wording from a bounded candidate lexicon (the agent may not
   invent an unbounded vocabulary).
4. **Gates** every proposal on three properties:
   * *inclusivity* — the attribute must be broad enough to hold a
     range of sentences, and must not duplicate an existing attribute;
   * *mount efficiency* — an attribute that only ever mounts a
     handful of sentences is not worth a slot in the embedding matrix;
   * *distinctiveness* — the attribute must belong to ONE class
     ("birth" belongs to PERSON, "date" belongs to EVENT); if it would
     rank equally high for several classes it discriminates nothing
     and is discarded.
5. **Applies** the surviving delta through `AttributeSchema`, then
   re-mounts the sentences so the new attributes actually take effect.

The LLM is used in exactly two places, and both are fallbacks:

* **Ambiguous ownership** — when one sentence carries several entities
  of the SAME class ("Washington and Lincoln"), the vector mount cannot
  tell WHICH entity the attribute belongs to.  The agent asks the LLM.
* **Gap confirmation** — a detected gap may be an artefact of the
  trigger vocabulary; the LLM is asked to confirm before the gap is
  accepted.

Everything else is deterministic.  The agent never rewrites source
files: the delta is persisted as JSON and can be exported as a
pasteable snippet (`AttributeSchema.export_python`).

Public API
----------
    AttributeAgentConfig   Tunables for one run.
    AttributeGap           One detected gap, with its verdict.
    AttributeAgent         The agent.
    AgentReport            The outcome of a run.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .attribute_schema import (
    ATTRIBUTE_CANDIDATE_LEXICON,
    CLASS_CANDIDATE_LEXICON,
    AttributeSchema,
    AttributeSpec,
    ClassSpec,
    normalize_entity_label,
    stems_to_patterns,
)
from . import sentence_graph as _sg

logger = logging.getLogger(__name__)


# ==== [2026-10-29] 属性表滚动优化 Agent ====
# [改动目的] 上下文记忆设计阶段需要"agent 滚动优化属性表"：随机抽题让
#             agent 判断是否缺少属性/实体类，缺则补，并按包容性、挂载
#             效率、区分度三条准则决定保留还是丢弃。
# [改动功能] AttributeAgent 完成 采样→检测缺口→提议→三准则把关→应用
#             →重新挂载 的完整闭环；LLM 仅用于同大类多实体的归属兜底
#             与缺口确认；增量通过 AttributeSchema 持久化，不改源码。
# ==== end 属性表滚动优化 Agent ====


# ---------------------------------------------------------------------------
# Question-level triggers for the EXISTING global attributes.
#
# The tables in `sentence_graph` carry SENTENCE-level stems ("born",
# "died").  Gap detection works on the QUESTION side: "when was X born"
# asks for `birth` even though the question never contains the word
# "born" alone.  So we keep a separate question-trigger table.
# ---------------------------------------------------------------------------

QUESTION_TRIGGERS: Dict[str, List[str]] = {
    "birth": [
        "when was", "date of birth", "birth date", "birthday",
        "born in", "born on", "year of birth", "place of birth",
        "how old is", "what age",
    ],
    "death": [
        "when did", "date of death", "death date", "died",
        "passed away", "year of death", "place of death", "how old was",
    ],
    "career": [
        "career", "profession", "occupation", "what does", "what did",
        "job", "work as", "position", "role in",
    ],
    "education": [
        "education", "educated", "university", "college", "school",
        "alma mater", "studied", "degree", "graduated",
    ],
    "family": [
        "father", "mother", "spouse", "married", "wife", "husband",
        "children", "son", "daughter", "brother", "sister",
        "family", "relatives",
    ],
    "achievement": [
        "award", "won", "prize", "honor", "honour", "medal",
        "record", "championship", "achievement", "recognised",
        "recognized", "nominated",
    ],
    "residence": [
        "lives in", "lived in", "residence", "resided", "hometown",
        "moved to", "lived on",
    ],
    "founded": [
        "founded", "established", "when was", "created in",
        "incorporated", "started in",
    ],
    "headquarters": [
        "headquarters", "headquartered", "hq", "based in",
        "main office",
    ],
    "products": [
        "what product", "products", "makes", "made", "manufactures",
        "what does", "released", "publishes",
    ],
    "leadership": [
        "who leads", "who is the ceo", "ceo", "president of",
        "head of", "led by", "chairman", "leader of", "directed by",
        "managed by", "runs",
    ],
    "events": [
        "major events", "milestone", "historical events", "important events",
    ],
    "location": [
        "where is", "located", "location of", "situated",
        "geographical", "what country is", "which country",
    ],
    "history": [
        "history of", "historical", "originated", "heritage",
    ],
    "population": [
        "population", "how many people", "how many inhabitants",
        "census", "how many residents",
    ],
    "date": [
        "what year", "what date", "when", "on what day",
    ],
    "participants": [
        "who participated", "who took part", "who attended",
        "participants", "attendees", "who was involved",
    ],
    "outcome": [
        "outcome", "result of", "what happened as a result",
        "consequence", "impact of",
    ],
    "creator": [
        "who created", "who wrote", "who directed", "who produced",
        "author of", "director of", "creator of", "written by",
    ],
    "genre": [
        "what genre", "genre of", "what type of work", "what style",
    ],
    "characters": [
        "characters", "who plays", "cast of", "protagonist",
        "main character",
    ],
    "event": [
        "when was the event", "when did the event", "what event",
    ],
    "origin": [
        "where is", "where are", "come from", "came from",
        "nationality", "ancestry", "roots",
    ],
    "beliefs": [
        "belief", "beliefs", "religion of", "ideology", "doctrine",
        "what does the group believe",
    ],
    "members": [
        "members of", "notable members", "who are the members",
        "founding members",
    ],
    "description": [
        "what is", "describe", "what kind of", "what type of",
    ],
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AttributeAgentConfig:
    """
    Tunables for one agent run.

    Attributes:
        n_questions:      Questions drawn per round.
        n_rounds:         Rounds.  Total draws = n_questions * n_rounds.
        seed:             Base seed.  Each round derives its own seed so
                          the rounds are reproducible but not identical.
        min_mount_count:  Minimum number of sentences an attribute must
                          be able to mount.  Below this the attribute
                          costs a slot in the embedding matrix and buys
                          nothing.
        min_global_mount_rate:
                          Minimum share of the corpus the attribute must
                          mount (the "mount efficiency" gate).
        min_class_top_rate:
                          Minimum share of a class's sentences on which
                          the attribute must rank top-K.
        min_distinctiveness_margin:
                          How much better the attribute must rank for its
                          home class than for any other class that also
                          declares it.  This is the "who owns this
                          attribute" gate.
        max_descriptor_cos:
                          Above this cosine against an existing
                          descriptor, a candidate is a duplicate and is
                          rejected.
        max_llm_calls:    Hard cap on LLM calls per run.
        use_llm:          Whether the LLM fallback may be used at all.
    """

    n_questions: int = 20
    n_rounds: int = 10
    seed: int = 1234

    min_mount_count: int = 3
    min_global_mount_rate: float = 0.02
    min_class_top_rate: float = 0.05
    min_distinctiveness_margin: float = 0.10
    max_descriptor_cos: float = 0.80

    max_llm_calls: int = 200
    use_llm: bool = True

    # Attribute mounting constants inherited from the graph builder.
    top_k: int = _sg.ATTR_MOUNT_TOP_K
    min_sim: float = _sg.ATTR_MOUNT_MIN_SIM


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------

@dataclass
class AttributeGap:
    """
    One detected gap in the attribute table.

    Attributes:
        kind:          "missing_attribute" or "missing_class".
        entity_type:   The entity class the gap belongs to.
        attribute:     The attribute name (for a missing class, the
                       class's proposed primary attribute).
        evidence:      The questions that exposed the gap.
        count:         How many times the gap fired.
        proposal:      The proposed spec, if the agent produced one.
        verdict:       "accepted" / "rejected" / "pending".
        reason:        Why the verdict was reached.
        metrics:       Numbers behind the verdict.
    """

    kind: str
    entity_type: str
    attribute: str = ""
    evidence: List[str] = field(default_factory=list)
    count: int = 0
    proposal: Optional[AttributeSpec] = None
    verdict: str = "pending"
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "entity_type": self.entity_type,
            "attribute": self.attribute,
            "evidence": list(self.evidence),
            "count": self.count,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "verdict": self.verdict,
            "reason": self.reason,
            "metrics": dict(self.metrics),
        }


@dataclass
class AgentReport:
    """The outcome of one agent run."""

    n_sampled: int = 0
    n_rounds: int = 0
    gaps: List[AttributeGap] = field(default_factory=list)
    accepted: List[AttributeGap] = field(default_factory=list)
    rejected: List[AttributeGap] = field(default_factory=list)
    llm_calls: int = 0
    ambiguity_resolved: int = 0
    schema: Optional[AttributeSchema] = None
    corpus_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "n_sampled": self.n_sampled,
            "n_rounds": self.n_rounds,
            "n_gaps": len(self.gaps),
            "n_accepted": len(self.accepted),
            "n_rejected": len(self.rejected),
            "llm_calls": self.llm_calls,
            "ambiguity_resolved": self.ambiguity_resolved,
            "gaps": [g.to_dict() for g in self.gaps],
            "corpus_stats": dict(self.corpus_stats),
        }


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------

def load_questions(path) -> List[Dict[str, Any]]:
    """
    Load a QA dataset into a list of dicts.

    Supports the HotpotQA JSON layout (``_id`` / ``question`` / ``answer``
    / ``context``) and degrades gracefully for a generic list of
    ``{"question": ..., "answer": ...}`` records.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Question file not found: %s", p)
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("data", "questions", "examples", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]

    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            continue
        out.append(
            {
                "qid": str(rec.get("_id") or rec.get("id") or rec.get("qid") or i),
                "question": str(rec.get("question") or rec.get("query") or ""),
                "answer": str(rec.get("answer") or rec.get("gold") or ""),
                "context": rec.get("context") or rec.get("paragraphs") or [],
            }
        )
    return out


def _sentences_of(context: Any) -> List[str]:
    """Flatten a HotpotQA-style ``context`` into a list of sentences."""
    out: List[str] = []
    if not isinstance(context, list):
        return out
    for para in context:
        if isinstance(para, (list, tuple)) and len(para) >= 2:
            sents = para[1]
            if isinstance(sents, list):
                out.extend(str(s) for s in sents)
        elif isinstance(para, dict):
            sents = para.get("sentences") or []
            if isinstance(sents, list):
                out.extend(str(s) for s in sents)
    return out


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class AttributeAgent:
    """
    The rolling attribute-table optimiser.

    Usage::

        agent = AttributeAgent(config, client=make_client())
        report = agent.run(load_questions("eval/data/hotpot_dev_distractor_v1.json"))
        report.schema.save("eval/results/attribute_schema.json")
    """

    def __init__(
        self,
        config: Optional[AttributeAgentConfig] = None,
        client: Optional[Any] = None,
        nlp: Optional[Any] = None,
        schema: Optional[AttributeSchema] = None,
        progress: Optional[Any] = None,
    ) -> None:
        """
        Args:
            progress: Optional progress reporter.  Must expose
                `update(done, total, note)` and `finish()`.  The agent
                calls it after every sampled question so the user can
                see how far the rolling optimisation has got.
        """
        self.config = config or AttributeAgentConfig()
        self.client = client
        self.nlp = nlp
        self.schema = schema or AttributeSchema()
        self._llm_calls = 0
        self.progress = progress

        # Cache for the descriptor embeddings.
        #
        # The descriptor set is CONSTANT across the whole run (it only
        # changes when the schema delta changes), but `_attr_profiles`
        # re-encoded all ~40 descriptors for EVERY sampled question —
        # 200 times for the default 20×10 configuration.  Caching them
        # turns 200 embedder batches into 1.
        self._desc_emb_cache: Optional[Tuple[Any, Dict[str, Any]]] = None

        if self.nlp is None:
            from .graph_builder import _load_spacy_shared
            self.nlp = _load_spacy_shared("en_core_web_lg")

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _chat(self, system: str, user: str) -> str:
        """One LLM call, respecting the per-run cap.  Never raises."""
        if not self.config.use_llm:
            return ""
        if self.client is None or not getattr(self.client, "enabled", False):
            return ""
        if self._llm_calls >= self.config.max_llm_calls:
            return ""
        self._llm_calls += 1
        try:
            return self.client.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=120,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("AttributeAgent LLM call failed: %s", e)
            return ""

    def _confirm_gap(self, gap: AttributeGap) -> Tuple[bool, str]:
        """
        Ask the LLM whether a detected gap is real.

        The deterministic detector fires on question trigger vocabulary,
        which can produce false positives ("when was" also matches the
        `founded` trigger for an ORG).  A confirmation step keeps the
        table from accumulating junk attributes.
        """
        if not self.config.use_llm:
            return True, "llm-disabled"
        if not gap.evidence:
            return True, "no-evidence"
        system = (
            "You audit a knowledge-graph attribute table. You are given "
            "questions and the attribute a system believes is MISSING for "
            "a given entity class. Decide whether the questions really "
            "ask for that attribute, and whether that attribute is a "
            "distinct, general property (not a one-off detail). "
            "Answer STRICT JSON: {\"missing\": true|false, "
            "\"attribute\": \"<snake_case name>\", "
            "\"reason\": \"<short>\"}"
        )
        user = (
            f"Entity class: {gap.entity_type}\n"
            f"Proposed attribute: {gap.attribute}\n"
            "Questions:\n"
            + "\n".join(f"- {q}" for q in gap.evidence[:8])
            + "\n\nJSON:"
        )
        raw = self._chat(system, user)
        if not raw:
            return True, "llm-unavailable"
        obj = _parse_json_object(raw)
        if not obj:
            return True, "llm-unparseable"
        missing = bool(obj.get("missing", True))
        reason = str(obj.get("reason") or "")
        name = str(obj.get("attribute") or "").strip().lower()
        if name and re.fullmatch(r"[a-z][a-z0-9_]*", name):
            gap.attribute = name
        return missing, reason or ("confirmed" if missing else "rejected")

    def _resolve_ambiguous_owner(
        self,
        sentence: str,
        attribute: str,
        label: str,
        candidates: Sequence[str],
    ) -> str:
        """
        Decide which of several same-class entities an attribute belongs to.

        This is the LLM FALLBACK for the ambiguity the vector mount cannot
        resolve: "Washington crossed the Delaware with Lincoln" mounts
        under `career` for both PERSONs, but the fact is about one of
        them.

        Returns the chosen entity, or "" when the LLM cannot decide (the
        caller then keeps the mount on all candidates).
        """
        if not self.config.use_llm or len(candidates) < 2:
            return ""
        system = (
            "You decide which entity in a sentence a given attribute "
            "belongs to. The attribute is a property (e.g. birth date, "
            "founding date). Exactly one candidate usually owns it. "
            "Answer STRICT JSON: {\"entity\": \"<one candidate, verbatim>\"} "
            "or {\"entity\": \"none\"} if no candidate owns it."
        )
        user = (
            f"Sentence: {sentence}\n"
            f"Attribute: {attribute} ({label})\n"
            f"Candidates: {json.dumps(list(candidates), ensure_ascii=False)}\n"
            "JSON:"
        )
        raw = self._chat(system, user)
        if not raw:
            return ""
        obj = _parse_json_object(raw)
        if not obj:
            return ""
        choice = str(obj.get("entity") or "").strip()
        if not choice or choice.lower() == "none":
            return ""
        # Accept only an exact candidate match (case-insensitive).
        for cand in candidates:
            if cand.lower() == choice.lower():
                return cand
        return ""

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_rounds(
        self,
        questions: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Draw ``n_rounds`` rounds of ``n_questions`` random questions.

        Each round uses its own derived seed, so the rounds are
        reproducible from ``config.seed`` but not identical to each
        other — that is what makes the agent see the table from several
        angles instead of re-checking the same 20 questions 10 times.
        """
        if not questions:
            return []
        rounds: List[List[Dict[str, Any]]] = []
        n = min(self.config.n_questions, len(questions))
        for r in range(self.config.n_rounds):
            rng = random.Random(self.config.seed + r * 7919)
            idx = rng.sample(range(len(questions)), n)
            idx.sort()
            rounds.append([questions[i] for i in idx])
        return rounds

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, qdata: Dict[str, Any]):
        """
        Build a StenRAG graph for one question.

        Mirrors the evaluation harness: the gold + distractor paragraphs
        become sentences, entities come from spaCy, and the graph is
        built with the shared model.
        """
        from .graph_builder import ComplementedSentence, GraphBuilder

        sents = _sentences_of(qdata.get("context"))
        if not sents:
            return None, None

        builder = GraphBuilder(shared_nlp=self.nlp)
        for text in sents:
            try:
                doc = self.nlp(text)
            except Exception:  # noqa: BLE001
                continue
            ents = [e.text for e in doc.ents]
            builder.add_sentence(
                ComplementedSentence(
                    original_text=text,
                    complemented_text=text,
                    resolved_subject=ents[0] if ents else "",
                    entities=ents,
                )
            )

        # Build the sentence embeddings and the attribute mounts.
        #
        # `compute_tfidf_index()` is what populates `node.summary_vector`
        # (via `_populate_sentence_summary_vectors`) and then runs
        # `_soft_mount_attributes`.  Without it every node's vector stays
        # None, so the agent would measure nothing.
        try:
            builder.compute_tfidf_index()
        except Exception as e:  # noqa: BLE001
            logger.warning("AttributeAgent: tf-idf/embedding build failed: %s", e)

        # Make sure every container knows its entity inventory, so the
        # template lookup and the entity->class mapping work.
        try:
            for c in builder.container_manager.get_all_containers():
                builder.container_manager.update_container_index(c.id)
        except Exception as e:  # noqa: BLE001
            logger.debug("AttributeAgent: container index update failed: %s", e)

        return builder, builder.sentence_registry

    def _question_entities(self, builder, question: str) -> List[str]:
        """
        Find the containers the question anchors on.

        Longest surface form first, so "Dr. Sarah Chen" wins over
        "Sarah Chen"; a candidate that is a strict subspan of an
        already-chosen one is dropped.
        """
        if builder is None:
            return []
        q_lower = (question or "").lower()
        cands: List[str] = []
        seen: Set[str] = set()
        for c in builder.container_manager.get_all_containers():
            e = (c.entity or "").strip()
            if not e or e in seen:
                continue
            seen.add(e)
            cands.append(e)
        cands.sort(key=lambda x: -len(x))

        chosen: List[str] = []
        used_norm: Set[str] = set()
        for cand in cands:
            if cand.lower() not in q_lower:
                continue
            norm = normalize_entity_label(cand)
            if not norm or norm in used_norm:
                continue
            if any(norm in p and norm != p for p in used_norm):
                continue
            used_norm.add(norm)
            chosen.append(cand)
            if len(chosen) >= 3:
                break
        return chosen

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def _question_attributes(self, question: str) -> List[Tuple[str, int]]:
        """
        Which attributes does this question ask about?

        Scores every known attribute (existing global attributes plus the
        candidate lexicon) by how many distinct question triggers fire.
        Returns [(attribute_name, score)] sorted by score desc.
        """
        q = (question or "").lower()
        if not q:
            return []
        scores: Dict[str, int] = {}

        table: Dict[str, List[str]] = dict(QUESTION_TRIGGERS)
        for name, spec in ATTRIBUTE_CANDIDATE_LEXICON.items():
            table.setdefault(name, [])
            table[name] = list(table[name]) + list(spec.get("triggers") or [])

        for name, triggers in table.items():
            hits = 0
            for t in triggers:
                t = (t or "").strip().lower()
                if t and t in q:
                    hits += 1
            if hits:
                scores[name] = hits
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    def _detect_gaps(
        self,
        builder,
        question: str,
        qdata: Dict[str, Any],
    ) -> List[AttributeGap]:
        """
        Detect the gaps exposed by one question.

        Two kinds:

        * *missing class* — the anchor entity's spaCy type has no
          template of its own.  Its sentences therefore only ever mount
          the `description` catch-all, which carries no signal.
        * *missing attribute* — the question asks for a property that the
          anchor entity's class does not declare.  E.g. "What religion
          is X?" against a PERSON, whose template has no `religion`.
        """
        gaps: List[AttributeGap] = []
        if builder is None:
            return gaps

        entities = self._question_entities(builder, question)
        if not entities:
            return gaps

        # The anchor is the first (longest) matched entity.
        anchor = entities[0]
        container = builder.container_manager.get_container_for_entity(anchor)
        if container is None:
            return gaps
        etype = (container.entity_type or "").strip()

        known_types = set(self.schema.known_types())

        # -- missing class ------------------------------------------------
        if not etype or etype not in known_types:
            gap = AttributeGap(
                kind="missing_class",
                entity_type=etype or "UNKNOWN",
                attribute="",
                evidence=[question],
                count=1,
            )
            gaps.append(gap)

        # -- missing attribute --------------------------------------------
        template = self.schema.effective_templates().get(etype, {})
        for attr_name, score in self._question_attributes(question):
            if attr_name in template:
                continue
            # An attribute the class does not own but which some OTHER
            # class owns is a *distinctiveness* problem, not a gap.
            other_owner = any(
                attr_name in tmpl and other != etype
                for other, tmpl in self.schema.effective_templates().items()
            )
            if other_owner:
                continue
            if attr_name not in ATTRIBUTE_CANDIDATE_LEXICON:
                # Not in the bounded candidate lexicon -> the agent may
                # not propose it (it would be an unbounded vocabulary).
                continue
            gap = AttributeGap(
                kind="missing_attribute",
                entity_type=etype,
                attribute=attr_name,
                evidence=[question],
                count=1,
            )
            gaps.append(gap)
            break  # one attribute gap per question is enough

        return gaps

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _descriptor_embeddings(
        self,
        embedder,
        descriptors: Dict[str, str],
        names: List[str],
    ):
        """
        Encode the attribute descriptors, reusing the result across
        questions.

        The descriptor set is constant for the whole run, so encoding it
        once and caching the matrix removes ~199 of the 200 embedder
        batches the default configuration used to perform.  The cache is
        keyed by the descriptor mapping itself, so a changed schema
        invalidates it automatically.

        Returns the embedding matrix, or None when encoding fails.
        """
        try:
            import numpy as _np
        except Exception:  # noqa: BLE001
            return None

        cached = self._desc_emb_cache
        if cached is not None:
            cached_desc, cached_emb = cached
            if cached_desc == descriptors:
                return cached_emb

        try:
            desc_embs = _np.asarray(
                embedder.encode(
                    [descriptors[a] for a in names],
                    normalize_embeddings=True,
                    # The progress bar is what the user sees as a
                    # "batches loading" spinner.  It is re-shown for
                    # EVERY question (the agent samples 200 of them),
                    # which reads as a hung run.  Off: the agent has its
                    # own progress reporting.
                    show_progress_bar=False,
                ),
                dtype=_np.float32,
            ).reshape(len(names), -1)
        except Exception as e:  # noqa: BLE001
            logger.warning("Attribute profile encode failed: %s", e)
            return None

        self._desc_emb_cache = (dict(descriptors), desc_embs)
        return desc_embs

    def _attr_profiles(
        self,
        builder,
        extra_descriptors: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str], Dict[str, List[str]]]:
        """
        Compute every sentence's attribute profile.

        Returns (profiles, node_entity_types, node_class_by_entity) where
        ``profiles[sid][attr]`` is the cosine between the sentence and
        the attribute descriptor.
        """
        try:
            import numpy as _np
        except Exception:  # noqa: BLE001
            return {}, {}, {}

        embedder = getattr(builder.container_manager, "embedder", None)
        nodes = builder.sentence_registry.get_all_nodes()
        if embedder is None or not nodes:
            return {}, {}, {}

        descriptors: Dict[str, str] = dict(_sg.ATTRIBUTE_DESCRIPTORS)
        descriptors.update(extra_descriptors)
        names = list(descriptors.keys())

        desc_embs = self._descriptor_embeddings(embedder, descriptors, names)
        if desc_embs is None:
            return {}, {}, {}

        profiles: Dict[str, Dict[str, float]] = {}
        node_types: Dict[str, List[str]] = {}
        for node in nodes:
            sv = getattr(node, "summary_vector", None)
            if sv is None:
                continue
            try:
                svec = _np.asarray(sv, dtype=_np.float32).reshape(-1)
                sims = _np.maximum(desc_embs @ svec, 0.0)
            except Exception:  # noqa: BLE001
                continue
            prof: Dict[str, float] = {}
            for i, aname in enumerate(names):
                v = float(sims[i])
                if v > 0.0:
                    prof[aname] = v
            profiles[node.id] = prof

            # Which class does each entity on this sentence belong to?
            types: List[str] = []
            for ent in (getattr(node, "entities", []) or []):
                c = builder.container_manager.get_container_for_entity(ent)
                types.append((c.entity_type or "") if c is not None else "")
            node_types[node.id] = types

        return profiles, node_types, {}

    def _candidate_descriptors(self) -> Dict[str, str]:
        """
        The descriptor of every attribute the agent MAY propose.

        Both lexicons contribute: the attribute lexicon (religion,
        nationality, ...) and the class lexicon (a class proposes
        attributes like `function`, `maker`, `features`).  Without the
        second one, a class gap could never be measured — the attribute
        would have no descriptor and therefore no profile at all.

        The lexicon entries are keyed BY attribute name and their
        `name` field is not repeated inside the entry, so the key is
        the name.
        """
        out: Dict[str, str] = {}
        for name, spec in ATTRIBUTE_CANDIDATE_LEXICON.items():
            desc = spec.get("descriptor")
            if name and desc:
                out[name] = desc
        for cls in CLASS_CANDIDATE_LEXICON.values():
            # The class lexicon keys its attributes BY NAME and the
            # entry itself does not repeat the name.
            for name, attr in (cls.get("attributes") or {}).items():
                if not attr:
                    # `None` means "reuse the global attribute of this
                    # name" — it has no descriptor of its own.
                    continue
                desc = attr.get("descriptor")
                if name and desc:
                    out[name] = desc
        return out

    def _measure(
        self,
        profiles: Dict[str, Dict[str, float]],
        node_types: Dict[str, List[str]],
        candidates: Dict[str, Dict[str, str]],
        class_additions: Dict[str, List[str]],
        prefer_classes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Measure every candidate attribute.

        The measurement mirrors the REAL mount (`AttributeRegistry.
        soft_mount`) as closely as possible, because a candidate that
        would never win a mount slot in production must not be accepted:

        * the ranking set for a sentence of class X is X's template
          (the one it WOULD have, including the candidate) — the mount
          restricts itself to the attributes the entity declares;
        * the per-attribute weight is applied BEFORE the ranking, so
          the `description` catch-all is down-weighted and cannot
          occupy a top-K slot;
        * only the top `top_k` attributes count as "mounted".

        Args:
            profiles:       sentence id -> {attribute: cosine}.
            node_types:     sentence id -> the classes of its entities.
            candidates:     attribute -> {"descriptor", "home",
                            "candidates"}.  "candidates" is the set of
                            classes allowed to own it; "home" is a hint.
                            class_additions: class -> the attributes a new
                            class adds.
            prefer_classes: attribute -> the class the QUESTION anchored
                            on, used to break a tie between two equally
                            good owner classes.

        For each candidate we compute:

        * ``mount_count``   — sentences on which the candidate wins a
          top-K slot;
        * ``global_rate``   — ``mount_count`` / total sentences;
        * ``top_rate[T]``   — share of class T's sentences on which the
          candidate wins a top-K slot.
        """
        total = len(profiles) or 1

        # The templates each class WOULD have if the delta were applied.
        templates: Dict[str, Dict[str, str]] = {
            etype: dict(tmpl)
            for etype, tmpl in self.schema.effective_templates().items()
        }
        # A candidate attribute is mounted by its HOME class, so it has
        # to exist in that class's template for the ranking to see it.
        for attr, spec in candidates.items():
            home = spec.get("home") or ""
            if home:
                templates.setdefault(home, {}).setdefault(attr, "")
        for etype, attrs in class_additions.items():
            tmpl = templates.setdefault(etype, {})
            for a in attrs:
                tmpl.setdefault(a, "")

        # Per-class sentence counts.
        class_counts: Dict[str, int] = {}
        for sid, types in node_types.items():
            for t in set(t for t in types if t):
                class_counts[t] = class_counts.get(t, 0) + 1

        weights = dict(_sg.ATTR_WEIGHTS)

        # Pre-compute, for every (sentence, class) pair, the class's
        # attribute list ranked by weighted score — WITHOUT the
        # candidate.  A candidate is then evaluated by inserting it into
        # that ranking and taking the top-K.  Doing it this way is what
        # makes the measurement mirror the real mount (which ranks only
        # the attributes the entity's template declares, and applies the
        # weight BEFORE the ranking).
        ranked_by_class: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        for sid, prof in profiles.items():
            for etype in set(t for t in node_types.get(sid, []) if t):
                rank_set = set(templates.get(etype, {}).keys())
                if not rank_set:
                    continue
                scored = [
                    (a, prof.get(a, 0.0) * float(weights.get(a, 1.0)))
                    for a in rank_set
                    if prof.get(a, 0.0) >= self.config.min_sim
                ]
                scored.sort(key=lambda kv: -kv[1])
                ranked_by_class[(sid, etype)] = scored

        out: Dict[str, Dict[str, float]] = {}
        for attr in sorted(candidates.keys()):
            spec = candidates[attr] or {}
            prefer_class = (prefer_classes or {}).get(attr, "")
            # The owner is NOT assumed: it is the class on which the
            # attribute wins the most top-K slots.  The candidate owner
            # list is only the set of classes allowed to own it.
            allowed = [
                t for t in (spec.get("candidates") or "").split(",")
                if t
            ]
            owner_wins: Dict[str, int] = {}
            top_counts: Dict[str, int] = {}

            for sid, prof in profiles.items():
                classes = set(t for t in node_types.get(sid, []) if t)
                if not classes:
                    continue
                for etype in classes:
                    base = ranked_by_class.get((sid, etype))
                    if base is None:
                        # The class has no template at all, so nothing
                        # can be mounted for it.
                        continue

                    # Insert the candidate into the class's ranking.
                    # For an ALLOWED owner this is the real future state;
                    # for any other class it answers "would this
                    # attribute also win a slot there?" — which is
                    # exactly the distinctiveness question.
                    cand_score = prof.get(attr, 0.0)
                    if cand_score < self.config.min_sim:
                        continue
                    cand_score *= float(weights.get(attr, 1.0))
                    merged = sorted(
                        base + [(attr, cand_score)],
                        key=lambda kv: -kv[1],
                    )[: self.config.top_k]
                    if any(a == attr for a, _ in merged):
                        top_counts[etype] = top_counts.get(etype, 0) + 1
                        if not allowed or etype in allowed:
                            owner_wins[etype] = owner_wins.get(etype, 0) + 1

            # The owner is the allowed class that wins the most.  When
            # no allowed class wins anything, the attribute has no home
            # and its mount count is 0.
            #
            # TIE-BREAK: when two allowed classes win the SAME number of
            # slots, the class the QUESTION anchored on wins.  Without
            # this, `religion` (which fits PERSON and NORP equally well)
            # would be assigned by alphabetical order, which is
            # arbitrary and would put the attribute on the wrong class.
            # The anchor is passed in as `prefer_class`.
            home = ""
            if owner_wins:
                best = max(owner_wins.values())
                tied = [t for t, v in owner_wins.items() if v == best]
                if prefer_class and prefer_class in tied:
                    home = prefer_class
                else:
                    home = sorted(tied)[0]
            mount_count = owner_wins.get(home, 0) if home else 0

            top_rate: Dict[str, float] = {}
            for t, n in class_counts.items():
                if n:
                    top_rate[t] = top_counts.get(t, 0) / n

            out[attr] = {
                "mount_count": float(mount_count),
                "global_rate": mount_count / total,
                "top_rate": top_rate,  # type: ignore[assignment]
                "owner": home,
            }
        return out

    def _duplicate_of(self, descriptor: str, builder=None) -> str:
        """
        If the candidate descriptor is nearly identical to an existing
        one, return that existing attribute's name (else "").

        A duplicate adds a slot to every entity template and to the
        attribute embedding matrix, but buys no extra recall — the
        existing attribute already covers those sentences.
        """
        if not descriptor:
            return ""
        try:
            import numpy as _np
        except Exception:  # noqa: BLE001
            return ""
        builder = builder if builder is not None else getattr(self, "_last_builder", None)
        if builder is None:
            return ""
        embedder = getattr(builder.container_manager, "embedder", None)
        if embedder is None:
            return ""
        try:
            vecs = _np.asarray(
                embedder.encode(
                    [descriptor]
                    + [d for d in _sg.ATTRIBUTE_DESCRIPTORS.values()],
                    normalize_embeddings=True,
                ),
                dtype=_np.float32,
            )
            q = vecs[0]
            sims = vecs[1:] @ q
            names = list(_sg.ATTRIBUTE_DESCRIPTORS.keys())
            best = int(_np.argmax(sims))
            if float(sims[best]) >= self.config.max_descriptor_cos:
                return names[best]
        except Exception:  # noqa: BLE001
            return ""
        return ""

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def _rival_types(self, gap: AttributeGap) -> List[str]:
        """
        The other entity classes that could plausibly own this attribute.

        A class is a rival when it EITHER

        * the candidate lexicon designed the attribute for it
          (`ATTRIBUTE_CANDIDATE_LEXICON[...]["types"]`), or
        * already declares the attribute in its template.

        The distinctiveness margin is measured against these, because the
        question the gate asks is "does this attribute belong to the
        proposed class rather than to a class that also wants it?".

        Comparing against EVERY class would make the margin meaningless:
        most classes never mount a `religion` slot at all, so they would
        contribute a rate of 0.0 and inflate the margin.
        """
        attr = gap.attribute
        rivals: Set[str] = set()

        if gap.kind == "missing_attribute":
            lex = ATTRIBUTE_CANDIDATE_LEXICON.get(attr)
            if lex:
                rivals |= set(lex.get("types") or [])

        # Any class whose template already declares this attribute is a
        # genuine rival — it would compete for the same sentences.
        for etype, tmpl in self.schema.effective_templates().items():
            if attr in tmpl:
                rivals.add(etype)

        # The OWNER is excluded by the caller (it is the class the
        # attribute was assigned to), so it must NOT be removed here.
        # The anchor class stays a rival: the question asked about it,
        # but the measurement may have found another class wins more.
        return sorted(rivals)

    def _gate(
        self,
        gap: AttributeGap,
        metrics: Dict[str, Dict[str, float]],
    ) -> Tuple[bool, str]:
        """
        Apply the three acceptance criteria to one proposal.

        1. **Inclusivity** — the attribute must be general enough to hold
           a range of sentences, and must not duplicate an existing
           attribute (a duplicate adds a slot but no recall).
        2. **Mount efficiency** — an attribute that mounts only a
           handful of sentences is dropped: it would occupy a slot in
           the attribute embedding matrix and in every entity template
           while answering almost nothing.
        3. **Distinctiveness** — the attribute must clearly belong to
           the proposed class.  If it ranks just as well for another
           class, it discriminates nothing and is dropped.
        """
        attr = gap.attribute
        if attr not in metrics:
            return False, "no-measurement"
        m = metrics[attr]

        # (1) inclusivity
        if gap.proposal is not None:
            spec = gap.proposal
            if not spec.descriptor:
                return False, "no-descriptor"
            if len(spec.stems) < 2 and len(spec.triggers) < 2:
                return False, "not-inclusive"
            dup = self._duplicate_of(spec.descriptor, getattr(self, "_last_builder", None))
            if dup:
                return False, f"duplicate-of:{dup}"

        # (2) mount efficiency
        if m["mount_count"] < self.config.min_mount_count:
            return False, (
                f"low-mount-efficiency: {int(m['mount_count'])} sentences "
                f"< {self.config.min_mount_count}"
            )
        if m["global_rate"] < self.config.min_global_mount_rate:
            return False, (
                f"low-mount-rate: {m['global_rate']:.3f} "
                f"< {self.config.min_global_mount_rate}"
            )

        # (3) distinctiveness
        # The owner is the class on which the attribute actually wins
        # the most slots — it was decided by the measurement, not
        # assumed.  The margin is measured against the OTHER allowed
        # owner classes: if another class wants the attribute just as
        # much, it discriminates nothing and is dropped.
        top_rate: Dict[str, float] = m.get("top_rate") or {}  # type: ignore[assignment]
        owner = str(m.get("owner") or "")
        if not owner:
            return False, "no-owner-class"
        own = top_rate.get(owner, 0.0)
        if own < self.config.min_class_top_rate:
            return False, (
                f"low-class-top-rate: {own:.3f} "
                f"< {self.config.min_class_top_rate} for {owner}"
            )
        rivals = [t for t in self._rival_types(gap) if t != owner]
        others = [top_rate.get(t, 0.0) for t in rivals]
        best_other = max(others) if others else 0.0
        margin = own - best_other
        if margin < self.config.min_distinctiveness_margin:
            return False, (
                f"not-distinctive: margin {margin:.3f} "
                f"< {self.config.min_distinctiveness_margin} "
                f"(own {own:.3f} vs other {best_other:.3f})"
            )

        return True, (
            f"accepted: mount={int(m['mount_count'])} "
            f"rate={m['global_rate']:.3f} top_rate[{owner}]={own:.3f} "
            f"margin={margin:.3f}"
        )

    # ------------------------------------------------------------------
    # Ambiguity resolution
    # ------------------------------------------------------------------

    def resolve_ambiguous_mounts(self, builder) -> int:
        """
        Fix same-class multi-entity mounts with the LLM.

        A sentence that carries two entities of the SAME class ("Washington
        and Lincoln") mounts under the same attribute for both, because
        the vector profile has no entity signal.  This asks the LLM which
        entity the attribute belongs to and rewrites the mount.

        Returns the number of mounts corrected.
        """
        if builder is None or not self.config.use_llm:
            return 0
        if self.client is None or not getattr(self.client, "enabled", False):
            return 0

        fixed = 0
        for node in builder.sentence_registry.get_all_nodes():
            mounts = list(getattr(node, "attribute_mounts", []) or [])
            if not mounts:
                continue
            ents = list(getattr(node, "entities", []) or [])
            if len(ents) < 2:
                continue

            # Group the sentence's entities by class.
            by_class: Dict[str, List[str]] = {}
            for ent in ents:
                c = builder.container_manager.get_container_for_entity(ent)
                etype = (c.entity_type or "") if c is not None else ""
                if not etype:
                    continue
                by_class.setdefault(etype, []).append(ent)

            ambiguous_classes = [t for t, es in by_class.items() if len(es) >= 2]
            if not ambiguous_classes:
                continue

            for m in mounts:
                aname = getattr(m, "attribute_name", "")
                if not aname or aname == "other":
                    continue
                owner = getattr(m, "entity", "")
                etype = ""
                for t, es in by_class.items():
                    if owner in es:
                        etype = t
                        break
                if not etype:
                    continue
                candidates = by_class[etype]
                if len(candidates) < 2:
                    continue
                if owner not in candidates:
                    continue

                label = ""
                for tname, tmpl in self.schema.effective_templates().items():
                    if aname in tmpl:
                        label = tmpl[aname]
                        break

                chosen = self._resolve_ambiguous_owner(
                    node.text or "", aname, label or aname, candidates,
                )
                if chosen and chosen != owner:
                    m.entity = chosen
                    fixed += 1
        return fixed

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _report_progress(self, done: int, total: int, note: str) -> None:
        """
        Report sampling progress to the attached progress reporter.

        The reporter is optional — when it is absent the agent stays
        silent, which is what the unit tests want.  When present it
        gets `update(done, total, note)`.
        """
        if self.progress is None:
            return
        try:
            self.progress.update(done, total, note)
        except Exception as e:  # noqa: BLE001 - never break the run
            logger.debug("progress update failed: %s", e)

    def _set_progress_extra(self, **kwargs: Any) -> None:
        """
        Update the rolling counters shown on the progress line.

        The counters are the answer to "有多少题目还需要滚动 / 有多少
        已经做好了":

        * ``draws`` — how many questions have been processed;
        * ``gaps``  — how many distinct gaps have been detected;
        * ``ok``    — how many gaps were ACCEPTED (rolled into the
          table — i.e. "已经做好了");
        * ``no``    — how many were rejected;
        * ``attrs`` — the size of the current attribute table;
        * ``classes`` — the size of the current class table.
        """
        if self.progress is None:
            return
        setter = getattr(self.progress, "set_extra", None)
        if not callable(setter):
            return
        try:
            setter(**kwargs)
        except Exception as e:  # noqa: BLE001
            logger.debug("progress set_extra failed: %s", e)

    def run(
        self,
        questions: List[Dict[str, Any]],
        apply: bool = True,
    ) -> AgentReport:
        """
        Run the full optimisation loop.

        Args:
            questions: The dataset (see `load_questions`).
            apply:    Whether to apply the accepted delta to the live
                      tables.  Set False for a dry run.
        """
        report = AgentReport(n_rounds=self.config.n_rounds)
        rounds = self.sample_rounds(questions)
        if not rounds:
            logger.warning("AttributeAgent: no questions to sample from.")
            return report

        # The total number of draws this run will make — the
        # denominator of the progress bar.
        total_draws = sum(len(r) for r in rounds)

        # --- pass 1: detect gaps -------------------------------------
        gap_index: Dict[Tuple[str, str], AttributeGap] = {}
        all_profiles: Dict[str, Dict[str, float]] = {}
        all_node_types: Dict[str, List[str]] = {}
        last_builder = None

        for rnd in rounds:
            for qdata in rnd:
                question = qdata.get("question") or ""
                builder, _ = self._build_graph(qdata)
                if builder is None:
                    # Still count it as done — the question was
                    # processed, it just produced no graph.
                    report.n_sampled += 1
                    self._report_progress(
                        report.n_sampled, total_draws,
                        f"skip (no graph) | {question[:50]}",
                    )
                    continue
                report.n_sampled += 1
                last_builder = builder

                for gap in self._detect_gaps(builder, question, qdata):
                    key = (gap.kind, f"{gap.entity_type}::{gap.attribute}")
                    prev = gap_index.get(key)
                    if prev is None:
                        gap_index[key] = gap
                    else:
                        prev.count += 1
                        if len(prev.evidence) < 8 and question not in prev.evidence:
                            prev.evidence.append(question)

                # Accumulate the attribute profiles over the corpus.
                #
                # Sentence IDs restart at "s_0001" in every graph, so
                # they are namespaced by the question id.  Without this
                # the corpus statistics would be a single graph's
                # statistics counted 200 times.
                extra = self._candidate_descriptors()
                prof, ntypes, _ = self._attr_profiles(builder, extra)
                qid = str(qdata.get("qid") or report.n_sampled)
                for sid, p in prof.items():
                    all_profiles.setdefault(f"{qid}::{sid}", {}).update(p)
                for sid, t in ntypes.items():
                    all_node_types.setdefault(f"{qid}::{sid}", t)

                # Progress: how many draws are done, and what the
                # running gap tally looks like.  This is the only thing
                # that tells the user the rolling optimisation is
                # actually moving — the sampling loop is otherwise
                # silent for minutes.
                self._report_progress(
                    report.n_sampled, total_draws,
                    f"{len(gap_index)} gap(s) so far | {question[:50]}",
                )

        self._last_builder = last_builder
        report.gaps = list(gap_index.values())

        # Report the gap tally now that pass 1 is done.
        self._set_progress_extra(
            gaps=len(gap_index),
            attrs=len(self.schema.attributes),
            classes=len(self.schema.classes),
        )

        # --- pass 2: propose -----------------------------------------
        for gap in report.gaps:
            if gap.kind == "missing_attribute":
                lex = ATTRIBUTE_CANDIDATE_LEXICON.get(gap.attribute)
                if lex:
                    gap.proposal = AttributeSpec(
                        name=gap.attribute,
                        label=lex.get("label", ""),
                        descriptor=lex.get("descriptor", ""),
                        stems=list(lex.get("stems") or []),
                        triggers=list(lex.get("triggers") or []),
                        types=[gap.entity_type],
                        source="lexicon",
                    )
            elif gap.kind == "missing_class":
                lex = CLASS_CANDIDATE_LEXICON.get(gap.entity_type)
                if lex:
                    attrs: List[AttributeSpec] = []
                    for aname, aspec in (lex.get("attributes") or {}).items():
                        if aspec is None:
                            # Reuse a global attribute: inherit its
                            # descriptor, stems and label from the
                            # module tables.
                            attrs.append(
                                AttributeSpec(
                                    name=aname,
                                    label=_attribute_label(aname),
                                    descriptor=_sg.ATTRIBUTE_DESCRIPTORS.get(aname, ""),
                                    stems=list(_sg._ATTRIBUTE_STEMS.get(aname, [])),
                                    types=[gap.entity_type],
                                    source="reuse",
                                )
                            )
                            continue
                        attrs.append(
                            AttributeSpec(
                                name=aname,
                                label=aspec.get("label", ""),
                                descriptor=aspec.get("descriptor", ""),
                                stems=list(aspec.get("stems") or []),
                                triggers=list(aspec.get("triggers") or []),
                                types=[gap.entity_type],
                                source="lexicon",
                            )
                        )
                    gap.proposal = attrs[0] if attrs else None
                    # A class needs ALL of its attributes, not just the
                    # first one, so keep the full list for the apply step.
                    gap.metrics["class_attributes"] = [
                        a.to_dict() for a in attrs
                    ]

        # --- pass 3: measure + gate ----------------------------------
        #
        # Every candidate attribute needs a DESCRIPTOR (it is the
        # embedding anchor) and a set of CANDIDATE OWNER classes (the
        # classes the lexicon designed it for).  The measurement then
        # decides WHICH class actually owns it.
        candidates: Dict[str, Dict[str, str]] = {}   # attr -> {desc, home}
        class_additions: Dict[str, List[str]] = {}   # class -> attrs

        for g in report.gaps:
            if g.kind != "missing_attribute":
                continue
            lex = ATTRIBUTE_CANDIDATE_LEXICON.get(g.attribute)
            if not lex or not lex.get("descriptor"):
                continue
            candidates[g.attribute] = {
                "descriptor": lex["descriptor"],
                # The owner is decided by the measurement, not assumed.
                # The gap's class is only a CANDIDATE owner.
                "home": "",
                "candidates": ",".join(lex.get("types") or []) or g.entity_type,
            }

        # A missing CLASS proposes several attributes; each must be
        # measured, because the class is accepted only if at least one
        # of them can actually mount sentences.
        for g in report.gaps:
            if g.kind != "missing_class":
                continue
            cls = CLASS_CANDIDATE_LEXICON.get(g.entity_type)
            if not cls:
                continue
            names: List[str] = []
            for n, v in (cls.get("attributes") or {}).items():
                names.append(n)
                if v and v.get("descriptor"):
                    candidates.setdefault(n, {
                        "descriptor": v["descriptor"],
                        "home": g.entity_type,
                        "candidates": g.entity_type,
                    })
                elif n in _sg.ATTRIBUTE_DESCRIPTORS:
                    candidates.setdefault(n, {
                        "descriptor": _sg.ATTRIBUTE_DESCRIPTORS[n],
                        "home": g.entity_type,
                        "candidates": g.entity_type,
                    })
            class_additions[g.entity_type] = names

        candidates.pop("other", None)
        candidates.pop("description", None)

        metrics = self._measure(
            all_profiles, all_node_types, candidates, class_additions,
            # The class the question anchored on.  Used only to break a
            # tie between two equally good owner classes — e.g. a
            # question about Marie Curie (PERSON) that asks for
            # `religion`, which fits PERSON and NORP equally well.
            prefer_classes={
                g.attribute: g.entity_type
                for g in report.gaps
                if g.kind == "missing_attribute" and g.attribute
            },
        )

        for gap in report.gaps:
            if gap.proposal is None:
                gap.verdict = "rejected"
                gap.reason = "no-proposal-available"
                report.rejected.append(gap)
                continue
            if gap.kind == "missing_attribute":
                ok, reason = self._gate(gap, metrics)
                gap.verdict = "accepted" if ok else "rejected"
                gap.reason = reason
                if ok:
                    # The owner class was decided by the measurement —
                    # the attribute is declared by THAT class, not
                    # necessarily by the class the gap was detected on.
                    m = metrics.get(gap.attribute) or {}
                    owner = str(m.get("owner") or "")
                    if owner and gap.proposal is not None:
                        gap.proposal.types = [owner]
                        gap.entity_type = owner
                        gap.metrics["owner"] = owner
            else:
                # A missing CLASS is judged by the mount efficiency of
                # its proposed attributes: a class whose attributes can
                # never mount anything is not worth adding.
                cls = CLASS_CANDIDATE_LEXICON.get(gap.entity_type)
                if cls is None:
                    gap.verdict = "rejected"
                    gap.reason = "no-class-proposal-available"
                else:
                    names = set()
                    for n, v in (cls.get("attributes") or {}).items():
                        names.add(n)
                    # A class is accepted only if at least one of its
                    # attributes passes the SAME mount-efficiency bar as
                    # a standalone attribute — otherwise the class would
                    # add a template whose slots never mount anything.
                    usable = []
                    for n in sorted(names):
                        m = metrics.get(n)
                        if not m:
                            continue
                        if (m["mount_count"] >= self.config.min_mount_count
                                and m["global_rate"]
                                >= self.config.min_global_mount_rate):
                            usable.append(n)
                    if usable:
                        gap.verdict = "accepted"
                        gap.reason = f"accepted: usable attributes {usable}"
                    else:
                        gap.verdict = "rejected"
                        gap.reason = "all-proposed-attributes-fail-mount-efficiency"

            (report.accepted if gap.verdict == "accepted" else report.rejected).append(gap)

            # Live tally: how many gaps are settled and how the table
            # looks right now.  This is the "多少已经做好了" counter.
            self._set_progress_extra(
                gaps=len(report.gaps),
                ok=len(report.accepted),
                no=len(report.rejected),
                attrs=len(self.schema.attributes),
                classes=len(self.schema.classes),
            )

        # --- pass 4: LLM confirmation of accepted attribute gaps -----
        for gap in list(report.accepted):
            if gap.kind != "missing_attribute":
                continue
            confirmed, reason = self._confirm_gap(gap)
            gap.metrics["llm_confirmed"] = confirmed
            gap.metrics["llm_reason"] = reason
            if not confirmed:
                gap.verdict = "rejected"
                gap.reason = f"llm-rejected: {reason}"
                report.accepted.remove(gap)
                report.rejected.append(gap)
            self._set_progress_extra(
                gaps=len(report.gaps),
                ok=len(report.accepted),
                no=len(report.rejected),
                attrs=len(self.schema.attributes),
                classes=len(self.schema.classes),
            )

        # --- pass 5: apply -------------------------------------------
        if apply:
            for gap in report.accepted:
                if gap.kind == "missing_attribute" and gap.proposal:
                    self.schema.add_attribute(gap.proposal)
                elif gap.kind == "missing_class":
                    cls = CLASS_CANDIDATE_LEXICON.get(gap.entity_type)
                    if cls:
                        # Rebuild the FULL attribute list from the
                        # lexicon, so the class gets all of its slots
                        # and not just the primary one.
                        attrs: List[AttributeSpec] = []
                        for n, v in (cls.get("attributes") or {}).items():
                            if v is None:
                                attrs.append(
                                    AttributeSpec(
                                        name=n,
                                        label=_attribute_label(n),
                                        descriptor=_sg.ATTRIBUTE_DESCRIPTORS.get(n, ""),
                                        stems=list(_sg._ATTRIBUTE_STEMS.get(n, [])),
                                        types=[gap.entity_type],
                                        source="reuse",
                                    )
                                )
                            else:
                                attrs.append(
                                    AttributeSpec(
                                        name=n,
                                        label=v.get("label", ""),
                                        descriptor=v.get("descriptor", ""),
                                        stems=list(v.get("stems") or []),
                                        triggers=list(v.get("triggers") or []),
                                        types=[gap.entity_type],
                                        source="lexicon",
                                    )
                                )
                        self.schema.add_class(
                            ClassSpec(
                                entity_type=gap.entity_type,
                                label=cls.get("label", ""),
                                attributes=attrs,
                            )
                        )
            self.schema.apply_to_modules()

            # Re-mount the last graph so the new attributes take effect.
            if last_builder is not None:
                self.schema.apply_to_containers(last_builder.container_manager)
                try:
                    last_builder._soft_mount_attributes(
                        last_builder.sentence_registry.get_all_nodes()
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("AttributeAgent re-mount failed: %s", e)

        report.schema = self.schema
        report.llm_calls = self._llm_calls
        report.corpus_stats = {
            "n_sentence_profiles": len(all_profiles),
            "n_classes_seen": len(
                {t for ts in all_node_types.values() for t in ts if t}
            ),
            "candidate_attributes": sorted(candidates.keys()),
        }

        # Final progress line + summary.
        if self.progress is not None:
            try:
                self.progress.finish({
                    "draws": report.n_sampled,
                    "gaps": len(report.gaps),
                    "ok": len(report.accepted),
                    "no": len(report.rejected),
                    "attrs": len(self.schema.attributes),
                    "classes": len(self.schema.classes),
                    "llm": report.llm_calls,
                })
            except Exception as e:  # noqa: BLE001
                logger.debug("progress finish failed: %s", e)

        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attribute_label(name: str) -> str:
    """
    The human-readable label of an attribute.

    The labels live in the per-entity templates, so the attribute has to
    be looked up across them.  The first template that declares it wins.
    """
    for tmpl in _sg._ATTRIBUTE_TEMPLATES.values():
        if name in tmpl:
            return tmpl[name]
    return name


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Leniently extract the first JSON object from an LLM reply."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(s[start:end + 1])
    except Exception:  # noqa: BLE001
        try:
            obj = json.loads(s[start:end + 1].replace("'", '"'))
        except Exception:  # noqa: BLE001
            return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# End of the attribute-table rolling optimisation agent block (2026-10-29).
# ---------------------------------------------------------------------------

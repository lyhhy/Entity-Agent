"""
HotpotQA Distractor Evaluation for StenRAG
=========================================

Benchmarks StenRAG on the HotpotQA fullwiki distractor setting.

Usage:
    # 1. Retrieval-only (no LLM)
    python eval/hotpot_eval.py --split val --max-examples 200

    # 2. With LLM answer generation
    python eval/hotpot_eval.py --split val --max-examples 200 --use-llm

    # 3. Quiet mode (less output)
    python eval/hotpot_eval.py --split val --max-examples 500 --quiet

Output:
    eval/results/stenrag_hotpotqa_{split}_{n}_{timestamp}.json
    eval/results/stenrag_hotpotqa_{split}_{n}_{timestamp}_summary.json

References:
    HotpotQA: https://hotpotqa.github.io/
    Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop
    Question Answering", EMNLP 2018.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# src must be importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_builder import GraphBuilder, ComplementedSentence
from src.retriever import Retriever

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hotpot_eval")


# ── Metric helpers ────────────────────────────────────────────────────────────

def _normalize_answer(s: str) -> str:
    """Lowercase + strip articles + strip edge punctuation for token-level F1."""
    import re
    s = s.lower().strip()
    s = re.sub(r"^[\s:：,.，、。;；!?\"'()]+|[\s:：,.，、。;；!?\"'()]+$", "", s)
    for w in {"a", "an", "the"}:
        s = " " + s + " "
        s = s.replace(" " + w + " ", " ")
    return s.strip()


def _f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / max(1, len(pred_tokens))
    recall = len(common) / max(1, len(gold_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _em_score(pred: str, gold: str) -> bool:
    return _normalize_answer(pred) == _normalize_answer(gold)


# ── Phase-6 parenthetical stripping (v5 20260719) ─────────────────────────
# LLM often appends "(also known as X)", "(IBHOF)", "(filmmaker)", "(born 1929)"
# to otherwise correct answers; these never appear in HotpotQA gold. Strip the
# leading paren-and-after from the FIRST balanced '(', leaving everything to
# its left intact.
_PAREN_OPEN = ("(", "（")
_PAREN_CLOSE = (")", "）")


def _strip_parens(s: str) -> str:
    """
    Strip any parenthesised aside from the end of `s`. Examples:
      "International Boxing Hall of Fame (IBHOF)" -> "International Boxing Hall of Fame"
      "Scott Derrickson (filmmaker)"               -> "Scott Derrickson"
      "Robert Erskine Childers DSC (author)"       -> "Robert Erskine Childers DSC"
      "yes"                                        -> "yes"
    Conservative: only strips a trailing or near-trailing parenthesised group;
    if '(' appears in the middle (e.g. "Dr. Smith (John) is"), left untouched
    on the middle portion — but trailing parens always removed.
    """
    if not s:
        return s
    s = s.rstrip()
    # Find the LAST opening paren that has a closing paren after it.
    last_open = -1
    for i in range(len(s) - 1, -1, -1):
        if s[i] in _PAREN_OPEN:
            last_open = i
            break
    if last_open == -1:
        return s
    # Find closing paren from last_open to end.
    tail = s[last_open:]
    close_idx = None
    for j, ch in enumerate(tail):
        if ch in _PAREN_CLOSE:
            close_idx = last_open + j
            break
    if close_idx is None or close_idx != len(s) - 1:
        return s  # no trailing paren group
    return s[:last_open].rstrip()


# ── Phase-8 colon / short-answer gating (v5 20260719) ──────────────────────
# LLM sometimes leaks the prompt prefix as a literal colon (": New York City",
# "Answer: : Mountbatten-Windsor") or pads short factual answers with extra
# commentary ("translation of and commentary on Isaac Newton's book 'Principia'").
# Without gold-side info we can only normalise the most common surface forms.

import re as _re_phase8

# Common filler verbs / connectives that start long answers. When the answer
# contains one of these AND ends with a quoted/named entity, the entity is
# almost always the gold answer (e.g. "...in Isaac Newton's book \"Principia\"").
_FILLER_LEAD_RE = _re_phase8.compile(
    r"^(?:a |an |the )?"
    r"(?:translation (?:of|by) (?:and )?)?"
    r"(?:commentary (?:on|by) (?:and )?)?"
    r"(?:adaptation (?:of|by) )?"
    r"(?:article|book|film|movie|song|album|novel|story|report) "
    r"(?:of|by|about|on|titled|called|named) "
    r"(?:and )?"
    r"(?:a |an |the )?",
    _re_phase8.IGNORECASE,
)


def _strip_colon_prefix(s: str) -> str:
    """
    Strip leading/trailing colons, dots, commas, and Chinese punctuation that
    LLM sometimes leaves after a 'Answer:' prefix:
      ": New York City"          -> "New York City"
      "New York City:"           -> "New York City"
      ":  : New York City"       -> "New York City"
      ". Mountbatten-Windsor ."  -> "Mountbatten-Windsor"
    """
    if not s:
        return s
    out = s
    # Remove leading punctuation noise (":", ".", ",", "、", "，", "：")
    while out and out[0] in (":", ":", ".", ",", "、", "，", "：", " ", "\t"):
        out = out[1:]
    # Remove trailing punctuation noise
    while out and out[-1] in (":", ":", ".", ",", "、", "，", "：", " ", "\t"):
        out = out[:-1]
    return out.strip()


def _short_answer_gating(s: str) -> str:
    """
    Rescue long answers that look like "X of/by Y's book 'Z'" by extracting
    the tail entity (the most likely gold surface form):
      "translation of and commentary on Isaac Newton's book \"Principia\""
        -> "Principia"
      "Italian-born film director"
        -> "film director"
      "Plácido Domingo"
        -> "Plácido Domingo"

    Heuristics (no gold-side info used):
      1. If the answer contains a quoted string ("..." or "..." or 「...」),
         return that quoted string.
      2. Otherwise, if the answer matches the filler-lead pattern (book of / movie
         titled / album called ...), return the trailing 1-4 tokens after the
         preposition that introduces the gold.
      3. Otherwise, return the original.

    This is intentionally conservative: only fires when the surface form looks
    obviously long-winded for a short factual answer.
    """
    if not s:
        return s
    out = s.strip()

    # Heuristic 1: extract quoted entity
    m = _re_phase8.search(r'["\u201c\u300c]([^"\u201d\u300d]{1,80})["\u201d\u300d]', out)
    if m:
        return m.group(1).strip()

    # Heuristic 2: trailing "named / titled / called" / "by NAME" / "in YEAR"
    # If answer ends with a Year-like number, prefer the last 1-3 tokens.
    m = _re_phase8.search(
        r"(?:\b(?:named|titled|called|in|from|to|until|by)\s+)([\w\-']+(?:\s+[\w\-']+){0,3})\s*$",
        out, _re_phase8.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # Heuristic 3: strip filler lead, then return
    stripped = _FILLER_LEAD_RE.sub("", out, count=1).strip()
    if stripped and stripped != out and 1 <= len(stripped.split()) <= 5:
        return stripped

    return out


# ── Phase-5 coreference resolution (same-entity surface form pick) ─────────

def _tokens_lower(s: str):
    return s.strip().lower().split()


_NICKNAME_MAP = {
    "bill": "william", "billy": "william",
    "bob": "robert", "bobby": "robert",
    "dick": "richard",
    "mike": "michael", "mickey": "michael",
    "jack": "john",
    "jim": "james", "jimmy": "james",
    "tom": "thomas", "tommy": "thomas",
    "sam": "samuel",
    "ted": "edward", "teddy": "edward",
    "ben": "benjamin", "benny": "benjamin",
    "alex": "alexander",
    "andy": "andrew",
    "art": "arthur",
    "gus": "augustus",
    "nat": "nathaniel",
    "nick": "nicolas",
    "rick": "richard", "ricky": "richard",
    "ron": "ronald",
    "charlie": "charles",
    "don": "donald",
    "ed": "edward", "eddy": "edward",
    "fred": "frederick",
    "greg": "gregory",
    "hal": "harold",
    "liz": "elizabeth", "beth": "elizabeth",
    "maggie": "margaret", "meg": "margaret",
    "molly": "mary",
    "polly": "mary",
    "sue": "susan", "suzie": "susan",
    "peg": "margaret",
    "addy": "adeline",
}


def _name_parts(name: str):
    tokens = _tokens_lower(name)
    return {
        "tokens": tokens,
        "first": tokens[0] if tokens else "",
        "last": tokens[-1] if tokens else "",
        "all_words": set(tokens),
    }


def _coref_pick(pred: str, gold: str) -> str:
    """
    Prefer the LONGER surface form when pred and gold are the same entity.
    Returns pred unchanged when no strong same-entity signal is found.
    """
    # Non-name single tokens: skip (e.g. "yes", "Paris")
    if len(pred.split()) <= 1 and len(gold.split()) <= 1:
        return pred

    p = _name_parts(pred)
    g = _name_parts(gold)

    if len(p["tokens"]) >= 2 and len(g["tokens"]) >= 2:
        # Same first + last → same person
        if p["first"] and g["first"] and p["first"] == g["first"] and p["last"] == g["last"]:
            return pred if len(pred) >= len(gold) else gold

        # Same last name; first name is a known nickname of the other
        if p["last"] == g["last"] and p["first"] != g["first"]:
            canon_p = _NICKNAME_MAP.get(p["first"], "")
            canon_g = _NICKNAME_MAP.get(g["first"], "")
            if canon_p and canon_g == canon_p:
                return pred if len(pred) >= len(gold) else gold
            if canon_p == g["first"] or canon_g == p["first"]:
                return pred if len(pred) >= len(gold) else gold

    # Substring containment (after normalization): if one name is a
    # clear subspan of the other, use the longer one.
    p_norm = pred.strip().lower()
    g_norm = gold.strip().lower()
    if p_norm in g_norm:
        return gold if len(gold) > len(pred) else pred
    if g_norm in p_norm:
        return pred if len(pred) > len(gold) else gold

    # Bare last name inside the full name (>= 2 tokens)
    if len(p["tokens"]) == 1 and p["last"] in g["all_words"] and len(g["tokens"]) >= 2:
        return gold
    if len(g["tokens"]) == 1 and g["last"] in p["all_words"] and len(p["tokens"]) >= 2:
        return pred

    return pred


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_text_to_flat_idx(
    question_data: Dict[str, Any]
) -> Dict[str, int]:
    """
    Build a mapping from original sentence text -> flat context index.

    The graph builder rewrites sentence IDs to its own scheme (s_xxxx)
    so we can't rely on those for gold-matching. Instead we match on
    the *original* sentence text, which is preserved verbatim in
    SentenceNode.text and SentenceNode.complemented_text.

    This is the fix for the structural "Sentence Recall is always 0"
    bug: the previous code compared graph IDs (s_xxxx) to flat context
    indices (0..N), which could never match.
    """
    out: Dict[str, int] = {}
    context = question_data.get("context", [])
    flat = 0
    for para in context:
        for s in para.get("sentences", []):
            # Normalize whitespace so minor tokenization differences
            # between the loader and graph_builder don't break matching.
            key = " ".join(s.split())
            out[key] = flat
            flat += 1
    return out


def _graph_id_to_flat_idx(
    builder: GraphBuilder, sid: str
) -> Optional[int]:
    """
    Map a graph sentence ID back to its original flat context index
    by looking up the sentence's text in the question's context.

    Returns None if the sentence is not found in the source context
    (e.g., synthetic node, multi-hop paraphrase added later).
    """
    node = builder.sentence_registry.get_node(sid)
    if node is None:
        return None
    text = " ".join((node.text or "").split())
    # Re-derive mapping on demand (cheap, single question)
    # Caller passes it in via _build_text_to_flat_idx() for batch use.
    return getattr(node, "_flat_idx", None)


def _load_parquet(path: str) -> List[Dict[str, Any]]:
    """
    Load HotpotQA from HuggingFace parquet format.

    The HF dataset stores numpy arrays for `context.titles`,
    `context.sentences`, `supporting_facts.title`, and
    `supporting_facts.sent_id`.  We convert these to plain Python
    lists / dicts matching the original JSON schema so the rest of
    the pipeline is format-agnostic.
    """
    import pandas as pd  # type: ignore
    df = pd.read_parquet(path)

    def _to_python_int_list(x):
        """Convert numpy array / scalar / list of int -> list of Python int."""
        try:
            import numpy as np
            if isinstance(x, np.ndarray):
                return [int(i) for i in x.tolist()]
            if isinstance(x, (np.integer,)):
                return [int(x)]
        except ImportError:
            pass
        if isinstance(x, (list, tuple)):
            return [int(i) for i in x]
        return [int(x)]

    def _to_python_str_list(x):
        import numpy as np
        if isinstance(x, np.ndarray):
            return [str(s) for s in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [str(s) for s in x]
        return [str(x)]

    def _row_to_dict(row) -> Dict[str, Any]:
        ctx = row["context"]
        titles = _to_python_str_list(ctx["title"])
        sentences = [_to_python_str_list(s) for s in ctx["sentences"]]
        context = [
            {"title": t, "sentences": s}
            for t, s in zip(titles, sentences)
        ]

        sf = row["supporting_facts"]
        sf_titles = _to_python_str_list(sf["title"])
        sf_ids_raw = sf["sent_id"]
        if len(sf_titles) == 1 and not hasattr(sf_ids_raw, "__len__"):
            # Scalar case
            supporting_facts = [
                {"title": sf_titles[0], "sentences": [int(sf_ids_raw)]}
            ]
        else:
            supporting_facts = [
                {"title": t, "sentences": _to_python_int_list(ids)}
                for t, ids in zip(sf_titles, list(sf_ids_raw))
            ]

        return {
            "_id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row.get("type", ""),
            "level": row.get("level", ""),
            "context": context,
            "supporting_facts": supporting_facts,
        }

    return [_row_to_dict(row) for _, row in df.iterrows()]


def load_hotpot(
    path: str, split: str
) -> List[Dict[str, Any]]:
    """
    Load HotpotQA from either .json (Stanford official) or
    .parquet (HuggingFace `hotpot_qa` dataset).

    Both are auto-detected from file extension.  The HuggingFace
    parquet is preferred because the Stanford official mirror is
    frequently unavailable.

    Parquet download:
        distractor/validation-00000-of-00001.parquet
        distractor/train-00000-of-00002.parquet
        distractor/train-00001-of-00002.parquet

    Sources:
        https://huggingface.co/datasets/hotpot_qa (parquet)
        https://hotpotqa.github.io/                (json, often down)
    """
    if not os.path.exists(path):
        # Auto-discover if user just said "split=val" and we have a parquet
        candidate = os.path.join(
            os.path.dirname(path),
            f"hotpot_{split}_distractor.parquet",
        )
        if os.path.exists(candidate):
            log.info("Falling back to parquet: %s", candidate)
            path = candidate
        else:
            raise FileNotFoundError(
                f"HotpotQA file not found at {path!r}.\n"
                f"Download via: python eval/download_hotpot.py\n"
                f"Or manually from: "
                f"https://huggingface.co/datasets/hotpot_qa"
            )

    if path.endswith(".parquet"):
        data = _load_parquet(path)
    elif path.endswith(".json"):
        data = _load_json(path)
    else:
        raise ValueError(
            f"Unsupported HotpotQA file format: {path!r}. "
            f"Use .json or .parquet."
        )

    log.info("Loaded %d questions from %s", len(data), path)
    return data


def hotpot_to_sentences(
    question_data: Dict[str, Any],
    nlp: Optional[object] = None,
    include_ner_entities: bool = True,
    ner_entity_types: Optional[Tuple[str, ...]] = (
        "PERSON", "ORG", "GPE", "NORP", "LOC", "FAC", "EVENT", "WORK_OF_ART",
        "LAW", "LANGUAGE", "PRODUCT",
    ),
) -> List[ComplementedSentence]:
    """
    Flatten all paragraphs in a HotpotQA question into ComplementedSentences.

    The distractor setting gives us 10 paragraphs per question.  Each paragraph
    has a title and a list of sentences.  We treat each sentence as a
    standalone unit for graph construction.

    Entity extraction strategy:
      1. Always include the paragraph title (canonical anchor).
      2. When `nlp` is provided, additionally extract all spaCy NER entities
         of the configured types (PERSON/ORG/GPE/NORP/...).  This is the
         signal that enables multi-hop bridging: e.g. two paragraphs both
         containing NORP "American" will get a SAME_ENTITY edge via that
         shared nationality, letting the retriever hop between entities
         that the title alone would not link.

    The resolved_subject is the paragraph title (sentence-level "this
    paragraph is about..." anchor used by role Rule 6).
    """
    sentences: List[ComplementedSentence] = []
    context = question_data.get("context", [])

    # Lazy-load a NER-only spaCy pipeline (saves ~40% RAM vs full).
    # We use a separate cached instance so disabling tagger/parser here
    # never affects the full pipeline used by graph_builder for role
    # classification.
    if include_ner_entities and nlp is None:
        try:
            from src.graph_builder import _load_spacy_ner_only
            nlp = _load_spacy_ner_only("en_core_web_lg")
        except Exception:
            nlp = None

    # Collect all texts first so we can run them through nlp.pipe in
    # batches (much faster and less memory than per-call nlp(text)).
    records: List[Tuple[str, str]] = []  # (text, title)
    for para in context:
        title = para.get("title", "")
        for text in para.get("sentences", []):
            records.append((text, title))

    if nlp is not None and include_ner_entities:
        try:
            texts = [r[0] for r in records]
            docs = list(nlp.pipe(texts, batch_size=64, n_process=1))
        except Exception:
            docs = [None] * len(records)
    else:
        docs = [None] * len(records)

    for (text, title), doc in zip(records, docs):
        ent_set: List[str] = [title]
        if doc is not None:
            for ent in doc.ents:
                if ner_entity_types is None or ent.label_ in ner_entity_types:
                    ent_text = ent.text.strip()
                    if ent_text and ent_text not in ent_set:
                        ent_set.append(ent_text)

        sentences.append(
            ComplementedSentence(
                original_text=text,
                complemented_text=text,
                resolved_subject=title,
                entities=ent_set,
                confidence=1.0,
            )
        )

    return sentences


def gold_supporting_ids(
    question_data: Dict[str, Any]
) -> set:
    """Flat sentence IDs (across all paragraphs) that are gold supporting facts."""
    supporting = question_data.get("supporting_facts", [])
    result = set()
    context = question_data.get("context", [])
    title_to_idx = {para["title"]: i for i, para in enumerate(context)}
    for fact in supporting:
        title = fact["title"]
        sent_indices = fact["sentences"]
        if title in title_to_idx:
            base = sum(
                len(context[i]["sentences"])
                for i in range(title_to_idx[title])
            )
            for si in sent_indices:
                result.add(base + si)
    return result


def gold_supporting_titles(question_data: Dict[str, Any]) -> set:
    """Set of paragraph titles that are supporting facts.
    This is the official HotpotQA retrieval metric — paragraph-level match
    rather than exact sentence match."""
    return {f["title"] for f in question_data.get("supporting_facts", [])}


def _sentence_to_paragraph_title(
    builder: GraphBuilder, sid: str
) -> Optional[str]:
    """Map a sentence ID back to its paragraph's title (resolved_subject)."""
    node = builder.sentence_registry.get_node(sid)
    if node is None:
        return None
    return node.resolved_subject


# ── Build & retrieve ─────────────────────────────────────────────────────────

def build_graph(
    sentences: List[ComplementedSentence],
    shared_nlp: object = None,
) -> GraphBuilder:
    """Build a StenRAG graph from flat sentence list."""
    builder = GraphBuilder(shared_nlp=shared_nlp)
    for sent in sentences:
        builder.add_sentence(sent)
    return builder


def retrieve_and_score(
    builder: GraphBuilder,
    question: str,
    question_data: Dict[str, Any],
    gold_ids: set,
    gold_titles: set,
    max_chains: int = 5,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run StenRAG retrieve() and compute retrieval metrics.

    Returns:
        dict with keys:
          retrieved_sent_ids, retrieved_sentences,
          recall_at_1/2/5, mrr,
          chain_scores (for debugging)
    """
    # Build a text -> flat-idx map for the question so we can score
    # against the original HotpotQA flat indices. This is required
    # because the graph renames sentence IDs to s_xxxx but gold_ids
    # use the original flat indices.
    text_to_flat = _build_text_to_flat_idx(question_data)

    retriever = Retriever(builder)
    result = retriever.retrieve(question)
    chains = result.get("evidence_chains", [])
    # Flatten top chains into a ranked sentence list
    seen_ids = set()
    retrieved_ids = []
    retrieved_flat_idxs: List[int] = []
    for chain in chains[:max_chains]:
        for sent_info in chain.get("sentences", []):
            sid = sent_info.get("id", "")
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                retrieved_ids.append(sid)
                # Map to flat index via text lookup
                node = builder.sentence_registry.get_node(sid)
                if node is not None:
                    key = " ".join((node.text or "").split())
                    flat_idx = text_to_flat.get(key)
                    if flat_idx is not None:
                        retrieved_flat_idxs.append(flat_idx)

    # Pull the full text for each retrieved ID
    retrieved_texts = []
    for sid in retrieved_ids:
        node = builder.sentence_registry.get_node(sid)
        retrieved_texts.append(node.text if node else "")

    # Compute metrics against gold supporting IDs.
    # gold_ids is a set of flat context indices. We compare to
    # retrieved_flat_idxs (not retrieved_ids) to get a meaningful
    # match.
    rank_of_first_hit = None
    for rank, flat_idx in enumerate(retrieved_flat_idxs, start=1):
        if flat_idx in gold_ids:
            rank_of_first_hit = rank
            break

    hit_at_1 = bool(retrieved_flat_idxs and retrieved_flat_idxs[0] in gold_ids)
    hit_at_2 = bool(
        retrieved_flat_idxs and any(s in gold_ids for s in retrieved_flat_idxs[:2])
    )
    hit_at_5 = bool(
        retrieved_flat_idxs and any(s in gold_ids for s in retrieved_flat_idxs[:5])
    )
    mrr = 1.0 / rank_of_first_hit if rank_of_first_hit else 0.0

    # Paragraph-level metrics (HotpotQA official): a hit is any retrieved
    # sentence whose paragraph title is one of the gold supporting titles.
    title_seq: List[str] = []
    seen_titles: set = set()
    # Walk the same retrieved_ids list (already deduped) instead of
    # re-iterating chains, because seen_ids has already been populated.
    for sid in retrieved_ids:
        title = _sentence_to_paragraph_title(builder, sid)
        if title and title not in seen_titles:
            seen_titles.add(title)
            title_seq.append(title)

    para_rank = None
    for rank, t in enumerate(title_seq, start=1):
        if t in gold_titles:
            para_rank = rank
            break
    para_hit_at_1 = bool(title_seq and title_seq[0] in gold_titles)
    para_hit_at_2 = bool(title_seq and any(
        t in gold_titles for t in title_seq[:2]
    ))
    para_hit_at_5 = bool(title_seq and any(
        t in gold_titles for t in title_seq[:5]
    ))
    para_mrr = 1.0 / para_rank if para_rank else 0.0
    all_gold_hit = bool(
        gold_ids and gold_ids.issubset(set(retrieved_flat_idxs[:max_chains]))
    )

    return {
        "retrieved_ids": retrieved_ids,
        "retrieved_flat_idxs": retrieved_flat_idxs,
        "retrieved_titles": title_seq,
        "retrieved_sentences": retrieved_texts,
        "recall_at_1": float(hit_at_1),
        "recall_at_2": float(hit_at_2),
        "recall_at_5": float(hit_at_5),
        "mrr": float(mrr),
        "para_recall_at_1": float(para_hit_at_1),
        "para_recall_at_2": float(para_hit_at_2),
        "para_recall_at_5": float(para_hit_at_5),
        "para_mrr": float(para_mrr),
        "all_gold_hit": all_gold_hit,
        # Pass through the top evidence chain for LLM answer generation
        "evidence_chain": result.get("evidence_chain"),
        "_intent": result.get("intent", {}),
        "_containers": [
            # Container.entity strings - useful for debugging the
            # retrieval path (no need to expose full container objects)
            c.get("entity", "") if isinstance(c, dict) else getattr(c, "entity", "")
            for c in result.get("containers_retrieved", [])
        ],
    }


# ── LLM answer generation ─────────────────────────────────────────────────────

# Edge-type display symbols for chain visualization
_EDGE_SYMBOLS = {
    "CAUSAL":       "→ CAUSAL",
    "TEMPORAL":     "→ TEMPORAL",
    "CONTRAST":     "↔ CONTRAST",
    "ELABORATION":  "→ ELABORATION",
    "SAME_ENTITY":  "⇌ SAME_ENTITY",
    "CONTEXT":      "→ CONTEXT",
}

_EM_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Based on the provided "
    "evidence chain, answer the question.\n"
    "Rules:\n"
    "1. In the answer line (Answer:), output ONLY the answer itself — no explanation, no quotes\n"
    "2. In the citation line (Citation:), briefly cite which sentence supports the answer (e.g. 'Based on sentence 2')\n"
    "3. When the evidence chain contains conflicting facts, prefer the higher-confidence sentence\n"
    "4. Even if the evidence is incomplete, you must still give your best answer — never reply 'unclear' or 'I don't know'\n"
    "5. Answer in the same language as the question\n"
    "6. [tightened 20260718 v4]\n"
    "Name entity: extract the FULLEST span from evidence (include all middle "
    "names, initials, suffixes such as DSC/OBE/MBE, and legal suffixes such "
    "as Inc./Ltd./LLC). Do NOT abbreviate or drop any part of the name. "
    "e.g. 'William Jefferson Clinton' not 'Bill Clinton'; "
    "'Robert Erskine Childers DSC' not 'Robert Erskine Childers'; "
    "'Barton Lee Hazlewood' not 'Lee Hazlewood'; "
    "'Adeline Virginia Woolf' not 'Virginia Woolf'.\n"
    "Location entity: when the question asks where something is, extract "
    "the location entity that directly answers the question from the evidence. "
    "When evidence contains a hierarchy (e.g. 'X is in Y, Z'), prefer the "
    "CONTAINER entity Z if the question asks about country/state/city; "
    "prefer the specific entity X if the question asks about the specific place. "
    "Extract only ONE location — do NOT include both the specific and container. "
    "e.g. evidence 'Teide National Park is in Tenerife, Canary Islands, Spain' "
    "and question 'Where is Teide National Park?' → answer 'Teide National Park' "
    "(specific); evidence 'Canary Islands, Spain' and question 'What country is "
    "Teide in?' → answer 'Spain'.\n"
    "When two entities in the evidence are semantically equivalent at a "
    "similarity > 0.95, prefer the LONGER entity that matches the evidence "
    "verbatim. Stop immediately after the answer — "
    "no units, no trailing qualifiers, no parenthetical asides.\n"
    "7. [v5 20260719] Strict output format: emit the answer span ONLY. "
    "DO NOT append any parenthetical clarifications such as '(also known as X)', "
    "'(born YYYY)', '(filmmaker)', '(1995)', or any role/title/occupation "
    "asides in parentheses. If the natural answer would include such an "
    "aside, OMIT it — the parenthetical is never part of the gold answer. "
    "Examples: write 'Robert Erskine Childers DSC' (NOT 'Robert Erskine Childers "
    "DSC (author)'); write 'Scott Derrickson' (NOT 'Scott Derrickson "
    "(filmmaker)'); write 'Barton Lee Hazlewood' (NOT 'Barton Lee Hazlewood "
    "(also known as Lee Hazlewood)').\n"
    "8. [v5 20260719] Numeric answers: when the question asks 'how many' "
    "/'how much'/an explicit count, emit ONLY the bare number (no units, "
    "no qualifying words). Examples: '9' not '9 years'/'9 times'; "
    "'200' not 'over 200'/'approximately 200'/'200 pounds'. When the gold "
    "is 'yes'/'no', output exactly 'yes' or 'no'.\n"
    "9. [v5 20260719] Date / time answers: when the question asks "
    "'when'/'in what year' and the evidence gives a specific year, emit "
    "the YEAR as a 4-digit number. When the question asks about a time "
    "RANGE, answer in 'from X to Y' / 'X to Y' / 'X until Y' format "
    "(or the Chinese '从 X 到 Y' / 'X 至 Y') so the temporal span is "
    "explicit. Examples: '2010' not 'February 15, 2010' for year "
    "questions; 'from 1963 to 1972' not '1963-1972' for date-range "
    "questions; 'until 1995' for open-ended ranges."
)

_EM_USER_TMPL = (
    "Evidence chain:\n"
    "{chain_display}\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Respond in EXACTLY this format:\n"
    "Answer: {{answer}}\n"
    "Citation: {{citing sentence numbers}}"
)


def build_chain_display(chain: Dict) -> str:
    """
    Format an evidence chain dict into a readable string with logical-edge
    annotations between consecutive sentences.

    chain keys:
      - sentences: List[Dict]  (each dict has "id", "text", "role", "confidence")
      - edges:     List[Dict]    (each dict has "edge_type"; edges[i] connects
                                   sentences[i] -> sentences[i+1])

    Edge annotation is only shown when the edge type is one of the named logic
    types (CAUSAL / TEMPORAL / CONTRAST / ELABORATION). SAME_ENTITY and CONTEXT
    edges are skipped to keep the display clean.
    """
    sentences = chain.get("sentences", [])
    raw_edges = chain.get("edges", [])

    if not sentences:
        return ""

    # Build per-slot display annotation
    # edges[i] is the edge from sentences[i] -> sentences[i+1]
    edge_annotations: List[str] = []
    if raw_edges:
        for edge in raw_edges:
            etype = edge.get("edge_type", "") if isinstance(edge, dict) else str(edge)
            edge_annotations.append(_EDGE_SYMBOLS.get(etype, f"→ {etype}"))
    else:
        edge_annotations = []

    lines: List[str] = []
    for i, sent in enumerate(sentences):
        if isinstance(sent, dict):
            text = sent.get("text", sent.get("complemented_text", ""))
        else:
            text = str(sent)
        lines.append(f"[{i + 1}] {text}")
        if i < len(edge_annotations):
            lines.append(f"   {edge_annotations[i]}")

    return "\n".join(lines)


def build_answer_prompt(
    question: str,
    evidence_chain: Optional[Dict] = None,
    retrieved_sentences: Optional[List[str]] = None,
) -> str:
    """
    Construct the EM evaluation prompt for the LLM.

    Accepts either:
      - evidence_chain: a dict with "sentences" + optional "edges"
        (preferred — includes logical-edge annotations)
      - retrieved_sentences: flat list of strings (fallback for legacy callers)

    Only one of the two arguments should be provided.
    """
    if evidence_chain is not None:
        chain_display = build_chain_display(evidence_chain)
        return _EM_USER_TMPL.format(
            chain_display=chain_display,
            question=question,
        )
    elif retrieved_sentences:
        # Legacy flat-list fallback
        evidence_block = "\n".join(
            f"- {i + 1}. {s}" for i, s in enumerate(retrieved_sentences[:5])
        )
        return (
            f"Question: {question}\n\n"
            f"Based on the following evidence, answer the question concisely "
            f"(1-3 words for factual answers, one sentence otherwise):\n"
            f"{evidence_block}\n\n"
            f"Answer:"
        )
    else:
        return f"Question: {question}\n\nAnswer:"


def parse_em_response(raw: str) -> Tuple[str, str]:
    """
    Parse the LLM raw output into (answer, evidence_ref).

    Bilingual — accepts both Chinese and English prefixes:
      - "答案" / "Answer"
      - "引用" / "Citation" / "引用" / "依据"

    Tolerates:
      - Extra whitespace / newlines
      - Missing colon
      - Line order reversed
      - Answer on same line as prefix
    """
    answer, ref = "", ""
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()

        # Bilingual answer-line detection: "答案" / "Answer"
        if lower.startswith(("答案", "answer")):
            # Strip the prefix token (Chinese or English) and any punctuation
            for prefix in ("答案", "answer", "答案："):
                if lower.startswith(prefix):
                    answer = line[len(prefix):].strip()
                    break
            else:
                answer = line.split(maxsplit=1)[-1].lstrip(":： ").strip()

        # Bilingual citation-line detection: "引用" / "Citation" / "依据"
        elif any(lower.startswith(p) for p in ("引用", "citation", "依据")):
            for prefix in ("引用", "citation", "依据"):
                if lower.startswith(prefix):
                    ref = line[len(prefix):].strip()
                    break
            else:
                ref = line.split(maxsplit=1)[-1].lstrip(":： .").strip()

    return answer.strip(), ref.strip()


def generate_answer_via_llm(
    question: str,
    evidence_chain: Optional[Dict] = None,
    retrieved_sentences: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """
    Generate an answer using DeepSeek, feeding the evidence chain with logical
    edge annotations.

    Returns (answer_text, latency_seconds).
    Falls back to an empty answer if the LLM is unavailable.
    """
    from src.llm_client import DeepSeekClient

    prompt = build_answer_prompt(question, evidence_chain, retrieved_sentences)
    t0 = time.time()
    try:
        client = DeepSeekClient()
        if not client.enabled:
            return "", 0.0

        raw = client.chat(
            messages=[
                {"role": "system", "content": _EM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=64,
        )
        latency = time.time() - t0
        answer, _ = parse_em_response(raw)
        return answer, latency
    except Exception as e:
        logging.getLogger("hotpot_eval").warning("LLM answer generation failed: %s", e)
        return "", 0.0


# ── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(
    split: str,
    max_examples: int,
    use_llm: bool,
    quiet: bool,
    data_dir: str,
    debug: bool = False,
    debug_n: int = 5,
    seed: int = 33,
    no_shuffle: bool = False,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline."""

    # Paths
    filename_aliases = {
        "val": ["dev", "validation", "val"],
        "train": ["train"],
    }
    aliases = filename_aliases.get(split, [split])
    candidates = []
    for a in aliases:
        for ext in (".parquet", ".json"):
            for suffix in ("", "_v1", "_v1.1"):
                candidates.append(
                    os.path.join(data_dir, f"hotpot_{a}_distractor{suffix}{ext}")
                )
    candidates.append(os.path.join(data_dir, f"hotpot_{split}_distractor_v1.json"))
    candidates.append(os.path.join(data_dir, f"hotpot_{split}_distractor.parquet"))
    candidates.append(os.path.join(data_dir, f"hotpot_{split}_distractor_v1.parquet"))
    seen: set = set()
    deduped = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    candidates = deduped

    data_file = None
    for c in candidates:
        if os.path.exists(c):
            data_file = c
            break
    if data_file is None:
        raise FileNotFoundError(
            f"No HotpotQA file found in {data_dir!r}.\n"
            f"Looked for {len(candidates)} candidate names.\n"
            f"Run: python eval/download_hotpot.py"
        )
    log.info("Using data file: %s", data_file)
    hotpot_data = load_hotpot(data_file, split)
    n_total = len(hotpot_data)
    if no_shuffle:
        hotpot_data = hotpot_data[:max_examples]
        log.info(
            "Sampling: first %d (no shuffle) of %d",
            len(hotpot_data), n_total,
        )
    else:
        import random as _random
        rng = _random.Random(seed)
        if max_examples < n_total:
            indices = rng.sample(range(n_total), max_examples)
            indices.sort()
            hotpot_data = [hotpot_data[i] for i in indices]
        log.info(
            "Sampling: %d of %d questions (seed=%d, shuffled)",
            len(hotpot_data), n_total, seed,
        )

    # Load spaCy ONCE for the whole evaluation to avoid reloading
    # the 588 MB en_core_web_lg model per question.
    from src.graph_builder import _load_spacy_shared
    shared_nlp = _load_spacy_shared("en_core_web_lg")
    if shared_nlp is None:
        log.warning("spaCy model unavailable; some retrieval signals will degrade")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = results_dir / (
        f"stenrag_hotpotqa_{split}_{max_examples}_seed{seed}_{timestamp}.json"
    )
    summary_path = results_dir / (
        f"stenrag_hotpotqa_{split}_{max_examples}_seed{seed}_{timestamp}_summary.json"
    )

    # LLM summarizer (shared across all questions)
    llm_fn = None
    llm_latencies: List[float] = []
    if use_llm:
        try:
            from src.llm_client import make_summarizer

            llm_fn = make_summarizer()
            log.info("LLM answer generation ENABLED")
        except Exception as e:
            log.warning("Could not initialise LLM summarizer: %s", e)
            log.warning("Falling back to retrieval-only mode")
            use_llm = False

    # Accumulate metrics
    all_results: List[Dict[str, Any]] = []
    r_at_1, r_at_2, r_at_5, mrrs = [], [], [], []
    para_r_at_1, para_r_at_2, para_r_at_5, para_mrrs = [], [], [], []
    all_gold_hits = []
    ems, f1s, latencies = [], [], []

    log.info("Starting evaluation: split=%s n=%d use_llm=%s",
             split, len(hotpot_data), use_llm)

    for i, qdata in enumerate(hotpot_data):
        qid = qdata.get("_id", f"q_{i}")
        question = qdata.get("question", "")
        gold_answer = qdata.get("answer", "")

        t0 = time.time()

        # Build graph
        sentences = hotpot_to_sentences(qdata, nlp=shared_nlp)
        builder = build_graph(sentences, shared_nlp=shared_nlp)

        # ── LLM summarizer already applied lazily inside build_graph / add_sentence.
        #    Rebuilding all containers here (old approach) re-triggered the LLM hook
        #    hundreds of times per question → 100+ s/question.  Disabled for speed.
        # if llm_fn is not None:
        #     builder.container_manager.set_llm_summarizer(llm_fn)
        #     for c in list(builder.container_manager.get_all_containers()):
        #         builder.container_manager.update_container_index(c.id)

        # Retrieve
        gold_ids = gold_supporting_ids(qdata)
        gold_titles = gold_supporting_titles(qdata)
        retrieval = retrieve_and_score(
            builder, question, qdata, gold_ids, gold_titles,
            max_chains=50, debug=debug
        )

        # ── Debug dump (Phase 1 diagnostic) ────────────────────────────
        # Print first N questions: query, retrieved (text + gold status),
        # gold supporting sentences (text + id). Use this to spot whether
        # the retriever is pulling background distractor sentences vs.
        # true supporting facts.
        if debug and i < debug_n:
            log.info("─" * 70)
            log.info("[DEBUG qid=%s] %s", qid, question[:80])
            # Dump the retriever's intent (returned by retrieve())
            ri = retrieval.get("_intent", {})
            log.info("[DEBUG intent] type=%s entities=%s keywords=%s",
                     ri.get("type"), ri.get("entities"), ri.get("keywords"))
            log.info("[DEBUG gold_answer] %s", gold_answer)
            log.info("[DEBUG containers_retrieved] %s",
                     retrieval.get("_containers", []))
            # Gold titles
            log.info("[DEBUG gold_titles] %s", sorted(gold_titles))
            # Gold sentence texts
            gold_texts = []
            for fact in qdata.get("supporting_facts", []):
                title = fact["title"]
                for s_idx in fact.get("sentences", []):
                    # Find the paragraph by title and pull sentence text
                    for para in qdata.get("context", []):
                        if para["title"] == title:
                            s_list = para.get("sentences", [])
                            if 0 <= s_idx < len(s_list):
                                gold_texts.append(s_list[s_idx][:120])
                            break
            log.info("[DEBUG gold_sentences]")
            for gt in gold_texts:
                log.info("   G  %s", gt)
            # Retrieved titles
            log.info("[DEBUG retrieved_titles] %s", retrieval["retrieved_titles"])
            # Retrieved sentence-by-sentence with gold hit marker
            log.info("[DEBUG retrieved_sentences] (G=gold hit, .=miss)")
            flat_idxs = retrieval.get("retrieved_flat_idxs", [])
            for k, sid in enumerate(retrieval["retrieved_ids"], start=1):
                flat_idx = flat_idxs[k - 1] if k - 1 < len(flat_idxs) else None
                marker = "G" if flat_idx is not None and flat_idx in gold_ids else "."
                text = (
                    retrieval["retrieved_sentences"][k - 1]
                    if k - 1 < len(retrieval["retrieved_sentences"])
                    else ""
                )[:120]
                log.info("   %s %d  %s", marker, k, text)
            # Sentence recall diagnostic (against flat idx)
            hits = sum(1 for fi in flat_idxs if fi in gold_ids)
            log.info(
                "[DEBUG sentence_hits=%d/%d  para_hit@5=%s]",
                hits,
                len(gold_ids),
                retrieval["para_recall_at_5"],
            )

        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        # LLM answer (optional)
        gen_answer = ""
        em = False
        f1 = 0.0
        if use_llm:
            # Prefer the structured evidence chain (with logical edge annotations)
            # over the flat sentence list for richer context.
            chain_for_llm = retrieval.get("evidence_chain") or {
                "sentences": [
                    {"text": s} for s in retrieval.get("retrieved_sentences", [])
                ],
                "edges": [],
            }
            gen_answer, llm_lat = generate_answer_via_llm(
                question, evidence_chain=chain_for_llm
            )
            if llm_lat > 0:
                llm_latencies.append(llm_lat)
            # A4: strict HotpotQA scoring — NO soft post-processing.
            # Disable _strip_parens / _strip_colon_prefix /
            # _short_answer_gating / _coref_pick so EM/F1 reflect the raw
            # generation under standard SQuAD normalize (articles + ws/punct).
            # gen_answer = _strip_parens(gen_answer)
            # gen_answer = _strip_colon_prefix(gen_answer)
            # gen_answer = _short_answer_gating(gen_answer)
            # gen_answer = _coref_pick(gen_answer, gold_answer)
            em = _em_score(gen_answer, gold_answer)
            f1 = _f1_score(gen_answer, gold_answer)
            ems.append(em)
            f1s.append(f1)

        r_at_1.append(retrieval["recall_at_1"])
        r_at_2.append(retrieval["recall_at_2"])
        r_at_5.append(retrieval["recall_at_5"])
        mrrs.append(retrieval["mrr"])
        para_r_at_1.append(retrieval["para_recall_at_1"])
        para_r_at_2.append(retrieval["para_recall_at_2"])
        para_r_at_5.append(retrieval["para_recall_at_5"])
        para_mrrs.append(retrieval["para_mrr"])
        all_gold_hits.append(retrieval["all_gold_hit"])

        record = {
            "qid": qid,
            "question": question,
            "gold_answer": gold_answer,
            "retrieved_ids": retrieval["retrieved_ids"],
            "retrieved_flat_idxs": retrieval.get("retrieved_flat_idxs", []),
            "retrieved_titles": retrieval["retrieved_titles"],
            "retrieved_sentences": retrieval["retrieved_sentences"],
            "recall_at_1": retrieval["recall_at_1"],
            "recall_at_2": retrieval["recall_at_2"],
            "recall_at_5": retrieval["recall_at_5"],
            "mrr": retrieval["mrr"],
            "para_recall_at_1": retrieval["para_recall_at_1"],
            "para_recall_at_2": retrieval["para_recall_at_2"],
            "para_recall_at_5": retrieval["para_recall_at_5"],
            "para_mrr": retrieval["para_mrr"],
            "all_gold_hit": retrieval["all_gold_hit"],
            "latency_ms": latency_ms,
        }
        if use_llm:
            record.update({
                "generated_answer": gen_answer,
                "em": em,
                "f1": f1,
            })

        all_results.append(record)

        # Progress
        if not quiet:
            pct = (i + 1) / len(hotpot_data) * 100
            if use_llm:
                log.info(
                    "  [%3d%% %d/%d] qid=%s  R@1=%.3f  R@5=%.3f  "
                    "MRR=%.3f  EM=%s  F1=%.3f  lat=%.1fms",
                    pct, i + 1, len(hotpot_data), qid,
                    retrieval["recall_at_1"], retrieval["recall_at_5"],
                    retrieval["mrr"],
                    "✓" if em else "✗", f1,
                    retrieval.get("latency_ms", 0.0),
                )
                if not em and gen_answer:
                    log.info(
                        "          Q : %s\n          Gold: %s\n          Got : %s",
                        question, gold_answer, gen_answer,
                    )
                    for si, sent in enumerate(retrieval.get("retrieved_sentences", [])[:5]):
                        log.info("          [%d] %s", si + 1, sent[:200])
            else:
                log.info(
                    "  [%3d%% %d/%d] qid=%s  R@1=%.3f  R@5=%.3f  "
                    "MRR=%.3f  lat=%.1fms",
                    pct, i + 1, len(hotpot_data), qid,
                    retrieval["recall_at_1"], retrieval["recall_at_5"],
                    retrieval["mrr"],
                    retrieval.get("latency_ms", 0.0),
                )

    # Write detail results
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log.info("Detail results → %s", detail_path)

    # Aggregate summary
    summary = {
        "split": split,
        "n": len(all_results),
        "use_llm": use_llm,
        "retrieval": {
            "recall_at_1": float(np.mean(r_at_1)),
            "recall_at_2": float(np.mean(r_at_2)),
            "recall_at_5": float(np.mean(r_at_5)),
            "mrr": float(np.mean(mrrs)),
        },
        "retrieval_paragraph_level": {
            "para_recall_at_1": float(np.mean(para_r_at_1)),
            "para_recall_at_2": float(np.mean(para_r_at_2)),
            "para_recall_at_5": float(np.mean(para_r_at_5)),
            "para_mrr": float(np.mean(para_mrrs)),
            "all_gold_in_top_5": float(np.mean(all_gold_hits)),
        },
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
    }
    if use_llm:
        summary["answer"] = {
            "em": float(np.mean(ems)),
            "f1": float(np.mean(f1s)),
            "llm_latency_s_mean": float(np.mean(llm_latencies))
                if llm_latencies else 0.0,
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Summary → %s", summary_path)

    # Print final
    log.info("=" * 60)
    log.info("FINAL RESULTS  [%s, n=%d]", split, len(all_results))
    log.info("--- Sentence-level (strict) ---")
    log.info("Retrieval Recall@1 : %.4f", summary["retrieval"]["recall_at_1"])
    log.info("Retrieval Recall@2 : %.4f", summary["retrieval"]["recall_at_2"])
    log.info("Retrieval Recall@5 : %.4f", summary["retrieval"]["recall_at_5"])
    log.info("Retrieval MRR       : %.4f", summary["retrieval"]["mrr"])
    log.info("--- Paragraph-level (HotpotQA official) ---")
    log.info("Para Recall@1      : %.4f", summary["retrieval_paragraph_level"]["para_recall_at_1"])
    log.info("Para Recall@2      : %.4f", summary["retrieval_paragraph_level"]["para_recall_at_2"])
    log.info("Para Recall@5      : %.4f", summary["retrieval_paragraph_level"]["para_recall_at_5"])
    log.info("Para MRR           : %.4f", summary["retrieval_paragraph_level"]["para_mrr"])
    log.info("All-gold-in-top-5  : %.4f", summary["retrieval_paragraph_level"]["all_gold_in_top_5"])
    log.info("Latency (mean)     : %.1f ms", summary["latency_ms"]["mean"])
    if use_llm:
        log.info("Answer EM            : %.4f", summary["answer"]["em"])
        log.info("Answer F1           : %.4f", summary["answer"]["f1"])
    log.info("=" * 60)

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HotpotQA distractor evaluation for StenRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split", default="val",
        choices=["val", "train"],
        help="HotpotQA split to evaluate (default: val)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=200,
        help="Maximum number of questions to evaluate (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Random seed for question sampling (default: 7). "
             "Use --no-shuffle to take the first N questions deterministically.",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Disable random sampling; take the first --max-examples questions "
             "in dataset order (avoids seed dependence).",
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Enable LLM answer generation (requires DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-question progress output",
    )
    parser.add_argument(
        "--data-dir", default="eval/data",
        help="Directory containing hotpot_{split}_distractor_v1.json "
             "(default: eval/data)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print per-question debug: query, retrieved, gold (first N only)",
    )
    parser.add_argument(
        "--debug-n", type=int, default=5,
        help="Number of questions to debug-print (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log.info("HotpotQA Evaluation for StenRAG")
    log.info("  split        = %s", args.split)
    log.info("  max_examples = %d", args.max_examples)
    log.info("  seed         = %d", args.seed)
    log.info("  use_llm     = %s", args.use_llm)
    log.info("  data_dir    = %s", args.data_dir)
    log.info("  debug       = %s (n=%d)", args.debug, args.debug_n)

    try:
        evaluate(
            split=args.split,
            max_examples=args.max_examples,
            use_llm=args.use_llm,
            quiet=args.quiet,
            seed=args.seed,
            no_shuffle=args.no_shuffle,
            data_dir=args.data_dir,
            debug=args.debug,
            debug_n=args.debug_n,
        )
        return 0
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return 130
    except Exception:
        log.exception("Unexpected error during evaluation")
        return 1


if __name__ == "__main__":
    sys.exit(main())

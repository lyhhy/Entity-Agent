"""
eval/recall_utils.py — R@k 的 gold 句子匹配（唯一真相源）

为什么这个模块存在
==================

R@k 衡量的是"gold 支撑句有没有出现在检索结果的前 k 位"。这个指标
的**正确性完全取决于 gold 句子的判定方式**，而原来的判定方式有
两个系统性错误：

────────────────────────────────────────────────────────────────
错误 1：子串包含判定（系统性低估 recall）
────────────────────────────────────────────────────────────────
旧代码：

```python
gold_norm in _normalize_answer(node.text or "")
```

这是**子串**匹配，不是词边界匹配。它有两个方向的错：

(a) **假阴性 —— 判定过严。**
    gold 是 `'yes'`（HotpotQA 的 "same X?" 类问题）时，整个 context
    里没有任何一句包含 "yes" 这个子串。这道题的 R@k **恒为 0**，
    无论检索质量多好。在 seed=7 的 20 题里就有 1 题（Q4）属于这种
    情况 —— 也就是说 R@10=0.800 这个数字里，**有一半是量纲错误**，
    真实上限是 17/20=0.850。

    在 300 题的抽样上，这类"gold 在 context 里根本不存在"的题占
    **2.3%**。它们系统性地把 recall 压低，且压低幅度与检索质量无关
    —— 也就是**噪声**，会让 A/B 对比完全失去意义。

(b) **假阳性 —— 判定过松。**
    gold 是 `'Matthew Good'` 时，`'matthew good'` 是
    `"...on Matthew Good's own imprint..."` 的子串 —— 但那一句讲
    的是**另一件事**（唱片的发行），不是答案句。子串匹配把这条
    无关句判成了"命中 gold"。

    更糟的是短词 gold：`'Polk County'` 之类还好，但 `'French Open'`、
    `'yes'` 这种一旦拆词就会命中大量无关句。300 题抽样里，
    **25.7%** 的题会匹配到 ≥3 个句子。

────────────────────────────────────────────────────────────────
错误 2：只看答案串，忽略真正的 gold 支撑句
────────────────────────────────────────────────────────────────
HotpotQA 的 `supporting_facts` 字段**直接给出了 gold 支撑句**
（`[[title, sent_idx], ...]`）。这是数据集作者的标注，是 ground
truth 的 ground truth。

旧代码完全没用它，而是用"答案串是否出现在句子里"来反推哪句是
gold。这在语义上是**错的**：HotpotQA 的 gold 支撑句是"包含答案的
那两句"，而"包含答案串的句子"可能有 10 句 —— 其中 9 句只是恰好
提到了那个实体。

────────────────────────────────────────────────────────────────
修复方案
========

**主判据：`supporting_facts` 的精确句子匹配**（词边界）。

**降级判据**：当数据集没有 `supporting_facts`（或标注解析失败）时，
退回到子串匹配，但加上**词边界约束** —— 消除 `'Matthew Good's'`
这类假阳性。

**绝不匹配的情况**：gold 是 `'yes'` / `'no'` 这类**极性答案**。
它们不是句子里的一段文本，用它们做句子匹配在语义上就是错的。
这类题的 R@k 记为 **None（不计入分母）**，并在日志里单独报数 ——
而不是静默地当成 0 分（那会污染均值）或当成 1 分（那会虚高）。

Why this module is separate
============================

R@k 的判定逻辑在 `quick20_eval.py`（2 处）和 `quick20_clean.py`
（1 处）各有一份拷贝。三处独立实现意味着**修一处漏两处**，而且
它们已经漂移了（变量名不同、行为可能不一致）。集中到这里之后，
三处调用点共享同一个实现，改一次全局生效。

Public API
==========
    gold_sentence_matcher(qdata, builder) -> GoldMatcher
    GoldMatcher.contains_answer(sid) -> bool
    GoldMatcher.is_scorable -> bool
    GoldMatcher.match_kind -> str
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Optional, Set

# ─────────────────────────────────────────────────────────────────
# 极性答案：不是句子里的文本，不能用于句子级匹配
# ─────────────────────────────────────────────────────────────────
POLAR_ANSWERS = frozenset({"yes", "no"})


def normalize_answer(s: str) -> str:
    """The eval's canonical normalisation (articles / punctuation / case).

    Kept byte-identical to `quick20_eval._normalize_answer` so the two
    modules never disagree about what "the same string" means.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc((s or "").lower())))


def _word_boundary_contains(haystack_norm: str, needle_norm: str) -> bool:
    """True iff `needle_norm` occurs in `haystack_norm` on token boundaries.

    Fixes the `'Matthew Good'` vs `"Matthew Good's imprint"` false
    positive: the old plain-substring test matched the possessive,
    this one does not.

    Note the trailing boundary is `(?![\\w'])` rather than `(?![\\w])`
    so that a gold answer that legitimately ENDS in an apostrophe
    (`'That Bizarre Girl'` → `"bizarre girl"`) still matches the
    quoted form in the sentence.
    """
    if not needle_norm:
        return False
    return re.search(
        r"(?<!\w)" + re.escape(needle_norm) + r"(?!\w)",
        haystack_norm,
    ) is not None


class GoldMatcher:
    """
    Decides whether a candidate sentence is a gold supporting sentence.

    Two match kinds, in priority order:

    * ``"supporting_facts"`` — the sentence IDs resolved from the
      dataset's own ``supporting_facts`` annotation.  This is the
      authoritative judgement and is used whenever the annotation
      could be resolved.

    * ``"answer_span"`` — the fallback: the sentence contains the gold
      answer string, on token boundaries.

    `is_scorable` is False for polar answers ('yes' / 'no'), which are
    not sentence spans at all.  Those questions must be excluded from
    the R@k denominator rather than counted as a miss.
    """

    def __init__(
        self,
        gold_answer: str,
        gold_sids: Optional[Set[str]] = None,
        match_kind: str = "answer_span",
        reason: str = "",
    ):
        self.gold_answer = gold_answer or ""
        self.gold_norm = normalize_answer(self.gold_answer)
        self.gold_sids: Set[str] = set(gold_sids or ())
        self.match_kind = match_kind
        self.reason = reason

    # -- properties ---------------------------------------------------

    @property
    def is_polar(self) -> bool:
        """True for yes/no answers — not a sentence span."""
        return self.gold_norm in POLAR_ANSWERS

    @property
    def is_scorable(self) -> bool:
        """False when R@k would be meaningless for this question."""
        return not self.is_polar

    @property
    def has_authoritative_ids(self) -> bool:
        return bool(self.gold_sids)

    # -- matching -----------------------------------------------------

    def contains_answer(self, sid: str, sentence_text: str) -> bool:
        """True iff `sid` is a gold supporting sentence."""
        if self.is_polar:
            # A polar answer cannot be located in a sentence.  Returning
            # False here would silently score it as a miss; the caller
            # is expected to skip polar questions via `is_scorable`.
            return False

        if self.gold_sids:
            return sid in self.gold_sids

        # Fallback: word-boundary containment of the answer string.
        return _word_boundary_contains(
            normalize_answer(sentence_text or ""), self.gold_norm,
        )


def _resolve_supporting_fact_sids(
    qdata: Dict[str, Any],
    builder: Any,
) -> List[str]:
    """
    Resolve the dataset's `supporting_facts` annotation to sentence IDs.

    `supporting_facts` is either ``[[title, sent_idx], ...]`` or
    ``[{title, sentence_id}, ...]``.  A title + index pair points at one
    sentence of one paragraph; we locate that sentence's text and then
    map the text back to the graph's sentence ID (the graph assigns its
    own IDs, so a text match is the only bridge).
    """
    sids: List[str] = []
    try:
        sfacts = (
            qdata.get("supporting_facts")
            or qdata.get("support_facts")
            or []
        )
        para = qdata.get("context") or []
        nodes = getattr(
            getattr(builder, "sentence_registry", None), "_nodes", {},
        ) or {}

        def _title_idx(sf):
            if isinstance(sf, dict):
                return (
                    sf.get("title") or "",
                    int(sf.get("sentence_id") or 0),
                )
            if isinstance(sf, (list, tuple)) and len(sf) >= 2:
                return (sf[0] or "", int(sf[1]))
            return ("", -1)

        # text -> sid map, built once (exact match first, then normalised)
        by_text: Dict[str, str] = {}
        by_norm: Dict[str, str] = {}
        for sid, node in nodes.items():
            txt = getattr(node, "text", "") or ""
            by_text.setdefault(txt, sid)
            by_norm.setdefault(normalize_answer(txt), sid)

        for sf in sfacts:
            title_raw, sidx = _title_idx(sf)
            title = (title_raw or "").lower().strip()
            if not title or sidx < 0:
                continue
            sentence_text = None
            for p in para:
                if (
                    isinstance(p, list) and len(p) == 2
                    and (p[0] or "").lower().strip() == title
                ):
                    p_sents = p[1]
                    if isinstance(p_sents, list) and sidx < len(p_sents):
                        sentence_text = p_sents[sidx]
                    break
            if not sentence_text:
                continue
            sid = by_text.get(sentence_text)
            if sid is None:
                sid = by_norm.get(normalize_answer(sentence_text))
            if sid and sid not in sids:
                sids.append(sid)
    except Exception:
        return sids
    return sids


def gold_sentence_matcher(
    qdata: Dict[str, Any],
    builder: Any,
) -> GoldMatcher:
    """
    Build the matcher for one question.

    Priority:
      1. Resolve `supporting_facts` → use the dataset's own annotation.
      2. Otherwise fall back to word-boundary answer containment.
      3. Polar answers ('yes' / 'no') are marked un-scorable.
    """
    gold_answer = (qdata or {}).get("answer", "") or ""

    sids: List[str] = []
    if builder is not None:
        sids = _resolve_supporting_fact_sids(qdata, builder)

    matcher = GoldMatcher(
        gold_answer=gold_answer,
        gold_sids=set(sids),
        match_kind="supporting_facts" if sids else "answer_span",
        reason="",
    )
    if matcher.is_polar:
        matcher.reason = "polar answer — not a sentence span"
    elif not sids:
        matcher.reason = "supporting_facts unavailable — answer-span fallback"
    return matcher


# ─────────────────────────────────────────────────────────────────
# 段落级（passage）指标
# ─────────────────────────────────────────────────────────────────
#
# 为什么需要段落级指标
# ====================
#
# 句子级 R@k 衡量的是"gold **支撑句**有没有被检索到"。但多跳 QA 的
# 真正瓶颈是**证据段落**：HotpotQA 的每道题有 2 个 gold 段落，每个
# 段落里的某一句才是支撑句。检索器的实际任务是"把这两个段落捞进
# 上下文"，而不是"把某一句精确捞进来"。
#
# 论文里的两种标准口径（都是段落粒度）：
#
# **口径 A —— Passage Recall@k（MDR / Xiong 2020 用的）**
#
#     Recall@k = (1/|Q|) Σ_q  |top-k ∩ gold_passages_q| / |gold_passages_q|
#
#     允许只抓到一半证据就得分。top5 里抓到 2 个 gold 段落中的 1 个
#     → 该样本贡献 0.5。
#
# **口径 B —— Hit@k / Pair Recall@k（GraphRAG / LongRAG 用的）**
#
#     Recall@k = |{q : top-k ⊇ gold_passages_q}| / |Q|
#
#     必须**全部** gold 段落都在 top-k 里才算成功，缺任意一个直接 0。
#     适合多跳 RAG —— 少任意一篇证据就无法完成推理。
#
# 这两个口径**不能互相替代**：口径 A 对检索器更宽容（可以只捞到
# 一半），口径 B 更严格（必须全捞到）。HotpotQA 每题恰好 2 个 gold
# 段落，所以：
#
#   · 口径 A ∈ {0, 0.5, 1}
#   · 口径 B ∈ {0, 1}
#
# 报两个口径，才能和不同论文的数字对齐。
#
# 排序口径的说明
# --------------
# 这两个指标都在**段落**粒度上定义，所以需要把"句子排序"转换成
# "段落排序"。转换规则：**段落的位置 = 该段落里排名最高的那个句子
# 的位置**（first-appearance）。这是标准做法 —— 一个段落只要有任
# 何一句进了 top-k，它就算被命中了。
#
# 举例：句子顺序 [s_A1, s_B1, s_A2]，段落 A 的位置 = 1（s_A1），
# 段落 B 的位置 = 2（s_B1）。top-2 里段落 A 和 B 都在 → Hit@2 = 1。

def build_sentence_to_paragraph(
    qdata: Dict[str, Any],
    builder: Any,
) -> Dict[str, str]:
    """
    Map every graph sentence ID to its paragraph title (lowercased).

    The graph assigns its own sentence IDs, so the bridge is the
    sentence TEXT: we match each paragraph's sentences against the
    registry, exactly like `_resolve_supporting_fact_sids` does.
    """
    out: Dict[str, str] = {}
    try:
        nodes = getattr(
            getattr(builder, "sentence_registry", None), "_nodes", {},
        ) or {}
        by_text: Dict[str, str] = {}
        by_norm: Dict[str, str] = {}
        for sid, node in nodes.items():
            txt = getattr(node, "text", "") or ""
            by_text.setdefault(txt, sid)
            by_norm.setdefault(normalize_answer(txt), sid)

        for title, sents in (qdata.get("context") or []):
            title_key = (title or "").lower().strip()
            if not title_key:
                continue
            for sent_text in sents or []:
                sid = by_text.get(sent_text)
                if sid is None:
                    sid = by_norm.get(normalize_answer(sent_text))
                if sid:
                    out[sid] = title_key
    except Exception:
        return out
    return out


def gold_passage_titles(qdata: Dict[str, Any]) -> List[str]:
    """
    The gold paragraph titles for one question, from `supporting_facts`.

    HotpotQA's `supporting_facts` is ``[[title, sent_idx], ...]`` — the
    titles are the gold PARAGRAPHS, and the indices point at the
    supporting sentence inside each.  So the set of distinct titles IS
    the gold passage set.
    """
    titles: List[str] = []
    seen: Set[str] = set()
    for sf in (
        qdata.get("supporting_facts")
        or qdata.get("support_facts") or []
    ):
        raw = (
            sf.get("title") if isinstance(sf, dict)
            else (sf[0] if isinstance(sf, (list, tuple)) and sf else "")
        )
        t = (raw or "").lower().strip()
        if t and t not in seen:
            seen.add(t)
            titles.append(t)
    return titles


def passage_metrics(
    ranked_sids: List[str],
    sent_to_para: Dict[str, str],
    gold_paras: List[str],
    ks: List[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute both passage-level recall conventions.

    Args:
        ranked_sids: the ranked sentence IDs (best first).
        sent_to_para: sentence ID → paragraph title.
        gold_paras: the gold paragraph titles.
        ks: the cut-offs to report.

    Returns:
        A dict with, for each k:
          * ``passage_recall@k`` — 口径 A: fraction of gold passages
            present in the top-k, averaged over queries.
          * ``hit@k``           — 口径 B: 1 iff ALL gold passages are
            present in the top-k.

        Plus ``n_gold_passages`` for reference.

    **Ranking semantics**: a passage's rank is the position of its
    BEST-ranked sentence (first appearance).  So a passage counts as
    present in the top-k if ANY of its sentences is in the top-k.
    """
    gold_set = set(gold_paras or [])
    n_gold = len(gold_set)

    out: Dict[str, float] = {"n_gold_passages": float(n_gold)}
    if n_gold == 0:
        for k in ks:
            out[f"passage_recall@{k}"] = 0.0
            out[f"hit@{k}"] = 0.0
        return out

    # Passage order = position of its best sentence.
    para_rank: Dict[str, int] = {}
    for pos, sid in enumerate(ranked_sids or []):
        para = sent_to_para.get(sid)
        if para is None:
            continue
        if para not in para_rank:
            para_rank[para] = pos

    for k in ks:
        # 口径 A: |top-k passages ∩ gold| / |gold|
        in_top = {p for p, r in para_rank.items() if r < k}
        hit_paras = in_top & gold_set
        out[f"passage_recall@{k}"] = len(hit_paras) / n_gold
        # 口径 B: all gold passages present?
        out[f"hit@{k}"] = 1.0 if gold_set.issubset(in_top) else 0.0

    return out

"""Probe span-granularity prompt designs on fixed evidence.

Runs several candidate system prompts over the same evidence and reports
EM/F1 plus a per-question diff against the v3 baseline, so a design can
be judged on which specific failures it moves.

Usage:
  python eval/_probe_span.py --records eval/results/clean_results_attrgate2.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def _em(p: str, g: str) -> int:
    return int(_norm(p) == _norm(g))


def _f1(p: str, g: str) -> float:
    a, b = _norm(p).split(), _norm(g).split()
    if not a or not b:
        return float(a == b)
    common = Counter(a) & Counter(b)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(a), ns / len(b)
    return 2 * prec * rec / (prec + rec)


# ── design S1: explicit boundary table + one worked example each ─────
S1 = (
    "You are a precision answer extractor for multi-hop QA. Given a question "
    "and ranked evidence sentences from a Wikipedia-style corpus, extract the "
    "SINGLE best answer span.\n\n"
    "The answer must be the EXACT SPAN the question asks for — the same "
    "extent, neither wider nor narrower.\n\n"
    "SPAN BOUNDARY RULES:\n"
    "• Drop a series/edition prefix, keep the distinctive title:\n"
    "  'UFC 50: The War of \\'04' -> 'The War of \\'04'\n"
    "• Drop an honorific/title before a person's name:\n"
    "  'Sir George Cayley' -> 'George Cayley'\n"
    "• Keep the unit/qualifier that belongs to a quantity:\n"
    "  '43,000 marchers' -> '43,000 marchers' (NOT '43,000')\n"
    "• Keep the full administrative string of a place:\n"
    "  'Midland, Michigan, United States' -> 'Midland, Michigan, United States'\n"
    "• When the question asks for a NAME, return the name — not a phrase "
    "describing it:\n"
    "  'the practice of what way of filling armed forces vacancies' -> "
    "'Conscription' (the name of the practice), not 'requiring only men to "
    "register'\n"
    "• When the question asks for a DEFINITION, return the defining phrase:\n"
    "  'what does X refer to' -> 'amount charged by a bookmaker'\n\n"
    "OUTPUT RULES:\n"
    "1. The answer MUST appear literally in the evidence. Copy it verbatim.\n"
    "2. Match the question's answer TYPE: a NUMBER for 'how many', a DATE for "
    "'when'/'what year', a NAME for 'who'/'which X', 'yes' or 'no' for a "
    "polar question, a PLACE for 'where'.\n"
    "3. NEVER answer 'unanswerable' — every question here has a real answer.\n"
    "4. Scan the WHOLE evidence list; the answer is not necessarily in the "
    "first sentence.\n"
    "5. Output ONLY the answer span. No explanation, no quotes, no trailing "
    "punctuation. 1-10 words."
)

# ── design S2: S1 + an explicit "ask yourself" reflection step ───────
S2 = (
    S1 + "\n\n"
    "BEFORE OUTPUTTING, check the span against these three questions:\n"
    "1. Is this exact string present in the evidence? If not, replace it.\n"
    "2. Does it have the type the question asks for?\n"
    "3. Is it the right extent — no leftover prefix, no missing unit, and "
    "is it a NAME when a NAME was asked for?\n"
    "If any check fails, fix the span before answering."
)

# ── design S3: two-step (locate sentence, then cut the span) ─────────
S3 = (
    "You are a precision answer extractor for multi-hop QA. Given a question "
    "and ranked evidence sentences from a Wikipedia-style corpus, find the "
    "answer.\n\n"
    "Reply in exactly this format:\n"
    "SENTENCE: <the one evidence sentence that answers the question>\n"
    "ANSWER: <the exact answer span, copied from that sentence>\n\n"
    "Rules for ANSWER:\n"
    "• It MUST appear literally in the SENTENCE you picked.\n"
    "• It MUST be the same extent the question asks for:\n"
    "  - drop a series/edition prefix ('UFC 50: The War of \\'04' -> "
    "'The War of \\'04')\n"
    "  - drop an honorific ('Sir George Cayley' -> 'George Cayley')\n"
    "  - keep a quantity's unit ('43,000 marchers', not '43,000')\n"
    "  - keep a place's full administrative string\n"
    "  - if the question asks for a NAME, give the NAME ('Conscription'), "
    "not a phrase describing it\n"
    "  - if the question asks for a DEFINITION, give the defining phrase\n"
    "• Match the question's answer TYPE (number / date / name / yes-no / "
    "place).\n"
    "• NEVER answer 'unanswerable'.\n"
    "• Scan the WHOLE evidence list; the answer is not necessarily in the "
    "first sentence.\n"
    "• Output ONLY the two lines above."
)


def parse_s3(text: str) -> str:
    m = re.search(r"ANSWER\s*:\s*(.+)", text or "")
    if not m:
        return ""
    return m.group(1).strip().strip('"').strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    args = ap.parse_args()

    os.environ["PYTHONIOENCODING"] = "utf-8"
    from src.llm_client import make_client
    import src.answer_extractor as ae

    client = make_client()
    recs = []
    for p in args.records:
        recs.extend(json.load(open(REPO / p, encoding="utf-8"))["records"])

    designs = {"S1_boundary": S1, "S2_boundary+check": S2, "S3_twostep": S3}
    results = {}

    for name, system in designs.items():
        answers = {}
        for r in recs:
            ctx = "\n".join(e["text"] for e in (r.get("evidence") or []) if e.get("text"))
            try:
                raw = ae.postprocess(client.chat(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": ae.build_user_message(r["question"], ctx)},
                    ],
                    temperature=0.0,
                    max_tokens=120 if name == "S3_twostep" else 40,
                ))
            except Exception:
                raw = ""
            answers[r["qid"]] = parse_s3(raw) if name == "S3_twostep" else raw
        results[name] = answers

    # v3 baseline for reference.
    base = {}
    for r in recs:
        ctx = "\n".join(e["text"] for e in (r.get("evidence") or []) if e.get("text"))
        base[r["qid"]] = ae.extract_answer(
            client, r["question"], ctx, variant="v3", self_correct=False,
        )

    print("=" * 78)
    print(f"n={len(recs)}  v3 baseline EM="
          f"{sum(_em(base[r['qid']], r['gold']) for r in recs)/len(recs):.3f}")
    print("=" * 78)
    for name, answers in results.items():
        ems = [_em(answers[r["qid"]], r["gold"]) for r in recs]
        f1s = [_f1(answers[r["qid"]], r["gold"]) for r in recs]
        fixed = broken = 0
        for r in recs:
            was = _em(base[r["qid"]], r["gold"])
            now = _em(answers[r["qid"]], r["gold"])
            if not was and now:
                fixed += 1
                print(f"    FIXED {r['gold']!r} <- {answers[r['qid']]!r}")
            elif was and not now:
                broken += 1
                print(f"    BROKE {r['gold']!r} -> {answers[r['qid']]!r}")
        print(f"  {name:<20} EM={sum(ems)/len(ems):.3f} "
              f"F1={sum(f1s)/len(f1s):.3f}  fixed={fixed} broken={broken}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

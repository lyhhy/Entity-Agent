"""Probe Self-Correction prompt variants on the pooled 512-question set.

The 512-question run showed the self-correction pass is a solid win
(+0.025 EM, bootstrap CI [+0.006, +0.047]) but that its 8 broken
answers share a shape: the reflection pass OVER-ELABORATES a correct
short answer ('Sully' -> 'Chesley Burnett "Sully" Sullenberger III',
'Laura Dern' -> 'Richard Dreyfuss').

It also showed the separate span-granularity prompt (v5) is HARMFUL:
its 11 true breaks are mostly over-truncation ('Chick tracts' ->
'Chick', 'Swiss made' -> 'Swiss'), the exact opposite of the intent.

This probe therefore varies the SELF-CORRECTION prompt only, and
measures fixed/broken against the v3 first pass.

Usage:
  python eval/_probe_sc512.py --records eval/results/pool_evidence512.json
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


# ── W1: current implementation (independent solve, no carried rules) ──
W1 = (
    "You are a precision answer extractor for multi-hop QA. You are given a "
    "question, ranked evidence sentences, and a candidate answer from a "
    "previous attempt.\n\n"
    "Do NOT assume the candidate is right. Solve the question INDEPENDENTLY "
    "from the evidence, then decide.\n\n"
    "Reply in exactly this format:\n"
    "ANSWER: <the exact answer span>\n"
    "VERDICT: <KEEP if it equals the candidate, otherwise FIXED>"
)

# ── W2: W1 + anti-over-elaboration + name-vs-definition ──────────────
W2 = (
    "You are a precision answer extractor for multi-hop QA. You are given a "
    "question, ranked evidence sentences, and a candidate answer from a "
    "previous attempt.\n\n"
    "Do NOT assume the candidate is right. Solve the question INDEPENDENTLY "
    "from the evidence, then decide.\n\n"
    "Answer rules:\n"
    "1. The answer MUST be a span that literally appears in the evidence. "
    "Copy it verbatim.\n"
    "2. Match the question's answer TYPE: a NUMBER for 'how many', a DATE "
    "for 'when'/'what year', a NAME for 'who'/'which X', exactly 'yes' or "
    "'no' for a polar question, a PLACE for 'where'.\n"
    "3. NAME vs DEFINITION: if the question asks for the NAME of something "
    "('which X', 'who', 'what album'), answer with the NAME. If it asks for "
    "the MEANING/DEFINITION ('refers to which portion', 'what does X mean'), "
    "answer with the descriptive phrase that defines the term, NOT the term "
    "itself.\n"
    "4. Match the PRECISION the question asked for. If the question asks for "
    "a person, a short name is right ('Sully', not 'Chesley Burnett "
    "\"Sully\" Sullenberger III'). If it asks for a place, give the place "
    "as the evidence spells it ('Nelson County', not 'Nelson'). If it asks "
    "for a quantity, give the quantity with its unit ('43,000 marchers').\n"
    "5. NEVER answer 'unanswerable'. Commit to the most plausible span.\n"
    "6. Scan the WHOLE evidence list; the answer is not necessarily in the "
    "first sentence.\n\n"
    "Reply in exactly this format:\n"
    "ANSWER: <the exact answer span>\n"
    "VERDICT: <KEEP if it equals the candidate, otherwise FIXED>"
)

# ── W3: W2 but the candidate wins ties ───────────────────────────────
W3 = (
    "You are a precision answer extractor for multi-hop QA. You are given a "
    "question, ranked evidence sentences, and a candidate answer from a "
    "previous attempt.\n\n"
    "Solve the question INDEPENDENTLY from the evidence, then compare your "
    "answer with the candidate.\n\n"
    "IMPORTANT: the candidate was produced by a model that already applied "
    "the rules below, and it is usually correct. Use it as a tie-breaker: if "
    "your independent answer is not CLEARLY better, keep the candidate.\n\n"
    "Answer rules:\n"
    "1. The answer MUST be a span that literally appears in the evidence. "
    "Copy it verbatim.\n"
    "2. Match the question's answer TYPE: a NUMBER for 'how many', a DATE "
    "for 'when'/'what year', a NAME for 'who'/'which X', exactly 'yes' or "
    "'no' for a polar question, a PLACE for 'where'.\n"
    "3. NAME vs DEFINITION: if the question asks for the NAME of something, "
    "answer with the NAME; if it asks for the MEANING/DEFINITION, answer with "
    "the descriptive phrase that defines the term, NOT the term itself.\n"
    "4. Match the PRECISION the question asked for: a short name for a person "
    "('Sully', not 'Chesley Burnett \"Sully\" Sullenberger III'), a place as "
    "spelled in the evidence ('Nelson County', not 'Nelson').\n"
    "5. NEVER answer 'unanswerable'. Commit to the most plausible span.\n"
    "6. Scan the WHOLE evidence list; the answer is not necessarily in the "
    "first sentence.\n\n"
    "Reply in exactly this format:\n"
    "ANSWER: <the exact answer span>\n"
    "VERDICT: <KEEP if the candidate should stand, otherwise FIXED>"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.environ["PYTHONIOENCODING"] = "utf-8"
    from src.llm_client import make_client
    import src.answer_extractor as ae

    client = make_client()
    recs = []
    for p in args.records:
        recs.extend(json.load(open(REPO / p, encoding="utf-8"))["records"])
    if args.limit:
        recs = recs[:args.limit]

    # First pass: v3, no self-correction. Cached across designs.
    base = {}
    for i, r in enumerate(recs):
        ctx = "\n".join(e["text"] for e in (r.get("evidence") or []) if e.get("text"))
        base[r["qid"]] = ae.extract_answer(
            client, r["question"], ctx, variant="v3", self_correct=False,
        )
        if (i + 1) % 50 == 0:
            print(f"  base {i+1}/{len(recs)}", flush=True)

    designs = {"W1_current": W1, "W2_precise": W2, "W3_tiebreak": W3}
    results = {}
    for name, system in designs.items():
        answers = {}
        for i, r in enumerate(recs):
            ctx = "\n".join(e["text"] for e in (r.get("evidence") or []) if e.get("text"))
            cand = base[r["qid"]]
            user = ae._SELF_CORRECT_USER.format(
                question=r["question"], context=ctx, candidate=cand)
            try:
                raw = ae.postprocess(client.chat(
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0.0, max_tokens=120))
            except Exception:
                raw = ""
            ans = ae._parse_tagged(raw, "ANSWER")
            verdict = ae._parse_tagged(raw, "VERDICT").upper()
            if not ans or verdict.startswith("KEEP"):
                answers[r["qid"]] = cand
            elif verdict.startswith("FIXED") and ae._span_in_context(ans, ctx):
                answers[r["qid"]] = ans
            else:
                answers[r["qid"]] = cand
        results[name] = answers
        print(f"  {name} done", flush=True)

    print("=" * 78)
    print(f"n={len(recs)}  v3 first-pass EM="
          f"{sum(_em(base[r['qid']], r['gold']) for r in recs)/len(recs):.4f}")
    print("=" * 78)
    for name, answers in results.items():
        ems = [_em(answers[r["qid"]], r["gold"]) for r in recs]
        f1s = [_f1(answers[r["qid"]], r["gold"]) for r in recs]
        fixed = broken = 0
        broke_ex = []
        for r in recs:
            was = _em(base[r["qid"]], r["gold"])
            now = _em(answers[r["qid"]], r["gold"])
            if not was and now:
                fixed += 1
            elif was and not now:
                broken += 1
                broke_ex.append((r["gold"], base[r["qid"]], answers[r["qid"]]))
        print(f"  {name:<14} EM={sum(ems)/len(ems):.4f} "
              f"F1={sum(f1s)/len(f1s):.4f}  fixed={fixed} broken={broken} "
              f"net={fixed-broken:+d}")
        for g, a, b in broke_ex[:10]:
            print(f"      BROKE {g!r:26} {a!r} -> {b!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

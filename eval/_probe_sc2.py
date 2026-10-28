"""Probe Self-Correction prompt variants.

The first self-correction implementation re-solved the question
independently but WITHOUT carrying over the extractor's own rules
(name-vs-definition, span granularity).  That made it break answers the
extractor had got right (e.g. it answered the DEFINITION question with
the defined TERM, "Vigorish", instead of the definition).

This probes self-correction prompts that carry those rules, so the
reflection pass is at least as well-informed as the first pass.

Usage:
  python eval/_probe_sc2.py --records eval/results/clean_results_attrgate2.json
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


RULES = (
    "Answer rules:\n"
    "1. The answer MUST be a span that literally appears in the evidence. "
    "Copy it verbatim.\n"
    "2. Match the question's answer TYPE: a NUMBER for 'how many', a DATE "
    "for 'when'/'what year', a NAME for 'who'/'which X', exactly 'yes' or "
    "'no' for a polar question, a PLACE for 'where'.\n"
    "3. NAME vs DEFINITION: if the question asks for the NAME of something "
    "('which X', 'who', 'what album', 'what colloquial umbrella'), answer "
    "with the NAME. If it asks for the MEANING/DEFINITION ('refers to which "
    "portion', 'what does X mean'), answer with the descriptive phrase that "
    "defines the term, NOT the term itself.\n"
    "4. SPAN GRANULARITY: return the answer at the exact extent the "
    "question asks for. Drop a series/edition prefix (\"UFC 50: The War of "
    "'04\" -> \"The War of '04\") and an honorific (\"Sir George Cayley\" -> "
    "\"George Cayley\"); keep a quantity's unit (\"43,000 marchers\") and a "
    "place's full administrative string.\n"
    "5. NEVER answer 'unanswerable'. Commit to the most plausible span.\n"
    "6. Scan the WHOLE evidence list; the answer is not necessarily in the "
    "first sentence.\n"
)

# V1: current implementation (independent solve, no carried-over rules)
V1 = (
    "You are a precision answer extractor for multi-hop QA. You are given a "
    "question, ranked evidence sentences, and a candidate answer from a "
    "previous attempt.\n\n"
    "Do NOT assume the candidate is right. Solve the question INDEPENDENTLY "
    "from the evidence, then decide.\n\n"
    "Reply in exactly this format:\n"
    "ANSWER: <the exact answer span>\n"
    "VERDICT: <KEEP if it equals the candidate, otherwise FIXED>"
)

# V2: independent solve + carried-over rules
V2 = (
    "You are a precision answer extractor for multi-hop QA. You are given a "
    "question, ranked evidence sentences, and a candidate answer from a "
    "previous attempt.\n\n"
    "Do NOT assume the candidate is right. Solve the question INDEPENDENTLY "
    "from the evidence, then decide.\n\n"
    + RULES +
    "\nReply in exactly this format:\n"
    "ANSWER: <the exact answer span>\n"
    "VERDICT: <KEEP if it equals the candidate, otherwise FIXED>"
)

# V3: independent solve + rules + explicit "the candidate may already be
# right, do not downgrade it"
V3 = (
    "You are a precision answer extractor for multi-hop QA. You are given a "
    "question, ranked evidence sentences, and a candidate answer from a "
    "previous attempt.\n\n"
    "Solve the question INDEPENDENTLY from the evidence, then compare your "
    "answer with the candidate.\n\n"
    "IMPORTANT: the candidate was produced by a model that already applied "
    "the rules below, and it is usually correct. Use it as a tie-breaker: if "
    "your independent answer is not CLEARLY better, keep the candidate.\n\n"
    + RULES +
    "\nReply in exactly this format:\n"
    "ANSWER: <the exact answer span>\n"
    "VERDICT: <KEEP if the candidate should stand, otherwise FIXED>"
)


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

    # Baseline: v3 first pass, no self-correction.
    base = {}
    for r in recs:
        ctx = "\n".join(e["text"] for e in (r.get("evidence") or []) if e.get("text"))
        base[r["qid"]] = ae.extract_answer(
            client, r["question"], ctx, variant="v3", self_correct=False,
        )

    designs = {"V1_current": V1, "V2_with_rules": V2, "V3_tiebreak": V3}
    results = {}
    for name, system in designs.items():
        answers = {}
        for r in recs:
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
            if not ans:
                answers[r["qid"]] = cand
            elif verdict.startswith("KEEP"):
                answers[r["qid"]] = cand
            elif verdict.startswith("FIXED") and ae._span_in_context(ans, ctx):
                answers[r["qid"]] = ans
            else:
                answers[r["qid"]] = cand
        results[name] = answers

    print("=" * 78)
    print(f"n={len(recs)}  v3 first-pass EM="
          f"{sum(_em(base[r['qid']], r['gold']) for r in recs)/len(recs):.3f}")
    print("=" * 78)
    for name, answers in results.items():
        ems = [_em(answers[r["qid"]], r["gold"]) for r in recs]
        f1s = [_f1(answers[r["qid"]], r["gold"]) for r in recs]
        fixed = broken = 0
        lines = []
        for r in recs:
            was = _em(base[r["qid"]], r["gold"])
            now = _em(answers[r["qid"]], r["gold"])
            if not was and now:
                fixed += 1
                lines.append(f"    FIXED {r['gold']!r} <- {answers[r['qid']]!r}")
            elif was and not now:
                broken += 1
                lines.append(f"    BROKE {r['gold']!r} -> {answers[r['qid']]!r}")
        print(f"  {name:<16} EM={sum(ems)/len(ems):.3f} "
              f"F1={sum(f1s)/len(f1s):.3f}  fixed={fixed} broken={broken}")
        for line in lines:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

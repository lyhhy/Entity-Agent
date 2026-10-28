"""Probe several self-correction designs + span-prompt phrasings.

Runs each candidate design over a fixed record set and reports, per
design: EM/F1 overall, how many EM-failures it FIXES, and how many
already-correct answers it BREAKS.  The fix/break split is the number
that matters — a design that fixes 0 and breaks 0 is a no-op, and one
that fixes 0 and breaks 2 is harmful.

Usage:
  python eval/_probe_selfcorrect.py --records eval/results/clean_results_attrgate2.json
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


# ── design B: independent grounded re-solve ──────────────────────────
B_SYSTEM = (
    "You are a precision answer extractor for multi-hop QA. Given a question "
    "and ranked evidence sentences, find the ONE evidence sentence that "
    "directly answers the question, then give the exact answer span from "
    "that sentence.\n\n"
    "Reply in exactly this format:\n"
    "SENTENCE: <the evidence sentence that answers the question>\n"
    "ANSWER: <the exact answer span, copied from that sentence>"
)

B_USER = (
    "Evidence sentences (ranked, best first):\n{context}\n\n"
    "Question: {question}\n\n"
    "Find the answering sentence and the exact span."
)

# ── design C: reflection, candidate shown, evidence quoted first ─────
C_SYSTEM = (
    "You are verifying a candidate answer against evidence.\n\n"
    "Step 1 — quote the evidence sentence that directly answers the question.\n"
    "Step 2 — compare it to the candidate.\n"
    "Step 3 — if the candidate is right, output KEEP. If it is wrong, output "
    "the correct span.\n\n"
    "Reply in exactly this format:\n"
    "SENTENCE: <the answering evidence sentence>\n"
    "VERDICT: <KEEP or the corrected answer span>"
)

C_USER = (
    "Evidence sentences (ranked, best first):\n{context}\n\n"
    "Question: {question}\n\n"
    "Candidate answer: {candidate}\n\n"
    "Quote the answering sentence, then give your verdict."
)

# ── design D: reflection, candidate NOT shown (no anchoring bias) ────
D_SYSTEM = (
    "You are a precision answer extractor for multi-hop QA. You are given a "
    "question and ranked evidence sentences, and a candidate answer from a "
    "previous attempt.\n\n"
    "Do NOT assume the candidate is right. Solve the question independently "
    "from the evidence, then decide.\n\n"
    "Reply in exactly this format:\n"
    "ANSWER: <the exact answer span>\n"
    "VERDICT: <KEEP if it equals the candidate, otherwise FIXED>"
)

D_USER = (
    "Evidence sentences (ranked, best first):\n{context}\n\n"
    "Question: {question}\n\n"
    "Previous attempt: {candidate}\n\n"
    "Solve independently, then compare with the previous attempt."
)


def parse_tagged(text: str, tag: str) -> str:
    m = re.search(rf"{tag}\s*:\s*(.+)", text or "")
    return m.group(1).strip() if m else ""


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

    # Baseline answers (v3, no self-correction).
    base = {}
    for r in recs:
        ctx = "\n".join(e["text"] for e in (r.get("evidence") or []) if e.get("text"))
        base[r["qid"]] = ae.extract_answer(
            client, r["question"], ctx, variant="v3", self_correct=False,
        )

    designs = {}
    for r in recs:
        ctx = "\n".join(e["text"] for e in (r.get("evidence") or []) if e.get("text"))
        cand = base[r["qid"]]

        def call(system, user):
            try:
                return ae.postprocess(client.chat(
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0.0, max_tokens=120,
                ))
            except Exception:
                return ""

        raw_b = call(B_SYSTEM, B_USER.format(context=ctx, question=r["question"]))
        ans_b = parse_tagged(raw_b, "ANSWER")

        raw_c = call(C_SYSTEM, C_USER.format(
            context=ctx, question=r["question"], candidate=cand))
        ans_c = parse_tagged(raw_c, "VERDICT")

        raw_d = call(D_SYSTEM, D_USER.format(
            context=ctx, question=r["question"], candidate=cand))
        ans_d = parse_tagged(raw_d, "ANSWER")
        verdict_d = parse_tagged(raw_d, "VERDICT").upper()

        designs.setdefault("B_grounded_resolve", {})[r["qid"]] = ans_b or cand
        designs.setdefault("C_reflect_quoted", {})[r["qid"]] = (
            "" if ans_c.upper().startswith("KEEP") else ans_c) or cand
        designs.setdefault("D_reflect_independent", {})[r["qid"]] = (
            ans_d if verdict_d.startswith("FIXED") and ans_d else cand)

    print("=" * 78)
    print(f"n={len(recs)}   baseline v3 EM="
          f"{sum(_em(base[r['qid']], r['gold']) for r in recs)/len(recs):.3f}")
    print("=" * 78)
    for name, answers in designs.items():
        ems = [_em(answers[r["qid"]], r["gold"]) for r in recs]
        f1s = [_f1(answers[r["qid"]], r["gold"]) for r in recs]
        fixed = broken = 0
        details = []
        for r in recs:
            was = _em(base[r["qid"]], r["gold"])
            now = _em(answers[r["qid"]], r["gold"])
            if not was and now:
                fixed += 1
                details.append(f"    FIXED   {r['gold']!r} <- {answers[r['qid']]!r}")
            elif was and not now:
                broken += 1
                details.append(f"    BROKE   {r['gold']!r} -> {answers[r['qid']]!r}")
        print(f"  {name:<24} EM={sum(ems)/len(ems):.3f} "
              f"F1={sum(f1s)/len(f1s):.3f}  fixed={fixed} broken={broken}")
        for d in details:
            print(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())

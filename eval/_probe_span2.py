"""Probe MINIMAL-DIFF span-granularity prompt variants.

Small models are sensitive to prompt length: a 9-rule prompt with a
long contract can dilute the signal that mattered.  This probes
v3 + ONE added line each, to find the smallest change that actually
moves the span-boundary failures.

Usage:
  python eval/_probe_span2.py --records eval/results/clean_results_attrgate2.json
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

    v3 = ae.ANSWER_PROMPT_VARIANTS["v3"]

    # Minimal diffs: insert ONE line into v3's rule 5 (comparison) area,
    # i.e. right before the final "Output ONLY the answer span" rule.
    def insert_before_output(prompt: str, extra: str) -> str:
        marker = "6. Output ONLY the answer span"
        i = prompt.find(marker)
        if i < 0:
            marker = "8. Output ONLY the answer span"
            i = prompt.find(marker)
        assert i > 0
        return prompt[:i] + extra + "\n" + prompt[i:]

    LINE_PREFIX = (
        "5b. SPAN GRANULARITY — return the answer at the exact extent the "
        "question asks for. Drop any series/edition prefix, honorific or "
        "title (\"UFC 50: The War of '04\" -> \"The War of '04\"; \"Sir "
        "George Cayley\" -> \"George Cayley\"), but keep units and full "
        "place strings (\"43,000 marchers\", not \"43,000\"). When the "
        "question asks for the NAME of something, return the NAME, not a "
        "phrase describing it (\"Conscription\", not \"requiring only men "
        "to register\").\n"
    )
    LINE_TERSE = (
        "5b. SPAN GRANULARITY — return the minimal span that answers the "
        "question and nothing more: no series/edition prefix, no honorific, "
        "no descriptive phrase in place of a name.\n"
    )
    LINE_EXAMPLE = (
        "5b. SPAN GRANULARITY — copy the answer at exactly the extent the "
        "question asks for. Examples: \"UFC 50: The War of '04\" -> \"The "
        "War of '04\"; \"Sir George Cayley\" -> \"George Cayley\"; \"43,000 "
        "marchers\" -> \"43,000 marchers\" (keep the unit).\n"
    )

    designs = {
        "v3 (baseline)": v3,
        "M1_full_line": insert_before_output(v3, LINE_PREFIX),
        "M2_terse_line": insert_before_output(v3, LINE_TERSE),
        "M3_examples": insert_before_output(v3, LINE_EXAMPLE),
    }

    answers_by = {}
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
                    temperature=0.0, max_tokens=40,
                ))
            except Exception:
                raw = ""
            answers[r["qid"]] = raw
        answers_by[name] = answers

    base = answers_by["v3 (baseline)"]
    print("=" * 78)
    print(f"n={len(recs)}")
    print("=" * 78)
    for name, answers in answers_by.items():
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
        print(f"  {name:<18} EM={sum(ems)/len(ems):.3f} "
              f"F1={sum(f1s)/len(f1s):.3f}  fixed={fixed} broken={broken}")
        if name != "v3 (baseline)":
            for line in lines:
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

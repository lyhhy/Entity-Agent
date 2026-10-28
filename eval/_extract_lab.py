"""Offline answer-extraction iteration harness.

Loads the evidence that `eval/quick20_clean.py` dumps into the records
JSON, re-runs ONLY the answer-extraction step against that fixed
evidence, and scores EM/F1.  This decouples extractor changes from
the (slow) retrieval pipeline.

Usage:
    python eval/_extract_lab.py --records <records.json> [...] \
        [--variant base|v1|v2] [--show-failures]
"""
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


# ── scoring (mirrors eval/quick20_eval.py) ───────────────────────────
def _normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def _em(pred: str, gold: str) -> int:
    return int(_normalize_answer(pred) == _normalize_answer(gold))


def _f1(pred: str, gold: str) -> float:
    p = _normalize_answer(pred).split()
    g = _normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec = ns / len(p)
    rec = ns / len(g)
    return 2 * prec * rec / (prec + rec)


# ── prompt variants ─────────────────────────────────────────────────
# The prompts themselves live in src/answer_extractor.py so that the
# live path and this harness build a byte-identical request.  A prompt
# that differs by even one punctuation mark shifts the answer
# distribution, which would make any measured delta meaningless.
def _variants():
    from src.answer_extractor import ANSWER_PROMPT_VARIANTS
    return sorted(ANSWER_PROMPT_VARIANTS)


def build_user_msg(question: str, evidence: list) -> str:
    lines = "\n".join(e["text"] for e in evidence if e["text"])
    from src.answer_extractor import build_user_message
    return build_user_message(question, lines)


def strip_answer(s: str) -> str:
    # Use the shared postprocess so the lab and the live path agree.
    from src.answer_extractor import postprocess
    return postprocess(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--variant", default="base", choices=_variants())
    ap.add_argument("--show-failures", action="store_true")
    args = ap.parse_args()

    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from src.llm_client import make_client
    from src.answer_extractor import (
        ANSWER_PROMPT_VARIANTS,
        extract_answer,
    )
    client = make_client()
    if not getattr(client, "enabled", False):
        print("[FATAL] LLM client disabled")
        raise SystemExit(1)

    if args.variant not in ANSWER_PROMPT_VARIANTS:
        print(f"[FATAL] unknown variant {args.variant!r}; "
              f"have {sorted(ANSWER_PROMPT_VARIANTS)}")
        raise SystemExit(1)
    system = ANSWER_PROMPT_VARIANTS[args.variant]

    all_recs = []
    for p in args.records:
        d = json.load(open(REPO / p, encoding="utf-8"))
        all_recs.extend(d["records"])

    ems, f1s, rows = [], [], []
    for r in all_recs:
        ev = r.get("evidence") or []
        if not ev:
            continue
        context = "\n".join(e["text"] for e in ev if e["text"])
        # Use the shared extractor so the lab and the live path apply the
        # same yes/no shortcut and the same abstention-retry logic.
        ans = extract_answer(
            client, r["question"], context, variant=args.variant,
        )
        em = _em(ans, r["gold"])
        f1 = _f1(ans, r["gold"])
        ems.append(em)
        f1s.append(f1)
        rows.append((r, ans, em, f1))

    n = len(ems)
    em_avg = sum(ems) / max(1, n)
    f1_avg = sum(f1s) / max(1, n)
    print("=" * 84)
    print(f"variant={args.variant}  n={n}  "
          f"EM={em_avg:.3f}  F1={f1_avg:.3f}")
    print("=" * 84)

    # Compare to the pipeline's own recorded answer.
    base_em = sum(r["em"] for r, _, _, _ in rows) / max(1, n)
    base_f1 = sum(r["f1"] for r, _, _, _ in rows) / max(1, n)
    print(f"  pipeline baseline : EM={base_em:.3f}  F1={base_f1:.3f}")
    print(f"  this variant      : EM={em_avg:.3f}  F1={f1_avg:.3f}")
    print(f"  delta             : EM={em_avg-base_em:+.3f}  "
          f"F1={f1_avg-base_f1:+.3f}")
    print()

    if args.show_failures:
        print("-" * 84)
        print("NON-EM CASES")
        print("-" * 84)
        for r, ans, em, f1 in rows:
            if em:
                continue
            tag = "PARTIAL" if f1 > 0 else "MISS"
            gold_in = any(e["is_gold"] for e in (r.get("evidence") or []))
            print(f"  [{tag}] {r['qid'][:8]} F1={f1:.3f} "
                  f"gold_in_evidence={gold_in}")
            print(f"    Q: {r['question']}")
            print(f"    G: {r['gold']}")
            print(f"    A: {ans}")
            print(f"    pipeline said: {r['answer']}")


if __name__ == "__main__":
    main()

"""A/B the answer-extraction changes on FIXED evidence.

Runs a matrix of (prompt variant) x (self-correct on/off) against the
evidence recorded in a records JSON, so the retrieval side is held
constant and only the answer stage moves.

Usage:
  python eval/_ab_matrix.py --records eval/results/clean_results_attrgate2.json \
      --variants v3 v5 --self-correct 0 1
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["v3", "v5"])
    ap.add_argument("--self-correct", nargs="+", default=["0", "1"])
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from src.llm_client import make_client
    from src.answer_extractor import extract_answer

    client = make_client()
    if not getattr(client, "enabled", False):
        print("[FATAL] LLM client disabled")
        return 1

    recs = []
    for p in args.records:
        recs.extend(json.load(open(REPO / p, encoding="utf-8"))["records"])

    # Baseline: what the pipeline itself recorded.
    b_em = sum(bool(r["em"]) for r in recs) / max(1, len(recs))
    b_f1 = sum(r["f1"] for r in recs) / max(1, len(recs))
    print("=" * 78)
    print(f"n={len(recs)}  pipeline-recorded baseline: EM={b_em:.3f} F1={b_f1:.3f}")
    print("=" * 78)

    results = {}
    for variant in args.variants:
        for sc in args.self_correct:
            sc_on = sc not in ("0", "false", "no", "off")
            answers = {}
            for r in recs:
                ev = r.get("evidence") or []
                context = "\n".join(e["text"] for e in ev if e.get("text"))
                answers[r["qid"]] = extract_answer(
                    client, r["question"], context,
                    variant=variant, self_correct=sc_on,
                )
            ems = [_em(answers[r["qid"]], r["gold"]) for r in recs]
            f1s = [_f1(answers[r["qid"]], r["gold"]) for r in recs]
            key = f"{variant}+sc{int(sc_on)}"
            results[key] = {
                "em": sum(ems) / max(1, len(ems)),
                "f1": sum(f1s) / max(1, len(f1s)),
                "answers": answers,
            }
            print(f"  {key:<12} EM={results[key]['em']:.3f}  "
                  f"F1={results[key]['f1']:.3f}  "
                  f"(dEM={results[key]['em']-b_em:+.3f} "
                  f"dF1={results[key]['f1']-b_f1:+.3f})")

    if args.show_failures:
        for key, res in results.items():
            print("-" * 78)
            print(f"NON-EM CASES — {key}")
            print("-" * 78)
            for r in recs:
                ans = res["answers"][r["qid"]]
                em = _em(ans, r["gold"])
                if em:
                    continue
                f1 = _f1(ans, r["gold"])
                tag = "PARTIAL" if f1 > 0 else "MISS"
                gold_in = any(e.get("is_gold") for e in (r.get("evidence") or []))
                print(f"  [{tag}] {r['qid'][:8]} F1={f1:.3f} "
                      f"gold_in_evidence={gold_in}")
                print(f"    Q: {r['question'][:120]}")
                print(f"    G: {r['gold']}")
                print(f"    A: {ans}")

    if args.out:
        out = {
            "n": len(recs),
            "baseline": {"em": b_em, "f1": b_f1},
            "variants": {
                k: {"em": v["em"], "f1": v["f1"], "answers": v["answers"]}
                for k, v in results.items()
            },
            "rows": [
                {"qid": r["qid"], "question": r["question"], "gold": r["gold"],
                 "pipeline_answer": r.get("answer", "")}
                for r in recs
            ],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

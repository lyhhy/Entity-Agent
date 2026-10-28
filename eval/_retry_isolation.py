"""
Isolate the abstention-retry rule from the prompt text.

Rationale: the "never abstain, always commit" rule is the only change
with a theoretical guarantee — a refusal scores exactly 0, while a
guess can score 0 or more, so it weakly dominates on any dataset whose
gold answers never contain a refusal.  The prompt *text* changes, by
contrast, are fitted to observed errors and may be overfitting.

This script therefore A/Bs the SAME prompt with retry on vs off.

Usage:
  python eval/_retry_isolation.py --records a.json b.json --variant base
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "eval"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    key = (_REPO_ROOT / "eval/api_key.txt").read_text(encoding="utf-8").strip()
    if not key:
        print("[FATAL] no key in eval/api_key.txt")
        return 1
    os.environ["DEEPSEEK_API_KEY"] = key
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from src.llm_client import DeepSeekClient
    from src.answer_extractor import extract_answer
    import quick20_eval as q20

    client = DeepSeekClient()
    if not getattr(client, "enabled", False):
        print("[FATAL] LLM client disabled")
        return 1

    recs = []
    for p in args.records:
        with open(p, encoding="utf-8") as f:
            recs.extend(json.load(f)["records"])

    res = {}
    for mode in ("retry_off", "retry_on"):
        answers = {}
        for r in recs:
            ev = r.get("evidence") or []
            context = "\n".join(e["text"] for e in ev if e.get("text"))
            answers[r["qid"]] = extract_answer(
                client, r["question"], context,
                variant=args.variant,
                force_choice_retry=(mode == "retry_on"),
            )
        ems = [q20._em_score(answers[r["qid"]], r["gold"]) for r in recs]
        f1s = [q20._f1_score(answers[r["qid"]], r["gold"]) for r in recs]
        res[mode] = {
            "em": sum(ems) / max(1, len(ems)),
            "f1": sum(f1s) / max(1, len(f1s)),
            "answers": answers,
        }
        print(f"{mode:>10}  EM={res[mode]['em']:.3f}  F1={res[mode]['f1']:.3f}")

    changed = [
        qid for qid in res["retry_off"]["answers"]
        if res["retry_off"]["answers"][qid] != res["retry_on"]["answers"][qid]
    ]
    print(f"  answers changed by retry: {len(changed)}/{len(recs)}")
    for qid in changed:
        r = next(x for x in recs if x["qid"] == qid)
        print(f"    {qid[:8]} G={r['gold']!r}")
        print(f"        off -> {res['retry_off']['answers'][qid]!r}")
        print(f"        on  -> {res['retry_on']['answers'][qid]!r}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"n": len(recs), "variant": args.variant, **res},
                      f, ensure_ascii=False, indent=2)
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

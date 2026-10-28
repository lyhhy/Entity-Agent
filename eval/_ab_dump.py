"""
A/B two prompt variants on the same records, dumping per-question answers.

Purpose: distinguish two very different explanations for "no delta":
  (a) the variant genuinely produces different answers that score the
      same  -> the tuning-set gain was overfitting;
  (b) the variant produced IDENTICAL answers            -> the variant
      never reached the LLM, i.e. a harness bug.

Usage:
  python eval/_ab_dump.py --records a.json b.json \
      --variants base v3 --out eval/results/ab_heldout.json
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
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", required=True)
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

    out: dict = {"n": len(recs), "variants": {}, "rows": []}
    for r in recs:
        ev = r.get("evidence") or []
        context = "\n".join(e["text"] for e in ev if e.get("text"))
        out["rows"].append({
            "qid": r["qid"],
            "question": r["question"],
            "gold": r["gold"],
            "pipeline_answer": r.get("answer", ""),
            "gold_in_evidence": any(e.get("is_gold") for e in ev),
            "n_evidence": len(ev),
        })

    for variant in args.variants:
        answers = {}
        for r in recs:
            ev = r.get("evidence") or []
            context = "\n".join(e["text"] for e in ev if e.get("text"))
            answers[r["qid"]] = extract_answer(
                client, r["question"], context, variant=variant,
            )
        ems = [q20._em_score(answers[r["qid"]], r["gold"]) for r in recs]
        f1s = [q20._f1_score(answers[r["qid"]], r["gold"]) for r in recs]
        out["variants"][variant] = {
            "em": sum(ems) / max(1, len(ems)),
            "f1": sum(f1s) / max(1, len(f1s)),
            "answers": answers,
        }
        print(f"{variant:>6}  EM={out['variants'][variant]['em']:.3f}  "
              f"F1={out['variants'][variant]['f1']:.3f}")

    # How many questions actually changed their answer?
    vs = args.variants
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a, b = vs[i], vs[j]
            diff = [
                qid for qid in out["variants"][a]["answers"]
                if out["variants"][a]["answers"][qid]
                != out["variants"][b]["answers"][qid]
            ]
            out.setdefault("diffs", {})[f"{a}_vs_{b}"] = {
                "n_changed": len(diff),
                "changed_qids": diff,
            }
            print(f"  {a} vs {b}: {len(diff)}/{len(recs)} answers differ")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

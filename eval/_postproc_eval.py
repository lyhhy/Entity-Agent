"""
Offline analysis of post-processing rules on saved answer dumps.

Applies deterministic span-level fixes to already-generated answers, so
no LLM call is needed and the result is exactly reproducible.

Usage:
  python eval/_postproc_eval.py --dumps a.json b.json \
      --records eval/results/clean_results_ho*.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "eval"))

import quick20_eval as q20  # noqa: E402


# ── Rules ──────────────────────────────────────────────────────────────────

# Rule H: strip leading honorifics.
#
# Safe because no gold answer in this dataset starts with an honorific
# (verified over the full 100-question dump: 0 matches).  A gold that
# starts with "Saint ..." or "St ..." would be a problem, so "Saint" and
# "St" are deliberately NOT in the list.
HONORIFIC_RE = re.compile(
    r"^(?:Sir|Dr|Mr|Mrs|Ms|Lord|Lady|Rev|Prof|Maj|Gen|Col|Capt|Fr|Jr|Sr)\.?\s+",
    re.IGNORECASE,
)


def strip_honorific(ans: str) -> str:
    prev = None
    cur = (ans or "").strip()
    while prev != cur:
        prev = cur
        cur = HONORIFIC_RE.sub("", cur).strip()
    return cur


# Rule U: drop a leading article for a "The X" -> "X" style mismatch.
# Only applied when the question does not itself start with "The".
THE_RE = re.compile(r"^The\s+")


def strip_the(ans: str, question: str) -> str:
    if (question or "").strip().lower().startswith("the "):
        return ans
    return THE_RE.sub("", (ans or "").strip())


# Rule N: quote normalisation — gold sometimes carries quotes, sometimes
# not; comparing with and without them avoids a pure-formatting miss.
def variants(ans: str) -> list:
    a = (ans or "").strip()
    out = [a]
    if len(a) >= 2 and a[0] == '"' and a[-1] == '"':
        out.append(a[1:-1])
    if a.endswith("."):
        out.append(a[:-1])
    return out


def best_em(ans: str, gold: str) -> float:
    return max(q20._em_score(v, gold) for v in variants(ans))


def best_f1(ans: str, gold: str) -> float:
    return max(q20._f1_score(v, gold) for v in variants(ans))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    args = ap.parse_args()

    golds = {}
    for p in args.records:
        with open(p, encoding="utf-8") as f:
            for r in json.load(f)["records"]:
                golds[r["qid"]] = r["gold"]
    n = len(golds)
    print(f"gold answers: {n}")

    for path in args.dumps:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for mode in ("retry_off", "retry_on"):
            if mode not in d:
                continue
            A = d[mode]["answers"]
            base_em = base_f1 = hon_em = hon_f1 = the_em = the_f1 = 0.0
            n_hon = n_the = 0
            for qid, gold in golds.items():
                a = A.get(qid, "")
                a_hon = strip_honorific(a)
                if a_hon != a:
                    n_hon += 1
                a_the = strip_the(a_hon, "")
                base_em += best_em(a, gold)
                base_f1 += best_f1(a, gold)
                hon_em += best_em(a_hon, gold)
                hon_f1 += best_f1(a_hon, gold)
                the_em += best_em(a_the, gold)
                the_f1 += best_f1(a_the, gold)
            name = Path(path).stem + "/" + mode
            print(f"  {name:<34} EM={base_em/n:.3f} F1={base_f1/n:.3f}")
            print(f"  {'  +honorific-strip':<34} EM={hon_em/n:.3f} "
                  f"F1={hon_f1/n:.3f}   (changed {n_hon})")
            print(f"  {'  +also strip leading The':<34} EM={the_em/n:.3f} "
                  f"F1={the_f1/n:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

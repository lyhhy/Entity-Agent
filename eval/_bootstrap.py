"""
Paired bootstrap over questions for a prompt-variant comparison.

A raw EM delta is not enough: the same configuration re-run differs by
~0.02 EM (measured on 160 held-out questions), so a delta has to be
compared against that noise, not against zero.

This script does a paired bootstrap: for each resample it draws
questions WITH replacement, and computes the delta EM/F1 of each variant
against the baseline on that same resample.  The reported interval is
the 2.5/97.5 percentile of the delta distribution.

Because the comparison is paired (same questions, same evidence, same
call), this is far more sensitive than comparing two independent runs.

Usage:
  python eval/_bootstrap.py --ab eval/results/ab_ho160.json \
      --baseline base --variants v1 v2 v3 --iters 10000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "eval"))

import quick20_eval as q20  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", nargs="+", required=True)
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    for path in args.ab:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        rows = {r["qid"]: r for r in d["rows"]}
        qids = list(rows.keys())
        n = len(qids)
        base = args.baseline
        variants = args.variants or [v for v in d["variants"] if v != base]

        # Pre-compute per-question scores once (cheap), then resample.
        ems = {v: [q20._em_score(d["variants"][v]["answers"][q], rows[q]["gold"])
                   for q in qids] for v in d["variants"]}
        f1s = {v: [q20._f1_score(d["variants"][v]["answers"][q], rows[q]["gold"])
                   for q in qids] for v in d["variants"]}

        print("=" * 78)
        print(f"{Path(path).name}  n={n}  iters={args.iters}  "
              f"baseline={base}")
        print("=" * 78)

        for v in variants:
            deltas_em = []
            deltas_f1 = []
            for _ in range(args.iters):
                idx = [rng.randrange(n) for _ in range(n)]
                de = (sum(ems[v][i] for i in idx)
                      - sum(ems[base][i] for i in idx)) / n
                df = (sum(f1s[v][i] for i in idx)
                      - sum(f1s[base][i] for i in idx)) / n
                deltas_em.append(de)
                deltas_f1.append(df)
            deltas_em.sort()
            deltas_f1.sort()

            def pct(xs, p):
                return xs[min(len(xs) - 1, max(0, int(p * len(xs))))]

            point_em = (sum(ems[v]) - sum(ems[base])) / n
            point_f1 = (sum(f1s[v]) - sum(f1s[base])) / n
            lo_em, hi_em = pct(deltas_em, 0.025), pct(deltas_em, 0.975)
            lo_f1, hi_f1 = pct(deltas_f1, 0.025), pct(deltas_f1, 0.975)
            sig = "SIG" if (lo_em > 0 or hi_em < 0) else "ns "
            print(f"  {v:>5} vs {base}:  dEM={point_em:+.4f} "
                  f"[{lo_em:+.4f}, {hi_em:+.4f}] {sig}   "
                  f"dF1={point_f1:+.4f} [{lo_f1:+.4f}, {hi_f1:+.4f}]")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

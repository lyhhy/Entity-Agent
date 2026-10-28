"""
Stratified comparison of prompt variants.

The overall EM on the full set is dominated by retrieval failures
(questions whose gold string is not present in the evidence at all) —
those cannot be fixed by any prompt, so they dilute the signal.

This script splits the set into:
  * "reachable": gold IS present in the evidence  -> prompt can matter
  * "unreachable": gold is NOT in the evidence     -> retrieval-limited

and reports EM/F1 for each variant on each stratum.

Usage:
  python eval/_stratify.py --ab eval/results/ab_ho160.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "eval"))

import quick20_eval as q20  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", nargs="+", required=True,
                    help="one or more _ab_dump.py output files")
    args = ap.parse_args()

    for path in args.ab:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        rows = {r["qid"]: r for r in d["rows"]}
        variants = list(d["variants"].keys())

        print("=" * 78)
        print(f"{Path(path).name}   n={d['n']}")
        print("=" * 78)

        # Base row for deltas.
        base = variants[0]
        base_ans = d["variants"][base]["answers"]

        strata = {
            "reachable  ": lambda r: r["gold_in_evidence"],
            "unreachable": lambda r: not r["gold_in_evidence"],
        }

        for sname, pred in strata.items():
            sub = [qid for qid, r in rows.items() if pred(r)]
            print(f"\n  [{sname}] n={len(sub)}")
            for v in variants:
                A = d["variants"][v]["answers"]
                ems = [q20._em_score(A[q], rows[q]["gold"]) for q in sub]
                f1s = [q20._f1_score(A[q], rows[q]["gold"]) for q in sub]
                n = max(1, len(sub))
                b_ems = [q20._em_score(base_ans[q], rows[q]["gold"]) for q in sub]
                b_f1s = [q20._f1_score(base_ans[q], rows[q]["gold"]) for q in sub]
                dem = sum(ems) / n - sum(b_ems) / n
                df1 = sum(f1s) / n - sum(b_f1s) / n
                tag = "" if v == base else f"   dEM={dem:+.3f} dF1={df1:+.3f}"
                print(f"    {v:>6}  EM={sum(ems)/n:.3f}  F1={sum(f1s)/n:.3f}{tag}")

        # Overall
        print(f"\n  [all      ] n={d['n']}")
        for v in variants:
            A = d["variants"][v]["answers"]
            ems = [q20._em_score(A[q], rows[q]["gold"]) for q in rows]
            f1s = [q20._f1_score(A[q], rows[q]["gold"]) for q in rows]
            n = max(1, len(rows))
            b_ems = [q20._em_score(base_ans[q], rows[q]["gold"]) for q in rows]
            b_f1s = [q20._f1_score(base_ans[q], rows[q]["gold"]) for q in rows]
            dem = sum(ems) / n - sum(b_ems) / n
            df1 = sum(f1s) / n - sum(b_f1s) / n
            tag = "" if v == base else f"   dEM={dem:+.3f} dF1={df1:+.3f}"
            print(f"    {v:>6}  EM={sum(ems)/n:.3f}  F1={sum(f1s)/n:.3f}{tag}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

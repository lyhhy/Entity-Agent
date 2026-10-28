"""Dump the ranked evidence for specific qids so the extractor's
failure modes can be inspected against the actual sentences.

Usage:
    python eval/_show_evidence.py <qid8> [<qid8> ...]
       --records <records.json> [...]
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qids", nargs="+")
    ap.add_argument("--records", nargs="+", required=True)
    args = ap.parse_args()

    want = set(args.qids)
    for p in args.records:
        d = json.load(open(REPO / p, encoding="utf-8"))
        for r in d["records"]:
            if r["qid"][:8] not in want:
                continue
            print("=" * 88)
            print(f"{r['qid'][:8]}   {p.split('clean_results_')[-1]}")
            print("=" * 88)
            print(f"Q : {r['question']}")
            print(f"G : {r['gold']}")
            print(f"A : {r['answer']}   (F1={r['f1']:.3f})")
            print("-" * 88)
            for e in r.get("evidence") or []:
                mark = "<<<GOLD" if e["is_gold"] else "       "
                print(f"  #{e['rank']:>2} {mark} {e['text']}")
            print()


if __name__ == "__main__":
    main()

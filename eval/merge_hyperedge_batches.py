"""
Merge multiple offline-hyperedge Parquet files into one.

Usage:
    python eval/merge_hyperedge_batches.py \\
        --inputs batch1.parquet batch2.parquet batch3.parquet batch4.parquet \\
        --out eval/offline_hyperedges/hyperedges.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge offline-hyperedge Parquet shards.",
    )
    ap.add_argument(
        "--inputs", nargs="+", required=True,
        help="Input Parquet files (in order).",
    )
    ap.add_argument(
        "--out", required=True,
        help="Output Parquet path (e.g. hyperedges.parquet).",
    )
    args = ap.parse_args()

    frames: List[pd.DataFrame] = []
    for inp in args.inputs:
        if not Path(inp).exists():
            print(f"[WARN] missing {inp}, skipping")
            continue
        df = pd.read_parquet(inp)
        print(f"  loaded {inp}: {len(df)} rows / {df['qid'].nunique()} qids")
        frames.append(df)
    if not frames:
        print("No input files; aborting.")
        return 1

    merged = pd.concat(frames, ignore_index=True)
    # Drop any duplicate (qid, paragraph_idx) — last wins.
    merged = merged.drop_duplicates(
        subset=["qid", "paragraph_idx"], keep="last",
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(
        args.out, engine="pyarrow", compression="zstd", index=False,
    )
    summary = {
        "n_questions": int(merged["qid"].nunique()),
        "n_paragraphs": int(len(merged)),
        "n_containers_total": int(merged["container_ids"].apply(len).sum()),
        "avg_paragraphs_per_question": float(
            merged.groupby("qid").size().mean(),
        ),
        "avg_containers_per_paragraph": float(
            merged["n_containers"].mean(),
        ),
    }
    summary_path = Path(args.out).with_name("summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"Merged: {len(merged)} rows → {args.out}")
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Sequentially run offline hyperedge build batches 3 and 4.

Batch 1 (0-1850)  -- DONE
Batch 2 (1850-3700) -- currently running (started manually)
This script polls for batch 2 completion, then runs:
  batch3: 3700-5550
  batch4: 5550-7405
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path(r"c:\Users\zsf\Desktop\lyh\StenRAG")
LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)

WAIT_FOR = "batch2"  # don't start batch3 until batch2 is finished
BATCHES = [
    (3700, 5550, "batch3"),
    (5550, 7405, "batch4"),
]


def already_done(suffix: str) -> bool:
    pq = REPO / "eval" / "offline_hyperedges" / f"hyperedges_{suffix}.parquet"
    sm = REPO / "eval" / "offline_hyperedges" / f"summary_{suffix}.json"
    return pq.exists() and sm.exists()


def run_one(start: int, end: int, suffix: str) -> int:
    if already_done(suffix):
        print(f"[skip] {suffix} already done")
        return 0
    log_path = LOG_DIR / f"build_{suffix}.log"
    cmd = (
        f"cd /d {REPO} && "
        f"python eval\\build_hyperedges_offline.py "
        f"--start {start} --end {end} "
        f"--out-dir eval\\offline_hyperedges --out-suffix {suffix}"
    )
    print(f"[start] {suffix} ({start}-{end}) -> log={log_path}")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            ["cmd", "/c", cmd],
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        rc = proc.wait()
    elapsed = time.time() - t0
    print(f"[done ] {suffix} rc={rc} elapsed={elapsed:.0f}s")
    return rc


def wait_for(suffix: str, poll_secs: int = 60) -> None:
    pq = REPO / "eval" / "offline_hyperedges" / f"hyperedges_{suffix}.parquet"
    sm = REPO / "eval" / "offline_hyperedges" / f"summary_{suffix}.json"
    print(f"[wait ] {suffix} -> {pq.name}")
    while not (pq.exists() and sm.exists()):
        time.sleep(poll_secs)
    print(f"[ready] {suffix} finished.")


def main() -> int:
    overall_t0 = time.time()
    wait_for(WAIT_FOR)
    for start, end, suffix in BATCHES:
        rc = run_one(start, end, suffix)
        if rc != 0:
            print(f"[fail ] {suffix} returned {rc}; aborting chain")
            return rc
    print(f"[all done] total elapsed = {time.time() - overall_t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

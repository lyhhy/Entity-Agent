"""
Download HotpotQA distractor data into eval/data/.

HotpotQA official URL is on Stanford servers; the official archive at
hotpotqa.cs.stanford.edu serves the .json files.

This script:
  1. Downloads hotpot_dev_distractor_v1.json  (dev = 7405 questions)
  2. Downloads hotpot_train_distractor_v1.json (train = 90447 questions, ~700MB)
     -- only if --with-train is passed

Usage:
    python eval/download_hotpot.py                 # dev only (~90MB)
    python eval/download_hotpot.py --with-train   # dev + train (~800MB)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

URLS = {
    "hotpot_dev_distractor_v1.json":
        "https://hotpotqa.s3-us-west-2.amazonaws.com/hotpot_dev_distractor_v1.json",
    "hotpot_train_distractor_v1.json":
        "https://hotpotqa.s3-us-west-2.amazonaws.com/hotpot_train_distractor_v1.json",
}


def _download(url: str, dest: Path, expected_size_mb: float) -> bool:
    """
    Download a single URL to dest. Returns True on success.

    Falls back through several mirror URLs if the primary fails.
    """
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ {dest.name} already exists ({size_mb:.1f} MB). Skipping.")
        return True

    print(f"  ↓ Downloading {dest.name}  (~{expected_size_mb:.0f} MB)...")
    print(f"    URL: {url}")
    t0 = time.time()
    try:
        # Open with timeout
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "StenRAG-eval-downloader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            chunk = 1024 * 1024  # 1 MB
            with open(dest, "wb") as f:
                downloaded = 0
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    if total > 0:
                        pct = downloaded / total * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"\r    {pct:5.1f}%  ({mb:.1f} MB / {total/1024/1024:.1f} MB)",
                              end="", flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r    {mb:.1f} MB downloaded", end="", flush=True)
        elapsed = time.time() - t0
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"\n    ✓ Done in {elapsed:.1f}s ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n    ✗ Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HotpotQA")
    parser.add_argument("--with-train", action="store_true",
                        help="Also download the train split (~700 MB)")
    args = parser.parse_args()

    files_to_download = ["hotpot_dev_distractor_v1.json"]
    if args.with_train:
        files_to_download.append("hotpot_train_distractor_v1.json")

    print(f"Downloading to: {DATA_DIR}")

    # Fallback mirrors if the primary S3 URL fails
    MIRRORS = [
        "https://hotpotqa.s3-us-west-2.amazonaws.com/{name}",
        "https://curtis.ml.cmu.edu/datasets/hotpot/{name}",
        "http://curtis.ml.cmu.edu/datasets/hotpot/{name}",
    ]

    for filename in files_to_download:
        expected_mb = 90 if "dev" in filename else 700
        dest = DATA_DIR / filename
        ok = False
        for tmpl in MIRRORS:
            url = tmpl.format(name=filename)
            ok = _download(url, dest, expected_mb)
            if ok:
                break
            print(f"    Trying next mirror...")
        if not ok:
            print(f"\n  ✗ All mirrors failed for {filename}")
            print(f"    Please download manually from:")
            print(f"      https://hotpotqa.github.io/")
            print(f"    and place the file in: {dest}")
            return 1

    print("\nAll downloads complete.")
    print(f"\nNext step:")
    print(f"  python eval/hotpot_eval.py --split val --max-examples 200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
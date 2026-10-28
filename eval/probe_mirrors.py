"""
Probe HotpotQA mirrors to find the live download URL.
"""

import urllib.request
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def probe(url, timeout=15):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Length"), len(r.read())
    except Exception as e:
        return f"ERR: {type(e).__name__}: {e}"[:200]


# Step 1: Find download links from the official page
print("=" * 60)
print("Step 1: scraping hotpotqa.github.io for download links")
print("=" * 60)
html = urllib.request.urlopen(
    urllib.request.Request("https://hotpotqa.github.io/", headers={"User-Agent": UA}),
    timeout=15,
).read().decode("utf-8", "ignore")
urls = set(re.findall(r"https?://[^\s\"'<>]+\.(?:json|zip)", html))
for u in sorted(urls):
    print(f"  found: {u}")

# Step 2: Probe candidate mirrors
print("\n" + "=" * 60)
print("Step 2: probing candidate URLs")
print("=" * 60)

CANDIDATES = [
    # Common mirrors (dev distractor is the smallest, ~90 MB)
    "https://hotpotqa.s3-us-west-2.amazonaws.com/hotpot_dev_distractor_v1.json",
    "https://hotpotqa.s3.amazonaws.com/hotpot_dev_distractor_v1.json",
    "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "https://huggingface.co/datasets/hotpot_qa/resolve/main/hotpot_dev_distractor_v1.json",
    "https://github.com/hotpotqa/hotpot/raw/master/hotpot_dev_distractor_v1.json",
    # Fallback HuggingFace dataset hub path
    "https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_dev_distractor_v1.json",
    # Stanford mirror (sometimes serves via http)
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    # HF datasets with revised loader path
    "https://huggingface.co/datasets/jenhil/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json",
    "https://huggingface.co/datasets/lukaemon/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json",
]

for url in CANDIDATES:
    print(f"\n  URL: {url}")
    print(f"    -> {probe(url)}")
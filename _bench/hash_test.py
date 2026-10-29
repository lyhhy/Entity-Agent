"""Pin PYTHONHASHSEED and re-run to test whether the n_he flip is
hash-order nondeterminism rather than a regression from the deletions."""
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "eval" / "data" / "hotpot_dev_distractor_v1.json"

CHILD = r'''
import json, os, random, sys, statistics, logging
from pathlib import Path
ROOT = Path(r"{root}")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
os.environ.setdefault("TRANSFORMERS_OFFLINE","1"); os.environ.setdefault("HF_HUB_OFFLINE","1")
logging.disable(logging.CRITICAL)
import quick20_eval as q20
from src.graph_builder import _load_spacy_shared
from src.dispatcher import build_dispatcher
from src.hyper_edge import HyperEdgeRegistry

data = json.load(open(r"{data}", encoding="utf-8"))
rng = random.Random({seed})
idx = sorted(rng.sample(range(len(data)), 20))
sample = [data[i] for i in idx]
nlp = _load_spacy_shared("en_core_web_lg")
out = []
for qdata in sample:
    builder, sentences, titles, para_sents, bridge_titles = q20.build_graph_from_hotpot(qdata, nlp)
    cm = builder.container_manager
    for c in cm.get_all_containers(): cm.update_container_index(c.id)
    builder.compute_tfidf_index()
    he = HyperEdgeRegistry(container_manager=cm)
    for _pt, _pc in (bridge_titles or {{}}).items():
        cs = [cm._containers.get(_c) for _c in _pc if cm._containers.get(_c) is not None]
        if cs:
            he.build_hyperedges_for_chunk(chunk_id=str(_pt), containers=cs, time_word="", paragraph_title=str(_pt))
    disp = build_dispatcher(builder, client=None, nlp=nlp, hyperedge_registry=he)
    res = disp.run(qdata["question"], bridge_titles=bridge_titles)
    pr = res.pipeline_result
    out.append({{"qid": qdata["_id"], "n_he": len(pr.step3_hyperedges),
                "n_total_he": len(he), "pool": pr.step6_pool_size,
                "n_final": len(pr.step10_final_sids)}})
print("@@JSON@@" + json.dumps(out))
'''

def run(seed, hashseed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    code = CHILD.format(root=ROOT, data=DATA, seed=seed)
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in r.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    print(r.stdout[-2000:]); print(r.stderr[-3000:])
    raise SystemExit("child failed")

print("PYTHONHASHSEED pinning test (seed=7, 20 questions)")
print("=" * 78)
results = {}
for hs in (0, 1, 2, 3):
    res = run(7, hs)
    results[hs] = {r["qid"]: r for r in res}
    n_he = sum(r["n_he"] for r in res)
    print(f"  PYTHONHASHSEED={hs}: total step3 hyperedges = {n_he}")

print("-" * 78)
base = results[0]
same = True
for hs in (1, 2, 3):
    for qid in base:
        a, b = base[qid], results[hs][qid]
        if a != b:
            same = False
            print(f"  DIFF seed={hs} {qid}: {a} vs {b}")
print("\nAll pinned-hash runs identical:", same)
print("=> n_he is hash-order dependent" if not same else "=> n_he is stable under hash pinning")

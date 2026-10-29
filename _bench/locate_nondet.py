"""Locate exactly which pipeline stage is hash-order dependent.

Runs the SAME question under different PYTHONHASHSEED values in separate
processes and dumps every intermediate stage, so we can see which one
first diverges.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "eval" / "data" / "hotpot_dev_distractor_v1.json"
QID = "5ab9d2e255429955dce3ed90"

CHILD = r'''
import json, os, random, sys, logging
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
target = [d for d in data if d["_id"] == "{qid}"][0]
nlp = _load_spacy_shared("en_core_web_lg")

builder, sentences, titles, para_sents, bridge_titles = q20.build_graph_from_hotpot(target, nlp)
cm = builder.container_manager
for c in cm.get_all_containers(): cm.update_container_index(c.id)
builder.compute_tfidf_index()
he = HyperEdgeRegistry(container_manager=cm)
for _pt, _pc in (bridge_titles or {{}}).items():
    cs = [cm._containers.get(_c) for _c in _pc if cm._containers.get(_c) is not None]
    if cs:
        he.build_hyperedges_for_chunk(chunk_id=str(_pt), containers=cs, time_word="", paragraph_title=str(_pt))
disp = build_dispatcher(builder, client=None, nlp=nlp, hyperedge_registry=he)
p = disp.pipeline
q = target["question"]

ents = p._step1_query_analysis(q)
a = p._step2_recall_a_containers(q, ents)
bt, bc = p._step2_5_bridge_titles(q, a, bridge_titles) if bridge_titles else ([], [])
a2 = list(a) + [c for c in bc if c not in a]
h = p._step3_hyperedges_for_containers(a2)
b = p._step4_recall_b_containers(q, h, a2, bridge_titles=bridge_titles)

print("@@JSON@@" + json.dumps({{
  "entities": ents,
  "a": a, "bt": bt, "bc": bc, "a2": a2,
  "hyperedges": h,
  "b": b,
  "he_container_map": {{k: sorted(v) for k, v in
      ((cid, [e.id for e in he.get_hyperedges_for_container(cid)])
       for cid in [c.id for c in cm.get_all_containers()])}},
}}))
'''


def run(hashseed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    code = CHILD.format(root=ROOT, data=DATA, qid=QID)
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in r.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    print(r.stdout[-1500:]); print(r.stderr[-2500:])
    raise SystemExit("child failed")


res = {hs: run(hs) for hs in (0, 1, 2, 3, 4)}

print("Comparing stages across PYTHONHASHSEED values")
print("=" * 78)
for stage in ("entities", "a", "bt", "bc", "a2", "hyperedges", "b"):
    vals = [json.dumps(res[hs][stage]) for hs in (0, 1, 2, 3, 4)]
    same = len(set(vals)) == 1
    print(f"\n  {stage:<12} {'SAME' if same else '*** DIFFERS ***'}")
    if not same:
        for hs in (0, 1, 2, 3, 4):
            print(f"      seed={hs}: {res[hs][stage]}")

print("\n" + "=" * 78)
print("container -> hyperedge map (this is built at graph time)")
for hs in (0, 1, 2, 3, 4):
    m = res[hs]["he_container_map"]
    sig = json.dumps({k: v for k, v in sorted(m.items()) if v})
    print(f"  seed={hs}: {sig[:300]}")

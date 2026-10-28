"""
v8 失败诊断: 对每题精确判定 gold 句是否在池里 / 容器是否在 A / gold 容器分裂情况.
跑一题单次拿到全部 step 状态, 不重复跑.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

# Force HF offline.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import logging
logging.basicConfig(level=logging.WARNING)

import argparse
from src.dispatcher import build_dispatcher
from src.hyper_edge import HyperEdgeRegistry


def _find_gold_sids(gold: str, builder) -> list:
    """Find sentence IDs whose text contains the gold answer (normalized)."""
    g = (gold or "").strip().lower()
    if not g:
        return []
    # Strip punctuation for matching.
    import re
    g_norm = re.sub(r"[^a-z0-9 ]+", " ", g).strip()
    if not g_norm:
        return []
    candidates = []
    for sid, node in builder.sentence_registry._nodes.items():
        text = (getattr(node, "text", "") or "").lower()
        text_norm = re.sub(r"[^a-z0-9 ]+", " ", text)
        # Word-boundary match on normalized text.
        if re.search(rf"\b{re.escape(g_norm)}\b", text_norm):
            candidates.append(sid)
    return candidates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=137)
    ap.add_argument("--data", default="eval/data/hotpot_dev_distractor_v1.json")
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument(
        "--offline-hyperedges",
        default="eval/offline_hyperedges/hyperedges.parquet",
    )
    ap.add_argument("--no-bridge", action="store_true")
    ap.add_argument("--no-gap-expand", action="store_true")
    args = ap.parse_args()

    import random
    import time
    from eval.quick20_eval import build_graph_from_hotpot as _bgh

    rng = random.Random(args.seed)
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    sample = data[: args.n]
    rng.shuffle(sample)

    nlp = None
    try:
        import spacy
        nlp = spacy.load("en_core_web_lg")
    except Exception:
        pass

    # Optional: build offline hyperedge store.
    offline_store = None
    if args.offline_hyperedges and Path(args.offline_hyperedges).exists():
        try:
            from src.offline_hyperedge_store import OfflineHyperedgeStore
            offline_store = OfflineHyperedgeStore.load(args.offline_hyperedges)
        except Exception as e:
            print(f"[warn] offline store load failed: {e}")

    summary = {
        "in_pool": 0,        # gold 句在 step6 pool
        "not_in_pool": 0,    # gold 句不在 step6 pool
        "in_A": 0,           # gold 容器在 A
        "not_in_A": 0,       # gold 容器不在 A
        "in_L2": 0,          # gold 容器在 L2
        "not_in_L2": 0,
        "split": 0,          # 容器分裂
    }
    cases = []  # 每题的详细诊断

    for i, qdata in enumerate(sample):
        qid = qdata["_id"]
        question = qdata["question"]
        gold = qdata["answer"]

        try:
            builder, sentences, titles, para_sents, bridge_titles = (
                _bgh(qdata, nlp)
            )
            cm = builder.container_manager
            for c in cm.get_all_containers():
                cm.update_container_index(c.id)
            builder.compute_tfidf_index()

            he = HyperEdgeRegistry(container_manager=cm)
            if offline_store is not None:
                para_cid_map = {}
                for p_title in titles:
                    cids = offline_store.get_containers_for_paragraph(qid, p_title)
                    if not cids:
                        cids = (bridge_titles or {}).get(p_title.lower().strip(), [])
                    if cids:
                        para_cid_map[p_title] = cids
                for p_title, cid_list in para_cid_map.items():
                    p_containers = [
                        cm._containers.get(cid) for cid in cid_list
                        if cm._containers.get(cid) is not None
                    ]
                    if p_containers:
                        he.build_hyperedges_for_chunk(
                            chunk_id=str(p_title),
                            containers=p_containers,
                            time_word="",
                            paragraph_title=str(p_title),
                        )
            else:
                for p_title, p_cids in (bridge_titles or {}).items():
                    p_containers = [
                        cm._containers.get(cid) for cid in p_cids
                        if cm._containers.get(cid) is not None
                    ]
                    if p_containers:
                        he.build_hyperedges_for_chunk(
                            chunk_id=str(p_title),
                            containers=p_containers,
                            time_word="",
                            paragraph_title=str(p_title),
                        )

            disp = build_dispatcher(
                builder, client=None, nlp=nlp,
                hyperedge_registry=he, strict_entity_match=False,
            )
            if args.no_gap_expand:
                disp.pipeline.gap_detector = None

            t0 = time.time()
            res = disp.run(question, bridge_titles=None if args.no_bridge else bridge_titles)
            latency = (time.time() - t0) * 1000
            pr = res.pipeline_result
        except Exception as e:
            print(f"Q{i+1} {qid}: ERROR {e}")
            continue

        # ── Diagnostic ──────────────────────────────────────────
        gold_sids = _find_gold_sids(gold, builder)
        a_set = set(pr.step2_a_containers)
        l2_cids = pr.debug.get("step4_l2_cids", []) if pr.debug else []
        l2_set = set(l2_cids)
        # step6 pool: we don't have it directly, but we can reconstruct
        # from L0 (A) + L1 (L2) + L2 (traced).
        # Easier: read step10 final sids; we know pool ⊇ final.
        # But we want exact pool. Look at step7_top_n_sids as proxy.
        # Best: rebuild by re-iterating A + L2 + traced.
        pool = set()
        for cid in pr.step2_a_containers + l2_cids:
            c = cm.get_container(cid)
            if c:
                for sid in (getattr(c, "sentence_ids", []) or []):
                    pool.add(sid)
        for sid in pr.step5_traced_sids:
            pool.add(sid)

        # Where are the gold sentences? In pool? In A container? In L2?
        gold_containers = set()
        for sid in gold_sids:
            node = builder.sentence_registry.get_node(sid)
            if node and hasattr(node, "containers"):
                gold_containers.update(getattr(node, "containers", []) or [])

        # Build complete candidate pool: A ∪ L2 ∪ traced sentences.
        # Compare against gold_containers: if gold is NOT in any
        # container that's also in A∪L2, the gold container itself
        # was never recalled.
        all_cids = a_set | l2_set
        gold_in_recalled_container = bool(gold_containers & all_cids)

        gold_in_pool = [sid for sid in gold_sids if sid in pool]
        gold_in_A = [cid for cid in gold_containers if cid in a_set]
        gold_in_L2 = [cid for cid in gold_containers if cid in l2_set]

        in_pool = bool(gold_in_pool)
        in_A = bool(gold_in_A)
        in_L2 = bool(gold_in_L2) if l2_cids else False

        if in_pool:
            summary["in_pool"] += 1
        else:
            summary["not_in_pool"] += 1
        if in_A:
            summary["in_A"] += 1
        else:
            summary["not_in_A"] += 1
        if l2_cids:
            if in_L2:
                summary["in_L2"] += 1
            else:
                summary["not_in_L2"] += 1

        if not in_pool and not in_A and not in_L2:
            if gold_in_recalled_container:
                case = "MISSING_BUT_CONTAINER_RECALLED"
            else:
                case = "MISSING_CONTAINER_NOT_RECALLED"

        # Container split check: are there sibling containers sharing
        # the same entity base name?
        # (Look at all containers in A ∪ L2, find ones whose entity
        # name has been seen split into multiple container IDs.)
        # For diagnostic, we can check container.entity names.
        split_cids = []
        for cid in (list(a_set) + list(l2_set)):
            c = cm.get_container(cid)
            if c is None:
                continue
            ent = getattr(c, "entity", "")
            # Count how many containers share this entity name.
            same_name = [c2 for c2 in cm.get_all_containers()
                         if getattr(c2, "entity", "") == ent]
            if len(same_name) > 1:
                split_cids.append((cid, ent, len(same_name)))
        if split_cids:
            summary["split"] += 1

        # Classify the case.
        if in_pool and in_A:
            case = "IN_POOL_AND_A"        # 配额问题
        elif in_pool and in_L2:
            case = "IN_POOL_L2"           # 排序/配额问题
        elif in_pool and not in_A and not in_L2:
            case = "IN_POOL_TRACED"       # 来自溯源
        elif not in_pool and in_A:
            case = "NOT_IN_POOL_BUT_IN_A" # step6 bug 或句子没挂上 A 容器
        elif not in_pool and in_L2:
            case = "NOT_IN_POOL_BUT_IN_L2" # 句子没挂上 L2 容器
        elif not in_pool and not in_A and not in_L2:
            if gold_in_recalled_container:
                case = "MISSING_BUT_CONTAINER_RECALLED"  # step6 bug
            else:
                case = "MISSING_CONTAINER_NOT_RECALLED"   # 召回失败
        else:
            case = "UNKNOWN"

        # Position in step10 final:
        final = pr.step10_final_sids or []
        gold_rank = next(
            (i + 1 for i, sid in enumerate(final) if sid in gold_sids),
            None,
        )

        cases.append({
            "i": i + 1,
            "qid": qid,
            "question": question[:70],
            "gold": gold,
            "gold_sids": gold_sids,
            "gold_containers": list(gold_containers),
            "in_pool": in_pool,
            "in_A": in_A,
            "in_L2": in_L2,
            "n_A": len(a_set),
            "n_L2": len(l2_set),
            "n_pool": len(pool),
            "n_traced": len(pr.step5_traced_sids),
            "case": case,
            "gold_rank": gold_rank,
            "split_cids": [t[0] for t in split_cids[:3]],
            "n_split_groups": len(set(t[1] for t in split_cids)),
            "em": (res and getattr(res, "answer", "") == gold),
            "latency_ms": latency,
        })

    # ── Print summary table ───────────────────────────────────
    print("=" * 80)
    print(f"v8 失败诊断  n={len(cases)}")
    print(f"  in_pool:        {summary['in_pool']} / {len(cases)}")
    print(f"  not_in_pool:    {summary['not_in_pool']}")
    print(f"  in_A:           {summary['in_A']}")
    print(f"  not_in_A:       {summary['not_in_A']}")
    if summary['in_L2'] + summary['not_in_L2'] > 0:
        print(f"  in_L2 (L2 non-empty):    {summary['in_L2']}")
        print(f"  not_in_L2:    {summary['not_in_L2']}")
    print(f"  容器有分裂:   {summary['split']} 题")
    print("=" * 80)
    print(f"{'i':<3} {'case':<25} {'rank':<5} {'in_pool':<8} "
          f"{'in_A':<6} {'in_L2':<6} {'n_pool':<7} {'EM':<3} Q")
    print("-" * 80)
    for c in cases:
        print(
            f"{c['i']:<3} {c['case']:<25} {str(c['gold_rank']):<5} "
            f"{str(c['in_pool']):<8} {str(c['in_A']):<6} {str(c['in_L2']):<6} "
            f"{c['n_pool']:<7} {('Y' if c['em'] else 'N'):<3} {c['question']}"
        )
    print("=" * 80)
    # Save full cases.
    out = Path("eval/diag") / f"v8_diag_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": cases}, f,
                  ensure_ascii=False, indent=2)
    print(f"-> saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

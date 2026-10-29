"""
clean_eval.py — 干净的 quick20 评测 driver

设计目的:
  - 主输出只显示每题的 Q / G / A / EM / F1 / R@1 / R@5 / R@10
  - 不再打印 step1.5 / step9.5 / pipeline path / debug 等流水线内部信息
  - 详细的 pipeline 调试信息统一写到
        eval/results/clean_<tag>.json  →  records[*].debug 字段
    以及
        eval/diag/<tag>/r1_failures.txt →  R@1 失败案例(R@10=1 但 R@1=0)
                                       的 top10 句子文本 + 位置

和 quick20_eval.py 关系:
  - 复用 quick20_eval 中的所有 build_graph / extract_answer / 评分函数
  - 通过 --suppress-step-logs 抑制 quick20 自身的 INFO 流水
  - 主入口在 main(): 跑一遍 20 题,只打印 Q/G/A/EM/F1/R@K

Usage:
  python eval/quick20_clean.py --n 20 --seed 137 --use-llm \\
      --llm-base-url http://127.0.0.1:8013/v1 --llm-model llama-3.1-8b \\
      --tag seed137

超边说明:
  超边一律在线构建 (in-process), 离线 parquet store 已移除。
  LLM / embedding 模型均从本地加载 (见 src/model_paths.py, src/llm_client.py)。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── Make repo importable ────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Force HF / sentence-transformers to local cache (mirror quick20_eval).
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── Quiet logging ───────────────────────────────────────────────────────────
# 用户的 stdout 必须干净: 把 step1.5 / step9.5 / pipeline path 等所有 INFO
# 全部重定向到文件.  仅保留我们自己控制台打印的 Q/G/A/EM/F1/R@K.
class _TeeLogHandler(logging.Handler):
    """把所有 quick20 / dispatcher 日志写到 diag/<tag>.log, 不进 stdout."""
    def __init__(self, path: Path):
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._fh.write(self.format(record) + "\n")
            self._fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# ── Import core functions from quick20_eval (un-modified) ──────────────────
import quick20_eval as q20  # noqa: E402

_NORMALIZE = q20._normalize_answer
_EM = q20._em_score
_F1 = q20._f1_score
_BUILD_GRAPH = q20.build_graph_from_hotpot
_EXTRACT_ANSWER = q20.extract_answer_from_dispatch
_APPLY_HYBRID = q20._apply_hybrid_config


# ── Pretty console writer (handles GBK terminal) ───────────────────────────
def _safe_print(s: str) -> None:
    try:
        sys.stdout.write(s)
        sys.stdout.flush()
    except UnicodeEncodeError:
        s2 = s.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        )
        sys.stdout.write(s2)
        sys.stdout.flush()


def _build_sparse_triggers(builder, paragraph_cids):
    """Mirror of sparse_hyperedge_eval.build_sparse_triggers for in-process use."""
    triggers = {cid: set() for cid in paragraph_cids}
    for _sid, node in builder.sentence_registry._nodes.items():
        for mount in getattr(node, "attribute_mounts", []) or []:
            attr = mount.attribute_name
            if not attr or attr == "other":
                continue
            for cid in set(node.containers) & set(paragraph_cids):
                triggers[cid].add(attr)
    return {cid: sorted(s) for cid, s in triggers.items() if s}


# ── JSON-safe coercion for the pipeline debug dump ─────────────────────────
def _jsonable(obj):
    """Recursively coerce a debug payload into JSON-serialisable types.

    The pipeline debug dict mixes plain Python types with numpy scalars
    (float32 from the embedder) and sets (e.g. `wanted_attrs`).  json.dump
    cannot serialise those, so coerce:
      numpy scalar → python scalar
      set/tuple    → list
      unknown      → str(obj)
    """
    try:
        import numpy as _np

        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ── Main loop ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="干净版 quick20 评测: 只输出 Q/G/A/EM/F1/R@K"
    )
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--data", default="eval/data/hotpot_dev_distractor_v1.json"
    )
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument(
        "--llm-base-url", default="",
        help="OpenAI-compatible base URL of the LLM server "
             "(default: $STENRAG_LLM_BASE_URL, else hosted DeepSeek)",
    )
    ap.add_argument(
        "--llm-model", default="",
        help="Model name served by the LLM server "
             "(default: $STENRAG_LLM_MODEL, else deepseek-chat)",
    )
    ap.add_argument(
        "--out",
        default="eval/results/clean_results.json",
    )
    ap.add_argument("--tag", default="", help="给输出文件名加后缀")
    ap.add_argument(
        "--hybrid-mode", default="hybrid",
        choices=("dense_only", "bm25_only", "hybrid", "bm25_heavy", "dense_heavy"),
    )
    ap.add_argument("--bm25-weight", type=float, default=-1.0)
    ap.add_argument("--no-bridge", action="store_true")
    ap.add_argument("--no-gap-expand", action="store_true")
    ap.add_argument(
        "--strict-entity-match", action="store_true",
    )
    ap.add_argument(
        "--no-fuzzy", action="store_true",
        help="Disable the full fuzzy entity->container matcher "
             "(src/fuzzy_match.py). Fuzzy matching is ON by default.",
    )
    ap.add_argument(
        "--fuzzy-threshold", type=float, default=0.58,
        help="Minimum similarity for a fuzzy surface to count as a match.",
    )
    ap.add_argument(
        "--attr-triggers", action="store_true",
        help="Build hyperedges with attribute triggers (navigability aid)",
    )
    ap.add_argument(
        "--attr-soft-match", dest="attr_soft_match",
        action="store_true", default=True,
        help="Enable symmetric-attribute soft matching in the rerank stage "
             "(default: ON).  Adds W_ATTR·attr_match to the final score and "
             "enables the hard attribute gate.",
    )
    ap.add_argument(
        "--no-attr-soft-match", dest="attr_soft_match",
        action="store_false",
        help="Disable symmetric-attribute soft matching and the hard "
             "attribute gate.",
    )
    ap.add_argument(
        "--top-k-final", type=int, default=-1,
        help="Override pipeline.top_k_final (number of sentences handed to the "
             "LLM).  -1 = keep the pipeline default.  Raising this raises the "
             "recall ceiling: gold-bearing sentences that survive step6 but "
             "rank below the cut are otherwise dropped before the LLM sees them.",
    )
    ap.add_argument(
        "--event-name-weight", type=float, default=None,
        help="Override the event-name weight inside S_h_E "
             "(default: $STENRAG_EVENT_NAME_WEIGHT, else 0.7). "
             "S_h_E = W·cos(event_name,q) + (1-W)·cos(sentence_vec,q).",
    )
    args = ap.parse_args()

    # Apply the CLI override BEFORE importing the pipeline module, so the
    # module-level constants pick it up.  This makes the weight settable
    # without relying on shell env-var quoting, which is easy to get
    # wrong and silently leaves the default in place.
    if args.event_name_weight is not None:
        os.environ["STENRAG_EVENT_NAME_WEIGHT"] = str(args.event_name_weight)

    # ── 抑制 quick20 自身的 INFO 流水 ──
    # 把所有 log.* 写到 diag/<tag>.log, stdout 只留我们的主输出
    out_path = Path(args.out)
    if args.tag:
        out_path = out_path.with_name(
            out_path.stem + f"_{args.tag}" + out_path.suffix
        )
    diag_dir = Path("eval/diag") / (args.tag or "default")
    diag_dir.mkdir(parents=True, exist_ok=True)
    log_path = diag_dir / "pipeline.log"
    handler = _TeeLogHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # Also catch the library loggers.
    for name in ("quick20", "src", "src.dispatcher", "src.pipeline",
                 "src.hyper_edge"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False
        lg.setLevel(logging.INFO)

    # ── 加载数据 ──
    with open(args.data, encoding="utf-8") as f:
        all_data = json.load(f)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(all_data)), min(args.n, len(all_data)))
    indices.sort()
    sample = [all_data[i] for i in indices]

    # ── spaCy ──
    from src.graph_builder import _load_spacy_shared
    nlp = _load_spacy_shared("en_core_web_lg")
    if nlp is None:
        _safe_print("[FATAL] spaCy unavailable; aborting.\n")
        return 1

    # NOTE: do NOT pre-fill _EMBEDDER_CACHE with None here.
    # _get_embedder() returns the cached value verbatim, so seeding it
    # with None permanently disables the SentenceTransformer.  That in
    # turn kills (a) the cosine S_h_F path (it silently falls back to
    # Jaccard) and (b) the symmetric-attribute soft match, whose gate
    # requires an embedder.  See quick20_eval.py's "Do NOT pre-fill
    # _EMBEDDER_CACHE with None" note.

    # ── LLM client ──
    client = None
    if args.use_llm:
        from src.llm_client import make_client
        # Make sure a local server is actually serving the model
        # (starts one if needed); a hosted DeepSeek key is used as-is.
        base_url = args.llm_base_url or os.environ.get(
            "STENRAG_LLM_BASE_URL", ""
        ).strip()
        if base_url:
            from src.local_llm_server import ensure_server
            try:
                base_url = ensure_server(
                    base_url=base_url,
                    model=args.llm_model or "",
                )
            except Exception as e:
                _safe_print(f"[FATAL] 本地 LLM 服务启动失败: {e}\n")
                return 1
        client = make_client(
            base_url=base_url or None,
            model=args.llm_model or None,
        )
        if not client.enabled:
            _safe_print("[FATAL] --use-llm 但未检测到可用的 LLM 配置\n")
            return 1
        _safe_print(
            f"[setup] llm: model={client.model} base_url={client.base_url}\n"
        )

    # ── Hyperedges ──
    # Online construction only: the offline parquet store was removed,
    # so hyperedges are always built in-process from the paragraph
    # containers (see the per-question loop below).
    _safe_print(f"[setup] hyperedges: online (in-process)\n")
    _safe_print(
        f"[setup] n={args.n} seed={args.seed} hybrid={args.hybrid_mode} "
        f"use_llm={args.use_llm} "
        f"fuzzy={not bool(getattr(args, 'no_fuzzy', False))} "
        f"fuzzy_thr={getattr(args, 'fuzzy_threshold', 0.58)} "
        f"strict={bool(args.strict_entity_match)}\n"
    )
    # Print the scoring knobs that are set via environment variables.
    # Without this it is impossible to tell from the log which
    # configuration produced a given result file — a mistake that has
    # already produced one misleading "no change" reading.
    try:
        from src import stengraph_pipeline as _sgp
        from src import answer_extractor as _ae
        _safe_print(
            f"[setup] W_EVENT_NAME={_sgp.W_EVENT_NAME:.2f} "
            f"W_EVENT_SENT={_sgp.W_EVENT_SENT:.2f} "
            f"W_EVENT={_sgp.W_EVENT:.2f} | "
            f"answer_variant={_ae.DEFAULT_VARIANT} "
            f"structural={_ae.STRUCTURAL_DEFAULT} "
            f"self_correct={_ae.SELF_CORRECT_DEFAULT}\n"
        )
    except Exception as _exc:  # pragma: no cover - diagnostics only
        _safe_print(f"[setup] (scoring knobs unavailable: {_exc})\n")
    _safe_print("=" * 80 + "\n")

    # ── Per-question loop ──
    from src.dispatcher import build_dispatcher
    from src.hyper_edge import HyperEdgeRegistry

    ems: List[bool] = []
    f1s: List[float] = []
    r1s: List[int] = []
    r5s: List[int] = []
    r10s: List[int] = []
    records: List[Dict[str, Any]] = []
    r1_failure_top10: List[Dict[str, Any]] = []  # R@10=1 但 R@1=0 的诊断样本

    for i, qdata in enumerate(sample):
        qid = qdata["_id"]
        question = qdata["question"]
        gold = qdata["answer"]

        t0 = time.time()
        try:
            builder, sentences, titles, para_sents, bridge_titles = (
                _BUILD_GRAPH(qdata, nlp)
            )
            cm = builder.container_manager
            for c in cm.get_all_containers():
                cm.update_container_index(c.id)
            builder.compute_tfidf_index()

            he = HyperEdgeRegistry(container_manager=cm)
            # Online construction: build the hyperedges for every
            # paragraph from its containers.
            for _p_title, _p_cids in (bridge_titles or {}).items():
                    _p_containers = [
                        cm._containers.get(_cid)
                        for _cid in _p_cids
                        if cm._containers.get(_cid) is not None
                    ]
                    if _p_containers:
                        if args.attr_triggers:
                            _triggers = _build_sparse_triggers(
                                builder, _p_cids
                            )
                            he.build_hyperedges_for_chunk(
                                chunk_id=str(_p_title),
                                containers=_p_containers,
                                time_word="",
                                paragraph_title=str(_p_title),
                                attribute_triggers=_triggers,
                            )
                        else:
                            he.build_hyperedges_for_chunk(
                                chunk_id=str(_p_title),
                                containers=_p_containers,
                                time_word="",
                                paragraph_title=str(_p_title),
                            )

            # 段落级超边融合 (rule-based, no LLM) 已移除:
            # merge_paragraph_hyperedges_into 随其依赖一并删除.

            disp = build_dispatcher(
                builder, client=client, nlp=nlp,
                hyperedge_registry=he,
                strict_entity_match=bool(args.strict_entity_match),
                attr_soft_match=bool(getattr(args, 'attr_soft_match', False)),
                fuzzy_match=not bool(getattr(args, 'no_fuzzy', False)),
                fuzzy_threshold=float(getattr(args, 'fuzzy_threshold', 0.58)),
            )
            if args.no_gap_expand:
                disp.pipeline.gap_detector = None
            if getattr(args, "top_k_final", -1) > 0:
                disp.pipeline.top_k_final = int(args.top_k_final)
            _APPLY_HYBRID(disp.pipeline, args.hybrid_mode, args.bm25_weight)

            res = disp.run(
                question,
                bridge_titles=None if args.no_bridge else bridge_titles,
            )
            # TEMP DIAG: print he_registry + first ef_score
            try:
                pr0 = res.pipeline_result
                ef0 = pr0.debug.get("ef_scores", [])
                logger0 = logging.getLogger("diag")
                logger0.info(
                    "DIAG he_registry=%s he_n=%d cm_n=%d pool_size=%d step7_n=%d step10_n=%d ef_n=%d ef0_event_id=%s ef0_S_h_E=%.3f",
                    type(disp.pipeline.he_registry).__name__,
                    len(disp.pipeline.he_registry),
                    len(disp.pipeline.cm.get_all_containers()),
                    getattr(pr0, "step6_pool_size", -1),
                    len(getattr(pr0, "step7_top_n_sids", [])),
                    len(getattr(pr0, "step10_final_sids", [])),
                    len(ef0),
                    (ef0[0].get("event_id") if ef0 else "?"),
                    (ef0[0].get("S_h_E", 0.0) if ef0 else 0.0),
                )
            except Exception as _e:
                logging.getLogger("diag").info("DIAG failed: %s", _e)
            answer = _EXTRACT_ANSWER(
                res, builder, gold, nlp, client=client, question=question,
            )
        except Exception as e:
            logging.getLogger("clean_eval").exception("Q %s failed", qid)
            answer = ""
            res = None
            builder = None

        em_hit = _EM(answer, gold)
        f1 = _F1(answer, gold)
        latency_ms = (time.time() - t0) * 1000
        ems.append(em_hit)
        f1s.append(f1)

        # ── R@K ──
        gold_norm = _NORMALIZE(gold)
        final_sids: List[str] = []
        if res is not None and res.pipeline_result is not None:
            final_sids = list(
                getattr(res.pipeline_result, "step10_final_sids", []) or []
            )

        def _contains_gold(sid: str) -> bool:
            if builder is None:
                return False
            n = builder.sentence_registry.get_node(sid)
            return n is not None and gold_norm in _NORMALIZE(n.text or "")

        r1 = 1 if (final_sids and _contains_gold(final_sids[0])) else 0
        r5 = 1 if any(_contains_gold(s) for s in final_sids[:5]) else 0
        r10 = 1 if any(_contains_gold(s) for s in final_sids[:10]) else 0
        r1s.append(r1); r5s.append(r5); r10s.append(r10)

        # ── 收集 R@1 失败 (R@10 命中但 R@1 没命中) 案例 ──
        if r10 == 1 and r1 == 0 and disp is not None:
            # ── 诊断:用 StenGraphPipeline._score_sentences_v8 收集 6 分量 breakdown ──
            pipeline = getattr(disp, "pipeline", None)
            try:
                if pipeline:
                    pipeline._collect_breakdown = True
                    pipeline._breakdown_buffer = {}
                    _ = pipeline._score_sentences_v8(question, final_sids[:10])
            except Exception:
                pass
            finally:
                if pipeline:
                    pipeline._collect_breakdown = False

            top10_payload = []
            for rank, sid in enumerate(final_sids[:10], 1):
                n = builder.sentence_registry.get_node(sid)
                bd = (pipeline._breakdown_buffer.get(sid, {}) if pipeline else {})
                top10_payload.append({
                    "rank": rank,
                    "sid": sid,
                    "is_gold": _contains_gold(sid),
                    "text": (n.text if n else "")[:200],
                    "score": bd,
                })
            r1_failure_top10.append({
                "qid": qid,
                "question": question,
                "gold": gold,
                "answer": answer,
                "n_final_sids": len(final_sids),
                "top10": top10_payload,
                "gold_in_rank": next(
                    (p["rank"] for p in top10_payload if p["is_gold"]),
                    None,
                ),
            })

        # ── 干净输出: 一行 header + 三行 Q/G/A ──
        mark = "✓" if em_hit else "✗"
        _safe_print(
            f"[{i+1:>2}/{args.n}] {qid[:8]}  EM={mark} F1={f1:.3f} "
            f"R@1={r1} R@5={r5} R@10={r10}  ({latency_ms:.0f}ms)\n"
        )
        _safe_print(f"  Q : {question}\n")
        _safe_print(f"  G : {gold}\n")
        _safe_print(f"  A : {answer}\n")
        _safe_print("\n")

        records.append({
            "qid": qid,
            "question": question,
            "gold": gold,
            "answer": answer,
            "em": em_hit,
            "f1": f1,
            "r1": r1,
            "r5": r5,
            "r10": r10,
            "latency_ms": latency_ms,
            # Pipeline debug dump (ef_scores with S_h_E / S_h_F / S_f /
            # attr_match / attr_term / attr_gate_pass / wanted_attrs,
            # plus the step6 pool and step10 final sids).  Persisted so the
            # attribute soft-match and the hard attribute gate can be
            # verified offline without re-running the pipeline.
            "debug": _jsonable(getattr(res.pipeline_result, "debug", {}) or {}),
            # Ranked evidence handed to the answer extractor.  Kept so the
            # extractor can be re-run offline (no pipeline re-run needed).
            "evidence": [
                {
                    "rank": i + 1,
                    "sid": sid,
                    "is_gold": _contains_gold(sid),
                    "text": (
                        builder.sentence_registry.get_node(sid).text
                        if builder is not None
                        and builder.sentence_registry.get_node(sid)
                        else ""
                    ),
                }
                for i, sid in enumerate(final_sids)
            ],
        })

    # ── Summary ──
    n = len(ems)
    em_avg = sum(ems) / max(1, n)
    f1_avg = sum(f1s) / max(1, n)
    r1_avg = sum(r1s) / max(1, n)
    r5_avg = sum(r5s) / max(1, n)
    r10_avg = sum(r10s) / max(1, n)
    _safe_print("=" * 80 + "\n")
    _safe_print(
        f"SUMMARY  n={n}  EM={em_avg:.3f}  F1={f1_avg:.3f}  "
        f"R@1={r1_avg:.3f}  R@5={r5_avg:.3f}  R@10={r10_avg:.3f}\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n": n, "em": em_avg, "f1": f1_avg,
            "r1": r1_avg, "r5": r5_avg, "r10": r10_avg,
            "records": records,
        }, f, ensure_ascii=False, indent=2)
    _safe_print(f"  → records: {out_path}\n")

    # ── R@1 诊断 dump ──
    diag_path = diag_dir / "r1_failures.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_total": n,
            "n_r10_hit": sum(r10s),
            "n_r1_hit": sum(r1s),
            "n_r10_hit_but_r1_miss": len(r1_failure_top10),
            "failures": r1_failure_top10,
        }, f, ensure_ascii=False, indent=2)
    _safe_print(f"  → R@1 失败诊断 (R@10=1 但 R@1=0): {diag_path}\n")
    _safe_print(
        f"  → pipeline 日志: {log_path}\n"
    )
    handler.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Attribute Mount Inspector — 看「属性 (实体名, 实体种类)」

用途
----
在滚动优化跑的过程中, 需要肉眼确认三件事:

1. **进度** — 抽了多少题、还剩多少题;
2. **属性表状态** — 现在有多少属性 / 多少实体类, 已经滚进去多少;
3. **挂载质量** — 每条句子到底挂到了哪些属性上, 归属哪个实体、
   实体是哪个种类 (PERSON / ORG / GPE / EVENT ...)。

第 3 点就是「属性(具体实体名字 实体种类)」的落地: 一行一个挂载点,
格式为::

    birth (Marie Curie, PERSON) w=0.83
    date (Expo A, EVENT) w=0.77

用法
----
::

    # 只看某一题的挂载明细
    python eval/attribute_mount_inspect.py --qid <id>

    # 抽 3 题看挂载 + 当前属性表
    python eval/attribute_mount_inspect.py --n 3

    # 导出全部挂载点为 JSON
    python eval/attribute_mount_inspect.py --n 5 --out eval/results/mounts.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from src.attribute_agent import load_questions  # noqa: E402


def _safe_print(s: str) -> None:
    """
    Write a line to stdout, surviving a non-UTF-8 console.

    The Windows console defaults to GBK (cp936), which cannot encode
    every character used here.  We reconfigure stdout to UTF-8 once at
    startup (Python 3.7+); if that is impossible we degrade to
    replacement characters so the output stays readable instead of
    crashing.
    """
    try:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        safe = s.encode(enc, errors="replace").decode(enc, errors="replace")
        try:
            sys.stdout.write(safe + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass


def _force_utf8_stdout() -> None:
    """Reconfigure stdout to UTF-8 so Chinese labels are not mojibake."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - not a TTY / older Python
            pass


_force_utf8_stdout()


def _mounts_of(node) -> List[Dict[str, Any]]:
    """Extract the (attribute, entity, entity_type, weight) mount points."""
    out: List[Dict[str, Any]] = []
    for m in (getattr(node, "attribute_mounts", []) or []):
        out.append({
            "attribute": getattr(m, "attribute_name", "") or "",
            "entity": getattr(m, "entity", "") or "",
            "entity_type": getattr(m, "entity_type", "") or "",
            "weight": round(float(getattr(m, "weight", 0.0) or 0.0), 3),
        })
    return out


def _table_state() -> Dict[str, Any]:
    """Current size of the attribute table and the class table."""
    from src import sentence_graph as sg
    from src.attribute_schema import AttributeSchema

    templates = AttributeSchema().effective_templates()
    return {
        "n_attributes": len(sg.ATTRIBUTE_NAMES),
        "n_classes": len(templates),
        "classes": sorted(templates.keys()),
        "attributes": sorted(sg.ATTRIBUTE_NAMES),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/data/hotpot_dev_distractor_v1.json")
    ap.add_argument("--n", type=int, default=3,
                    help="How many questions to inspect (default 3).")
    ap.add_argument("--qid", default="",
                    help="Inspect one specific question id instead.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--schema", default="",
                    help="Load an attribute-schema delta before inspecting.")
    ap.add_argument("--out", default="",
                    help="Write the mount points to this JSON file.")
    ap.add_argument("--no-sentences", action="store_true",
                    help="Do not print the sentence text for each mount.")
    args = ap.parse_args()

    # ── Load an optional schema delta ─────────────────────────────
    if args.schema:
        from src.attribute_schema import AttributeSchema
        schema = AttributeSchema.load(args.schema)
        schema.apply_to_modules()
        _safe_print(f"[schema] loaded {len(schema.attributes)} attr(s), "
                    f"{len(schema.classes)} class(es) from {args.schema}")

    # ── Current table state ───────────────────────────────────────
    state = _table_state()
    _safe_print("=" * 78)
    _safe_print(f"属性表: {state['n_attributes']} 个属性 / "
                f"{state['n_classes']} 个实体类")
    _safe_print(f"实体类: {', '.join(state['classes'])}")
    _safe_print("=" * 78)

    questions = load_questions(args.data)
    if not questions:
        _safe_print(f"[error] no questions in {args.data}")
        return 1

    if args.qid:
        picked = [q for q in questions if q["qid"] == args.qid]
        if not picked:
            _safe_print(f"[error] qid {args.qid} not found")
            return 1
        sample = picked
    else:
        import random
        rng = random.Random(args.seed)
        idx = sorted(rng.sample(range(len(questions)),
                                 min(args.n, len(questions))))
        sample = [questions[i] for i in idx]

    from src.graph_builder import GraphBuilder, ComplementedSentence, _load_spacy_shared

    nlp = _load_spacy_shared("en_core_web_lg")
    if nlp is None:
        _safe_print("[error] spaCy unavailable")
        return 1

    all_mounts: List[Dict[str, Any]] = []

    for qi, q in enumerate(sample):
        _safe_print("")
        _safe_print(f"─── [{qi + 1}/{len(sample)}] {q['qid'][:16]} ───")
        _safe_print(f"Q: {q['question'][:100]}")

        sents: List[str] = []
        for para in (q.get("context") or []):
            if isinstance(para, (list, tuple)) and len(para) >= 2:
                sents.extend(str(s) for s in para[1])
            elif isinstance(para, dict):
                sents.extend(str(s) for s in (para.get("sentences") or []))
        if not sents:
            _safe_print("   (no sentences)")
            continue

        builder = GraphBuilder(shared_nlp=nlp)
        for text in sents:
            doc = nlp(text)
            ents = [e.text for e in doc.ents]
            builder.add_sentence(ComplementedSentence(
                original_text=text,
                complemented_text=text,
                resolved_subject=ents[0] if ents else "",
                entities=ents,
            ))
        builder.compute_tfidf_index()
        for c in builder.container_manager.get_all_containers():
            builder.container_manager.update_container_index(c.id)

        nodes = builder.sentence_registry.get_all_nodes()
        n_mounted = 0
        for node in nodes:
            mounts = _mounts_of(node)
            if not mounts:
                continue
            n_mounted += 1
            if not args.no_sentences:
                _safe_print(f"   句: {(node.text or '')[:88]}")
            for m in mounts:
                etype = m["entity_type"] or "?"
                line = (f"      → {m['attribute']} "
                        f"({m['entity']}, {etype}) w={m['weight']:.2f}")
                _safe_print(line)
                all_mounts.append({
                    "qid": q["qid"],
                    "sentence_id": node.id,
                    "sentence": (node.text or "")[:300],
                    **m,
                })
        _safe_print(f"   [{n_mounted}/{len(nodes)} 句有挂载]")

    # ── Summary ───────────────────────────────────────────────────
    if all_mounts:
        by_attr: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        for m in all_mounts:
            by_attr[m["attribute"]] = by_attr.get(m["attribute"], 0) + 1
            key = f"{m['attribute']}@{m['entity_type'] or '?'}"
            by_type[key] = by_type.get(key, 0) + 1
        _safe_print("")
        _safe_print("=" * 78)
        _safe_print(f"共 {len(all_mounts)} 个挂载点")
        _safe_print("按属性: " + ", ".join(
            f"{k}={v}" for k, v in sorted(by_attr.items(), key=lambda kv: -kv[1])
        ))
        _safe_print("按 属性@种类: " + ", ".join(
            f"{k}={v}" for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])
        ))
        _safe_print("=" * 78)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "table_state": state,
                "n_mounts": len(all_mounts),
                "mounts": all_mounts,
            }, f, ensure_ascii=False, indent=2)
        _safe_print(f"[out] {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

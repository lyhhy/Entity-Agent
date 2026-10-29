"""
Attribute Agent Evaluation Driver
=================================

Runs the context-memory attribute-table rolling optimisation agent over
a QA dataset and writes a report.

This driver is the "harness" for the agent: it owns the sampling
parameters, the persistence of the resulting schema delta and the
before/after measurement, so the agent module itself stays free of
file-system and CLI concerns.

Usage::

    # Dry run — detect gaps, propose, gate, report; nothing is applied.
    python eval/attribute_agent_eval.py --data eval/data/hotpot_dev_distractor_v1.json \
        --n 20 --rounds 10 --seed 1234 --dry-run

    # Apply — write the accepted delta and print a pasteable snippet.
    python eval/attribute_agent_eval.py --data eval/data/hotpot_dev_distractor_v1.json \
        --n 20 --rounds 10 --seed 1234 --out eval/results/attribute_schema.json

The dataset is NOT bundled with the repo (see .gitignore: `eval/data/`).
Point `--data` at a local HotpotQA dev file.

The module also exposes `run_attribute_agent()` — the reusable entry
point.  `eval/quick20_eval.py` imports it so the rolling optimisation
can run inline, right before the EM/F1 measurement, on exactly the
same question sample that is being evaluated.  That way one command
produces BOTH the schema delta and its measured effect.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Force HF / sentence-transformers to use the local cache only — the
# evaluation host has no outbound access, and a Hub HEAD request on
# every model load blocks the run for minutes.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("attribute_agent_eval")

from src.attribute_agent import (  # noqa: E402
    AttributeAgent,
    AttributeAgentConfig,
    AgentReport,
    load_questions,
)


# ---------------------------------------------------------------------------
# Reusable entry point
# ---------------------------------------------------------------------------

def run_attribute_agent(
    questions,
    *,
    n_questions: int = 20,
    n_rounds: int = 10,
    seed: int = 1234,
    min_mount_count: int = 3,
    min_global_rate: float = 0.02,
    min_class_top_rate: float = 0.05,
    min_distinctiveness: float = 0.10,
    max_llm_calls: int = 200,
    use_llm: bool = True,
    apply: bool = True,
    client=None,
    schema=None,
    nlp=None,
    progress=None,
    report_path: Optional[str] = None,
    schema_path: Optional[str] = None,
) -> AgentReport:
    """
    Run the rolling attribute-table optimisation and return the report.

    This is the reusable core of `main()`.  `eval/quick20_eval.py`
    calls it inline so that the schema delta it produces is applied to
    the very graphs that are then measured for EM/F1.

    Args:
        questions:       Either raw HotpotQA records or already-loaded
                         question dicts (see `load_questions`).
        apply:          Apply the accepted delta to the live module
                         tables.  When False it is a dry run.
        schema:         An existing `AttributeSchema` to roll forward
                         from (already applied to the modules by the
                         caller).
        nlp:            A pre-loaded spaCy pipeline.  Passing the
                         evaluation harness's model avoids loading
                         `en_core_web_lg` a second time.
        progress:       Optional progress reporter (see
                         `src/progress.py`).  When given, the run
                         prints a live line showing how many draws are
                         done, how many gaps were found, and how many
                         attributes have already been rolled in.
        report_path:    Where to write the full run report (JSON).
        schema_path:    Where to persist the accepted schema delta.
                        When None, the delta is NOT written to disk.

    Returns:
        The `AgentReport`.  When `apply` is True and the report carries
        a schema, the module tables in `src.sentence_graph` have already
        been patched — so any graph built afterwards inherits the new
        attributes and classes.
    """
    # Accept either raw records or pre-loaded question dicts.
    if questions and isinstance(questions[0], dict) and "context" in questions[0] \
            and "qid" not in questions[0]:
        questions = [
            {
                "qid": str(rec.get("_id") or rec.get("id") or i),
                "question": str(rec.get("question") or rec.get("query") or ""),
                "answer": str(rec.get("answer") or rec.get("gold") or ""),
                "context": rec.get("context") or rec.get("paragraphs") or [],
            }
            for i, rec in enumerate(questions)
        ]

    if client is None:
        try:
            from src.llm_client import make_client, describe_llm_config
            client = make_client()
            log.info("LLM client: %s",
                     "yes" if client and client.enabled else "no (no key)")
            log.info("LLM config: %s", describe_llm_config())
        except Exception as e:  # noqa: BLE001
            log.warning("LLM init failed: %s", e)

    config = AttributeAgentConfig(
        n_questions=n_questions,
        n_rounds=n_rounds,
        seed=seed,
        min_mount_count=min_mount_count,
        min_global_mount_rate=min_global_rate,
        min_class_top_rate=min_class_top_rate,
        min_distinctiveness_margin=min_distinctiveness,
        max_llm_calls=max_llm_calls,
        use_llm=use_llm,
    )

    agent = AttributeAgent(
        config, client=client, schema=schema, nlp=nlp, progress=progress,
    )
    report = agent.run(questions, apply=apply)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    if report_path:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        log.info("Report  → %s", rp)

    if schema_path and report.schema is not None and report.schema.attributes:
        sp = report.schema.save(schema_path)
        log.info("Schema  → %s", sp)
        snippet = report.schema.export_python()
        snippet_path = sp.with_suffix(".snippet.py")
        with open(snippet_path, "w", encoding="utf-8") as f:
            f.write(snippet)
        log.info("Snippet → %s", snippet_path)

    return report


def log_report(report: AgentReport) -> None:
    """Print the human-readable summary of an agent run."""
    log.info("=" * 68)
    log.info("Sampled %d question draws over %d rounds",
             report.n_sampled, report.n_rounds)
    log.info("Gaps: %d   accepted: %d   rejected: %d   LLM calls: %d",
             len(report.gaps), len(report.accepted), len(report.rejected),
             report.llm_calls)
    log.info("-" * 68)
    for gap in report.gaps:
        log.info("[%s] %s :: %s  (fired %dx)",
                 gap.verdict.upper(), gap.kind, gap.entity_type, gap.count)
        log.info("    attribute: %s", gap.attribute or "-")
        log.info("    reason:    %s", gap.reason)
        if gap.evidence:
            log.info("    evidence:  %s", gap.evidence[0][:100])
    log.info("=" * 68)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="eval/data/hotpot_dev_distractor_v1.json",
        help="QA dataset (HotpotQA-style JSON).",
    )
    ap.add_argument("--n", type=int, default=20,
                    help="Questions drawn per round (default 20).")
    ap.add_argument("--rounds", type=int, default=10,
                    help="Rounds of sampling (default 10).")
    ap.add_argument("--seed", type=int, default=1234,
                    help="Base seed (default 1234).")
    ap.add_argument("--min-mount-count", type=int, default=3,
                    help="Min sentences an attribute must mount.")
    ap.add_argument("--min-global-rate", type=float, default=0.02,
                    help="Min corpus share an attribute must mount.")
    ap.add_argument("--min-class-top-rate", type=float, default=0.05,
                    help="Min share of a class's sentences on which the "
                         "attribute must rank top-K.")
    ap.add_argument("--min-distinctiveness", type=float, default=0.10,
                    help="Required margin of the attribute's class rate "
                         "over every other class that declares it.")
    ap.add_argument("--max-llm-calls", type=int, default=200,
                    help="Hard cap on LLM calls for this run.")
    ap.add_argument("--no-llm", action="store_true",
                    help="Disable the LLM fallbacks entirely.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report gaps and verdicts without applying them.")
    ap.add_argument(
        "--out",
        default="eval/results/attribute_schema.json",
        help="Where to persist the accepted schema delta.",
    )
    ap.add_argument(
        "--report",
        default="eval/results/attribute_agent_report.json",
        help="Where to persist the full run report.",
    )
    ap.add_argument(
        "--schema",
        default="",
        help="Path to a persisted attribute-schema delta "
             "(eval/results/attribute_schema.json).  When set, the delta "
             "is loaded and applied BEFORE the agent runs, so the agent "
             "rolls the table forward from the current state.",
    )
    ap.add_argument(
        "--no-progress", action="store_true",
        help="Do not print the live progress line (draws done / gaps "
             "found / attributes already rolled in).",
    )
    args = ap.parse_args()

    questions = load_questions(args.data)
    if not questions:
        log.error("No questions loaded from %s", args.data)
        log.info("The dataset is not bundled with the repo — download it "
                 "first (see eval/README.md) or point --data at your own file.")
        return 1
    log.info("Loaded %d questions from %s", len(questions), args.data)

    # Load the current schema delta, if any.
    schema = None
    if args.schema:
        from src.attribute_schema import AttributeSchema
        schema = AttributeSchema.load(args.schema)
        log.info("Loaded schema delta: %d attribute(s), %d class(es) from %s",
                 len(schema.attributes), len(schema.classes), args.schema)
        schema.apply_to_modules()

    # Optional LLM client — used only for gap confirmation and for the
    # same-class multi-entity ownership fallback.
    client = None
    try:
        from src.llm_client import make_client
        client = make_client()
        log.info("LLM client: %s",
                 "yes" if client and client.enabled else "no (no key)")
    except Exception as e:  # noqa: BLE001
        log.warning("LLM init failed: %s", e)

    # Progress reporter — a single live line showing how many draws are
    # done, how many gaps were found, and how many attributes have
    # already been rolled into the table.
    progress = None
    if not args.no_progress:
        try:
            from src.progress import ConsoleReporter
            progress = ConsoleReporter(prefix="agent")
        except Exception as e:  # noqa: BLE001
            log.warning("progress reporter unavailable: %s", e)

    report = run_attribute_agent(
        questions,
        n_questions=args.n,
        n_rounds=args.rounds,
        seed=args.seed,
        min_mount_count=args.min_mount_count,
        min_global_rate=args.min_global_rate,
        min_class_top_rate=args.min_class_top_rate,
        min_distinctiveness=args.min_distinctiveness,
        max_llm_calls=args.max_llm_calls,
        use_llm=not args.no_llm,
        apply=not args.dry_run,
        client=client,
        schema=schema,
        progress=progress,
        report_path=args.report,
        schema_path=args.out,
    )

    log_report(report)
    if report.schema is None or not report.schema.attributes:
        log.info("No accepted attribute changes; schema not written.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

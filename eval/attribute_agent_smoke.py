"""
Attribute Agent Smoke Test
==========================

Runs the agent end-to-end on a tiny SYNTHETIC dataset, so the whole
loop (sample -> detect -> propose -> gate -> apply -> re-mount) is
exercised without the HotpotQA download.

The synthetic corpus deliberately contains:
  * a PERSON whose sentences are about nationality / religion, which the
    PERSON template does not declare;
  * sentences about an entity type with no template (FAC), so the
    "missing class" branch fires.

Run::

    python eval/attribute_agent_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.attribute_agent import (  # noqa: E402
    AttributeAgent,
    AttributeAgentConfig,
    load_questions,
)


SYNTHETIC = [
    {
        "_id": "syn-0",
        "question": "What religion is Marie Curie?",
        "answer": "Catholic",
        "context": [
            ["Marie Curie", [
                "Marie Curie was a Polish and naturalised-French physicist "
                "and chemist who conducted pioneering research on radioactivity.",
                "Marie Curie was raised as a Catholic and remained a devout "
                "Catholic throughout her life.",
                "Marie Curie was born in Warsaw, Poland, in 1867.",
                "Marie Curie shared the 1903 Nobel Prize in Physics with "
                "her husband.",
            ]],
        ],
    },
    {
        "_id": "syn-1",
        "question": "Where is the Eiffel Tower located?",
        "answer": "Paris",
        "context": [
            ["Eiffel Tower", [
                "The Eiffel Tower is a wrought-iron lattice tower on the "
                "Champ de Mars in Paris, France.",
                "The Eiffel Tower was built as the centrepiece of the 1889 "
                "World's Fair.",
                "The Eiffel Tower is 330 metres tall and was the world's "
                "tallest man-made structure until 1930.",
                "The Eiffel Tower was designated a UNESCO World Heritage "
                "Site in 1994.",
            ]],
        ],
    },
    {
        "_id": "syn-2",
        "question": "Who is the director of Inception?",
        "answer": "Christopher Nolan",
        "context": [
            ["Inception", [
                "Inception is a 2010 science fiction action film written and "
                "directed by Christopher Nolan.",
                "Inception stars Leonardo DiCaprio, Joseph Gordon-Levitt and "
                "Elliot Page.",
                "Inception was nominated for eight Academy Awards and won "
                "four, including Best Cinematography.",
                "Inception grossed over 800 million dollars worldwide.",
            ]],
        ],
    },
    {
        "_id": "syn-3",
        "question": "What nationality is Albert Einstein?",
        "answer": "German",
        "context": [
            ["Albert Einstein", [
                "Albert Einstein was a German-born theoretical physicist who "
                "published the theory of relativity.",
                "Albert Einstein held German citizenship and later became a "
                "Swiss and United States citizen.",
                "Albert Einstein was awarded the Nobel Prize in Physics in "
                "1921.",
                "Albert Einstein was born in Ulm, in the Kingdom of "
                "Wurttemberg, in 1879.",
            ]],
        ],
    },
]


def main() -> int:
    print("=" * 68)
    print("Attribute Agent Smoke Test (synthetic corpus)")
    print("=" * 68)

    with tempfile.TemporaryDirectory() as td:
        data_path = Path(td) / "synthetic.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(SYNTHETIC, f, ensure_ascii=False)

        questions = load_questions(data_path)
        print(f"Loaded {len(questions)} synthetic questions.")

        config = AttributeAgentConfig(
            n_questions=4,
            n_rounds=2,
            seed=1234,
            min_mount_count=1,
            min_global_mount_rate=0.01,
            min_class_top_rate=0.01,
            min_distinctiveness_margin=0.0,
            use_llm=False,
        )
        agent = AttributeAgent(config)
        report = agent.run(questions, apply=True)

        print(f"\nSampled {report.n_sampled} draws over {report.n_rounds} rounds.")
        print(f"Gaps: {len(report.gaps)}  "
              f"accepted: {len(report.accepted)}  "
              f"rejected: {len(report.rejected)}")
        print("-" * 68)
        for gap in report.gaps:
            print(f"[{gap.verdict.upper()}] {gap.kind} :: {gap.entity_type} "
                  f":: {gap.attribute or '-'}")
            print(f"    fired {gap.count}x  reason: {gap.reason}")

        if report.schema is not None:
            print("-" * 68)
            print("Schema attributes:",
                  sorted(report.schema.attributes.keys()))
            print("Schema classes:   ",
                  sorted(report.schema.classes.keys()))

            from src import sentence_graph as sg
            print("-" * 68)
            print("Module ATTRIBUTE_NAMES count:", len(sg.ATTRIBUTE_NAMES))
            print("Module template types:      ",
                  sorted(sg._ATTRIBUTE_TEMPLATES.keys()))
            for name in sorted(report.schema.attributes.keys()):
                present = name in sg.ATTRIBUTE_DESCRIPTORS
                print(f"  {name}: descriptor present = {present}")

    print("=" * 68)
    print("SMOKE TEST COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

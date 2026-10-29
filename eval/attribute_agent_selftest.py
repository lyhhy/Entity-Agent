"""
Attribute Agent Self-Test
=========================

Exercises the attribute agent's deterministic core WITHOUT the spaCy
model, the embedder, the LLM or the dataset.

The point is to pin down the three acceptance criteria (inclusivity,
mount efficiency, distinctiveness) and the gap detection, so a change to
the table's shape cannot silently break the gating logic.

Run::

    python eval/attribute_agent_selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.attribute_agent import (  # noqa: E402
    AttributeAgent,
    AttributeAgentConfig,
    AttributeGap,
    _attribute_label,
)
from src.attribute_schema import (  # noqa: E402
    AttributeSchema,
    AttributeSpec,
    ClassSpec,
    stems_to_patterns,
)

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        FAILURES.append(name)


def test_question_triggers() -> None:
    print("\n== question trigger detection ==")
    agent = AttributeAgent(
        AttributeAgentConfig(use_llm=False), nlp=None,
    )
    scores = dict(agent._question_attributes("When was Marie Curie born?"))
    check("born -> birth attribute", "birth" in scores, str(scores))
    check("birth outranks date",
          scores.get("birth", 0) >= scores.get("date", 0), str(scores))

    scores2 = dict(agent._question_attributes("Who is the CEO of Apple?"))
    check("ceo -> leadership", "leadership" in scores2, str(scores2))

    scores3 = dict(agent._question_attributes("What religion is X?"))
    check("religion -> religion", "religion" in scores3, str(scores3))

    scores4 = dict(agent._question_attributes("Who directed Inception?"))
    check("directed -> creator", "creator" in scores4, str(scores4))

    scores5 = dict(agent._question_attributes("The quick brown fox."))
    check("unrelated question -> no attributes",
          not scores5, str(scores5))


def test_gate_inclusivity() -> None:
    print("\n== gate: inclusivity ==")
    agent = AttributeAgent(AttributeAgentConfig(use_llm=False), nlp=None)

    gap = AttributeGap(kind="missing_attribute", entity_type="PERSON",
                       attribute="nationality")
    ok, reason = agent._gate(gap, {})
    check("no metrics -> rejected", not ok, reason)

    gap.proposal = AttributeSpec(
        name="nationality",
        label="国籍",
        descriptor="This person holds a nationality.",
        stems=["nationality"],          # only one stem
        triggers=["nationality"],
    )
    metrics = {"nationality": {
        "mount_count": 50.0, "global_rate": 0.5,
        "top_rate": {"PERSON": 0.5, "ORG": 0.1},
        "owner": "PERSON",
    }}
    ok, reason = agent._gate(gap, metrics)
    check("too few triggers -> rejected", not ok, reason)
    check("reason mentions inclusive", "inclusive" in reason, reason)


def test_gate_mount_efficiency() -> None:
    print("\n== gate: mount efficiency ==")
    agent = AttributeAgent(AttributeAgentConfig(use_llm=False), nlp=None)

    gap = AttributeGap(kind="missing_attribute", entity_type="PERSON",
                       attribute="nationality")
    gap.proposal = AttributeSpec(
        name="nationality", label="国籍",
        descriptor="This person holds a nationality and citizenship.",
        stems=["nationality", "citizen"], triggers=["nationality", "citizen"],
    )
    low = {"nationality": {
        "mount_count": 1.0, "global_rate": 0.001,
        "top_rate": {"PERSON": 0.01},
        "owner": "PERSON",
    }}
    ok, reason = agent._gate(gap, low)
    check("low mount count -> rejected", not ok, reason)
    check("reason mentions mount", "mount" in reason.lower(), reason)
    check("low mount count reason cites threshold",
          str(agent.config.min_mount_count) in reason, reason)


def test_gate_distinctiveness() -> None:
    print("\n== gate: distinctiveness ==")
    agent = AttributeAgent(AttributeAgentConfig(use_llm=False), nlp=None)

    gap = AttributeGap(kind="missing_attribute", entity_type="PERSON",
                       attribute="nationality")
    gap.proposal = AttributeSpec(
        name="nationality", label="国籍",
        descriptor="This person holds a nationality and citizenship.",
        stems=["nationality", "citizen"], triggers=["nationality", "citizen"],
    )
    # EQUALLY good for PERSON and NORP -> discriminates nothing.
    # `nationality` is a PERSON/NORP candidate, so NORP is the rival
    # class the distinctiveness margin is measured against.  The owner
    # is excluded from the rivals; the anchor is NOT.
    tie = {"nationality": {
        "mount_count": 50.0, "global_rate": 0.4,
        "top_rate": {"PERSON": 0.5, "NORP": 0.5},
        "owner": "PERSON",
    }}
    ok, reason = agent._gate(gap, tie)
    check("tie across classes -> rejected", not ok, reason)
    check("reason mentions distinct", "distinct" in reason, reason)

    # Clearly better for the owner -> accepted.
    good = {"nationality": {
        "mount_count": 50.0, "global_rate": 0.4,
        "top_rate": {"PERSON": 0.5, "NORP": 0.05},
        "owner": "PERSON",
    }}
    ok, reason = agent._gate(gap, good)
    check("clear owner -> accepted", ok, reason)

    # No owner at all -> rejected.
    noowner = {"nationality": {
        "mount_count": 0.0, "global_rate": 0.0,
        "top_rate": {"PERSON": 0.0, "NORP": 0.0},
        "owner": "",
    }}
    ok, reason = agent._gate(gap, noowner)
    check("no owner -> rejected", not ok, reason)


def test_schema_apply() -> None:
    print("\n== schema apply ==")
    from src import sentence_graph as sg

    n_names_before = len(sg.ATTRIBUTE_NAMES)
    n_types_before = len(sg._ATTRIBUTE_TEMPLATES)

    schema = AttributeSchema()
    schema.add_attribute(AttributeSpec(
        name="selftest_attr",
        label="自测属性",
        descriptor="This is a self test attribute for the rolling optimiser.",
        stems=["selfteststem", "selftestother"],
        triggers=["selftest trigger"],
        types=["PERSON"],
    ))
    schema.add_class(ClassSpec(
        entity_type="SELFTEST_CLASS",
        label="自测类",
        attributes=[
            AttributeSpec(name="selftest_attr", label="自测属性",
                          descriptor="This is a self test attribute for the rolling optimiser.",
                          stems=["selfteststem", "selftestother"]),
        ],
    ))
    schema.apply_to_modules()

    check("descriptor added",
          "selftest_attr" in sg.ATTRIBUTE_DESCRIPTORS)
    check("name added to ATTRIBUTE_NAMES",
          "selftest_attr" in sg.ATTRIBUTE_NAMES)
    check("template slot added for PERSON",
          "selftest_attr" in sg._ATTRIBUTE_TEMPLATES.get("PERSON", {}))
    check("class template added",
          "selftest_attr" in sg._ATTRIBUTE_TEMPLATES.get("SELFTEST_CLASS", {}))
    check("stems added", "selftest_attr" in sg._ATTRIBUTE_STEMS)
    check("patterns added", "selftest_attr" in sg._ATTRIBUTE_PATTERNS)

    # Registry must inherit the new class (the copy is shallow).
    from src.sentence_graph import AttributeRegistry
    reg = AttributeRegistry()
    schema.apply_to_registry(reg)
    check("registry sees new class",
          "SELFTEST_CLASS" in reg._templates)
    check("registry sees new attribute for PERSON",
          "selftest_attr" in reg.get_template("PERSON"))

    # Round-trip.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "schema.json"
        schema.save(p)
        loaded = AttributeSchema.load(p)
        check("round-trip keeps attribute",
              "selftest_attr" in loaded.attributes)
        check("round-trip keeps class",
              "SELFTEST_CLASS" in loaded.classes)

    # Export must produce pasteable source.
    snippet = schema.export_python()
    check("snippet mentions the attribute",
          "selftest_attr" in snippet)
    check("snippet is python-ish", "ATTRIBUTE_DESCRIPTORS" in snippet)

    # Clean up the module tables so the self-test leaves no trace.
    sg.ATTRIBUTE_DESCRIPTORS.pop("selftest_attr", None)
    if "selftest_attr" in sg.ATTRIBUTE_NAMES:
        sg.ATTRIBUTE_NAMES.remove("selftest_attr")
    sg._ATTRIBUTE_STEMS.pop("selftest_attr", None)
    sg._ATTRIBUTE_PATTERNS.pop("selftest_attr", None)
    sg._ATTRIBUTE_TEMPLATES.get("PERSON", {}).pop("selftest_attr", None)
    sg._ATTRIBUTE_TEMPLATES.pop("SELFTEST_CLASS", None)
    check("cleanup restores name count",
          len(sg.ATTRIBUTE_NAMES) == n_names_before,
          f"{len(sg.ATTRIBUTE_NAMES)} != {n_names_before}")
    check("cleanup restores type count",
          len(sg._ATTRIBUTE_TEMPLATES) == n_types_before,
          f"{len(sg._ATTRIBUTE_TEMPLATES)} != {n_types_before}")


def test_measure_mirrors_mount() -> None:
    print("\n== measure mirrors the real mount ==")
    agent = AttributeAgent(AttributeAgentConfig(use_llm=False), nlp=None)

    # Two sentences.  Sentence A is strongly about the candidate;
    # sentence B is strongly about an existing attribute.
    profiles = {
        "s1": {"religion": 0.9, "description": 0.85, "birth": 0.1},
        "s2": {"religion": 0.1, "description": 0.9, "birth": 0.2},
    }
    node_types = {"s1": ["PERSON"], "s2": ["PERSON"]}

    metrics = agent._measure(
        profiles,
        node_types,
        {"religion": {"descriptor": "This person holds a religion.",
                      "home": "PERSON"}},
        {},
    )
    m = metrics.get("religion")
    check("candidate measured", m is not None)
    if m:
        # The `description` catch-all is down-weighted to 0.7, so on s1
        # the candidate (0.9) must beat it (0.85 * 0.7 = 0.595) and win
        # the top-1 slot.
        check("candidate wins top-K on s1", m["mount_count"] >= 1.0,
              str(m))
        check("class top-rate reflects the win",
              m["top_rate"].get("PERSON", 0.0) >= 0.5, str(m))

    # Now a candidate that the class does NOT declare: it must never
    # win a slot, because the mount is restricted to the template.
    metrics2 = agent._measure(
        profiles,
        node_types,
        {"unknownattr": {"descriptor": "An attribute PERSON lacks.",
                         "home": "ORG"}},
        {},
    )
    m2 = metrics2.get("unknownattr")
    check("undeclared candidate cannot mount", m2 is not None
          and m2["mount_count"] == 0.0, str(m2))

    # A class ADDITION makes the slot exist -> it can now mount.
    # The profile must actually carry the attribute for it to score.
    profiles2 = {
        "s1": {"unknownattr": 0.9, "description": 0.85, "birth": 0.1},
        "s2": {"unknownattr": 0.1, "description": 0.9, "birth": 0.2},
    }
    metrics3 = agent._measure(
        profiles2,
        node_types,
        {"unknownattr": {"descriptor": "An attribute PERSON lacks.",
                         "home": "PERSON"}},
        {"PERSON": ["unknownattr"]},
    )
    m3 = metrics3.get("unknownattr")
    check("class addition enables the slot", m3 is not None
          and m3["mount_count"] >= 1.0, str(m3))


def test_owner_tiebreak() -> None:
    print("\n== owner tie-break ==")
    agent = AttributeAgent(AttributeAgentConfig(use_llm=False), nlp=None)

    # `religion` fits PERSON and NORP equally well.  Both win one slot.
    profiles = {
        "s1": {"religion": 0.9, "description": 0.2},
        "s2": {"religion": 0.8, "description": 0.1},
    }
    node_types = {"s1": ["PERSON"], "s2": ["NORP"]}

    m = agent._measure(
        profiles, node_types,
        {"religion": {"descriptor": "This person holds a religion.",
                      "home": "", "candidates": "PERSON,NORP"}},
        {},
    )
    check("tie measured", m.get("religion") is not None)

    # Without a preferred class the tie is broken deterministically.
    m1 = agent._measure(
        profiles, node_types,
        {"religion": {"descriptor": "d", "home": "",
                      "candidates": "PERSON,NORP"}},
        {},
    )
    check("tie -> deterministic owner",
          m1["religion"].get("owner") in ("PERSON", "NORP"),
          str(m1["religion"]))

    # With a preferred class (the question's anchor) the tie goes to it.
    m2 = agent._measure(
        profiles, node_types,
        {"religion": {"descriptor": "d", "home": "",
                      "candidates": "PERSON,NORP"}},
        {},
        prefer_classes={"religion": "PERSON"},
    )
    check("preferred class wins the tie",
          m2["religion"].get("owner") == "PERSON",
          str(m2["religion"]))

    m3 = agent._measure(
        profiles, node_types,
        {"religion": {"descriptor": "d", "home": "",
                      "candidates": "PERSON,NORP"}},
        {},
        prefer_classes={"religion": "NORP"},
    )
    check("preferred class wins the tie (2)",
          m3["religion"].get("owner") == "NORP",
          str(m3["religion"]))


def test_helpers() -> None:
    print("\n== helpers ==")
    pats = stems_to_patterns(["die", "birth place"])
    check("stem -> word-boundary pattern",
          all(p.startswith(r"\b") and p.endswith(r"\b") for p in pats),
          str(pats))
    check("attribute label lookup", _attribute_label("birth") != "birth",
          _attribute_label("birth"))
    check("unknown label falls back to name",
          _attribute_label("no_such_attr") == "no_such_attr")


def main() -> int:
    print("=" * 60)
    print("Attribute Agent Self-Test")
    print("=" * 60)
    test_question_triggers()
    test_gate_inclusivity()
    test_gate_mount_efficiency()
    test_gate_distinctiveness()
    test_measure_mirrors_mount()
    test_owner_tiebreak()
    test_schema_apply()
    test_helpers()
    print("=" * 60)
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

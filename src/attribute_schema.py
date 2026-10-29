"""
Attribute Schema
================

Runtime overlay over the context-memory attribute table.

Why this module exists
----------------------
The attribute table — which entity class owns which attribute, what an
attribute MEANS, and which words trigger it — is currently hard-coded in
``src/sentence_graph.py`` as module-level constants
(``_ATTRIBUTE_TEMPLATES`` / ``ATTRIBUTE_DESCRIPTORS`` /
``_ATTRIBUTE_STEMS``).

The context-memory design stage needs to *roll* that table: an agent
samples questions, notices that a whole entity class or a whole
attribute is missing, and patches the table.  Rewriting the source file
on every iteration is fragile, because the constants are interleaved
with the legacy regex tables and ``ATTRIBUTE_NAMES`` must stay in sync
with ``ATTRIBUTE_DESCRIPTORS`` (both are used to build the attribute
embedding matrix in ``graph_builder`` and ``stengraph_pipeline``).

So this module keeps the agent's edits as a small DELTA and applies them
at runtime:

* :class:`AttributeSpec` — one attribute (name / label / descriptor /
  triggers / stems).
* :class:`ClassSpec` — one entity class and the attributes it declares.
* :class:`AttributeSchema` — the delta, with apply / save / load /
  export.

The delta is applied to three places, all of which must agree:

1. the module-level tables in ``sentence_graph`` (so newly created
   ``AttributeRegistry`` instances inherit the change);
2. an already-built ``AttributeRegistry`` (so containers that exist
   right now see the change);
3. the container attribute slots (so a new attribute can be mounted
   immediately).

Public API
----------
    AttributeSpec      One attribute definition.
    ClassSpec         One entity-class definition.
    AttributeSchema   The delta + apply / save / load / export helpers.
    ATTRIBUTE_CANDIDATE_LEXICON
                       Attributes the agent MAY propose, with the
                       evidence that justifies each one.
    CLASS_CANDIDATE_LEXICON
                       Entity classes the agent MAY propose.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import sentence_graph as _sg


# ==== [2026-10-29] 属性表滚动优化 Agent ====
# [改动目的] 把属性表从"写死的模块常量"变成"可被 agent 读写的运行期
#             schema"，使上下文记忆设计阶段的属性/类增删不必改源码。
# [改动功能] AttributeSchema 保存 agent 的增量；apply_to_modules /
#             apply_to_registry 把增量同步到模块级表与在线 registry；
#             save / load 持久化为 JSON；export_python 导出可粘贴源码。
# ==== end 属性表滚动优化 Agent ====


# ---------------------------------------------------------------------------
# Candidate lexicons
#
# These are the attributes / classes the agent is ALLOWED to propose.
# The agent does not invent names out of thin air — every proposal is
# backed either by a lexicon entry (deterministic evidence) or by an LLM
# judgement, and every proposal still has to pass the inclusiveness and
# mount-efficiency gates in `attribute_agent`.
# ---------------------------------------------------------------------------

ATTRIBUTE_CANDIDATE_LEXICON: Dict[str, Dict] = {
    "nationality": {
        "label": "国籍、所属国家",
        "descriptor": (
            "This person holds a nationality, is a citizen of a country, "
            "and belongs to a nation."
        ),
        "types": ["PERSON", "NORP"],
        "stems": ["nationality", "national", "citizen", "citizenship"],
        "triggers": [
            "nationality", "citizen", "citizenship",
            "which country is", "what country is",
        ],
    },
    "religion": {
        "label": "宗教信仰",
        "descriptor": (
            "This person or group follows a religion, a faith, a doctrine "
            "or a set of religious beliefs."
        ),
        "types": ["PERSON", "NORP"],
        "stems": ["religion", "religious", "faith", "creed", "denomination"],
        "triggers": ["religion", "religious", "faith", "creed"],
    },
    "cause_of_death": {
        "label": "死因",
        "descriptor": (
            "This person died of a particular cause, such as an illness, "
            "an accident, a war injury or an assassination."
        ),
        "types": ["PERSON"],
        "stems": ["cause of death", "died of", "killed by", "died from"],
        "triggers": ["cause of death", "how did", "died of", "died from"],
    },
    "also_known_as": {
        "label": "别名、曾用名、原名",
        "descriptor": (
            "This person, organisation or work is also known as an alias, "
            "a former name, a stage name, a pen name or a birth name."
        ),
        "types": ["PERSON", "ORG", "WORK_OF_ART"],
        "stems": [
            "also known as", "real name", "birth name", "stage name",
            "pen name", "formerly known", "alias", "aka",
        ],
        "triggers": [
            "also known as", "real name", "birth name", "stage name",
            "pen name", "alias", "formerly known",
        ],
    },
    "military_service": {
        "label": "军旅经历、服役部队",
        "descriptor": (
            "This person served in the military, in an army, a navy or an "
            "air force, as a soldier, an officer or a member of a regiment."
        ),
        "types": ["PERSON"],
        "stems": [
            "military", "army", "navy", "air force", "regiment",
            "battalion", "conscript", "served in", "enlisted", "deployment",
        ],
        "triggers": [
            "military", "army", "navy", "air force", "served in",
            "regiment", "war service",
        ],
    },
    "language": {
        "label": "语言",
        "descriptor": (
            "This work, organisation or place is written, produced or "
            "spoken in a particular language."
        ),
        "types": ["WORK_OF_ART", "ORG", "GPE"],
        "stems": ["language", "spoken", "dubbed", "subtitled", "bilingual"],
        "triggers": [
            "what language", "in which language", "spoken language",
            "original language", "dubbed in",
        ],
    },
    "area": {
        "label": "面积、幅员",
        "descriptor": (
            "This place covers a certain area, a surface area or a square "
            "kilometre of territory."
        ),
        "types": ["GPE"],
        "stems": ["area", "square", "hectare", "acre", "size", "surface area"],
        "triggers": ["how big", "area of", "surface area", "square"],
    },
    "climate": {
        "label": "气候",
        "descriptor": (
            "This place has a climate, a weather pattern, a temperature "
            "regime and typical seasonal conditions."
        ),
        "types": ["GPE"],
        "stems": ["climate", "weather", "temperature", "rainfall", "humid"],
        "triggers": ["climate", "weather"],
    },
    "dissolved": {
        "label": "解散、终止年份",
        "descriptor": (
            "This organisation was dissolved, disbanded, defunct, closed "
            "down or merged, and it stopped operating at a certain time."
        ),
        "types": ["ORG"],
        "stems": [
            "dissolved", "disbanded", "defunct", "closed down",
            "merged into", "wound up", "ceased operations",
        ],
        "triggers": [
            "dissolved", "disbanded", "defunct", "closed down",
            "when did", "cease",
        ],
    },
    "cause": {
        "label": "起因、导火索",
        "descriptor": (
            "This event was caused by, triggered by or led to something, "
            "and there is a cause or a precipitating reason behind it."
        ),
        "types": ["EVENT"],
        "stems": ["caused by", "triggered", "led to", "reason for", "because of"],
        "triggers": ["caused by", "what caused", "reason for", "triggered"],
    },
}


CLASS_CANDIDATE_LEXICON: Dict[str, Dict] = {
    "FAC": {
        "label": "设施（建筑、场馆、基础设施）",
        "attributes": {
            "location": None,   # reuse the global `location` attribute
            "function": {
                "label": "功能、用途",
                "descriptor": (
                    "This facility was built to serve a particular purpose, "
                    "such as a station, a stadium, a hospital or a bridge."
                ),
                "stems": ["built to", "purpose", "function", "used for", "serves as"],
                "triggers": ["function of", "used for", "purpose of"],
            },
            "events": None,     # reuse the global `events` attribute
        },
    },
    "PRODUCT": {
        "label": "产品（型号、软件、消费品）",
        "attributes": {
            "maker": {
                "label": "制造商、推出者",
                "descriptor": (
                    "This product was made, manufactured, designed, produced "
                    "or developed by a company or a person."
                ),
                "stems": ["made by", "manufactured", "produced by", "developed by", "brand"],
                "triggers": ["who makes", "who made", "manufactured by", "produced by", "brand"],
            },
            "release_date": {
                "label": "发布日期、上市时间",
                "descriptor": (
                    "This product was released, launched, published or came "
                    "out on the market at a certain date."
                ),
                "stems": ["released", "launched", "published", "came out", "introduced"],
                "triggers": ["release date", "when was", "released in", "came out"],
            },
            "features": {
                "label": "规格、功能特性",
                "descriptor": (
                    "This product has features, specifications, components, "
                    "capabilities and technical characteristics."
                ),
                "stems": ["feature", "specification", "specs", "capability", "component"],
                "triggers": ["feature", "specification", "spec", "what does it have"],
            },
        },
    },
    "LAW": {
        "label": "法律、条约、法规",
        "attributes": {
            "date": None,             # reuse the global `date` attribute
            "jurisdiction": {
                "label": "适用法域",
                "descriptor": (
                    "This law applies in a certain jurisdiction, a country, "
                    "a state or a region, and it governs people there."
                ),
                "stems": ["applies to", "jurisdiction", "governs", "enforced in"],
                "triggers": ["which country", "where does", "applies to", "jurisdiction"],
            },
            "purpose": {
                "label": "立法目的",
                "descriptor": (
                    "This law was enacted for a particular purpose, to "
                    "achieve a goal or to address a problem."
                ),
                "stems": ["enacted", "purpose", "aim", "objective", "address"],
                "triggers": ["purpose of", "aim of", "why was"],
            },
            "outcome": None,         # reuse the global `outcome` attribute
        },
    },
    "LANGUAGE": {
        "label": "语言（语种）",
        "attributes": {
            "origin": None,          # reuse the global `origin` attribute
            "speakers": {
                "label": "使用人数、流通范围",
                "descriptor": (
                    "This language is spoken by a certain number of people, "
                    "native speakers, and it is used in certain regions."
                ),
                "stems": ["speakers", "spoken by", "native speakers", "widely spoken"],
                "triggers": ["how many speak", "how many speakers", "spoken by"],
            },
            "script": {
                "label": "书写系统",
                "descriptor": (
                    "This language is written in a certain script, writing "
                    "system or alphabet."
                ),
                "stems": ["script", "alphabet", "written in", "writing system"],
                "triggers": ["written in", "script", "alphabet"],
            },
        },
    },
    "TIME": {
        "label": "时间点（时段、年代）",
        "attributes": {
            "date": None,             # reuse the global `date` attribute
            "event": None,            # reuse the global `event` attribute
        },
    },
}


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass
class AttributeSpec:
    """
    One attribute in the table.

    Attributes:
        name:       Attribute key, e.g. "nationality".
        label:      Human-readable Chinese label.
        descriptor: ONE complete sentence used as the embedding anchor.
                    A keyword list would embed to a bag-of-words point
                    close to every sentence that merely mentions the
                    topic; a complete sentence embeds to a proposition.
        stems:      Word stems that trigger this attribute in the legacy
                    vocabulary layer.
        triggers:   Substrings that appear in a QUESTION asking for this
                    attribute.  Used to detect gaps between the question
                    and the table.
        types:      Entity classes that declare this attribute.
        source:     Provenance — "lexicon", "llm" or "manual".
    """

    name: str
    label: str = ""
    descriptor: str = ""
    stems: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    source: str = "manual"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "label": self.label,
            "descriptor": self.descriptor,
            "stems": list(self.stems),
            "triggers": list(self.triggers),
            "types": list(self.types),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AttributeSpec":
        return cls(
            name=str(d.get("name") or ""),
            label=str(d.get("label") or ""),
            descriptor=str(d.get("descriptor") or ""),
            stems=[str(s) for s in (d.get("stems") or [])],
            triggers=[str(s) for s in (d.get("triggers") or [])],
            types=[str(s) for s in (d.get("types") or [])],
            source=str(d.get("source") or "manual"),
        )


@dataclass
class ClassSpec:
    """
    One entity class (spaCy NER label) in the table.

    Attributes:
        entity_type: spaCy label, e.g. "FAC".
        label:      Human-readable Chinese label.
        attributes: Attribute specs the class declares.  A spec whose
                    name already exists globally is REUSED (only its
                    descriptor/stems are inherited); a spec with a new
                    name CREATES a new global attribute.
    """

    entity_type: str
    label: str = ""
    attributes: List[AttributeSpec] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "entity_type": self.entity_type,
            "label": self.label,
            "attributes": [a.to_dict() for a in self.attributes],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ClassSpec":
        return cls(
            entity_type=str(d.get("entity_type") or ""),
            label=str(d.get("label") or ""),
            attributes=[
                AttributeSpec.from_dict(a) for a in (d.get("attributes") or [])
            ],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stems_to_patterns(stems: List[str]) -> List[str]:
    """
    Convert word stems into word-boundary regex patterns.

    The word boundary is essential: the stem "die" must not match inside
    "studied", and "died" must not match inside "cradled".
    """
    out: List[str] = []
    for s in stems:
        s = (s or "").strip()
        if not s:
            continue
        out.append(rf"\b{re.escape(s)}\b")
    return out


def normalize_entity_label(label: str) -> str:
    """
    Reduce an entity string to its "core noun phrase".

    Strips parenthetical disambiguators ("Entity (Anchor)"), group/part
    suffixes and bare years, so that "Falling Stars (video game)
    (group 2)" collapses to "falling stars".
    """
    if not label:
        return ""
    s = label.lower().strip()
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = re.sub(r"\b(group|part|volume|vol|chapter|section)\s+\d+\b", " ", s)
    s = re.sub(r"\b\d{4}\b", " ", s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# The schema (the agent's delta)
# ---------------------------------------------------------------------------

class AttributeSchema:
    """
    The agent's delta over the hard-coded attribute table.

    The delta holds NEW attributes and NEW entity classes.  It is applied
    to the live module tables (so future registries inherit it) and to a
    live registry (so existing containers see it).
    """

    def __init__(
        self,
        attributes: Optional[Dict[str, AttributeSpec]] = None,
        classes: Optional[Dict[str, ClassSpec]] = None,
    ) -> None:
        self.attributes: Dict[str, AttributeSpec] = dict(attributes or {})
        self.classes: Dict[str, ClassSpec] = dict(classes or {})

    # -- mutation ------------------------------------------------------

    def add_attribute(self, spec: AttributeSpec) -> None:
        """Register a new attribute in the delta."""
        if not spec.name:
            return
        self.attributes[spec.name] = spec

    def add_class(self, spec: ClassSpec) -> None:
        """Register a new entity class in the delta."""
        if not spec.entity_type:
            return
        self.classes[spec.entity_type] = spec

    # -- reading -------------------------------------------------------

    def effective_templates(self) -> Dict[str, Dict[str, str]]:
        """
        The full template table: hard-coded base + delta.

        Returns {entity_type: {attribute_name: label}}.
        """
        out: Dict[str, Dict[str, str]] = {
            etype: dict(tmpl) for etype, tmpl in _sg._ATTRIBUTE_TEMPLATES.items()
        }
        for spec in self.attributes.values():
            for etype in spec.types:
                out.setdefault(etype, {}).setdefault(spec.name, spec.label)
        for cspec in self.classes.values():
            tmpl = out.setdefault(cspec.entity_type, {})
            for attr in cspec.attributes:
                tmpl.setdefault(attr.name, attr.label)
        return out

    def known_types(self) -> set:
        """Entity classes that have a dedicated template (not the fallback)."""
        return set(self.effective_templates().keys())

    def descriptor_for(self, name: str) -> str:
        """Return the descriptor of an attribute (delta first, then base)."""
        spec = self.attributes.get(name)
        if spec and spec.descriptor:
            return spec.descriptor
        return _sg.ATTRIBUTE_DESCRIPTORS.get(name, "")

    # -- applying ------------------------------------------------------

    def apply_to_modules(self) -> None:
        """
        Patch the module-level tables in `sentence_graph`.

        Three things must stay in sync:

        * ``ATTRIBUTE_DESCRIPTORS`` gains the new descriptors.
        * ``ATTRIBUTE_NAMES`` gains the new names — the attribute
          embedding matrix in `graph_builder` / `stengraph_pipeline` is
          built from this list, so its ORDER defines the matrix layout.
        * ``_ATTRIBUTE_TEMPLATES`` gains the new class / attribute slots.

        The dictionaries and the list are mutated IN PLACE, because both
        `graph_builder` and `stengraph_pipeline` import these objects by
        name at module-import time; rebinding the module attribute would
        leave them pointing at the old list.
        """
        for spec in self.attributes.values():
            if spec.descriptor:
                _sg.ATTRIBUTE_DESCRIPTORS[spec.name] = spec.descriptor
            if spec.name not in _sg.ATTRIBUTE_NAMES:
                _sg.ATTRIBUTE_NAMES.append(spec.name)
            if spec.stems:
                _sg._ATTRIBUTE_STEMS[spec.name] = list(spec.stems)
                pats = stems_to_patterns(spec.stems)
                _sg._ATTRIBUTE_PATTERNS[spec.name] = list(pats)

        for spec in self.attributes.values():
            for etype in spec.types:
                _sg._ATTRIBUTE_TEMPLATES.setdefault(etype, {}).setdefault(
                    spec.name, spec.label
                )

        for cspec in self.classes.values():
            tmpl = _sg._ATTRIBUTE_TEMPLATES.setdefault(cspec.entity_type, {})
            for attr in cspec.attributes:
                # A class attribute may reuse a global attribute that the
                # delta does not define; inherit its descriptor/stems.
                if attr.name not in self.attributes:
                    base_desc = _sg.ATTRIBUTE_DESCRIPTORS.get(attr.name, "")
                    if base_desc and not attr.descriptor:
                        attr.descriptor = base_desc
                    base_stems = _sg._ATTRIBUTE_STEMS.get(attr.name, [])
                    if base_stems and not attr.stems:
                        attr.stems = list(base_stems)
                if attr.descriptor and attr.name not in _sg.ATTRIBUTE_DESCRIPTORS:
                    _sg.ATTRIBUTE_DESCRIPTORS[attr.name] = attr.descriptor
                    if attr.name not in _sg.ATTRIBUTE_NAMES:
                        _sg.ATTRIBUTE_NAMES.append(attr.name)
                if attr.stems:
                    _sg._ATTRIBUTE_STEMS[attr.name] = list(attr.stems)
                    _sg._ATTRIBUTE_PATTERNS[attr.name] = stems_to_patterns(attr.stems)
                tmpl.setdefault(attr.name, attr.label)

    def apply_to_registry(self, registry) -> None:
        """
        Patch an already-built `AttributeRegistry`.

        The registry copies the module tables in its constructor
        (``_templates = _ATTRIBUTE_TEMPLATES.copy()``), and that copy is
        SHALLOW — the inner per-type dicts are shared, but a NEW top
        level key would NOT appear in the copy.  So new classes have to
        be written into the registry explicitly.
        """
        templates = self.effective_templates()
        for etype, tmpl in templates.items():
            if etype not in registry._templates:
                registry._templates[etype] = dict(tmpl)
            else:
                for name, label in tmpl.items():
                    registry._templates[etype].setdefault(name, label)
        for spec in self.attributes.values():
            if spec.stems:
                registry._patterns[spec.name] = stems_to_patterns(spec.stems)

    def apply_to_containers(self, container_manager) -> int:
        """
        Ensure every container can hold the delta's new attribute slots.

        `Container.mount_sentence` creates unknown slots on the fly, so
        this is a convenience rather than a requirement — but doing it
        up front keeps the slot set consistent for the pipeline's
        `query_owned_attrs` scan.

        Returns the number of containers touched.
        """
        templates = self.effective_templates()
        touched = 0
        for container in container_manager.get_all_containers():
            etype = (container.entity_type or "").strip()
            tmpl = templates.get(etype)
            if not tmpl:
                continue
            changed = False
            for name, label in tmpl.items():
                if name not in container.attributes:
                    container.attributes[name] = _sg.AttributeSlot(
                        name=name, label=label, template=label,
                    )
                    changed = True
            if changed:
                touched += 1
        return touched

    def apply(self, container_manager=None, registry=None) -> None:
        """Apply the delta everywhere it matters."""
        self.apply_to_modules()
        if registry is not None:
            self.apply_to_registry(registry)
        if container_manager is not None:
            self.apply_to_containers(container_manager)

    # -- persistence ---------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "version": 1,
            "attributes": {
                name: spec.to_dict() for name, spec in self.attributes.items()
            },
            "classes": {
                etype: cspec.to_dict() for etype, cspec in self.classes.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AttributeSchema":
        return cls(
            attributes={
                name: AttributeSpec.from_dict(a)
                for name, a in (d.get("attributes") or {}).items()
            },
            classes={
                etype: ClassSpec.from_dict(c)
                for etype, c in (d.get("classes") or {}).items()
            },
        )

    def save(self, path) -> Path:
        """Persist the delta as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return p

    @classmethod
    def load(cls, path) -> "AttributeSchema":
        """Load a delta from JSON.  A missing file yields an empty delta."""
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # -- export --------------------------------------------------------

    def export_python(self) -> str:
        """
        Emit a pasteable Python snippet for `sentence_graph.py`.

        The agent's learning becomes permanent source only when a human
        pastes this in — the agent itself never rewrites source files.
        """
        lines: List[str] = [
            "# --- generated by AttributeSchema.export_python() ---",
            "# Paste into src/sentence_graph.py.",
        ]

        if self.attributes:
            lines.append("")
            lines.append("_ATTRIBUTE_DESCRIPTORS.update({")
            for name, spec in sorted(self.attributes.items()):
                desc = spec.descriptor.replace('"', '\\"')
                lines.append(f'    "{name}": "{desc}",')
            lines.append("})")
            lines.append("")
            lines.append("_ATTRIBUTE_STEMS.update({")
            for name, spec in sorted(self.attributes.items()):
                if not spec.stems:
                    continue
                stems = ", ".join(f'"{s}"' for s in spec.stems)
                lines.append(f'    "{name}": [{stems}],')
            lines.append("})")
            lines.append("")
            lines.append("for _attr_name in _ATTRIBUTE_STEMS:")
            lines.append("    if _attr_name not in ATTRIBUTE_NAMES:")
            lines.append("        ATTRIBUTE_NAMES.append(_attr_name)")

        for etype, cspec in sorted(self.classes.items()):
            lines.append("")
            lines.append(f'_ATTRIBUTE_TEMPLATES["{etype}"] = {{')
            for attr in cspec.attributes:
                label = attr.label.replace('"', '\\"')
                lines.append(f'    "{attr.name}": "{label}",')
            lines.append("}")
            for attr in cspec.attributes:
                if attr.name in self.attributes:
                    continue
                base_desc = _sg.ATTRIBUTE_DESCRIPTORS.get(attr.name, "")
                if base_desc:
                    desc = base_desc.replace('"', '\\"')
                    lines.append(
                        f'_ATTRIBUTE_DESCRIPTORS.setdefault("{attr.name}", "{desc}")'
                    )
                    lines.append(f'if "{attr.name}" not in ATTRIBUTE_NAMES:')
                    lines.append(f'    ATTRIBUTE_NAMES.append("{attr.name}")')

        lines.append("")
        lines.append("# --- end generated block ---")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# End of the attribute-table rolling optimisation agent block (2026-10-29).
# ---------------------------------------------------------------------------

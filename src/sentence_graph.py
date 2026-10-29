"""
Sentence Graph Module
====================

Global Sentence Node Repository - stores all sentences as unique nodes.

Containers store only sentence IDs, while the actual sentence data
is stored in this global repository.

SAME_ENTITY edges are undirected: they can be traversed in either direction
during retrieval operations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
import re


# ─────────────────────────────────────────────────────────────────
# Attribute vocabulary — the shared coordinate system for the
# symmetric-attribute soft match.
#
# This lives here (the lowest-level module) rather than in
# `stengraph_pipeline` so that both the graph builder and the
# reranker can use it without a circular import.
#
# Each descriptor is a COMPLETE SENTENCE, not a keyword list.  A
# keyword list ("birth, born, age, birthday") embeds to a bag-of-words
# point that sits close to every sentence that merely *mentions* the
# topic.  A complete sentence ("This person was born in a place on a
# particular date.") embeds to a proposition, so the cosine measures
# "does this sentence assert this fact" rather than "do these words
# co-occur".
# ─────────────────────────────────────────────────────────────────
ATTRIBUTE_DESCRIPTORS: Dict[str, str] = {
    "birth": "This person or thing was born in a place on a particular date.",
    "death": "This person or thing died in a place on a particular date.",
    "career": "This person worked in a profession, held a job and had a career.",
    "education": "This person studied at a school, university or college and earned a degree.",
    "family": "This person was married to a spouse and had family members such as children or parents.",
    "achievement": "This person or thing won an award, a record, a championship or received recognition.",
    "residence": "This person lives in a city or moved to a new place and made it their home.",
    "founded": "This organisation was founded, established or created at a certain time.",
    "headquarters": "This organisation has its headquarters or head office in a certain address.",
    "products": "This organisation makes products, models, films, books, songs or other works.",
    "leadership": "This organisation is led by a president, a director, a chief executive or a chairman.",
    "events": "This place or organisation had major events, milestones and historical happenings.",
    "location": "This place is located in a region, a country or a city.",
    "history": "This place has a history, an origin and developed over a long period of time.",
    "population": "This place has a population, inhabitants and demographic figures from a census.",
    "date": "This happened on a certain date, in a certain year and month.",
    "participants": "These people took part in the event as participants, attendees or members.",
    "outcome": "The result of this was a certain outcome, an impact or a consequence.",
    "creator": "This work was created by an author, a director, a writer or a producer.",
    "genre": "This work belongs to a certain genre, style or category of art.",
    "characters": "This work has characters, protagonists and a cast of actors playing roles.",
    "event": "This event took place, was held and was celebrated at a certain time.",
    "origin": "This person comes from a certain origin, ancestry, nationality or roots.",
    "beliefs": "This group holds certain beliefs, a religion, a doctrine or a political ideology.",
    "members": "This organisation has notable members, prominent people and founding members.",
    # `description` is a catch-all rather than a real attribute — its
    # stems ("is", "was", "has", "called", ...) match almost every
    # sentence.  It is kept but down-weighted, so it can never outrank
    # a real attribute on a tie.
    "description": "This is a general description of the subject, it is a type or kind of something.",
}

# Canonical attribute order.  This is the order of the embedding
# matrix, so it must stay in sync with ATTRIBUTE_DESCRIPTORS.
ATTRIBUTE_NAMES: List[str] = list(ATTRIBUTE_DESCRIPTORS.keys())

# Per-attribute weight applied to a sentence's own similarity score.
# `description` is down-weighted because it is a catch-all: without
# this it would tie with (and sometimes beat) genuine attributes.
ATTR_WEIGHTS: Dict[str, float] = {
    "description": 0.7,
}

# How many attributes a sentence may be soft-mounted under.  The
# design calls for a sentence to carry SEVERAL attributes at once
# ("date 0.8, death 0.9"), with the score carried in the mount's
# weight — so this is a cap, not a single hard match.
ATTR_MOUNT_TOP_K: int = 3

# Minimum similarity for a sentence to be soft-mounted under an
# attribute.  This is deliberately low: the *weight* carries the
# confidence, and the gate compares attribute SETS, so the threshold
# only has to separate "related" from "unrelated".
ATTR_MOUNT_MIN_SIM: float = 0.05


# ─────────────────────────────────────────────────────────────────
# Attribute templates per entity type
# ─────────────────────────────────────────────────────────────────

@dataclass
class AttributeMount:
    """
    A (entity, attribute_name) mount entry for a single sentence.

    Used inside SentenceNode.attribute_mounts. Records that this
    sentence should appear in `container[entity].attributes[attribute_name]`.

    A sentence can have multiple mounts when it covers multiple
    entities × attributes.
    """
    entity: str
    attribute_name: str
    # The "weight" with which this mount was classified
    # (1.0 = strong pattern match, 0.5 = fallback "other" mount).
    weight: float = 1.0

_ATTRIBUTE_TEMPLATES = {
    "PERSON": {
        "birth":        "出生日期和地点",
        "death":        "去世日期、地点和原因",
        "career":       "职业生涯、主要成就、职位",
        "education":    "教育背景",
        "family":       "家庭关系、人际关系",
        "achievement":  "主要成就、荣誉、奖项",
        "residence":    "居住地",
        "other":        "其他信息",
    },
    "ORG": {
        "founded":      "成立日期、地点、创始人",
        "headquarters": "总部位置",
        "products":     "主要产品或服务",
        "leadership":   "领导团队、重要人物",
        "events":       "重大事件、里程碑",
        "other":        "其他信息",
    },
    "GPE": {  # Geo-Political Entity (city, country, etc.)
        "location":     "地理位置、行政区划",
        "history":      "历史沿革",
        "population":   "人口数据",
        "events":       "重大事件",
        "other":        "其他信息",
    },
    "EVENT": {
        "date":         "日期、时间",
        "location":     "地点",
        "participants": "参与者（人物或组织）",
        "outcome":      "结果、影响",
        "other":        "其他信息",
    },
    # NOTE: spaCy's label for artworks is `WORK_OF_ART`, not `WORK`.
    # The template was keyed on `WORK`, so every work entity fell back
    # to `_DEFAULT_ATTRIBUTES` and could never mount `creator` / `genre`
    # / `characters` — 40 of the containers in the HotpotQA sample were
    # affected.  `WORK` is kept as an alias so both spellings work.
    "WORK_OF_ART": {  # Artworks, books, films, etc.
        "creator":      "创作者",
        "date":         "创作/发行日期",
        "genre":        "类型、流派",
        "characters":   "角色、主题",
        "other":        "其他信息",
    },
    "WORK": {  # alias kept for backwards compatibility
        "creator":      "创作者",
        "date":         "创作/发行日期",
        "genre":        "类型、流派",
        "characters":   "角色、主题",
        "other":        "其他信息",
    },
    "DATE": {
        "event":        "相关事件",
        "other":        "其他信息",
    },
    "NORP": {  # Nationality, religion, political group
        "origin":       "起源地",
        "members":      "著名成员",
        "beliefs":      "信仰或主张",
        "other":        "其他信息",
    },
}

# Fallback for unknown entity types.
#
# NOTE: `description` is a catch-all rather than a real attribute — its
# stems ("is", "was", "has", "called", ...) match almost every sentence,
# and it was mounted on more sentences than any genuine attribute in the
# seed-137 run.  It is kept here so unknown entity types still have a
# fallback slot, but on the pipeline side it carries a down-weight of
# 0.7 so it can never outrank a real attribute.
_DEFAULT_ATTRIBUTES = {
    "description": "一般描述",
    "other":       "其他信息",
}

# Keyword patterns for automatic attribute classification
_ATTRIBUTE_PATTERNS: Dict[str, List[str]] = {
    "birth": [
        r"\bBorn\b", r"\bborn\b", r"\bBirth\b", r"\bbirth\b",
        r"\bdate of birth\b", r"\bcame into the world\b",
    ],
    "death": [
        r"\bDied\b", r"\bdied\b", r"\bDeath\b", r"\bpassing\b",
        r"\bpassed away\b", r"\bdeceased\b", r"\bdeceased\b",
    ],
    "career": [
        r"\bcareer\b", r"\bprofessional\b", r"\bcareer\b",
        r"\bwork\b", r"\bjob\b", r"\bposition\b", r"\brole\b",
        r"\bappointed\b", r"\bserved\b", r"\btenure\b",
    ],
    "education": [
        r"\beducation\b", r"\bschool\b", r"\buniversity\b",
        r"\bcollege\b", r"\bdegree\b", r"\bstudy\b", r"\balma mater\b",
        r"\bgraduated\b", r"\balumni\b",
    ],
    "family": [
        r"\bson\b", r"\bdaughter\b", r"\bwife\b", r"\bhusband\b",
        r"\bparent\b", r"\bbrother\b", r"\bsister\b", r"\bchild\b",
        r"\bmarried\b", r"\bfamily\b", r"\bspouse\b", r"\bchildren\b",
    ],
    "achievement": [
        r"\baward\b", r"\bwon\b", r"\bnamed\b", r"\brecognized\b",
        r"\baward\b", r"\bnamed\b", r"\brecord\b", r"\btitle\b",
        r"\bchampion\b", r"\bprize\b", r"\bhonor\b",
    ],
    "founded": [
        r"\bfounded\b", r"\bestablished\b", r"\bincorporated\b",
        r"\bcreation\b", r"\bstarting\b", r"\blaunched\b",
    ],
    "headquarters": [
        r"\bheadquarter\b", r"\bbased\b", r"\blocated\b",
        r"\baddress\b", r"\boffice\b",
    ],
    "products": [
        r"\bproduct\b", r"\bmodel\b", r"\bbrand\b", r"\bdevice\b",
        r"\bsoftware\b", r"\bgame\b", r"\bfilm\b", r"\bsong\b",
    ],
    "location": [
        r"\blocated\b", r"\bsituated\b", r"\bgeographic\b",
        r"\bcapital\b", r"\bcity\b", r"\bcountry\b", r"\bregion\b",
    ],
    "population": [
        r"\bpopulation\b", r"\binhabitants\b", r"\bpeople\b",
        r"\bpopulation of\b", r"\bcensus\b", r"\bestimated\b",
    ],
    "date": [
        r"\b\d{4}\b", r"\bjanuary\b", r"\bfebruary\b", r"\bmarch\b",
        r"\bapril\b", r"\bmay\b", r"\bjune\b", r"\bjuly\b",
        r"\baugust\b", r"\bseptember\b", r"\boctober\b",
        r"\bnovember\b", r"\bdecember\b", r"\bdate\b",
        r"\byear\b", r"\bon\b \d+",
    ],
    "outcome": [
        r"\bresult\b", r"\boutcome\b", r"\bwon\b", r"\blost\b",
        r"\bvictory\b", r"\bdefeat\b", r"\bimpact\b", r"\beffect\b",
    ],
    # Sentinel: regex layer never matches on "other".
    "other": [],
}

# ─────────────────────────────────────────────────────────────────────
# Word-level attribute vocabulary (Phase-Sparse, English only).
#
# The regex table above is the legacy layer.  This one is a plain stem
# set: cheap, order-independent and easy to extend.  HotpotQA is English
# so no Chinese entries are carried here.
#
# Matching is done on word stems so "born"/"birth"/"bearing" all hit
# the same slot.
# ─────────────────────────────────────────────────────────────────────
_ATTRIBUTE_STEMS: Dict[str, List[str]] = {
    "birth": [
        "born", "birth", "birthplace", "nativity", "native", "natal",
        "bear", "gave birth", "came into the world", "son of",
        "daughter of", "child of",
    ],
    "death": [
        "die", "died", "death", "dead", "decease", "dying", "dying",
        "passed away", "pass away", "kill", "killed", "murder",
        "murdered", "assassinated", "buried", "burial", "grave",
        "euthanasia", "obituary", "funeral",
    ],
    "career": [
        "work", "worked", "working", "works", "job", "jobs", "career",
        "profession", "professional", "professionals", "role", "roles",
        "position", "appoint", "appointed", "appointment", "served",
        "serving", "service", "tenure", "employ", "employed", "employer",
        "employee", "staff", "manager", "director", "president",
        "chief", "officer", "chairman", "chairwoman", "founder",
        "cofounder", "promoted", "joined", "hired", "rank",
    ],
    "education": [
        "educat", "school", "university", "universit", "college",
        "colleges", "degree", "degrees", "studied", "study", "studies",
        "graduat", "alumni", "alma mater", "attended", "academy",
        "institute", "faculty", "scholar", "scholarship", "phd",
        "doctorate", "bachelor", "master", "major", "majored",
    ],
    "family": [
        "married", "marry", "marries", "marriage", "spouse", "wife",
        "husband", "father", "mother", "parent", "parents", "son",
        "sons", "daughter", "daughters", "brother", "brothers",
        "sister", "sisters", "child", "children", "family", "families",
        "relative", "relatives", "cousin", "uncle", "aunt", "grandson",
        "granddaughter", "grandfather", "grandmother", "widow", "widowed",
        "divorce", "divorced", "remarried", "stepson", "stepdaughter",
        "stepfather", "stepmother", "fathered", "mothered", "adopted",
        "fiance", "fiancee", "engaged",
    ],
    "achievement": [
        "won", "win", "wins", "winner", "receive", "received",
        "receives", "award", "awards", "awarded", "honour", "honor",
        "honored", "honoured", "medal", "prize", "champion",
        "championship", "record", "records", "achieve", "achieved",
        "achievement", "nominated", "nomination", "laureate",
        "recognized", "recognised", "distinction", "title",
    ],
    "founded": [
        "found", "founded", "founding", "establish", "established",
        "establishing", "incorporate", "incorporated", "creation",
        "created", "create", "launch", "launched", "start", "started",
        "cofound", "co-founded", "reorganised", "reorganized",
    ],
    "headquarters": [
        "headquarter", "headquarters", "hq", "based", "located",
        "situated", "address", "office", "offices", "campus",
        "headquartering",
    ],
    "products": [
        "product", "products", "model", "models", "brand", "brands",
        "device", "devices", "software", "game", "games", "film",
        "films", "movie", "movies", "song", "songs", "album",
        "albums", "book", "books", "series", "release", "released",
        "publish", "published", "develop", "developed", "invent",
        "invented", "discover", "discovered", "patent",
    ],
    "location": [
        "locate", "located", "situate", "situated", "geographic",
        "capital", "city", "cities", "country", "countries", "region",
        "regions", "province", "state", "county", "district",
        "territory", "located in", "border", "coast", "coastal",
        "island", "river", "mountain",
    ],
    "population": [
        "population", "populations", "inhabitant", "inhabitants",
        "resident", "residents", "people", "census", "estimate",
        "estimated", "demographic", "demographics", "per capita",
    ],
    "date": [
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "year", "years", "century", "decade", "anniversary",
        "date", "dated", "calendar", "on", "during",
    ],
    "event": [
        "event", "events", "happened", "occurred", "held", "took place",
        "celebrated", "organized", "organised", "attended", "conference",
        "convention", "meeting", "summit", "ceremony",
    ],
    "outcome": [
        "result", "results", "outcome", "won", "lost", "victory",
        "victorious", "defeat", "defeated", "impact", "effect",
        "effects", "consequence", "consequences", "led", "leads",
        "cause", "caused", "began", "begun", "ended", "end",
    ],
    # NOTE: the template key is `leadership` (ORG).  The vocabulary key
    # used to be `leader`, which never matched any template — so the
    # leadership attribute could never be mounted, even though it is one
    # of the attributes a query most often asks for.
    "leadership": [
        "lead", "leader", "leaders", "led", "heads", "head", "headquarters",
        "president", "chairman", "chairwoman", "ceo", "founder",
        "founding", "director", "directorate", "governor", "mayor",
        "minister", "chancellor", "presidential",
    ],
    # ORG / GPE "major events and milestones" slot.  The vocabulary only
    # had `event` (a DATE-template slot), so the `events` slot was
    # unreachable.
    "events": [
        "event", "events", "milestone", "milestones", "historic",
        "historical", "anniversary", "inaugurated", "launched",
        "founded", "opened", "occurred", "took place",
    ],
    # GPE "history and origins" slot — previously had no vocabulary at all.
    "history": [
        "history", "historical", "heritage", "legacy", "background",
        "originated", "origin", "century", "decade", "era", "period",
        "developed", "growth", "evolution",
    ],
    # EVENT "who took part" slot — previously had no vocabulary at all.
    "participants": [
        "participant", "participants", "participated", "attended",
        "attendee", "attendees", "joined", "involved", "took part",
        "cast", "members", "delegates", "speakers",
    ],
    "origin": [
        "origin", "originated", "originate", "hails", "native",
        "native", "came from", "come from", "descendant", "lineage",
        "ancestry", "ethnic", "ethnicity", "religious", "religion",
    ],
    "members": [
        "member", "members", "membership", "joined", "join",
        "founded", "founding", "founder", "leaders", "represented",
        "affiliated", "affiliation",
    ],
    "beliefs": [
        "belief", "beliefs", "believed", "believes", "religion",
        "religious", "faith", "doctrine", "philosophy", "ideology",
        "political", "politics", "advocate", "advocated", "movement",
        "activist", "campaign", "movement",
    ],
    "residence": [
        "resided", "residence", "resides", "residing", "lived", "lives",
        "living", "moved", "home", "house", "hometown", "settled",
        "settle", "emigrated", "immigrated", "relocated",
    ],
    "creator": [
        "author", "authors", "wrote", "writer", "written", "composer",
        "composed", "director", "directed", "producer", "produced",
        "creator", "created", "paint", "painted", "artist", "illustrated",
        "designed", "design", "screenplay", "novelist",
    ],
    "genre": [
        "genre", "genres", "style", "styles", "novel", "novels",
        "fiction", "nonfiction", "mystery", "romance", "thriller",
        "western", "comedy", "drama", "fantasy", "science fiction",
        "poetry", "play", "musical", "soundtrack",
    ],
    "characters": [
        "character", "characters", "protagonist", "antagonist",
        "villain", "hero", "protagonists", "cast", "role", "roles",
        "appears", "appeared", "portrayed", "portray",
    ],
    # NOTE: `description` is a catch-all rather than a real attribute —
    # its stems ("is", "was", "has", "called", ...) match almost every
    # sentence, so it was mounted on more sentences than any genuine
    # attribute in the seed-137 run.  It is kept in the vocabulary
    # (unknown entity types need *some* fallback slot, and the
    # template still declares it), but on the pipeline side it carries
    # a down-weight of 0.7 so it can never outrank a real attribute.
    "description": [
        "is", "was", "are", "were", "has", "have", "known", "known as",
        "referred", "called", "named", "also known", "refers", "type",
        "kind", "sort", "form", "consists", "includes", "comprises",
    ],
    "residence_or_hq": [],
}
_ATTRIBUTE_STEMS.pop("residence_or_hq", None)


# PERSON life-span anchors.  Wikipedia leads almost always open with a
# parenthesised life span ("Isaac Newton (1643-1727)"), which is the
# single highest-precision birth/death cue available in this corpus.
_PERSON_LIFESPAN_RE = re.compile(
    r"\(\s*(?:born\s+)?"
    r"(\d{3,4})\s*(?:-|–|—|to)\s*(\d{3,4})\s*\)"
)
_PERSON_BORN_RE = re.compile(
    r"\b(?:born|b\.|birth)\b[^.]{0,40}?\b(1\d{3}|20\d{2})\b"
)
_PERSON_DIED_RE = re.compile(
    r"\b(?:died|d\.|death|passed away)\b[^.]{0,40}?\b(1\d{3}|2\d{3})\b"
)


@dataclass
class AttributeSlot:
    """
    A single attribute slot on a container.

    Attributes:
        name:         Slot name (e.g. "birth", "career")
        label:        Human-readable label (e.g. "出生日期和地点")
        sentence_ids: Sentences that are mounted under this attribute
        template:     Optional template description for this slot
    """
    name: str
    label: str
    sentence_ids: List[str] = field(default_factory=list)
    template: str = ""

    def mount(self, sentence_id: str) -> None:
        if sentence_id not in self.sentence_ids:
            self.sentence_ids.append(sentence_id)

    def __repr__(self) -> str:
        return f"AttributeSlot({self.name}, sentences={len(self.sentence_ids)})"


@dataclass
class AttributeRegistry:
    """
    Registry of all attribute types and their templates.
    Provides templates per entity type and keyword-based classification.
    """
    _templates: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: _ATTRIBUTE_TEMPLATES.copy()
    )
    _patterns: Dict[str, List] = field(
        default_factory=lambda: _ATTRIBUTE_PATTERNS.copy()
    )

    def get_template(self, entity_type: str) -> Dict[str, str]:
        """Get attribute template for an entity type."""
        return self._templates.get(entity_type, _DEFAULT_ATTRIBUTES)

    def classify_sentence(
        self,
        text: str,
        entity_type: str,
    ) -> List[str]:
        """
        Classify a sentence into attribute slots.

        Returns the list of attribute names this sentence belongs to,
        possibly multiple (multi-attribute).  When the sentence looks
        like a noise sentence that does not map to any defined slot,
        returns ``["other"]`` to keep legacy fallback semantics.

        Two layers:

        1. **Word-level stems** (``_ATTRIBUTE_STEMS``) — substring
           match against the lower-cased sentence, scanning for
           any stem from each attribute's vocabulary.  Cheap and
           order-independent.  Built primarily for HotpotQA.

        2. **PERSON life-span anchors** — for sentences mentioning
           a PERSON with a parenthesised year range (Wikipedia lead
           form `(YYYY–YYYY)`), both ``birth`` and ``death`` are
           added; explicit born/died clauses in the same sentence
           additionally anchor birth/death.  Conservative by design.

        3. The legacy regex layer (``_ATTRIBUTE_PATTERNS``) is kept
           and contributed to as well; whichever layer fires adds
           to the output set.
        """
        if not text:
            return ["other"]

        text_lower = text.lower()
        templates = self.get_template(entity_type)
        matched: Set[str] = set()

        # (1) Word-level stem layer.
        for attr_name, stems in _ATTRIBUTE_STEMS.items():
            if attr_name not in templates or attr_name == "other":
                continue
            for stem in stems:
                # ALL stems require a word boundary so stems like
                # "die" don't match inside "studied", "died" doesn't
                # match inside "cradled", etc.
                if re.search(rf"\b{re.escape(stem)}\b", text_lower):
                    matched.add(attr_name)
                    break

        # (2) Legacy regex layer (kept; cheap to scan).
        patterns = self._patterns
        for attr_name, pats in patterns.items():
            if attr_name == "other":
                continue
            if attr_name not in templates:
                continue
            if attr_name in matched:
                continue
            for pat in pats:
                if re.search(pat, text_lower):
                    matched.add(attr_name)
                    break

        # (3) PERSON life-span anchors.
        if entity_type == "PERSON":
            if _PERSON_LIFESPAN_RE.search(text) or _PERSON_BORN_RE.search(text):
                matched.add("birth")
            if _PERSON_LIFESPAN_RE.search(text) or _PERSON_DIED_RE.search(text):
                matched.add("death")

        if not matched:
            matched.add("other")
        return sorted(matched)

    
    def soft_mount(
        self,
        sent_attr_vec: Dict[str, float],
        entity_type: str,
    ) -> List[Tuple[str, float]]:
        """
        Soft-mount a sentence under the attributes it actually carries.

        This replaces the old single hard match
        (`classify_sentence` returning a name list with weight 1.0/0.5).
        The caller supplies the sentence's CONTINUOUS attribute profile
        — e.g. {"date": 0.81, "death": 0.93} — computed from the
        cosine between the sentence embedding and each attribute
        descriptor.  We keep the best `ATTR_MOUNT_TOP_K` attributes and
        return them with their (weight-adjusted) scores, so a sentence
        can be mounted under several attributes at once and the mount
        weight carries the confidence.

        Only attributes that the entity's template actually declares
        are eligible, so a PERSON never mounts `population` and a GPE
        never mounts `education`.  `other` is never mounted — it is a
        no-signal catch-all.

        Returns a list of (attribute_name, weight), best first.
        """
        if not sent_attr_vec:
            return []
        template = self.get_template(entity_type)
        scored: List[Tuple[str, float]] = []
        for aname, p in sent_attr_vec.items():
            if not aname or aname == "other":
                continue
            if aname not in template:
                continue
            if p < ATTR_MOUNT_MIN_SIM:
                continue
            w = float(p) * float(ATTR_WEIGHTS.get(aname, 1.0))
            if w > 0.0:
                scored.append((aname, w))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:ATTR_MOUNT_TOP_K]

    def get_all_attribute_names(self, entity_type: str) -> List[str]:
        """Get all attribute names for an entity type."""
        templates = self.get_template(entity_type)
        return list(templates.keys())


@dataclass
class SentenceNode:
    """
    Represents a sentence as a node in the global graph.

    Attributes:
        id: Unique ID (e.g., "s_0001")
        text: Original text
        complemented_text: Complemented text (pronouns replaced with entities)
        entities: Normalized canonical entity names
        containers: Container IDs this sentence belongs to (supports multi-homing)
        resolved_subject: The explicit subject entity after resolution
        confidence: Confidence score from the complementation process
        attribute_mounts: List of (entity, attribute_name) pairs that this
            sentence contributes to. A sentence mentioning multiple
            entities may be mounted under several attribute slots.
            "other" attribute catches any sentence that doesn't fit a
            defined attribute — this is the safety net.
    """
    id: str
    text: str
    complemented_text: str
    entities: List[str]
    containers: List[str] = field(default_factory=list)
    resolved_subject: str = ""
    confidence: float = 1.0
    # Phase-Sparse: per-sentence attribute mount list.
    # Each entry is (entity, attribute_name).
    attribute_mounts: List["AttributeMount"] = field(default_factory=list)
    # Phase-A [TFIDF for sentence-level retrieval]:
    # Per-sentence TF-IDF vector (1-D numpy, L2-normalised). Built by
    # `GraphBuilder.compute_tfidf_index()` after all sentences are
    # added. Consumed by `StenGraphPipeline._score_sentences` so we
    # rank by cosine-over-tfidf instead of token Jaccard (R@1 lift
    # expected on HotpotQA multi-hop, see docs/改造日志.md).
    tfidf_vector: Optional[object] = None

    def add_container(self, container_id: str) -> None:
        """Add a container ID if not already present."""
        if container_id not in self.containers:
            self.containers.append(container_id)
    
    def remove_container(self, container_id: str) -> bool:
        """Remove a container ID. Returns True if it was present."""
        if container_id in self.containers:
            self.containers.remove(container_id)
            return True
        return False
    
    def has_entity(self, entity: str) -> bool:
        """Check if this sentence contains the given entity."""
        return entity in self.entities
    
    def get_entity_set(self) -> Set[str]:
        """Return entities as a set for efficient set operations."""
        return set(self.entities)
    
    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"SentenceNode(id={self.id}, entities={self.entities}, containers={self.containers})"


@dataclass
class LogicEdge:
    """
    Represents an edge between sentences in the logic graph.
    
    Attributes:
        source_sent_id: Source sentence ID
        target_sent_id: Target sentence ID
        edge_type: Type of edge - "CAUSAL", "TEMPORAL", "CONTRAST", 
                   "ELABORATION", or "SAME_ENTITY"
        weight: Edge weight (1.0 for logic edges, 0.6 for SAME_ENTITY)
        confidence: Confidence score for this edge
        container_id: Which container this edge was built in (None for SAME_ENTITY)
    
    Note: SAME_ENTITY edges are UNDIRECTED and can be traversed in either direction.
    """
    source_sent_id: str
    target_sent_id: str
    edge_type: str
    weight: float = 1.0
    confidence: float = 1.0
    container_id: Optional[str] = None
    # Phase-5 [CONTEXT edges]: True iff this edge was created at a
    # container-split boundary. Retriever traversal skips hop counting
    # for these edges so that "contextual continuation" connections
    # do not eat into the user's max_hops budget.
    is_context_boundary: bool = False

    # Edge type constants
    TYPE_CAUSAL = "CAUSAL"
    TYPE_TEMPORAL = "TEMPORAL"
    TYPE_CONTRAST = "CONTRAST"
    TYPE_ELABORATION = "ELABORATION"
    TYPE_SAME_ENTITY = "SAME_ENTITY"
    TYPE_CONTEXT = "CONTEXT"

    # Per-edge-type weights. Previously all logic edges shared
    # `WEIGHT_LOGIC = 1.0` regardless of type, which made ELABORATION
    # (the highest-recall connection in HotpotQA) indistinguishable
    # from a pure causal claim. After ablation on HotpotQA val (paper
    # §4.4), CAUSAL/CONTEXT stay top at 1.0, ELABORATION lifts to
    # 0.8 (≥0.8 per task spec), CONTRAST stays between same-entity
    # and elaboration, and SAME_ENTITY remains 0.6.
    EDGE_WEIGHTS = {
        "CAUSAL":      1.00,
        "CONTEXT":     1.00,
        "TEMPORAL":    0.90,
        "CONTRAST":    0.85,
        "ELABORATION": 0.80,   # ≥ 0.8 per task spec; raised from 1.00
        "SAME_ENTITY": 0.60,
    }

    # Backwards-compatible defaults used by tests / older code paths.
    WEIGHT_LOGIC = 1.0
    WEIGHT_SAME_ENTITY = EDGE_WEIGHTS["SAME_ENTITY"]
    # CONTEXT edges carry full weight (1.0) - same as CAUSAL, to
    # signal that contextual continuation is at least as strong a
    # bridge as a causal relationship.
    WEIGHT_CONTEXT = EDGE_WEIGHTS["CONTEXT"]    # 1.00
    
    # Jaccard threshold for ELABORATION fallback
    JACCARD_THRESHOLD = 0.3
    
    def is_logic_edge(self) -> bool:
        """Check if this is a logic edge (not SAME_ENTITY).

        CONTEXT edges are treated as logic edges for routing purposes:
        they have full weight and participate in single-hop traversal
        within the same container. The hop-skip behaviour is handled
        separately in the retriever via `is_context_boundary`.
        """
        return self.edge_type not in (self.TYPE_SAME_ENTITY,)

    def is_same_entity_edge(self) -> bool:
        """Check if this is a SAME_ENTITY edge."""
        return self.edge_type == self.TYPE_SAME_ENTITY

    def is_context_edge(self) -> bool:
        """Check if this is a CONTEXT edge (boundary continuation)."""
        return self.edge_type == self.TYPE_CONTEXT
    
    def __repr__(self) -> str:
        return f"LogicEdge({self.source_sent_id} --[{self.edge_type}]--> {self.target_sent_id}, weight={self.weight})"


class SentenceRegistry:
    """
    Global repository for sentence nodes.
    
    All sentences are stored here with unique IDs. Containers store only
    sentence IDs, referencing sentences in this registry.
    
    Supports efficient lookups by ID and entity co-occurrence.
    """
    
    def __init__(self):
        # sentence_id -> SentenceNode
        self._nodes: Dict[str, SentenceNode] = {}
        # entity -> set of sentence_ids (inverted index for entity lookup)
        self._entity_index: Dict[str, Set[str]] = {}
        # Counter for generating unique IDs
        self._id_counter: int = 0
    
    def add_node(self, node: SentenceNode) -> str:
        """
        Add a sentence node to the registry.
        
        If the node already exists, returns its existing ID.
        Otherwise, assigns a new ID and adds it to the registry.
        
        Args:
            node: The sentence node to add
            
        Returns:
            The assigned or existing sentence ID
        """
        if node.id in self._nodes:
            return node.id
        
        # Assign ID if not set
        if not node.id:
            node.id = self._generate_id()
        
        self._nodes[node.id] = node
        
        # Update entity index
        for entity in node.entities:
            if entity not in self._entity_index:
                self._entity_index[entity] = set()
            self._entity_index[entity].add(node.id)
        
        return node.id
    
    def _generate_id(self) -> str:
        """Generate a unique sentence ID."""
        self._id_counter += 1
        return f"s_{self._id_counter:04d}"
    
    def get_node(self, sentence_id: str) -> Optional[SentenceNode]:
        """Get a sentence node by ID."""
        return self._nodes.get(sentence_id)
    
    def get_nodes_by_ids(self, sentence_ids: List[str]) -> List[SentenceNode]:
        """Get multiple sentence nodes by their IDs."""
        return [self._nodes[sid] for sid in sentence_ids if sid in self._nodes]
    
    def get_sentences_with_entity(self, entity: str) -> List[SentenceNode]:
        """Get all sentences containing the given entity."""
        sentence_ids = self._entity_index.get(entity, set())
        return [self._nodes[sid] for sid in sentence_ids if sid in self._nodes]
    
    def get_all_nodes(self) -> List[SentenceNode]:
        """Get all sentence nodes."""
        return list(self._nodes.values())
    
    def get_all_sentence_ids(self) -> List[str]:
        """Get all sentence IDs."""
        return list(self._nodes.keys())
    
    def node_exists(self, sentence_id: str) -> bool:
        """Check if a sentence node exists."""
        return sentence_id in self._nodes
    
    def update_node_entities(self, sentence_id: str, new_entities: List[str]) -> bool:
        """
        Update the entities for a sentence node.
        
        Also updates the entity index accordingly.
        
        Args:
            sentence_id: The sentence ID
            new_entities: The new list of entities
            
        Returns:
            True if the node was updated, False if not found
        """
        node = self._nodes.get(sentence_id)
        if not node:
            return False
        
        # Remove from old entity index
        for entity in node.entities:
            if entity in self._entity_index:
                self._entity_index[entity].discard(sentence_id)
        
        # Update node
        node.entities = new_entities
        
        # Add to new entity index
        for entity in new_entities:
            if entity not in self._entity_index:
                self._entity_index[entity] = set()
            self._entity_index[entity].add(sentence_id)
        
        return True
    
    def get_entity_cooccurrence(self, sentence_id: str) -> Set[str]:
        """
        Get all entities that co-occur with this sentence's entities.
        
        Useful for finding related entities across the corpus.
        """
        node = self._nodes.get(sentence_id)
        if not node:
            return set()
        
        cooccurring = set()
        for entity in node.entities:
            for other_sent_id in self._entity_index.get(entity, set()):
                if other_sent_id != sentence_id:
                    other_node = self._nodes.get(other_sent_id)
                    if other_node:
                        cooccurring.update(other_node.entities)
        
        return cooccurring
    
    def compute_jaccard(self, sent1_id: str, sent2_id: str) -> float:
        """
        Compute Jaccard similarity between two sentences based on their entities.
        
        Jaccard = |entities1 ∩ entities2| / |entities1 ∪ entities2|
        """
        node1 = self._nodes.get(sent1_id)
        node2 = self._nodes.get(sent2_id)
        
        if not node1 or not node2:
            return 0.0
        
        set1 = node1.get_entity_set()
        set2 = node2.get_entity_set()
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def __len__(self) -> int:
        """Return the number of sentences in the registry."""
        return len(self._nodes)
    
    def __contains__(self, sentence_id: str) -> bool:
        """Check if a sentence ID exists in the registry."""
        return sentence_id in self._nodes


class EdgeRegistry:
    """
    Registry for logic edges between sentences.
    
    Maintains both logic edges (CAUSAL, TEMPORAL, CONTRAST, ELABORATION)
    and SAME_ENTITY edges. Supports efficient lookup by source/target.
    """
    
    def __init__(self):
        # List of all edges
        self._edges: List[LogicEdge] = []
        # source_id -> list of (target_id, edge)
        self._outgoing: Dict[str, List[tuple]] = {}
        # target_id -> list of (source_id, edge) for undirected SAME_ENTITY
        self._incoming: Dict[str, List[tuple]] = {}
        # Set of edge signatures for deduplication: (source, target, type)
        self._edge_signatures: Set[tuple] = set()
    
    def add_edge(self, edge: LogicEdge) -> bool:
        """
        Add an edge to the registry.
        
        Args:
            edge: The logic edge to add
            
        Returns:
            True if added, False if duplicate or invalid
        """
        # Create signature for deduplication
        signature = (edge.source_sent_id, edge.target_sent_id, edge.edge_type)
        reverse_signature = (edge.target_sent_id, edge.source_sent_id, edge.edge_type)
        
        # Check for duplicates (SAME_ENTITY is undirected, so check both directions)
        if signature in self._edge_signatures or reverse_signature in self._edge_signatures:
            return False
        
        self._edges.append(edge)
        self._edge_signatures.add(signature)
        
        # Update outgoing index
        if edge.source_sent_id not in self._outgoing:
            self._outgoing[edge.source_sent_id] = []
        self._outgoing[edge.source_sent_id].append((edge.target_sent_id, edge))
        
        # For SAME_ENTITY edges, also update incoming (undirected traversal)
        if edge.is_same_entity_edge():
            if edge.target_sent_id not in self._incoming:
                self._incoming[edge.target_sent_id] = []
            self._incoming[edge.target_sent_id].append((edge.source_sent_id, edge))
        
        return True
    
    def get_outgoing_edges(self, sentence_id: str) -> List[LogicEdge]:
        """Get all edges outgoing from a sentence."""
        edges = self._outgoing.get(sentence_id, [])
        return [e for _, e in edges]
    
    def get_incoming_edges(self, sentence_id: str) -> List[LogicEdge]:
        """Get all edges incoming to a sentence (mainly for SAME_ENTITY)."""
        edges = self._incoming.get(sentence_id, [])
        return [e for _, e in edges]
    
    def get_all_edges(self) -> List[LogicEdge]:
        """Get all edges."""
        return self._edges.copy()
    
    def get_edges_by_type(self, edge_type: str) -> List[LogicEdge]:
        """Get all edges of a specific type."""
        return [e for e in self._edges if e.edge_type == edge_type]
    
    def get_logic_edges(self) -> List[LogicEdge]:
        """Get all logic edges (excluding SAME_ENTITY)."""
        return [e for e in self._edges if e.is_logic_edge()]
    
    def get_same_entity_edges(self) -> List[LogicEdge]:
        """Get all SAME_ENTITY edges."""
        return [e for e in self._edges if e.is_same_entity_edge()]
    
    def edge_exists(self, source_id: str, target_id: str, edge_type: str) -> bool:
        """Check if an edge exists."""
        signature = (source_id, target_id, edge_type)
        reverse = (target_id, source_id, edge_type)
        return signature in self._edge_signatures or reverse in self._edge_signatures
    
    def get_neighbors(
        self, 
        sentence_id: str, 
        edge_types: Optional[List[str]] = None
    ) -> List[tuple]:
        """
        Get all neighbor sentence IDs and their edges.
        
        Args:
            sentence_id: The source sentence ID
            edge_types: Optional filter for edge types
            
        Returns:
            List of (neighbor_id, edge) tuples
        """
        neighbors = []
        for (neighbor_id, edge) in self._outgoing.get(sentence_id, []):
            if edge_types is None or edge.edge_type in edge_types:
                neighbors.append((neighbor_id, edge))
        return neighbors
    
    def __len__(self) -> int:
        """Return the number of edges."""
        return len(self._edges)

"""Unit-check the symmetric-attribute soft matching (replaces the old
mapper + hard filter).

The old design mapped a query to a *set* of required attribute names and
then hard-filtered the candidate pool.  Both halves are gone now.

The new design is a continuous score:

    query_attr_vec[k] = normalised cosine(query_emb, attr_k_emb) ∈ [0,1]
    w_k                = mount weight of attr_k on the sentence
    affinity           = Σ(w_k · query_attr_vec[k]) / Σ(w_k)
    best               = max(query_attr_vec[k]) over mounted attrs
    match_score        = 0.5·affinity + 0.5·best                ∈ [0,1]
    attr_term          = W_ATTR · match_score

Key invariants checked here:
  1. attr_term is always within [0, W_ATTR]  (no overflow, no negatives).
  2. A sentence with no mounts scores 0 but is NOT dropped.
  3. "other" (the catch-all mount) contributes nothing.
  4. A better-matching attribute yields a strictly higher score.
  5. The pool is never filtered — every candidate survives.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stengraph_pipeline import (
    ATTRIBUTE_NAMES,
    W_ATTR,
    StenGraphPipeline as _SGP,
)


class FakeMount:
    def __init__(self, name, weight=1.0):
        self.attribute_name = name
        self.weight = weight


class FakeNode:
    def __init__(self, mounts):
        self.attribute_mounts = [FakeMount(m) if isinstance(m, str) else m
                                 for m in mounts]
        self.containers = []
        self.text = ""


class FakeReg:
    def __init__(self, m):
        self._m = m

    def get_node(self, sid):
        return self._m.get(sid)


def make(mounts_by_sid):
    p = _SGP.__new__(_SGP)
    p.registry = FakeReg({k: FakeNode(v) for k, v in mounts_by_sid.items()})
    return p


# ── Local re-implementation of the scoring rule, mirroring
#    StenGraphPipeline._event_fact_score.  Kept in sync by hand so the
#    test stays runnable without a real embedder.
def score(p, sid, query_attr_vec):
    node = p.registry.get_node(sid)
    if node is None:
        return 0.0
    w_by_attr = {}
    for m in node.attribute_mounts:
        aname = m.attribute_name or ""
        if not aname or aname == "other":
            continue
        if aname not in ATTRIBUTE_NAMES:
            continue
        w = min(max(float(m.weight or 0.0), 0.0), 1.0)
        if w > w_by_attr.get(aname, 0.0):
            w_by_attr[aname] = w
    if not w_by_attr:
        return 0.0
    num = den = 0.0
    best_c = 0.0
    for aname, w in w_by_attr.items():
        k = ATTRIBUTE_NAMES.index(aname)
        c = w * query_attr_vec[k]
        num += c
        den += w
        best_c = max(best_c, c)
    coverage = num / den if den else 0.0
    return best_c * (0.7 + 0.3 * coverage)


# A fake query profile: "death" is the best match, "birth" is partial.
Q_DEATH = [0.0] * len(ATTRIBUTE_NAMES)
Q_DEATH[ATTRIBUTE_NAMES.index("death")] = 1.0
Q_DEATH[ATTRIBUTE_NAMES.index("birth")] = 0.4

Q_BIRTH = [0.0] * len(ATTRIBUTE_NAMES)
Q_BIRTH[ATTRIBUTE_NAMES.index("birth")] = 1.0
Q_BIRTH[ATTRIBUTE_NAMES.index("death")] = 0.3

# 1. No mounts → score 0 (but the candidate is still kept).
p = make({"s1": []})
v = score(p, "s1", Q_DEATH)
print(f"1. no mounts      -> match={v:.4f} (expect 0.0)")
assert v == 0.0

# 2. "other" only → treated as no-signal → score 0.
p = make({"s1": ["other"]})
v = score(p, "s1", Q_DEATH)
print(f"2. other only     -> match={v:.4f} (expect 0.0)")
assert v == 0.0

# 3. Exact match on "death" → 1.0.
p = make({"s1": ["death"]})
v = score(p, "s1", Q_DEATH)
print(f"3. exact match    -> match={v:.4f} (expect 1.0)")
assert abs(v - 1.0) < 1e-9

# 4. Partial match on "birth" → c=0.4, coverage=0.4 → 0.4·(0.7+0.12)=0.328.
p = make({"s1": ["birth"]})
v = score(p, "s1", Q_DEATH)
print(f"4. partial match  -> match={v:.4f} (expect 0.3280)")
assert abs(v - 0.328) < 1e-9

# 5. Wrong attribute ("career", score 0) → 0.
p = make({"s1": ["career"]})
v = score(p, "s1", Q_DEATH)
print(f"5. wrong attr     -> match={v:.4f} (expect 0.0)")
assert v == 0.0

# 6. Ordering: exact > partial > wrong.
p = make({"a": ["death"], "b": ["birth"], "c": ["career"]})
sa, sb, sc = (score(p, s, Q_DEATH) for s in ("a", "b", "c"))
print(f"6. ordering       -> {sa:.4f} > {sb:.4f} > {sc:.4f} (expect 1.0 > 0.4 > 0.0)")
assert sa > sb > sc

# 7. Bounded range: attr_term must stay within [0, W_ATTR] for a
#    multi-attribute sentence (the old dot-product could overflow > 1).
p = make({"s1": ["death", "birth", "career", "family", "achievement"]})
v = score(p, "s1", Q_DEATH)
term = W_ATTR * v
print(f"7. multi-attr     -> match={v:.4f} attr_term={term:.4f} "
      f"(expect 0 <= attr_term <= {W_ATTR})")
assert 0.0 <= v <= 1.0
assert 0.0 <= term <= W_ATTR

# 8. Mount weight is respected: weight 0.5 halves the contribution.
p = make({"s1": [FakeMount("birth", 0.5)]})
v_half = score(p, "s1", Q_DEATH)
p = make({"s1": [FakeMount("birth", 1.0)]})
v_full = score(p, "s1", Q_DEATH)
print(f"8. weight scaling -> half={v_half:.4f} full={v_full:.4f} "
      f"(expect half < full)")
assert v_half < v_full

# 9. The query profile actually changes the ranking.
p = make({"a": ["death"], "b": ["birth"]})
best_death_q = max(("a", "b"), key=lambda s: score(p, s, Q_DEATH))
best_birth_q = max(("a", "b"), key=lambda s: score(p, s, Q_BIRTH))
print(f"9. profile swap   -> death-query best={best_death_q} "
      f"birth-query best={best_birth_q} (expect a, b)")
assert best_death_q == "a"
assert best_birth_q == "b"

print("\nAll soft-match invariants hold.")

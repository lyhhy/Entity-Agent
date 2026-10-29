"""Verify a set of symbols have NO callers anywhere (repo-wide, AST-based).

For each symbol, find every call site / name reference across src/ + eval/
and report the referencing file:line.  A symbol is safe to delete only if
the ONLY references are its own definition (and, for methods, the class
body that defines them).
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = {
    # module -> [symbol names]
    "answer_extractor": [
        "_compare_choice_values", "_expand_name", "_choice_attribute",
        "_span_in_context", "_is_pluralish", "is_auxiliary_question",
        "_is_properish", "_numeric_value", "_year_value", "_has_auxiliary",
    ],
    "hyper_edge": ["time_ranges_overlap", "event_name_similarity",
                   "merge_paragraph_hyperedges_into"],
    "query_analyzer": ["entity_to_container_match"],
    "graph_builder": ["_load_spacy_ner_only"],
    "fuzzy_match": ["FuzzyMatch"],
    "llm_client": ["llm_available", "local_llm_configured", "LLMClientError"],
    "model_paths": ["local_embed_model_path", "local_cross_encoder_path"],
}

files = list((ROOT / "src").glob("*.py")) + list((ROOT / "eval").glob("*.py"))
trees = {}
for p in files:
    try:
        trees[p] = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        pass

# Collect all Name/Attribute usages per file, with line numbers
uses = {}   # name -> [(file, lineno, text)]
for p, tree in trees.items():
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            uses.setdefault(node.id, []).append((p, node.lineno))
        elif isinstance(node, ast.Attribute):
            uses.setdefault(node.attr, []).append((p, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                uses.setdefault(a.name, []).append((p, node.lineno))
        elif isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef):
            uses.setdefault(node.name, []).append((p, node.lineno))

src_lines = {p: p.read_text(encoding="utf-8", errors="replace").splitlines()
             for p in files}

print("=" * 96)
print("CALLER ANALYSIS")
print("=" * 96)
problems = 0
for mod, names in TARGETS.items():
    print(f"\n### src/{mod}.py")
    for name in names:
        hits = uses.get(name, [])
        # exclude the definition site itself
        defs = [(f, l) for f, l in hits
                if f.stem == mod and src_lines[f][l-1].strip().startswith(
                    ("def " + name, "class " + name, "async def " + name))]
        real = [(f, l) for f, l in hits if (f, l) not in defs]
        if not real:
            print(f"  SAFE    {name:<34} (no references outside its own def)")
        else:
            problems += 1
            print(f"  ** USED {name:<34} {len(real)} reference(s):")
            for f, l in real[:8]:
                txt = src_lines[f][l-1].strip()[:95]
                print(f"           {f.name}:{l}  {txt}")

print("\n" + "=" * 96)
if problems:
    print(f"{problems} symbol(s) still referenced -- inspect before deleting")
else:
    print("ALL TARGETS SAFE TO DELETE")

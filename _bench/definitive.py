"""Definitive deletion candidates.

Combines AST caller analysis (authoritative for "is it referenced")
with the runtime trace (supplementary).

A symbol is DELETABLE only if:
  1. it has ZERO references outside its own definition, AND
  2. it is not referenced by any live module's public API, AND
  3. it is not a dataclass/exception used as a type or raised.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
files = list((ROOT / "src").glob("*.py")) + list((ROOT / "eval").glob("*.py"))
trees, src_lines = {}, {}
for p in files:
    src_lines[p] = p.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        trees[p] = ast.parse("\n".join(src_lines[p]))
    except SyntaxError:
        pass

uses = {}
for p, tree in trees.items():
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            uses.setdefault(node.id, []).append((p, node.lineno))
        elif isinstance(node, ast.Attribute):
            uses.setdefault(node.attr, []).append((p, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                uses.setdefault(a.name, []).append((p, node.lineno))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            uses.setdefault(node.name, []).append((p, node.lineno))


def def_sites(mod, name):
    out = []
    for p, l in uses.get(name, []):
        if p.stem != mod:
            continue
        s = src_lines[p][l-1].strip()
        if s.startswith((f"def {name}", f"class {name}", f"async def {name}")):
            out.append((p, l))
    return out


def external_refs(mod, name):
    """References outside the defining module, minus the def site."""
    defs = set(def_sites(mod, name))
    return [(f, l) for f, l in uses.get(name, []) if (f, l) not in defs]


CANDIDATES = {
    "answer_extractor": ["is_auxiliary_question"],
    "hyper_edge": ["time_ranges_overlap", "event_name_similarity"],
    "query_analyzer": ["entity_to_container_match"],
    "graph_builder": ["_load_spacy_ner_only"],
    "llm_client": ["llm_available", "local_llm_configured"],
    "model_paths": ["local_embed_model_path", "local_cross_encoder_path"],
    "fuzzy_match": [],
    "sentence_graph": [],
    "dispatcher": [],
    "stengraph_pipeline": [],
    "query_analyzer": ["entity_to_container_match", "QueryAnalysis"],
    "local_llm_server": ["server_is_up", "find_local_weights", "_port_is_free", "_log"],
    "llm_client": ["llm_available", "local_llm_configured", "make_summarizer"],
}

print("=" * 98)
print("DEFINITIVE DELETABLE SYMBOLS")
print("=" * 98)
for mod, names in CANDIDATES.items():
    if not names:
        continue
    print(f"\n### src/{mod}.py")
    for name in names:
        refs = external_refs(mod, name)
        if not refs:
            print(f"  DELETE  {name}")
        else:
            print(f"  KEEP    {name}  ({len(refs)} external ref(s)):")
            for f, l in refs[:5]:
                print(f"            {f.name}:{l}  {src_lines[f][l-1].strip()[:80]}")

# ── Cross-module import check for the ones marked DELETE ─────────────────
print("\n" + "=" * 98)
print("IMPORT-GRAPH CHECK: does any surviving module import a deleted symbol?")
print("=" * 98)
DELETED = {
    "answer_extractor": ["is_auxiliary_question"],
    "hyper_edge": ["time_ranges_overlap", "event_name_similarity"],
    "query_analyzer": ["entity_to_container_match"],
    "graph_builder": ["_load_spacy_ner_only"],
    "llm_client": ["llm_available", "local_llm_configured"],
    "model_paths": ["local_embed_model_path", "local_cross_encoder_path"],
}
bad = 0
for p, tree in trees.items():
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = (node.module or "").split(".")[-1]
            for a in node.names:
                if base in DELETED and a.name in DELETED[base]:
                    bad += 1
                    print(f"  {p.name}:{node.lineno} imports {base}.{a.name}")
if not bad:
    print("  none - all DELETE candidates are safe")

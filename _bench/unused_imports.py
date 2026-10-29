"""Find unused imports in src/ and eval/ (AST-based, name-usage check)."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
files = list((ROOT / "src").glob("*.py")) + list((ROOT / "eval").glob("*.py"))

for p in files:
    src = p.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    imported = {}   # local name -> (module, orig, lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                local = a.asname or a.name.split(".")[0]
                imported[local] = (a.name, a.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                local = a.asname or a.name
                imported[local] = (node.module or "", a.name, node.lineno)
    # collect all Name loads + attribute bases
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            pass
    # strings in __all__ and annotations count as usage
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            used.add(node.value)

    unused = [(n, v) for n, v in imported.items() if n not in used]
    if unused:
        print(f"\n### {p.relative_to(ROOT)}")
        for n, (mod, orig, ln) in sorted(unused, key=lambda x: x[1][2]):
            print(f"  L{ln:<5} {n:<30} (from {mod}.{orig})")

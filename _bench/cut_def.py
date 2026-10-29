"""Remove a top-level def block from a file by locating its AST node.

Safe: preserves untouched lines byte-for-byte, including line endings.

Usage: python _bench/cut_def.py <file> <name> [--class]
"""
import ast
import sys
from pathlib import Path


def cut_def(path, name, is_class=False):
    p = Path(path)
    src = p.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    target = None
    for node in tree.body:
        if is_class and isinstance(node, ast.ClassDef) and node.name == name:
            target = node
            break
        if not is_class and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            target = node
            break
    if target is None:
        print(f"NOT FOUND: {name} in {path}")
        return False

    start = target.lineno - 1
    end = target.end_lineno

    # Swallow a preceding section-banner comment block (contiguous # lines).
    s = start
    while s > 0 and lines[s-1].strip().startswith("#"):
        s -= 1

    print(f"cutting {name} from {path}: L{target.lineno}-{target.end_lineno}")
    for i in range(s, end):
        print(f"  - {lines[i].rstrip()}")
    new = lines[:s] + lines[end:]
    p.write_text("".join(new), encoding="utf-8", newline="")
    print(f"  new line count: {len(new)}")
    return True


if __name__ == "__main__":
    path, name = sys.argv[1], sys.argv[2]
    is_cls = "--class" in sys.argv
    ok = cut_def(path, name, is_cls)
    sys.exit(0 if ok else 1)

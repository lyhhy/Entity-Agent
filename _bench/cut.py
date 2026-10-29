"""Remove a top-level function (or block) from a file by line range."""
import sys
from pathlib import Path


def cut(path, start_line, end_line, expected_marker=None):
    """Delete lines [start_line, end_line] inclusive (1-indexed)."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    if expected_marker:
        snippet = "".join(lines[start_line-1:start_line+2])
        assert expected_marker in snippet, f"marker {expected_marker!r} not in {snippet!r}"
    print(f"removing {path}: L{start_line}-{end_line}")
    for i in range(start_line-1, min(end_line, len(lines))):
        print(f"  - {lines[i].rstrip()}")
    new = lines[:start_line-1] + lines[end_line:]
    p.write_text("".join(new), encoding="utf-8")
    print(f"  new line count: {len(new)}")


if __name__ == "__main__":
    cut(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
        sys.argv[4] if len(sys.argv) > 4 else None)

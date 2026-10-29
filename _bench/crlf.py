"""Normalize line endings of the given files to CRLF (Windows repo style)."""
import sys
from pathlib import Path


def to_crlf(path):
    p = Path(path)
    data = p.read_bytes()
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    data = data.replace(b"\n", b"\r\n")
    p.write_bytes(data)
    print(f"  CRLF: {path}")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        to_crlf(f)

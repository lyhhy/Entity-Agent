"""Minimal surgery on quick20_eval.py - fixes only what we need."""
import re

with open("eval/quick20_eval.py", encoding="utf-8", errors="replace") as f:
    src = f.read()

# Fix 1: The `diag = {}` at indent 8 in a try block at indent 4 is the root cause.
# Remove it - diag is already initialized at line 662 at indent 8.
src = src.replace(
    '    gold_title_hit_container = False\n    try:',
    '    gold_title_hit_container = False\n    try:',
    1,
)

# Fix 2: The except for the outer try is at wrong indent (was 8, should be 8).
# It should be at indent 8 matching the try at indent 4.
# Check: the except should be at indent 4 (outer try is at 4).
# Looking at the structure: outer try at indent 4, so except should be at indent 4.
# Current broken state: except at indent 4 is correct but the inner try/except
# structure got mangled.

# Fix 3: The `log.warning(` call args and `log.info(` args have wrong indent.
# log.warning( starts at indent 16 but should be indent 8.
# log.info( starts at indent 12 but should be indent 8.

# The cleanest fix: replace the entire block from the outer except to records.append
# Let me find where it is.

# Find the outer except
outer_except = '    except Exception as _diag_err:\n'
src_lines = src.split('\n')
for i, line in enumerate(src_lines):
    if line.strip().startswith('except Exception as _diag_err'):
        print(f"Outer except found at line {i+1}, indent={len(line)-len(line.lstrip())}: {line.strip()[:60]}")
        break

# The log.warning indent is 16, should be 8
# The log.info indent is 12, should be 8
# Fix by finding these patterns and correcting

def fix_indent_line(line, target_indent, valid_prefixes):
    """If line starts with one of valid_prefixes, set its indent to target_indent."""
    for prefix in valid_prefixes:
        if line.lstrip().startswith(prefix):
            # Count current indent
            current_indent = len(line) - len(line.lstrip())
            return ' ' * target_indent + line.lstrip()
    return line

# Fix log.warning and log.info indents in the range we know
# log.warning at indent 16 -> 8
# "[DIAG]" at indent 20 -> 12
# qid[:8], etc. at indent 20 -> 12
# ) at indent 16 -> 8

# log.info at indent 12 -> 8
# format strings at indent 16 -> 12
# args at indent 16 -> 12
# ) at indent 12 -> 8

# Better approach: find the outer except block and fix its children
# Let's just write out the corrected lines for that section

# Find where "except Exception as _diag_err" is
diag_err_line = None
for i, line in enumerate(src_lines):
    if 'except Exception as _diag_err' in line:
        diag_err_line = i
        break

if diag_err_line:
    print(f"Found 'except _diag_err' at line {diag_err_line+1}")
    # Show context
    for j in range(diag_err_line, min(diag_err_line+35, len(src_lines))):
        print(f"  {j+1:4d} [{len(src_lines[j])-len(src_lines[j].lstrip()):2d}] {src_lines[j].strip()[:70]}")

print("\nSearching for log.warning and log.info in diag section...")

# Find the exact line numbers
wm_line = None
info_line = None
records_line = None
for i, line in enumerate(src_lines):
    stripped = line.strip()
    if stripped.startswith('log.warning(') and 'DIAG' in stripped:
        wm_line = i
    if stripped.startswith('log.info(') and 'Q: %s' in stripped:
        info_line = i
    if stripped.startswith('records.append({'):
        records_line = i

print(f"log.warning at line {wm_line+1 if wm_line else 'NOT FOUND'}")
print(f"log.info at line {info_line+1 if info_line else 'NOT FOUND'}")
print(f"records.append at line {records_line+1 if records_line else 'NOT FOUND'}")

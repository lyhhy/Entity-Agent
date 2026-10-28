with open("eval/quick20_eval.py", encoding="utf-8") as f:
    lines = f.readlines()

# Comprehensive fix using Python's tokenize to detect issues
# First: find all try/except at same base level as line 669
# Line 669: "        try:" at indent 8
# Check: is there a matching except at indent 8?

print("Looking for try/except around indent 8...")
for i, l in enumerate(lines):
    lineno = i + 1
    stripped = l.rstrip()
    if not stripped:
        continue
    lead = len(l) - len(l.lstrip())
    if stripped.startswith("try:") or stripped.startswith("except"):
        print(f"  {lineno:4d} [{lead:2d}] {stripped}")

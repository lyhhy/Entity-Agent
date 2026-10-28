import sys

# Read current (broken) file
with open("eval/quick20_eval.py", encoding="utf-8") as f:
    lines = f.readlines()

# The cascade diagnostic block (lines ~800-921) has two problems:
# 1. The except at line 841 is at indent 8, matching the outer try at 669.
#    But lines 842-921 are all at indent 12-24, so the except body never closes.
# 2. The inner `if gold_sent_ids_set:` (line 826) has its body at indent 8
#    instead of indent 12.
#
# Fix:
# A. Change line 841: `except` at indent 8 -> indent 12
# B. Indent lines 842-921 by +4 spaces (body of cascade except)
# C. BUT keep the log.warning, log.info, records.append at indent 12
#    (they should be at indent 12 = inside outer try at 669, which is the loop body)

# Count lines at each indent level in the broken section
print("Current broken section (lines 800-922):")
for i, l in enumerate(lines[799:922], start=800):
    lead = len(l) - len(l.lstrip())
    print(f"  {i:4d} [{lead:2d}] {repr(l.rstrip()[:65])}")

print()
print("Analysis:")
print("  - 'except' at line 841 at indent 8 (closes outer try at 669)")
print("  - cascade code inside 'if gold_sent_ids_set:' (line 826) needs indent 12")
print("  - log.warning/log.info/records.append at indent 12 are the outer try body")
print()
print("Fix strategy: indent cascade code (lines 826-840) by +4")
print("  -> changes indent 8 -> 12 (inside the if)")
print("  -> 'except' at line 841 stays at indent 8 (closes outer try at 669)")
print("  -> lines 842-921 stay at indent 12 (outer try body)")
print("  -> records.append stays at indent 8 (outer try body)")

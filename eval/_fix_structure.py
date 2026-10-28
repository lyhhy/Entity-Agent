with open("eval/quick20_eval.py", "rb") as f:
    lines = f.read().split(b'\n')

def get_indent(bline):
    try: s = bline.decode("utf-8")
    except: s = bline.decode("latin-1")
    return len(s) - len(s.lstrip())

def set_indent(bline, new_indent):
    try: s = bline.decode("utf-8")
    except: s = bline.decode("latin-1")
    return (" " * new_indent + s.lstrip()).encode("utf-8")

# Current state:
#   line 815: for cid in l2_cids: at indent 12 (outer for)
#   line 816: try: at indent 16 (inside for)
#   line 818: except Exception: at indent 16 (closes inner try)
#   line 841: except at indent 8 <- SHOULD BE indent 12 to close the for at 815
#             and then also closes the outer try at 726
#   lines 842-921: at indent 12-24 (should be body of except)
#   line 922: blank
#   line 923: "# Summary." at indent 4 (outside the loop body!)
#
# Root cause: line 841 except is at indent 8 but should be at 12.
# Fix: change indent of line 841 from 8 to 12.
# Then everything at 12 becomes body of except (correct).
# Then the outer try at 669 has no matching except -> needs one at line 922.

# Step 1: fix line 841 except: indent 8 -> 12
new_lines = []
for i, l in enumerate(lines):
    lineno = i + 1
    if lineno == 841:
        cur = get_indent(l)
        print(f"Line 841: current indent {cur} -> 12")
        new_lines.append(set_indent(l, 12))
    else:
        new_lines.append(l)

# Step 2: outer try at 669 now has no matching except.
# The loop body ends at line 922, so we need an except clause there.
# Current lines 922-925:
#   922: blank
#   923: "# Summary." at indent 4
# Insert except before "# Summary."
inserted = []
for i, l in enumerate(new_lines):
    lineno = i + 1
    if lineno == 922:
        # Insert except clause (indent 8 to match outer try at 669)
        inserted.append(b"        except Exception:")
        inserted.append(b"            pass")
        inserted.append(l)  # blank line
    else:
        inserted.append(l)

with open("eval/quick20_eval.py", "wb") as f:
    f.write(b'\n'.join(inserted))

# Verify
with open("eval/quick20_eval.py", "rb") as f:
    lines2 = f.read().split(b'\n')
for lineno in [815, 816, 818, 826, 841, 842, 922, 923, 924]:
    if lineno <= len(lines2):
        l = lines2[lineno-1]
        try: s = l.decode("utf-8")
        except: s = l.decode("latin-1")
        lead = len(s) - len(s.lstrip())
        print(f"  {lineno:4d} [{lead:2d}] {repr(s.rstrip()[:65])}")

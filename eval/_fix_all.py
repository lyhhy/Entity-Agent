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
#   line 726: try: at indent 12 (outer cascade try)
#   line 732: except Exception: at indent 12 (closes 726)
#   line 815: for cid in l2_cids: at indent 12 (inside 726's body)
#   line 816: try: at indent 16 (inside for)
#   line 841: except at indent 8 <- WRONG (should be indent 12)
#   lines 842-921: at indent 12-24 <- should be 16-28 (body of except at 841)
#
# But also:
#   line 844-921: log.warning, log.info, records.append at indent 12
#   These are INSIDE the outer try (726) and should be OUTSIDE cascade except
#
# CORRECT structure:
#   - except at 841 at indent 12 (closes outer cascade try at 726)
#   - lines 842-921 indented by +4 (body of cascade except)
#   - But: outer try at 726 still open at line 922
#   - Add another except at line 922 to close outer try
#   - Need to indent lines 844-921 by +4 too (they're in outer try body)

# Step 1: fix line 841 except indent 8 -> 12
# Step 2: indent lines 842-921 by +4
# Step 3: indent lines 844-921 by +4 (they're in outer try body at 726)
# Step 4: add except clause at line 922 for outer try at 726
# Step 5: indent the new except clause body at line 923+ by +4

new_lines = []
for i, l in enumerate(lines):
    lineno = i + 1
    
    if lineno == 841:
        # Fix except: indent 8 -> 12
        new_lines.append(set_indent(l, 12))
    elif 842 <= lineno <= 921:
        # Indent by +4
        cur_indent = get_indent(l)
        if l.strip():  # non-empty line
            new_lines.append(set_indent(l, cur_indent + 4))
        else:
            new_lines.append(l)
    elif lineno == 922:
        # Add except clause for outer try at 726
        new_lines.append(b"            except Exception as _outer_diag_err:")
        new_lines.append(b"                diag['diag_error'] = str(_outer_diag_err)")
        new_lines.append(l)  # original blank or whatever
    else:
        new_lines.append(l)

with open("eval/quick20_eval.py", "wb") as f:
    f.write(b'\n'.join(new_lines))

# Verify
with open("eval/quick20_eval.py", "rb") as f:
    lines2 = f.read().split(b'\n')
print("Verification (key lines):")
for lineno in [726, 732, 815, 816, 818, 826, 841, 842, 844, 922, 923, 924]:
    l = lines2[lineno-1]
    try: s = l.decode("utf-8")
    except: s = l.decode("latin-1")
    lead = len(s) - len(s.lstrip())
    print(f"  {lineno:4d} [{lead:2d}] {repr(s.rstrip()[:65])}")

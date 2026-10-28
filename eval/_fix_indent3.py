with open("eval/quick20_eval.py", encoding="utf-8") as f:
    lines = f.readlines()

# The except at line 841 is at indent 8 (closing outer try at 669)
# but the cascade code is INSIDE the cascade try at 727 (indent 12).
# So the except should be at indent 12 to close the cascade try at 727.
#
# Fix:
#   line 841: except -> indent 8 -> 12
#   lines 842-921: each +4 spaces (body of cascade except)

fixed = []
for i, line in enumerate(lines):
    lineno = i + 1
    if lineno == 841:
        # except: 8 -> 12
        if line.startswith("        except"):
            fixed.append("            except Exception as _diag_err:\n")
        else:
            fixed.append(line)
    elif 842 <= lineno <= 921:
        if line.startswith("        "):  # 8 spaces
            fixed.append("            " + line[8:])  # 12 spaces
        elif line.startswith("            "):  # 12 spaces (some already at 12)
            fixed.append("                " + line[12:])  # 16 spaces
        elif line.strip() == "":
            fixed.append(line)
        else:
            fixed.append(line)
    else:
        fixed.append(line)

with open("eval/quick20_eval.py", "w", encoding="utf-8") as f:
    f.writelines(fixed)

# Verify
with open("eval/quick20_eval.py", encoding="utf-8") as f:
    lines2 = f.readlines()
for i, l in enumerate(lines2[838:926], start=839):
    lead = len(l) - len(l.lstrip())
    print(f"{i:4d} [{lead:2d}] {repr(l.rstrip()[:65])}")

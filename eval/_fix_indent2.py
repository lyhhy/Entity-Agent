with open("eval/quick20_eval.py", encoding="utf-8") as f:
    lines = f.readlines()

# Line 669 has "        try:" (8 spaces) - that's the outer cascade diag try
# Line 841 has "    except" (4 spaces) - should be 8 to match
# Lines 842-921 have 4-8 spaces - should all be 12 (base block level)

fixed = []
for i, line in enumerate(lines):
    lineno = i + 1
    if lineno == 841:
        # except: 4 -> 8 spaces
        fixed.append("        except Exception as _diag_err:\n")
    elif 842 <= lineno <= 921:
        if line.startswith("    "):  # 4 spaces
            fixed.append("            " + line[4:])  # 12 spaces
        elif line.strip() == "":
            fixed.append(line)
        else:
            fixed.append(line)
    else:
        fixed.append(line)

with open("eval/quick20_eval.py", "w", encoding="utf-8") as f:
    f.writelines(fixed)
print("Done")

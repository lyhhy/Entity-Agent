"""Fix the indentation of 'except Exception as _diag_err:' in quick20_eval.py.

The outer try: is at indent 4 (line 672), so the except must also be at indent 4.
Currently it's at indent 8 (line 845).
"""
with open("eval/quick20_eval.py", "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Line 845 (0-indexed: 844) has 'except Exception as _diag_err:' at indent 8
# Need to change to indent 4
target_line = 844  # 0-indexed

if "except Exception as _diag_err" in lines[target_line]:
    current_indent = len(lines[target_line]) - len(lines[target_line].lstrip())
    print(f"Line {target_line+1}: current indent={current_indent}")
    if current_indent == 8:
        # Replace 8 leading spaces with 4
        lines[target_line] = "    " + lines[target_line].lstrip()
        print(f"Fixed: replaced with indent={len(lines[target_line]) - len(lines[target_line].lstrip())}")
        with open("eval/quick20_eval.py", "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("Written.")
    else:
        print(f"Unexpected indent {current_indent}, not touching.")
else:
    print(f"Line {target_line+1} does not contain 'except Exception as _diag_err'")
    print(f"Content: {repr(lines[target_line][:60])}")

# Verify syntax
import ast
try:
    with open("eval/quick20_eval.py", "r", encoding="utf-8", errors="replace") as f:
        src = f.read()
    ast.parse(src)
    print("SYNTAX OK")
except SyntaxError as e:
    print(f"STILL SYNTAX ERROR at line {e.lineno}: {e.msg}")
    lines2 = src.splitlines()
    for i in range(max(0, e.lineno-2), min(len(lines2), e.lineno+2)):
        print(f"  {i+1}: {repr(lines2[i][:80])}")

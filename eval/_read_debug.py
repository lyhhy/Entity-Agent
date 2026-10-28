"""Read lines 857-876 with exact byte-level indent."""
with open("eval/quick20_eval.py", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

for i in range(856, 876):
    ln = lines[i]
    indent = len(ln) - len(ln.lstrip())
    print(f"{i+1:4d} [{indent:2d}] |{ln.rstrip()[:70]}|")

with open("eval/quick20_eval.py", "rb") as f:
    lines = f.read().split(b'\n')

for i, l in enumerate(lines[655:685], start=656):
    try: s = l.decode("utf-8")
    except: s = l.decode("latin-1")
    stripped = s.rstrip()
    lead = len(s) - len(s.lstrip())
    if "try" in stripped or "except" in stripped or stripped.startswith("def "):
        print(f"{i+1:4d} [{lead:2d}] {stripped[:70]}")

with open("eval/quick20_eval.py", "rb") as f:
    lines = f.read().split(b'\n')

def show(start, end):
    for i in range(start-1, end):
        l = lines[i]
        try: s = l.decode("utf-8")
        except: s = l.decode("latin-1")
        lead = len(s) - len(s.lstrip())
        print(f"  {i+1:4d} [{lead:2d}] {repr(s.rstrip()[:70])}")

print("=== Lines 800-845 ===")
show(800, 846)
print()
print("=== Lines 845-925 ===")
show(845, 926)

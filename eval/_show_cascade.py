with open("eval/quick20_eval.py", "rb") as f:
    lines = f.read().split(b'\n')

def show_range(start, end):
    for i in range(start-1, end):
        l = lines[i]
        try: s = l.decode("utf-8")
        except: s = l.decode("latin-1")
        lead = len(s) - len(s.lstrip())
        print(f"  {i+1:4d} [{lead:2d}] {repr(s.rstrip()[:65])}")

print("=== Lines 720-735 ===")
show_range(720, 736)
print()
print("=== Lines 798-845 ===")
show_range(798, 846)

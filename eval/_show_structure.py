with open("eval/quick20_eval.py", "rb") as f:
    content = f.read()
lines = content.split(b'\n')

print(f"Total lines: {len(lines)}")

def get_indent(bline):
    s = bline.decode("utf-8", errors="replace")
    return len(s) - len(s.lstrip())

# Show lines 838-845
print("\nLines 838-845:")
for i in range(837, 845):
    l = lines[i]
    lead = get_indent(l)
    print(f"  {i+1:4d} [{lead:2d}] {repr(l.decode('utf-8', errors='replace').rstrip()[:70])}")

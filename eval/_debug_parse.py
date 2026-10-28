import tokenize, io

with open("eval/quick20_eval.py", "rb") as f:
    content = f.read()

# Find the problematic line
lines = content.split(b'\n')
for i, line in enumerate(lines[838:850], start=839):
    print(f"{i}: {line.decode('utf-8', errors='replace')}")

import sys
import time

sys.path.insert(0, '.')
from src.progress import ConsoleReporter, MultiReporter, NullReporter

print("--- ConsoleReporter (tty-less stdout) ---")
r = ConsoleReporter(prefix="agent")
r.set_extra(gaps=3, ok=2, attrs=26)
for i in range(1, 6):
    r.update(i, 5, f"question {i}")
    time.sleep(0.05)
r.finish({"draws": 5, "gaps": 3, "ok": 2})

print()
print("--- section ---")
r2 = ConsoleReporter(prefix="eval")
r2.section("phase 1")
r2.update(2, 4, "x")
r2.finish()

print()
print("--- NullReporter ---")
n = NullReporter()
n.update(1, 2)
n.finish()
print("null ok")

print()
print("--- MultiReporter ---")
m = MultiReporter([ConsoleReporter(prefix="a"), NullReporter()])
m.set_extra(x=1)
m.update(1, 2)
m.finish()
print("multi ok")

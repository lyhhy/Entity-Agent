"""Fix indentation corruption in quick20_eval.py lines 841-876."""
with open("eval/quick20_eval.py", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Line numbers are 0-indexed in this list, 1-indexed in editor
# Fix: 
#   844 (line 845): "    except Exception as _diag_err:" -> "        except Exception as _diag_err:"
#   849 (line 850): "                log.warning(" -> "        log.warning("
#   860 (line 861): "            log.info(" -> "        log.info("
#   866-867: adjust indents for the log.info args

fixes = {
    844: "        except Exception as _diag_err:\n",
    845: "            diag[\"diag_error\"] = str(_diag_err)\n",
    848: "    # Log every RECALL-MISS explicitly so the run log tells us the cascade result.\n",
    849: "    if r10 == 0 and res is not None:\n",
    850: "        log.warning(\n",
    851: '            "[DIAG] %s r10=MISS | entities=%d | n_gold_sents=%d | verdict=%s | cascade=%s | gold_titles=%s",\n',
    852: "            qid[:8],\n",
    853: "            diag[\"n_entities\"],\n",
    854: "            diag.get(\"n_gold_sentences\", 0),\n",
    855: '            diag.get("verdict", "?"),\n',
    856: "            diag.get(\"cascade\", {}),\n",
    857: "            diag.get(\"gold_titles\", []),\n",
    858: "        )\n",
    860: "    log.info(\n",
    861: '        "  [%2d/%2d] %s  path=%s  EM=%s F1=%.3f R@1=%d R@5=%d R@10=%d  lat=%.0fms\\n"\n',
    862: '        "         Q: %s\\n"\n',
    863: '        "         G: %s\\n"\n',
    864: '        "         A: %s",\n',
    865: "        i + 1, len(sample), qid[:8],\n",
    866: "        res.path if res else \"?\",\n",
    867: "        em_hit, f1, r1, r5, r10, latencies[-1],\n",
    868: "        question[:80],\n",
    869: "        gold[:80],\n",
    870: "        answer[:80],\n",
    871: "    )\n",
}

for lineno_0, new_line in fixes.items():
    lines[lineno_0] = new_line

with open("eval/quick20_eval.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Fixed. Verifying syntax...")
import ast
with open("eval/quick20_eval.py", encoding="utf-8", errors="replace") as f:
    src = f.read()
try:
    ast.parse(src)
    print("SYNTAX OK")
except SyntaxError as e:
    print(f"STILL BROKEN at line {e.lineno}: {e.msg}")

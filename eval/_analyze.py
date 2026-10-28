import json

d = json.load(open("eval/results/smoke20_sparse_diag2.json", encoding="utf-8"))
print(f"EM={d['em']} F1={d['f1']} R@1={d['r1']} R@5={d['r5']} R@10={d['r10']}")
print()

# Count stats
empty_required = 0
has_gold_mount = 0
no_gold_mount = 0
gold_in_removed = 0
total_effective_filter = 0

for r in d["records"]:
    fd = r.get("filter_diag", {})
    qid = r["qid"][:12]
    r10 = r["r10"]
    req = fd.get("required_attrs", [])
    gold_mounts = fd.get("gold_mounts", [])
    removed = fd.get("removed_sids", [])
    in_removed = fd.get("gold_in_removed", [])
    gold_et = fd.get("gold_entity_type", "?")
    verdict = r.get("diag", {}).get("verdict", "?")
    casc = r.get("diag", {}).get("cascade", {})
    verdict_info = r.get("diag", {}).get("verdict", "?")
    removed_count = len(removed)
    gold_removed_count = len(in_removed)

    if not req:
        empty_required += 1
    if gold_mounts:
        has_gold_mount += 1
    else:
        no_gold_mount += 1
    if removed_count > 0:
        total_effective_filter += 1
    if gold_removed_count > 0:
        gold_in_removed += 1

    marker = ">>>" if r10 == 0 else "   "
    print(f"{marker} [{r10}] {qid}")
    print(f"       required={req}")
    print(f"       gold_mounts={gold_mounts}  gold_et={gold_et}")
    print(f"       removed={removed_count}  gold_in_removed={gold_removed_count}")
    if r10 == 0:
        print(f"       verdict={verdict_info}  cascade=l1:{casc.get('gold_in_step4_l1')} l2:{casc.get('gold_in_step4_l2')} s10:{casc.get('gold_in_step10_top10')}")
    print()

print("=" * 60)
print("SUMMARY")
print(f"  Mapper: {empty_required}/20 queries have EMPTY required attrs")
print(f"  Gold mount: {no_gold_mount}/{no_gold_mount+has_gold_mount} gold sentences have NO mount")
print(f"  Filter effective: {total_effective_filter}/20 questions had sentences removed")
print(f"  Gold in removed: {gold_in_removed} questions")

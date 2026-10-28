import json
import sys
import io
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ap = argparse.ArgumentParser()
ap.add_argument(
    "--tag", default="seed137",
    help="子目录名, 对应 eval/diag/<tag>/r1_failures.json",
)
ap.add_argument("--limit", type=int, default=0, help="只看前 N 个失败案例 (0=全部)")
args = ap.parse_args()

d = json.load(open(
    rf'c:\Users\zsf\Desktop\lyh\StenRAG\eval\diag\{args.tag}\r1_failures.json',
    encoding='utf-8',
))
print(f"[{args.tag}]  R@10 hit:", d['n_r10_hit'], '/ R@1 hit:', d['n_r1_hit'],
      '/ R@10 hit but R@1 miss:', d['n_r10_hit_but_r1_miss'])
print()

fails = d['failures']
if args.limit:
    fails = fails[:args.limit]
for fail in fails:
    print('=' * 72)
    print('Q :', fail['question'][:80])
    print('G :', fail['gold'])
    print('A :', fail['answer'])
    print('gold 在 top10 排第', fail['gold_in_rank'])
    for s in fail['top10']:
        flag = '**GOLD**' if s['is_gold'] else '       '
        rank = s['rank']
        sid = s['sid']
        text = s['text'][:120]
        print(f'  [{rank:>2}] {sid} {flag}: {text}')
    print()

# HotpotQA Evaluation on StenRAG

## Dataset

This evaluation uses the **HotpotQA fullwiki distractor** setting.

### Download

```bash
# Option A: wget (Linux / WSL)
mkdir -p eval/data
wget https://hotpotqa.cs.stanford.edu/data/hotpot_dev_distractor_v1.json
wget https://hotpotqa.cs.stanford.edu/data/hotpot_train_distractor_v1.json
mv hotpot_dev_distractor_v1.json eval/data/
mv hotpot_train_distractor_v1.json eval/data/

# Option B: manual download
# https://hotpotqa.cs.stanford.edu/
# Download "HotpotQA (full wiki data, with distractor paragraphs)"
# Place hotpot_dev_distractor_v1.json in eval/data/
```

### Format

Each entry in the JSON has:

```json
{
  "_id": "5a9b27df6e2a682006df2ae9",
  "question": "What sitcom ran on the same network as The Office?",
  "answer": "Parks and Recreation",
  "context": [
    {
      "title": "The Office (American TV series)",
      "sentences": ["Sentence 1 ...", "Sentence 2 ...", ...]
    },
    ...
  ],
  "supporting_facts": [
    {"title": "...", "sentences": [0, 3]},
    ...
  ]
}
```

Each question has **10 paragraphs** (8 distractor + 2 supporting). Sentences are
pre-split; the "sentences" list is the unit of graph construction.

---

## Running Evaluation

### Retrieval-Only (No LLM Answer Generation)

Tests pure retrieval quality — how well StenRAG surfaces supporting sentences.

```bash
# Val set, first 200 examples (quick smoke test)
python eval/hotpot_eval.py --split val --max-examples 200

# Full val set
python eval/hotpot_eval.py --split val --max-examples 1000

# Train set
python eval/hotpot_eval.py --split train --max-examples 5000
```

### With LLM Answer Generation

Adds end-to-end answer generation via DeepSeek / OpenAI. Output includes
end-to-end EM / F1 in addition to retrieval metrics.

```bash
# Set API key (once)
export DEEPSEEK_API_KEY=sk-...   # Linux/macOS
# $env:DEEPSEEK_API_KEY="sk-..." # PowerShell

# Run with LLM
python eval/hotpot_eval.py --split val --max-examples 200 --use-llm
```

---

## Metrics

| Metric | Description |
|---|---|
| **Retrieval Recall@1** | Top-1 retrieved sentence is in supporting facts |
| **Retrieval Recall@2** | At least one of top-2 is in supporting facts |
| **Retrieval Recall@5** | At least one of top-5 is in supporting facts |
| **Retrieval MRR** | Mean reciprocal rank of first correct sentence |
| **EM (answer-level)** | Exact string match with gold answer |
| **F1 (answer-level)** | Token-level F1 with gold answer |
| **Latency** | Per-question wall time (ms) |

---

## Output

Results are written to:

```
eval/results/
  stenrag_hotpotqa_val_200_YYYYMMDD_HHMMSS.json
```

Each line is a JSON object per question:

```json
{
  "qid": "5a9b27df6e2a682006df2ae9",
  "question": "What sitcom ran on the same network as The Office?",
  "gold_answer": "Parks and Recreation",
  "retrieved_sentences": ["Parks and Recreation aired on NBC...", "..."],
  "retrieval_hit_at_1": true,
  "retrieval_hit_at_2": true,
  "retrieval_mrr": 1.0,
  "generated_answer": "Parks and Recreation",   // only if --use-llm
  "em": true,                                   // only if --use-llm
  "f1": 1.0,                                    // only if --use-llm
  "latency_ms": 127.3
}
```

---

## Baselines to Compare

For reference, published HotpotQA distractor baselines:

| System | EM | F1 |
|---|---|---|
| QFE (EMNLP 2018) | 24.0 | — |
| GRN (NeurIPS 2018) | 32.3 | — |
| CGF (ACL 2019) | 38.8 | — |
| HGN (EMNLP 2019) | 50.2 | 61.0 |
| **StenRAG (ours)** | TBD | TBD |
| DECOMPRC (AAAI 2020) | 55.9 | 67.1 |
| Transformer-XH (ICLR 2021) | 58.9 | 70.5 |
| S2AG (ACL 2021, SentGraph) | 67.6 | 78.4 |
| P aden (Talmor & Berant 2019) | 72.1 | 82.8 |
| GOLD-BOLD | 79.3 | 87.2 |
| Upper bound (oracle) | 91.0 | 95.5 |

*Note: End-to-end EM/F1 numbers depend heavily on the answer generation LLM.
Retrieval-only Recall@K is a more reliable StenRAG evaluation metric.*

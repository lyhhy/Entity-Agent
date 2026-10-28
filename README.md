# StenRAG: Sentence-Level Logic Graph with Cross-Container Entity Fusion

A graph-based retrieval system that builds sentence-level logic graphs with:
- Container-based indexing (sentences grouped by entities)
- Intra-container logic edges (Causal/Temporal/Contrast relations)
- Cross-container entity edges (SAME_ENTITY edges)
- Multi-homing support (sentences can belong to multiple containers)
- Lightweight Core/Satellite classification (no LLM, SpaCy-only)
- Optional LLM summarization via DeepSeek / OpenAI-compatible API

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Modules

- `entity_resolver.py` - Standardized entity name resolution
- `sentence_graph.py` - Global sentence node repository
- `container_manager.py` - Container aggregation with split rules
- `graph_builder.py` - Logic edge builder + Core/Satellite classification
- `retriever.py` - Graph traversal for online retrieval
- `llm_client.py` - OpenAI-compatible LLM client (DeepSeek, vLLM, etc.)

## Usage

### Basic Retrieval

```python
from src.graph_builder import GraphBuilder
from src.retriever import Retriever

# Build graph from sentences
builder = GraphBuilder()
for sentence in sentences:
    builder.add_sentence(sentence)

# Retrieve evidence
retriever = Retriever(builder)
result = retriever.retrieve("What technology does Chen's chip design use?")
```

### LLM Summaries (Optional)

LLM summarization is **opt-in**. By default containers use a lightweight
template summary (`"entity - relates to kw1, kw2, kw3"`). To enable real LLM
summaries:

1. Copy `.env.example` to `.env` and fill in your API key:

   ```bash
   cp .env.example .env
   # edit .env and set DEEPSEEK_API_KEY=sk-...
   ```

2. Inject the summarizer at build time:

   ```python
   import os
   from dotenv import load_dotenv   # pip install python-dotenv
   load_dotenv()                  # loads DEEPSEEK_API_KEY

   from src.llm_client import make_summarizer
   from src.graph_builder import GraphBuilder

   builder = GraphBuilder()

   # Register the LLM summarizer (auto-detects DEEPSEEK_API_KEY)
   builder.container_manager.set_llm_summarizer(make_summarizer())

   # Now container summaries use DeepSeek instead of the template
   builder.add_sentence(my_sentence)
   ```

   Or use a custom function:

   ```python
   def my_summarizer(entity: str, context: str) -> str:
       # entity: canonical entity name
       # context: up to 5 sentences about the entity
       return call_your_llm(entity, context)

   builder.container_manager.set_llm_summarizer(my_summarizer)
   ```

   Supported providers: **DeepSeek**, OpenAI, Azure OpenAI, vLLM, Ollama
   (any OpenAI-compatible Chat Completions API).

## Evaluation

Benchmark on HotpotQA fullwiki distractor setting:

```bash
# Download HotpotQA first (see eval/README.md), then:
python eval/hotpot_eval.py --split val --max-examples 1000
```

See `eval/README.md` for full instructions including LLM-backed evaluation
(end-to-end answer generation) and comparison baselines.

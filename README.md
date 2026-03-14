# AI-Agents

Standalone, reusable agents for use in different projects or for moving to another repository.

Dependencies are injected via **interfaces**; you can implement your own database, LLM, and RAG as needed.

## Project structure

```
AI-Agents/
├── README.md
├── Agents/                    # Agent implementations
│   └── rag/                   # RAG chatbot (first agent)
│       ├── __init__.py
│       ├── agent.py           # RAG orchestration logic
│       ├── interfaces.py      # RagQueryFn, ListSourceIdsFn, ChatCompleteFn
│       └── llamaindex_backend.py  # LlamaIndex: ingest, index, rag_query
├── Infra/                     # Shared infrastructure
│   └── llm/                   # LLM provider (Gemini chat_complete)
│       ├── __init__.py
│       └── gemini.py
├── scripts/
│   └── run_rag_chat.py        # CLI: ingest + chat for RAG
├── requirements-rag.txt
└── RAG-TODO.md                # RAG stack checklist (all done)
```

## Agents

| Agent | Path | Description |
|-------|------|-------------|
| **RAG (chatbot)** | `Agents/rag/` | Knowledge-based chatbot: RAG retrieval + LLM response. Requires `rag_query`, `list_source_ids`, `chat_complete`. |

## RAG chatbot (LlamaIndex + Gemini)

The RAG agent is fully wired with:

- **LLM:** `Infra/llm/` — Gemini `chat_complete` (shared by all agents).
- **Backend:** `Agents/rag/llamaindex_backend.py` — LlamaIndex: load docs, HuggingFace embeddings, vector index, `rag_query`, `list_source_ids`.

### Setup

```bash
pip install -r requirements-rag.txt
```

Set `GEMINI_API_KEY` in `.env` (or env) for chat. Embeddings use HuggingFace locally by default (no key); override with `RAG_EMBED_MODEL` if needed.

### Run the chatbot

From the project root:

```bash
# Ingest + chat in one process (no persist):
python -m scripts.run_rag_chat run --path ./data --source-id my-docs

# Or: ingest once, persist, then chat in another run:
python -m scripts.run_rag_chat ingest --path ./data --source-id my-docs --persist-dir ./storage
python -m scripts.run_rag_chat chat --persist-dir ./storage --source-id my-docs
```

### Use in code

```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent  # project root
sys.path.insert(0, str(ROOT))

from Agents.rag import RagAgent
from Agents.rag.llamaindex_backend import build_index_from_path, create_rag_functions
from Infra.llm import chat_complete

index = build_index_from_path("./data", source_id="my-docs", persist_dir="./storage")
rag_query, list_source_ids = create_rag_functions(index, {"agent-1": ["my-docs"]})
agent = RagAgent(config={"top_k": 8})
run = agent.build()

# async:
result = await run(
    tenant_id="1", agent_id="agent-1", conversation_id="c1",
    message={"content": "Your question"},
    rag_query=rag_query, list_source_ids=list_source_ids, chat_complete=chat_complete,
    config={"chatbot_mode": "only_sources"},
)
# result["answer"], result["citations"]
```

## RAG stack status (RAG-TODO.md)

All items for the first agent (RAG chatbot) are done:

| # | Component | Status |
|---|-----------|--------|
| 1 | LLM API (`Infra/llm`, Gemini) | Done |
| 2 | Embeddings (HuggingFace in LlamaIndex backend) | Done |
| 3 | Vector store (LlamaIndex in-memory / persist) | Done |
| 4 | Ingest pipeline (load → chunk → embed → store) | Done |
| 5 | RAG query (`rag_query` in llamaindex_backend) | Done |
| 6 | Source mapping (`list_source_ids`) | Done |
| 7 | Wiring & scripts (`run_rag_chat` CLI) | Done |

## Installing in another project

- Copy the repo (or `Agents/`, `Infra/`, `scripts/`) and add the project root to `PYTHONPATH` so that `Agents` and `Infra` are importable.
- No hard dependency on any external app; Python 3.10+ and the packages in `requirements-rag.txt` are sufficient for the RAG chatbot.

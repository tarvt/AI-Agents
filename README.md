# AI-Agents

Standalone, reusable agents for use in different projects or for moving to another repository.

Dependencies are injected via **interfaces**; you can implement your own database, LLM, and RAG as needed.

## Project structure

```
AI-Agents/
├── README.md
├── Agents/                         # One folder per agent
│   ├── rag/                       # RAG chatbot
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── interfaces.py
│   │   └── llamaindex_backend.py
│   ├── financial_analysis/        # L6: market + trading strategy
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── crew.py
│   └── job_application/           # L7: resume + interview prep
│       ├── __init__.py
│       ├── agent.py
│       └── crew.py
├── Infra/
│   ├── llm/                       # Gemini chat_complete
│   └── crewai_common.py           # Shared CrewAI tools/LLM
├── scripts/
│   ├── run_rag_chat.py
│   └── run_crew.py                # CLI for financial_analysis / job_application
├── requirements-rag.txt
├── requirements-crew.txt
└── RAG-TODO.md
```

## Agents and libraries

| Agent | Path | Description | Libraries |
|-------|------|-------------|-----------|
| **RAG (chatbot)** | `Agents/rag/` | Knowledge-based Q&A: RAG retrieval + LLM response. | **LlamaIndex**, **HuggingFace** (embeddings), **Infra/llm** (Gemini) |
| **Financial analysis** | `Agents/financial_analysis/` | Multi-agent: Data Analyst, Trading Strategy, Trade Advisor, Risk Advisor (L6). | **CrewAI**, **crewai-tools**, **LangChain** (OpenAI) |
| **Job application** | `Agents/job_application/` | Multi-agent: Researcher, Profiler, Resume Strategist, Interview Preparer (L7). | **CrewAI**, **crewai-tools**, **LangChain** (OpenAI) |

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

## Financial analysis agent (L6)

CrewAI multi-agent: Data Analyst, Trading Strategy Developer, Trade Advisor, Risk Advisor.

### Setup

```bash
pip install -r requirements-crew.txt
```

Set **OPENAI_API_KEY** and **SERPER_API_KEY**. Optional: **OPENAI_MODEL_NAME** (default `gpt-3.5-turbo`).

### Run

```bash
python -m scripts.run_crew financial_analysis --stock AAPL --risk-tolerance Medium
```

### Use in code

```python
import asyncio
from Agents.financial_analysis import FinancialAnalysisAgent

agent = FinancialAnalysisAgent(config={"verbose": True})
run = agent.build()
result = asyncio.run(run(inputs={
    "stock_selection": "AAPL",
    "risk_tolerance": "Medium",
    "trading_strategy_preference": "Day Trading",
}))
# result["result"]
```

## Job application agent (L7)

CrewAI multi-agent: Tech Job Researcher, Profiler, Resume Strategist, Interview Preparer. Outputs `tailored_resume.md` and `interview_materials.md`.

### Run

```bash
python -m scripts.run_crew job_application --job-url "https://..." --github "https://github.com/..." --writeup "Your bio..." --resume ./resume.md --output-dir ./
```

### Use in code

```python
import asyncio
from Agents.job_application import JobApplicationAgent

agent = JobApplicationAgent(config={"resume_path": "./resume.md", "output_dir": "./"})
run = agent.build()
result = asyncio.run(run(inputs={
    "job_posting_url": "https://...",
    "github_url": "https://github.com/...",
    "personal_writeup": "Candidate bio...",
}))
# result["result"], result["outputs"]
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
- **RAG:** `pip install -r requirements-rag.txt` (+ `GEMINI_API_KEY` for chat).
- **Financial analysis / Job application:** `pip install -r requirements-crew.txt` (+ `OPENAI_API_KEY`, `SERPER_API_KEY`).

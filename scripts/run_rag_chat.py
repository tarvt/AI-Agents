# -*- coding: utf-8 -*-
"""
Run the RAG chatbot: ingest documents (once) and then answer questions using LlamaIndex + Gemini.

Usage:
  1. Put your documents (PDF, TXT, MD) in a folder, e.g. ./data or ./helper-notbooks.
  2. Set env: GEMINI_API_KEY (for chat). Optional: RAG_EMBED_MODEL, RAG_PERSIST_DIR.
  3. Ingest (first time):
       python -m scripts.run_rag_chat ingest --path ./data --source-id my-docs [--persist-dir ./storage]
  4. Chat (with in-memory index from step 3, or load from disk):
       python -m scripts.run_rag_chat chat [--persist-dir ./storage]
  Or run both in one process (ingest then chat, no persist):
       python -m scripts.run_rag_chat run --path ./data --source-id my-docs
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_llm():
    from Infra.llm import chat_complete
    return chat_complete


def _ensure_rag_backend():
    from Agents.rag.llamaindex_backend import (
        build_index_from_path,
        load_index,
        create_rag_functions,
        load_index_and_sources,
    )
    return build_index_from_path, load_index, create_rag_functions, load_index_and_sources


def cmd_ingest(args):
    """Build index from path and optionally persist."""
    build_index_from_path, _, _, _ = _ensure_rag_backend()
    path = Path(args.path)
    if not path.exists():
        print(f"Error: path not found: {path}")
        return 1
    persist_dir = Path(args.persist_dir) if args.persist_dir else None
    print(f"Ingesting {path} with source_id={args.source_id} ...")
    index = build_index_from_path(
        path,
        source_id=args.source_id,
        persist_dir=persist_dir,
        recursive=getattr(args, "recursive", True),
    )
    print("Done. Index is in memory.", "Persisted to", persist_dir if persist_dir else "")
    return 0


def cmd_load_and_chat(args):
    """Load index from persist_dir and run chat loop."""
    _, load_index, create_rag_functions, load_index_and_sources = _ensure_rag_backend()
    from Agents.rag import RagAgent

    persist_dir = Path(args.persist_dir or "./storage")
    if not persist_dir.exists():
        print(f"Error: persist_dir not found: {persist_dir}. Run 'ingest' first with --persist-dir")
        return 1

    index, agent_sources = load_index_and_sources(
        persist_dir,
        agent_sources={args.agent_id or "default": [args.source_id or "default"]},
    )
    rag_query, list_source_ids = create_rag_functions(index, agent_sources, similarity_top_k=args.top_k or 10)
    chat_complete = _ensure_llm()
    agent = RagAgent(config={"top_k": args.top_k or 8, "max_ctx_chars": 5000})
    run = agent.build()
    agent_id = args.agent_id or "default"

    async def one_turn(user_text: str):
        result = await run(
            tenant_id="1",
            agent_id=agent_id,
            conversation_id="cli",
            message={"content": user_text},
            rag_query=rag_query,
            list_source_ids=list_source_ids,
            chat_complete=chat_complete,
            config={"chatbot_mode": "only_sources"},
        )
        return result.get("answer") or "(no answer)"

    async def loop():
        print("RAG Chat (LlamaIndex + Gemini). Type 'quit' to exit.\n")
        while True:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                break
            answer = await one_turn(line)
            print(f"Bot: {answer}\n")

    asyncio.run(loop())
    return 0


def cmd_run(args):
    """Ingest from path (no persist) and run chat in the same process."""
    build_index_from_path, _, create_rag_functions, _ = _ensure_rag_backend()
    from Agents.rag import RagAgent

    path = Path(args.path)
    if not path.exists():
        print(f"Error: path not found: {path}")
        return 1
    print(f"Building index from {path} (source_id={args.source_id}) ...")
    index = build_index_from_path(path, source_id=args.source_id, persist_dir=None)
    agent_sources = {args.agent_id or "default": [args.source_id]}
    rag_query, list_source_ids = create_rag_functions(index, agent_sources, similarity_top_k=args.top_k or 10)
    chat_complete = _ensure_llm()
    agent = RagAgent(config={"top_k": args.top_k or 8, "max_ctx_chars": 5000})
    run = agent.build()
    agent_id = args.agent_id or "default"

    async def one_turn(user_text: str):
        result = await run(
            tenant_id="1",
            agent_id=agent_id,
            conversation_id="cli",
            message={"content": user_text},
            rag_query=rag_query,
            list_source_ids=list_source_ids,
            chat_complete=chat_complete,
            config={"chatbot_mode": "only_sources"},
        )
        return result.get("answer") or "(no answer)"

    async def loop():
        print("RAG Chat (LlamaIndex + Gemini). Type 'quit' to exit.\n")
        while True:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                break
            answer = await one_turn(line)
            print(f"Bot: {answer}\n")

    asyncio.run(loop())
    return 0


def main():
    parser = argparse.ArgumentParser(description="RAG chatbot: ingest and chat with LlamaIndex + Gemini")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Build index from a folder and optionally persist")
    ingest_p.add_argument("--path", "-p", default="./data", help="Path to documents (folder or file)")
    ingest_p.add_argument("--source-id", "-s", default="default", help="Source ID for these documents")
    ingest_p.add_argument("--persist-dir", "-d", default=None, help="Directory to persist index (optional)")
    ingest_p.add_argument("--no-recursive", action="store_false", dest="recursive", help="Do not load subdirs")
    ingest_p.set_defaults(func=cmd_ingest)

    chat_p = sub.add_parser("chat", help="Load index from persist_dir and run chat")
    chat_p.add_argument("--persist-dir", "-d", default="./storage", help="Directory where index was persisted")
    chat_p.add_argument("--agent-id", default="default", help="Agent ID")
    chat_p.add_argument("--source-id", default="default", help="Source ID to allow for this agent")
    chat_p.add_argument("--top-k", type=int, default=8, help="Retrieval top_k")
    chat_p.set_defaults(func=cmd_load_and_chat)

    run_p = sub.add_parser("run", help="Ingest from path (no persist) and chat in same process")
    run_p.add_argument("--path", "-p", required=True, help="Path to documents")
    run_p.add_argument("--source-id", "-s", default="default", help="Source ID")
    run_p.add_argument("--agent-id", default="default", help="Agent ID")
    run_p.add_argument("--top-k", type=int, default=8)
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

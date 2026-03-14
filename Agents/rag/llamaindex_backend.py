# -*- coding: utf-8 -*-
"""
LlamaIndex-based RAG backend: ingest documents, build vector index, and expose
rag_query + list_source_ids for use with rag.agent.RagAgent.

Usage:
  1. Ingest: build_index_from_path("./data", source_id="my-docs", persist_dir="./storage")
  2. Load: index, agent_sources = load_index_and_sources("./storage")
  3. Create run: rag_query, list_source_ids = create_rag_functions(index, agent_sources)
  4. Pass rag_query, list_source_ids, chat_complete into RagAgent.build()(..., ...)
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List

# LlamaIndex imports (optional deps: pip install llama-index llama-index-embeddings-huggingface)
try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
    from llama_index.core.schema import Document
except ImportError as e:
    raise ImportError(
        "LlamaIndex is required for the RAG backend. Install with: "
        "pip install llama-index llama-index-embeddings-huggingface"
    ) from e

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None  # type: ignore[misc, assignment]

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _get_embed_model():
    """Return embedding model (HuggingFace by default so no API key is needed)."""
    if HuggingFaceEmbedding is None:
        raise ImportError(
            "Install llama-index-embeddings-huggingface for embeddings: "
            "pip install llama-index-embeddings-huggingface"
        )
    model_name = os.getenv("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    return HuggingFaceEmbedding(model_name=model_name)


def load_documents(
    path: str | Path,
    source_id: str,
    recursive: bool = True,
    required_exts: List[str] | None = None,
) -> List[Document]:
    """
    Load documents from a directory (or single file) and tag each with source_id.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        reader = SimpleDirectoryReader(input_files=[str(path)])
    else:
        reader = SimpleDirectoryReader(
            input_dir=str(path),
            recursive=recursive,
            required_exts=required_exts or [".pdf", ".txt", ".md", ".docx"],
        )

    documents = reader.load_data()
    for doc in documents:
        doc.metadata["source_id"] = source_id
        doc.metadata.setdefault("title", path.name if path.is_file() else path.name)
    return documents


def build_index_from_path(
    path: str | Path,
    source_id: str,
    persist_dir: str | Path | None = None,
    embed_model: Any | None = None,
    recursive: bool = True,
    required_exts: List[str] | None = None,
) -> VectorStoreIndex:
    """
    Load documents from path, build a VectorStoreIndex, and optionally persist it.
    Returns the index. If persist_dir is set, the index is saved so you can load it later.
    """
    docs = load_documents(path, source_id=source_id, recursive=recursive, required_exts=required_exts)
    if not docs:
        raise ValueError(f"No documents loaded from {path}")

    embed_model = embed_model or _get_embed_model()
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model, show_progress=True)

    if persist_dir:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_dir))

    return index


def build_index_from_documents(
    documents: List[Document],
    embed_model: Any | None = None,
    persist_dir: str | Path | None = None,
) -> VectorStoreIndex:
    """Build index from a list of LlamaIndex Document objects (each should have metadata.source_id)."""
    embed_model = embed_model or _get_embed_model()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, show_progress=True)
    if persist_dir:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_dir))
    return index


def load_index(
    persist_dir: str | Path,
    embed_model: Any | None = None,
) -> VectorStoreIndex:
    """Load a previously persisted index from disk."""
    persist_dir = Path(persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(f"Persist dir not found: {persist_dir}")
    embed_model = embed_model or _get_embed_model()
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage_context, embed_model=embed_model)


def _node_to_hit(node: Any) -> Dict[str, Any]:
    """Convert a LlamaIndex NodeWithScore (or node) to our RAG hit format."""
    n = node.node if hasattr(node, "node") else node
    score = getattr(node, "score", None)
    if score is None:
        score = 1.0
    meta = getattr(n, "metadata", None) or {}
    text = getattr(n, "text", None) or ""
    node_id = getattr(n, "node_id", None) or ""
    return {
        "id": node_id,
        "chunk": text,
        "title": meta.get("title") or meta.get("file_name") or "Source",
        "url": meta.get("url") or "",
        "source_id": meta.get("source_id") or "",
        "score": float(score) if score is not None else 1.0,
    }


def create_rag_query(
    index: VectorStoreIndex,
    similarity_top_k: int = 10,
) -> Any:
    """
    Return an async function that matches RagQueryFn:
    async def rag_query(tenant_id, query, top_k, conversation_id, filters) -> List[Dict].
    Filters may contain "source_id_list": list of source_ids to restrict to.
    """

    def _retrieve_sync(query: str, top_k: int, source_id_list: List[str] | None) -> List[Dict[str, Any]]:
        retriever = index.as_retriever(similarity_top_k=min(top_k * 2, 50))  # fetch extra then filter
        nodes = retriever.retrieve(query)
        hits = [_node_to_hit(nd) for nd in nodes]
        if source_id_list:
            allowed = set(source_id_list)
            hits = [h for h in hits if h.get("source_id") in allowed]
        return hits[:top_k]

    async def rag_query(
        *,
        tenant_id: str,
        query: str,
        top_k: int,
        conversation_id: str | None = None,
        filters: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        source_id_list = (filters or {}).get("source_id_list")
        k = top_k or similarity_top_k
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _retrieve_sync, query, k, source_id_list)

    return rag_query


def create_list_source_ids(agent_sources: Dict[str, List[str]]) -> Any:
    """
    agent_sources: map agent_id -> list of source_id (e.g. {"agent-1": ["papers", "faq"]}).
    Returns an async function list_source_ids(agent_id) -> List[str].
    """

    async def list_source_ids(agent_id: str) -> List[str]:
        return list(agent_sources.get(agent_id, []))

    return list_source_ids


def load_index_and_sources(
    persist_dir: str | Path,
    agent_sources: Dict[str, List[str]] | None = None,
    embed_model: Any | None = None,
) -> tuple[VectorStoreIndex, Dict[str, List[str]]]:
    """
    Load index from persist_dir and return (index, agent_sources).
    If agent_sources is not provided, a default mapping is used: one agent "default" with one source "default".
    You can override agent_sources when calling create_rag_functions.
    """
    index = load_index(persist_dir, embed_model=embed_model)
    sources = agent_sources or {"default": ["default"]}
    return index, sources


def create_rag_functions(
    index: VectorStoreIndex,
    agent_sources: Dict[str, List[str]],
    similarity_top_k: int = 10,
) -> tuple[Any, Any]:
    """Convenience: return (rag_query, list_source_ids) for use with RagAgent.run()."""
    return (
        create_rag_query(index, similarity_top_k=similarity_top_k),
        create_list_source_ids(agent_sources),
    )

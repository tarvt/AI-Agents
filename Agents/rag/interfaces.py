# -*- coding: utf-8 -*-
"""
Injected protocols for RAG Agent.
Any project can supply these functions/objects with its own implementation.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Protocol

# RAG function type: retrieve chunks by query
RagQueryFn = Callable[..., Awaitable[List[Dict[str, Any]]]]
# Suggested signature:
# async def rag_query(
#     *,
#     tenant_id: str,
#     query: str,
#     top_k: int,
#     conversation_id: str | None = None,
#     filters: dict | None = None,
# ) -> List[Dict[str, Any]]
# Each item: at least "chunk", "title", "url", "id", "source_id", "score"

# Function type: list of source IDs attached to the agent
ListSourceIdsFn = Callable[[str], Awaitable[List[str]]]
# async def list_source_ids(agent_id: str) -> List[str]

# LLM function type: generate response from messages
ChatCompleteFn = Callable[..., Awaitable[Dict[str, Any]]]
# Suggested signature:
# async def chat_complete(
#     *,
#     model: str,
#     messages: List[Dict[str, str]],
#     temperature: float = 0.3,
#     max_tokens: int = 2048,
# ) -> Dict[str, Any]  # at least {"content": str}, optional "usage"


class ChatCompleteProtocol(Protocol):
    """Protocol for chat_complete (allows using a callable object)."""

    async def __call__(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        ...

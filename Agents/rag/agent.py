# -*- coding: utf-8 -*-
"""
RAG Agent — standalone with no dependency on app.
All dependencies (rag_query, list_source_ids, chat_complete) are injected when run() is called.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List

# Default small-talk set (short Q&A for casual conversation).
# Each item: list of keywords/phrases to match, response (English only).
_SMALL_TALK: List[tuple] = [
    (["hello", "hi", "hey"], "Hello! How can I help you today?"),
    (["good morning", "good afternoon", "good evening"], "Hi! How can I help you?"),
    (["how are you", "how do you do"], "I'm good thanks, and you?"),
    (["what's up", "whats up", "how is it going"], "All good! How can I help?"),
    (["thanks", "thank you", "thx"], "You're welcome!"),
    (["bye", "goodbye", "see you"], "Goodbye, have a great day!"),
]


def _normalize_for_small_talk(text: str) -> str:
    """Normalize and strip extra spaces for small-talk matching."""
    t = (text or "").strip().lower()
    return " ".join(t.split())


SMART_REPLY_STRATEGY_INSTRUCTIONS: Dict[str, str] = {
    "concise": "Keep the answer very brief, one or two sentences; no extra preamble.",
    "detailed": "Give a detailed, step-by-step answer; add examples if needed.",
    "empathetic": "Use an empathetic, supportive tone; acknowledge the user's feelings.",
    "formal": "Use a formal, professional tone; avoid slang.",
}
SMART_REPLY_DEFAULT_STRATEGIES = ["concise", "detailed", "empathetic", "formal"]
SMART_REPLY_DEFAULT_COUNT = 4
SMART_REPLY_MIN_COUNT = 2
SMART_REPLY_MAX_COUNT = 6


class RagAgent:
    """RAG agent with injected dependencies. No import from app."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.default_top_k = int(config.get("top_k", 8))
        self.default_max_ctx_chars = int(config.get("max_ctx_chars", 5000))
        self.strict = bool(config.get("strict_answers", True))

    def build(self) -> Callable[..., Awaitable[dict]]:
        async def run(
            *,
            tenant_id: str,
            agent_id: str,
            conversation_id: str,
            message: dict,
            rag_query: Callable[..., Awaitable[List[Dict[str, Any]]]],
            list_source_ids: Callable[[str], Awaitable[List[str]]],
            chat_complete: Callable[..., Awaitable[Dict[str, Any]]],
            history: List[Dict] | None = None,
            config: dict | None = None,
            **_,
        ) -> dict:
            cfg = config or {}
            retrieval = cfg.get("retrieval") or {}
            generation = cfg.get("generation") or {}
            chatbot_mode = cfg.get("chatbot_mode", "only_sources")
            fallback_message = cfg.get("fallback_message")
            response_language = cfg.get("response_language", "auto")
            small_talk_enabled = bool(cfg.get("small_talk_enabled", False))
            source_links_in_response = bool(cfg.get("source_links_in_response", False))
            smart_reply_enabled = bool(cfg.get("smart_reply_enabled", False))
            smart_reply_strategies = cfg.get("smart_reply_strategies") or SMART_REPLY_DEFAULT_STRATEGIES
            strategy_ids = [str(s) for s in smart_reply_strategies] if isinstance(smart_reply_strategies, list) else list(SMART_REPLY_DEFAULT_STRATEGIES)
            smart_reply_count = cfg.get("smart_reply_count", SMART_REPLY_DEFAULT_COUNT)
            smart_reply_count = max(SMART_REPLY_MIN_COUNT, min(SMART_REPLY_MAX_COUNT, int(smart_reply_count)))
            model = cfg.get("model") or "models/gemini-2.5-flash"
            base_system_prompt = cfg.get("base_system_prompt") or (
                "You are a helpful, concise, and strictly factual support agent. "
                "Only answer from the provided CONTEXT. "
                "The TEXT CONTENT within each SOURCE block is the primary source of truth: if the text contains the answer, use it even if the source title/label seems unrelated. "
                "If the user asks about a word or concept (e.g. 'hi' or 'example in English'), look for it in EVERY SOURCE below. "
                "Do NOT state that the information is missing unless you have checked every SOURCE provided. "
                "CRITICAL: You MUST always respond in clear English, regardless of the language used inside the CONTEXT. "
                "If the user only sends a greeting (like 'hello') without a question, "
                "respond warmly in English and invite them to ask questions about the content."
            )
            role_and_persona = cfg.get("role_and_persona") or "You are an expert knowledge base assistant."
            tone_and_style = cfg.get("tone_and_style") or (
                "Maintain a professional, objective, and polite tone. Be direct and avoid unnecessary elaboration."
            )
            main_objective = cfg.get("main_objective") or (
                "Your goal is to accurately answer the user's question using ONLY the content provided in the CONTEXT."
            )
            system_prompt = f"{role_and_persona}\n\n{tone_and_style}\n\n{main_objective}\n\n{base_system_prompt}"
            top_k = int(retrieval.get("top_k", self.default_top_k))
            max_ctx_chars = int(retrieval.get("max_ctx_chars", self.default_max_ctx_chars))
            temperature = float(generation.get("temperature", 0.3))
            max_output_tokens = int(generation.get("max_output_tokens", 2048))

            source_ids = await list_source_ids(agent_id)
            if chatbot_mode == "only_general":
                source_ids = []
            dynamic_filters = {}
            if source_ids:
                dynamic_filters["source_id_list"] = source_ids
            elif chatbot_mode in ("all_questions", "only_sources", "assistant"):
                return {
                    "answer": "I can't answer that because no knowledge sources are assigned to this agent.",
                    "citations": [],
                    "tool_calls": [],
                }

            user_text = (message.get("content") or "").strip()

            greetings_en = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "bye", "goodbye"]
            is_greeting = (
                user_text.lower() in [g.lower() for g in greetings_en]
                or any(user_text.lower().startswith(g.lower()) for g in greetings_en)
            )
            if is_greeting and len(user_text.split()) <= 3:
                greeting_response = (
                    "Hello! I'm a knowledge-based chatbot. Feel free to ask me any questions about your content."
                )
                if smart_reply_enabled:
                    return {
                        "answer": None,
                        "smart_reply_suggestions": [{"strategy": "default", "text": greeting_response, "citations": []}],
                        "citations": [],
                        "tool_calls": [],
                    }
                return {"answer": greeting_response, "citations": [], "tool_calls": []}

            if small_talk_enabled and len(user_text.split()) <= 6:
                normalized = _normalize_for_small_talk(user_text)
                for keywords, resp in _SMALL_TALK:
                    if any(kw in normalized or normalized in kw for kw in keywords):
                        answer_small = resp
                        if smart_reply_enabled:
                            return {"answer": None, "smart_reply_suggestions": [{"strategy": "default", "text": answer_small, "citations": []}], "citations": [], "tool_calls": []}
                        return {"answer": answer_small, "citations": [], "tool_calls": []}

            hits: List[Dict[str, Any]] = []
            if chatbot_mode != "only_general":
                hits = await rag_query(
                    tenant_id=tenant_id,
                    query=user_text,
                    top_k=top_k,
                    conversation_id=conversation_id,
                    filters=dynamic_filters,
                )

            def _get_fallback_answer() -> str:
                if fallback_message and str(fallback_message).strip():
                    return str(fallback_message).strip()
                return "I couldn't find this in your knowledge base. Please rephrase your question or add the relevant document."

            if chatbot_mode == "only_sources":
                if not hits:
                    fallback_text = _get_fallback_answer()
                    if smart_reply_enabled:
                        return {"answer": None, "smart_reply_suggestions": [{"strategy": "default", "text": fallback_text, "citations": []}], "citations": [], "tool_calls": []}
                    return {"answer": fallback_text, "citations": [], "tool_calls": []}
            elif chatbot_mode in ("all_questions", "assistant"):
                if self.strict and not hits and chatbot_mode == "all_questions":
                    fallback_text = _get_fallback_answer()
                    if smart_reply_enabled:
                        return {"answer": None, "smart_reply_suggestions": [{"strategy": "default", "text": fallback_text, "citations": []}], "citations": [], "tool_calls": []}
                    return {"answer": fallback_text, "citations": [], "tool_calls": []}

            context, citations = self._build_context(hits, max_ctx_chars) if hits else ("", [])

            if chatbot_mode == "only_general":
                general_system_prompt = (
                    f"{role_and_persona}\n\n{tone_and_style}\n\n"
                    "You are a helpful assistant. Answer the user's questions based on your general knowledge. "
                    "CRITICAL: You MUST always respond in clear English."
                )
                msgs = [{"role": "system", "content": general_system_prompt}]
            elif chatbot_mode == "assistant":
                assistant_system_prompt = (
                    f"{role_and_persona}\n\n{tone_and_style}\n\n"
                    "You are a helpful assistant. "
                    "If CONTEXT is provided below, prioritize it. "
                    "If CONTEXT is empty or doesn't contain the answer, use your general knowledge to help the user. "
                    "CRITICAL: You MUST always respond in clear English."
                )
                msgs = [{"role": "system", "content": assistant_system_prompt}]
                if context:
                    msgs.append({"role": "system", "content": f"CONTEXT:\n{context}"})
            else:
                msgs = [{"role": "system", "content": system_prompt}]
                if context:
                    msgs.append({"role": "system", "content": f"CONTEXT:\n{context}"})
                elif chatbot_mode == "all_questions":
                    msgs.append({"role": "system", "content": "Note: No relevant context found in knowledge base. Use your general knowledge to answer the user's question."})

            if history and len(history) > 0:
                recent_history = history[-6:] if len(history) > 6 else history
                history_summary_parts = []
                for m in recent_history:
                    role = m.get("role", "user")
                    content = (m.get("content") or "").strip()
                    if role in ("user", "assistant") and content:
                        role_label = "User" if role == "user" else "Assistant"
                        content_preview = content[:200] + ("..." if len(content) > 200 else "")
                        history_summary_parts.append(f"{role_label}: {content_preview}")
                if history_summary_parts:
                    history_summary = "\n".join(history_summary_parts)
                    msgs.append({
                        "role": "system",
                        "content": f"PREVIOUS CONVERSATION CONTEXT (for reference only):\n{history_summary}\n\nNote: Only answer the CURRENT question below, using the CONTEXT above.",
                    })

            # Force the model to answer in English regardless of the original language.
            if response_language != "en":
                response_language = "en"
            lang_note = "[Note: Please respond in English.]"
            user_message_with_instruction = f"{user_text}\n\n{lang_note}"
            msgs.append({"role": "user", "content": user_message_with_instruction})

            if smart_reply_enabled:
                selected = strategy_ids[:smart_reply_count]
                suggestions: List[Dict[str, Any]] = []
                for strategy_id in selected:
                    instruction = SMART_REPLY_STRATEGY_INSTRUCTIONS.get(strategy_id) or "Answer in a professional tone."
                    user_with_strategy = user_message_with_instruction + f"\n\n[Strategy: {instruction}]"
                    msgs_strategy = msgs[:-1] + [{"role": "user", "content": user_with_strategy}]
                    comp = await chat_complete(model=model, messages=msgs_strategy, temperature=temperature, max_tokens=max_output_tokens)
                    text = (comp.get("content") or "").strip()
                    if chatbot_mode in ("only_sources", "all_questions") and self.strict and not context.strip():
                        text = _get_fallback_answer()
                    if source_links_in_response and citations:
                        lines = []
                        for i, c in enumerate(citations, 1):
                            title = (c.get("title") or "Source").strip()
                            url = c.get("url")
                            lines.append(f"[{i}] {title}: {url}" if url else f"[{i}] {title}")
                        if lines:
                            text = (text or "").rstrip() + "\n\nSources:\n" + "\n".join(lines)
                    suggestions.append({"strategy": strategy_id, "text": text, "citations": citations})
                return {"answer": None, "smart_reply_suggestions": suggestions, "citations": citations, "tool_calls": []}

            completion = await chat_complete(model=model, messages=msgs, temperature=temperature, max_tokens=max_output_tokens)
            answer = (completion.get("content") or "").strip()
            if chatbot_mode in ("only_sources", "all_questions") and self.strict and not context.strip():
                answer = _get_fallback_answer()
            if source_links_in_response and citations:
                lines = []
                for i, c in enumerate(citations, 1):
                    title = (c.get("title") or "Source").strip()
                    url = c.get("url")
                    lines.append(f"[{i}] {title}: {url}" if url else f"[{i}] {title}")
                if lines:
                    answer = (answer or "").rstrip() + "\n\nSources:\n" + "\n".join(lines)
            return {"answer": answer, "citations": citations, "tool_calls": []}

        return run

    def _build_context(self, hits: List[Dict], max_ctx_chars: int) -> tuple[str, List[Dict]]:
        """Build context and citations from hits."""
        chunks, total = [], 0
        out_citations: List[Dict] = []
        seen_source_keys: set = set()
        for i, h in enumerate(hits):
            content = (h.get("chunk") or "").strip()
            if not content:
                continue
            title = h.get("title") or "Untitled Source"
            header = f"--- SOURCE {i + 1}: {title} ---\n"
            entry = header + content + "\n"
            remaining = max_ctx_chars - total
            if remaining <= 0:
                break
            cite = {k: h.get(k) for k in ("id", "title", "url", "score", "source_id")}
            source_key = (cite.get("source_id"), cite.get("title"), cite.get("url"))
            if len(entry) > remaining:
                if remaining > 100:
                    chunks.append(entry[:remaining])
                    total += remaining
                    if source_key not in seen_source_keys:
                        seen_source_keys.add(source_key)
                        out_citations.append(cite)
                break
            chunks.append(entry)
            total += len(entry)
            if source_key not in seen_source_keys:
                seen_source_keys.add(source_key)
                out_citations.append(cite)
        return "\n".join(chunks), out_citations

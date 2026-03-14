# -*- coding: utf-8 -*-
"""Shared CrewAI helpers: LLM and tools (used by financial_analysis and job_application agents)."""
from __future__ import annotations

import os

try:
    from crewai_tools import ScrapeWebsiteTool, SerperDevTool
except ImportError:
    ScrapeWebsiteTool = SerperDevTool = None  # type: ignore[misc, assignment]


def get_tools():
    if SerperDevTool is None or ScrapeWebsiteTool is None:
        raise ImportError("crewai_tools is required: pip install crewai-tools")
    return SerperDevTool(), ScrapeWebsiteTool()


def get_llm():
    """CrewAI uses OpenAI; set OPENAI_API_KEY and OPENAI_MODEL_NAME."""
    model = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0.7)
    except ImportError:
        from langchain_community.chat_models import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0.7)

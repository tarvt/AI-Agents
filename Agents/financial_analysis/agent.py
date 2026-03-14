# -*- coding: utf-8 -*-
"""
FinancialAnalysisAgent — multi-agent crew for market analysis and trading strategy (L6).
Uses CrewAI. Requires OPENAI_API_KEY and SERPER_API_KEY.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict

from .crew import build_crew

DEFAULT_INPUTS = {
    "stock_selection": "AAPL",
    "initial_capital": "100000",
    "risk_tolerance": "Medium",
    "trading_strategy_preference": "Day Trading",
    "news_impact_consideration": True,
}


class FinancialAnalysisAgent:
    """Agent: Data Analyst, Trading Strategy Developer, Trade Advisor, Risk Advisor (CrewAI)."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.verbose = bool(config.get("verbose", True))

    def build(self) -> Callable[..., Awaitable[Dict[str, Any]]]:
        async def run(
            *,
            inputs: Dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            inp = {**DEFAULT_INPUTS, **(inputs or {})}
            crew = build_crew(verbose=self.verbose)
            try:
                raw = await asyncio.to_thread(crew.kickoff, inputs=inp)
                return {"result": str(raw) if raw is not None else ""}
            except Exception as e:
                return {"result": "", "error": str(e)}

        return run

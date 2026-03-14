# -*- coding: utf-8 -*-
"""
JobApplicationAgent — multi-agent crew to tailor resume and prepare interview materials (L7).
Uses CrewAI. Requires OPENAI_API_KEY, SERPER_API_KEY; resume path and output_dir configurable.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict

from .crew import build_crew

DEFAULT_INPUTS = {
    "job_posting_url": "",
    "github_url": "",
    "personal_writeup": "",
}


class JobApplicationAgent:
    """Agent: Tech Job Researcher, Profiler, Resume Strategist, Interview Preparer (CrewAI)."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.verbose = bool(config.get("verbose", True))
        self.resume_path = config.get("resume_path", "./fake_resume.md")
        self.output_dir = config.get("output_dir", ".")

    def build(self) -> Callable[..., Awaitable[Dict[str, Any]]]:
        async def run(
            *,
            inputs: Dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            inp = inputs or {}
            resume_path = inp.get("resume_path") or self.resume_path
            output_dir = inp.get("output_dir") or self.output_dir
            crew = build_crew(resume_path=resume_path, output_dir=output_dir, verbose=self.verbose)
            kickoff_inputs = {**DEFAULT_INPUTS, **inp}
            kickoff_inputs.pop("resume_path", None)
            kickoff_inputs.pop("output_dir", None)
            try:
                raw = await asyncio.to_thread(crew.kickoff, inputs=kickoff_inputs)
                return {
                    "result": str(raw) if raw is not None else "",
                    "outputs": {
                        "tailored_resume": f"{output_dir}/tailored_resume.md",
                        "interview_materials": f"{output_dir}/interview_materials.md",
                    },
                }
            except Exception as e:
                return {"result": "", "error": str(e), "outputs": {}}

        return run

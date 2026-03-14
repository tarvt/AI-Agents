# -*- coding: utf-8 -*-
"""
Run a CrewAI-based agent: financial_analysis (L6) or job_application (L7).

Requires: OPENAI_API_KEY, SERPER_API_KEY.
Optional: OPENAI_MODEL_NAME (default gpt-3.5-turbo).

Usage:
  python -m scripts.run_crew financial_analysis --stock AAPL --risk-tolerance Medium
  python -m scripts.run_crew job_application --job-url <url> --github <url> --writeup "..." --resume ./resume.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run CrewAI agent (financial_analysis or job_application)")
    parser.add_argument(
        "agent",
        choices=["financial_analysis", "job_application"],
        help="Which agent to run",
    )
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory (job_application)")
    parser.add_argument("--resume", "-r", default="./fake_resume.md", help="Resume path (job_application)")
    parser.add_argument("--stock", default="AAPL", help="Stock symbol (financial_analysis)")
    parser.add_argument("--risk-tolerance", default="Medium", help="Risk tolerance (financial_analysis)")
    parser.add_argument("--trading-preference", default="Day Trading", help="Trading preference (financial_analysis)")
    parser.add_argument("--job-url", required=False, help="Job posting URL (job_application)")
    parser.add_argument("--github", required=False, help="GitHub profile URL (job_application)")
    parser.add_argument("--writeup", required=False, default="", help="Personal writeup (job_application)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    if args.agent == "financial_analysis":
        from Agents.financial_analysis import FinancialAnalysisAgent
        agent = FinancialAnalysisAgent(config={"verbose": not args.quiet})
        run = agent.build()
        inputs = {
            "stock_selection": args.stock,
            "risk_tolerance": args.risk_tolerance,
            "trading_strategy_preference": args.trading_preference,
            "initial_capital": "100000",
            "news_impact_consideration": True,
        }
    else:
        from Agents.job_application import JobApplicationAgent
        agent = JobApplicationAgent(config={
            "verbose": not args.quiet,
            "resume_path": args.resume,
            "output_dir": args.output_dir,
        })
        run = agent.build()
        inputs = {
            "job_posting_url": args.job_url or "",
            "github_url": args.github or "",
            "personal_writeup": args.writeup or "",
            "resume_path": args.resume,
            "output_dir": args.output_dir,
        }

    result = asyncio.run(run(inputs=inputs))

    if result.get("error"):
        print("Error:", result["error"], file=sys.stderr)
        sys.exit(1)
    print(result.get("result", ""))
    if result.get("outputs"):
        print("\nGenerated files:", json.dumps(result["outputs"], indent=2))


if __name__ == "__main__":
    main()

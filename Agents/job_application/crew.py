# -*- coding: utf-8 -*-
"""L7-style CrewAI crew: Researcher, Profiler, Resume Strategist, Interview Preparer."""
from __future__ import annotations

import os

try:
    from crewai import Agent, Task, Crew
except ImportError as e:
    raise ImportError("CrewAI required: pip install crewai crewai-tools") from e

try:
    from crewai_tools import FileReadTool, MDXSearchTool
except ImportError:
    FileReadTool = MDXSearchTool = None  # type: ignore[misc, assignment]

from Infra.crewai_common import get_tools, get_llm


def build_crew(
    resume_path: str,
    output_dir: str = ".",
    verbose: bool = True,
) -> Crew:
    search_tool, scrape_tool = get_tools()
    if FileReadTool is None or MDXSearchTool is None:
        raise ImportError("crewai_tools (FileReadTool, MDXSearchTool) required: pip install crewai-tools")

    read_resume = FileReadTool(file_path=resume_path)
    semantic_search_resume = MDXSearchTool(mdx=resume_path)

    researcher = Agent(
        role="Tech Job Researcher",
        goal="Make sure to do amazing analysis on job posting to help job applicants",
        tools=[scrape_tool, search_tool],
        verbose=verbose,
        backstory=(
            "As a Job Researcher, your prowess in navigating and extracting critical information from job postings "
            "is unmatched. Your skills help pinpoint the necessary qualifications and skills sought by employers, "
            "forming the foundation for effective application tailoring."
        ),
    )

    profiler = Agent(
        role="Personal Profiler for Engineers",
        goal="Do incredible research on job applicants to help them stand out in the job market",
        tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
        verbose=verbose,
        backstory=(
            "Equipped with analytical prowess, you dissect and synthesize information from diverse sources "
            "to craft comprehensive personal and professional profiles, laying the groundwork for personalized "
            "resume enhancements."
        ),
    )

    resume_strategist = Agent(
        role="Resume Strategist for Engineers",
        goal="Find all the best ways to make a resume stand out in the job market.",
        tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
        verbose=verbose,
        backstory=(
            "With a strategic mind and an eye for detail, you excel at refining resumes to highlight the most "
            "relevant skills and experiences, ensuring they resonate perfectly with the job's requirements."
        ),
    )

    interview_preparer = Agent(
        role="Engineering Interview Preparer",
        goal="Create interview questions and talking points based on the resume and job requirements",
        tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
        verbose=verbose,
        backstory=(
            "Your role is crucial in anticipating the dynamics of interviews. With your ability to formulate "
            "key questions and talking points, you prepare candidates for success, ensuring they can confidently "
            "address all aspects of the job they are applying for."
        ),
    )

    research_task = Task(
        description=(
            "Analyze the job posting URL provided ({job_posting_url}) to extract key skills, experiences, "
            "and qualifications required. Use the tools to gather content and identify and categorize the requirements."
        ),
        expected_output=(
            "A structured list of job requirements, including necessary skills, qualifications, and experiences."
        ),
        agent=researcher,
        async_execution=True,
    )

    profile_task = Task(
        description=(
            "Compile a detailed personal and professional profile using the GitHub ({github_url}) URLs, "
            "and personal write-up ({personal_writeup}). Utilize tools to extract and synthesize information."
        ),
        expected_output=(
            "A comprehensive profile document that includes skills, project experiences, contributions, "
            "interests, and communication style."
        ),
        agent=profiler,
        async_execution=True,
    )

    resume_strategy_task = Task(
        description=(
            "Using the profile and job requirements from previous tasks, tailor the resume to highlight "
            "the most relevant areas. Employ tools to adjust and enhance the resume content. Do not make up "
            "information. Update every section (summary, work experience, skills, education) to better reflect "
            "the candidate's abilities and how they match the job posting."
        ),
        expected_output=(
            "An updated resume that effectively highlights the candidate's qualifications and experiences "
            "relevant to the job."
        ),
        output_file=os.path.join(output_dir, "tailored_resume.md"),
        context=[research_task, profile_task],
        agent=resume_strategist,
    )

    interview_task = Task(
        description=(
            "Create a set of potential interview questions and talking points based on the tailored resume "
            "and job requirements. Use tools to generate relevant questions and discussion points to help "
            "the candidate highlight the main points of the resume and how it matches the job posting."
        ),
        expected_output=(
            "A document containing key questions and talking points that the candidate should prepare "
            "for the initial interview."
        ),
        output_file=os.path.join(output_dir, "interview_materials.md"),
        context=[research_task, profile_task, resume_strategy_task],
        agent=interview_preparer,
    )

    return Crew(
        agents=[researcher, profiler, resume_strategist, interview_preparer],
        tasks=[research_task, profile_task, resume_strategy_task, interview_task],
        verbose=verbose,
    )

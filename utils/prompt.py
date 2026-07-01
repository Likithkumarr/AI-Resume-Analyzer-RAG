"""
=============================================================================
Prompt Templates Module
=============================================================================
Contains the two LangChain PromptTemplates used in the Two-Stage Pipeline:
  1. EXTRACTION_PROMPT - Instructs LLM to extract raw skill lists only.
  2. FEEDBACK_PROMPT   - Instructs LLM to write qualitative HR evaluation
                         based on the deterministic matching results.
=============================================================================
"""
from langchain_core.prompts import PromptTemplate

# ── Stage 1: Skill Extraction Prompt ─────────────────────────────────────────
# This prompt tells the LLM to act ONLY as a data extractor.
# It must NOT match, score, or compare — just extract raw lists.
EXTRACTION_PROMPT_TEMPLATE = """
You are a highly analytical AI specialized in data extraction.
Your ONLY job is to extract technical skills, tools, frameworks, and soft skills from the provided text.

Job Description:
{question}

Resume Context:
{context}

Extract EVERY technical skill, tool, software, cloud platform, programming language, database, and soft skill you can find.
- Return two flat arrays of strings.
- Do NOT perform any matching, comparison, or scoring. Just extract the raw lists.

Respond ONLY with a valid JSON object. No markdown, no explanation.

{{
    "extracted_resume_skills": ["<skill_1>", "<skill_2>"],
    "extracted_job_skills": ["<skill_1>", "<skill_2>"]
}}
"""

extraction_prompt = PromptTemplate(
    template=EXTRACTION_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ── Stage 2: Qualitative Feedback Prompt ─────────────────────────────────────
# This prompt receives the DETERMINISTIC results from Python's RapidFuzz matcher.
# The LLM is forbidden from changing the score or skill lists.
# It only generates human-readable HR qualitative evaluation.
FEEDBACK_PROMPT_TEMPLATE = """
You are a senior HR recruiter and expert ATS (Applicant Tracking System).
The technical skill matching has already been performed by a deterministic ATS algorithm.
Do NOT recalculate the scores or change the matching skills.

Here are the deterministic results from the ATS algorithm:
- ATS Match Score: {ats_score}%
- Matching Skills: {matching_skills}
- Missing Required Skills: {missing_skills}
- Extra Skills (in resume but not JD): {extra_skills}

Based ONLY on these facts, generate a detailed qualitative evaluation for the candidate.

Respond ONLY with a valid JSON object. No markdown formatting, no explanation.

{{
    "candidate_suitability": "<string: brief one paragraph summary of their fit based on the score and missed skills>",
    "strengths": ["<strength_1>", "<strength_2>"],
    "weaknesses": ["<weakness_1>", "<weakness_2>"],
    "recommended_skills_to_learn": ["<skill_1>", "<skill_2>"],
    "resume_improvement_suggestions": ["<suggestion_1>", "<suggestion_2>"],
    "overall_feedback": "<string: detailed overall feedback paragraph>",
    "hiring_recommendation": "<string: exactly one of: Strongly Recommended, Recommended, Consider, Not Recommended>"
}}
"""

feedback_prompt = PromptTemplate(
    template=FEEDBACK_PROMPT_TEMPLATE,
    input_variables=["ats_score", "matching_skills", "missing_skills", "extra_skills"]
)

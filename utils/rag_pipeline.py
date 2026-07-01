"""
=============================================================================
RAG Pipeline Module
=============================================================================
Orchestrates the Two-Stage AI + Deterministic matching pipeline:
  Stage 1: LLM extracts raw skills from resume & JD text.
  Stage 2: Python (RapidFuzz) performs accurate set intersection + ATS score.
  Stage 3: LLM writes qualitative feedback based on the hard math results.
=============================================================================
"""
import json
import re
import json_repair
from langchain_ollama import OllamaLLM
from utils.prompt import extraction_prompt, feedback_prompt
from utils.skill_matcher import compare_skills, calculate_ats_score


def extract_json(output_text):
    """
    Extracts a JSON object from raw LLM output, stripping any conversational
    text or markdown formatting the model may have added.

    Args:
        output_text (str): Raw string output from the LLM.

    Returns:
        str: Clean JSON string.
    """
    # Find the first { ... } block in the output
    json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
    if json_match:
        output_text = json_match.group(0)

    # Strip common markdown code block wrappers
    if output_text.startswith("```json"):
        output_text = output_text[7:]
    if output_text.endswith("```"):
        output_text = output_text[:-3]

    return output_text.strip()


def analyze_resume_with_rag(resume_text, job_description):
    """
    Main pipeline function. Takes full resume text and job description,
    runs a Two-Stage pipeline and returns a comprehensive analysis dictionary.

    Args:
        resume_text (str):       Complete text of the candidate's resume.
        job_description (str):   Job description text from the user.

    Returns:
        dict: Analysis results including ats_score, matching_skills,
              missing_skills, extra_skills, strengths, weaknesses, etc.
              OR a dict with key "error" if the pipeline fails.
    """
    # Initialize Ollama LLM with low temperature for deterministic output
    llm = OllamaLLM(model="llama3.2", temperature=0.1)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: SKILL EXTRACTION
    # The LLM reads the full resume and JD and simply outputs two flat lists.
    # It is explicitly told NOT to match or score anything.
    # ─────────────────────────────────────────────────────────────────────────
    extract_input = extraction_prompt.format(
        context=resume_text,
        question=job_description
    )

    try:
        extract_result = llm.invoke(extract_input)
        extract_text = extract_json(extract_result)

        # Use json_repair to handle minor LLM formatting errors (missing commas, etc.)
        extracted_data = json_repair.loads(extract_text)
        if not isinstance(extracted_data, dict):
            raise json.JSONDecodeError("json_repair failed on extraction", extract_text, 0)

        resume_skills = extracted_data.get("extracted_resume_skills", [])
        job_skills = extracted_data.get("extracted_job_skills", [])

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 2: DETERMINISTIC MATCHING (Pure Python, No LLM)
        # RapidFuzz compares the two lists with alias normalization & fuzzy
        # matching. The LLM never touches this step.
        # ─────────────────────────────────────────────────────────────────────
        matching, missing, extra = compare_skills(resume_skills, job_skills)
        ats_score = calculate_ats_score(len(matching), len(missing))

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 3: QUALITATIVE FEEDBACK
        # The hard math results are fed back into the LLM.
        # It writes human-readable feedback, strengths, weaknesses, etc.
        # ─────────────────────────────────────────────────────────────────────
        feedback_input = feedback_prompt.format(
            ats_score=ats_score,
            matching_skills=", ".join(matching) if matching else "None",
            missing_skills=", ".join(missing) if missing else "None",
            extra_skills=", ".join(extra) if extra else "None"
        )

        feedback_result = llm.invoke(feedback_input)
        feedback_text = extract_json(feedback_result)

        final_data = json_repair.loads(feedback_text)
        if not isinstance(final_data, dict):
            raise json.JSONDecodeError("json_repair failed on feedback", feedback_text, 0)

        # Merge the deterministic results into the final response dict
        final_data["ats_score"] = ats_score
        final_data["matching_skills"] = matching
        final_data["missing_skills"] = missing
        final_data["extra_skills"] = extra

        return final_data

    except Exception as e:
        error_msg = f"Analysis Pipeline Error: {str(e)}"
        print(error_msg)
        return {
            "error": f"Failed to process analysis.\n\nError: {str(e)}\n\nPlease try again."
        }

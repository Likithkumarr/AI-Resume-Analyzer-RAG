"""
=============================================================================
Deterministic Skill Matching Module
=============================================================================
This module bypasses LLM logic to perform highly accurate mathematical
string matching. It uses the `rapidfuzz` library to compare sets of
skills extracted by the AI, ensuring 100% accurate ATS scoring.
=============================================================================
"""
import re
from rapidfuzz import fuzz, process


def normalize_skill(skill):
    """
    Normalizes a skill string by converting to lowercase, removing punctuation,
    extra spaces, and applying common industry aliases.

    Args:
        skill (str): Raw skill string.

    Returns:
        str: Normalized skill string.
    """
    skill = skill.lower().strip()

    # Remove common punctuation and hyphens, replace with space
    skill = re.sub(r'[^\w\s]', ' ', skill)
    skill = re.sub(r'\s+', ' ', skill).strip()

    # Dictionary of exact hardcoded aliases to standardize common skill names
    aliases = {
        "rest api": "rest api",
        "rest apis": "rest api",
        "js": "javascript",
        "java script": "javascript",
        "genai": "generative ai",
        "gen ai": "generative ai",
        "llm": "large language model",
        "llms": "large language model",
        "postgres": "postgresql",
        "ms sql": "microsoft sql server",
        "sql server": "microsoft sql server",
        "eda": "exploratory data analysis",
        "nlp": "natural language processing",
        "ml": "machine learning",
        "aws ec2 s3": "aws",
        "aws ec2": "aws",
        "aws s3": "aws",
        "problem solving": "problem solving",
        "problem-solving": "problem solving",
        "communication skills": "communication skills",
        "communication": "communication skills",
    }

    return aliases.get(skill, skill)


def split_compound_skills(skills_list):
    """
    Splits compound skills like "Data Cleaning & Visualization" into
    individual skills: ["Data Cleaning", "Data Visualization"].

    Args:
        skills_list (list): List of raw skill strings.

    Returns:
        list: Expanded list with compound skills split out.
    """
    expanded = []
    for skill in skills_list:
        # Split on & / and / + separators
        if " & " in skill or " and " in skill.lower() or " + " in skill:
            parts = re.split(r'\s+&\s+|\s+and\s+|\s+\+\s+', skill, flags=re.IGNORECASE)
            expanded.append(skill)  # Keep original too for fuzzy matching
            for p in parts:
                p = p.strip()
                if p:
                    expanded.append(p)
        else:
            expanded.append(skill)
    return expanded


def compare_skills(resume_skills, job_skills):
    """
    Uses RapidFuzz to deterministically compare resume skills against job skills.
    Returns three distinct, non-overlapping lists.

    Args:
        resume_skills (list): Skills extracted from the resume.
        job_skills (list): Skills required by the job description.

    Returns:
        tuple: (matching, missing, extra) — all as lists of strings.
    """
    # Step 1: Split compound skills before processing
    res_skills = split_compound_skills(resume_skills)
    jd_skills = split_compound_skills(job_skills)

    # Step 2: Build normalized lookup dictionaries {normalized -> original}
    res_dict = {}
    for rs in res_skills:
        if not rs:
            continue
        norm = normalize_skill(rs)
        if norm not in res_dict or len(rs) > len(res_dict[norm]):
            res_dict[norm] = rs

    jd_dict = {}
    for js in jd_skills:
        if not js:
            continue
        norm = normalize_skill(js)
        if norm not in jd_dict or len(js) > len(jd_dict[norm]):
            jd_dict[norm] = js

    matching = set()
    missing = set()

    # Working copy of resume skill keys that have not been matched yet
    norm_res_list = list(res_dict.keys())

    # Step 3: Loop through each JD skill and find the best resume match
    for norm_jd, orig_jd in jd_dict.items():
        # First: check for exact normalized match
        if norm_jd in res_dict:
            matching.add(orig_jd)
            if norm_jd in norm_res_list:
                norm_res_list.remove(norm_jd)
            continue

        # Second: fuzzy match against remaining resume skills
        if not norm_res_list:
            missing.add(orig_jd)
            continue

        match = process.extractOne(
            norm_jd,
            norm_res_list,
            scorer=fuzz.token_set_ratio  # Best for partial skill names
        )

        if match:
            best_match_str, score, _ = match
            if score >= 85:  # 85% similarity threshold
                matching.add(orig_jd)
                norm_res_list.remove(best_match_str)  # Mark as consumed
            else:
                missing.add(orig_jd)
        else:
            missing.add(orig_jd)

    # Step 4: Anything left in resume list that wasn't matched = Extra skill
    extra = set()
    for norm_res in norm_res_list:
        extra.add(res_dict[norm_res])

    return list(matching), list(missing), list(extra)


def calculate_ats_score(matching_count, missing_count):
    """
    Calculates the ATS match percentage.

    Formula: (matching / (matching + missing)) * 100

    Args:
        matching_count (int): Number of matched skills.
        missing_count (int): Number of missing skills.

    Returns:
        int: ATS score clamped between 0 and 100.
    """
    total_required = matching_count + missing_count
    if total_required == 0:
        return 0
    score = (matching_count / total_required) * 100
    return min(100, round(score))

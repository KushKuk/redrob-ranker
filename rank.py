"""Intelligent Candidate Ranking System
========================================================
Architecture: Multi-signal scoring pipeline with semantic-aware feature extraction.

Scoring components (weighted sum → behavioral multiplier):
  1. Title/Role Fit          (0.25) — Is this person actually an AI/ML engineer?
  2. Core Skills Match       (0.25) — Do they have the must-have technical skills?
  3. Career Quality          (0.20) — Product company exp, not pure services/research
  4. Experience Years        (0.10) — 5-9 yrs ideal range
  5. Soft Disqualifiers      (-penalty) — consulting-only, CV/speech-only, title-chaser
  6. Behavioral Availability (0.15) — Engagement signals, recency, response rate
  7. Location/Logistics       (0.05) — India-based, relocation willingness, notice period

Honeypot detection: flags impossible profiles (expert skills with 0 months, 
future dates, suspicious career timelines) and penalizes heavily.
"""

import csv
import json
import math
import re
import sys
import argparse
from datetime import date, datetime
from pathlib import Path

# JD KNOWLEDGE BASE

# Titles that signal this person IS an AI/ML engineer
POSITIVE_TITLE_TERMS = {
    "machine learning", "ml engineer", "ai engineer", "applied ml", "applied ai",
    "nlp engineer", "data scientist", "research engineer", "senior engineer",
    "software engineer", "backend engineer", "full stack", "platform engineer",
    "search engineer", "ranking engineer", "recommendation", "retrieval",
    "founding engineer", "staff engineer", "principal engineer",
}

# Titles that strongly disqualify (not the role, no matter the skills)
DISQUALIFYING_TITLE_TERMS = {
    "marketing", "hr ", "human resources", "operations manager", "sales",
    "product manager", "finance", "accountant", "content writer", "designer",
    "ux ", "ui ", "graphic", "recruiter", "talent", "ceo", "cto", "coo",
    "vp ", "vice president", "director of", "program manager",
    "business analyst", "consultant", "project manager",
}

# Must-have technical skills (JD section: "Things you absolutely need")
CORE_SKILLS = {
    # Embeddings & retrieval
    "sentence-transformers", "sentence transformers", "embeddings", "vector search",
    "dense retrieval", "semantic search", "bi-encoder", "cross-encoder",
    "rag", "retrieval augmented generation",
    # Models
    "bge", "e5", "openai embeddings", "ada", "text-embedding",
    # Vector DBs
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "chromadb", "chroma",
    # Search
    "hybrid search", "bm25", "sparse retrieval", "solr",
    # General ML
    "pytorch", "tensorflow", "hugging face", "huggingface", "transformers",
    "llm", "large language model", "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    # Ranking/IR
    "learning to rank", "ndcg", "mrr", "map", "ranking", "reranking", "re-ranking",
    "xgboost", "lightgbm", "recommendation system", "recommender",
    # Python ML
    "scikit-learn", "sklearn", "numpy", "pandas", "mlflow", "wandb",
    "weights & biases", "dvc",
    # NLP
    "nlp", "natural language processing", "text classification", "named entity",
    "ner", "bert", "gpt", "llama", "mistral", "gemma",
    # Infra
    "docker", "kubernetes", "aws", "gcp", "azure", "spark", "kafka",
    "airflow", "mlops", "model serving", "triton", "onnx",
    # Evaluation
    "a/b testing", "ab testing", "evaluation framework", "offline evaluation",
}

# Nice-to-have skills
BONUS_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning llms",
    "xgboost", "lightgbm", "learning to rank",
    "distributed systems", "large-scale inference",
    "open source", "github",
}

# Industries that indicate product-company AI work
PRODUCT_INDUSTRIES = {
    "artificial intelligence", "machine learning", "technology", "software",
    "saas", "fintech", "edtech", "healthtech", "e-commerce", "internet",
    "data analytics", "cloud computing", "information technology",
}

# Pure consulting/services companies to penalize
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "l&t infotech", "mindtree", "niit", "patni", "mastech", "igate",
}

# Pure CV/speech/robotics keywords — wrong domain
WRONG_DOMAIN_SKILLS = {
    "computer vision", "image classification", "object detection", "segmentation",
    "opencv", "yolo", "image recognition", "speech recognition", "asr",
    "text to speech", "tts", "speech synthesis", "robotics", "ros",
    "slam", "autonomous driving", "lidar",
}

# India locations that are acceptable per JD
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru",
    "gurgaon", "gurugram", "ncr", "india",
}

# HELPERS

def normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def text_contains_any(text: str, terms: set) -> bool:
    t = normalize(text)
    return any(term in t for term in terms)


def count_matches(text: str, terms: set) -> int:
    t = normalize(text)
    return sum(1 for term in terms if term in t)


def parse_date(s) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def days_since(d: date | None, today: date) -> int:
    if d is None:
        return 9999
    return (today - d).days

# HONEYPOT DETECTION

def detect_honeypot(candidate: dict, today: date) -> tuple[bool, str]:
    """
    Returns (is_honeypot, reason).
    Checks for impossible/contradictory profile signals.
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    sigs = candidate.get("redrob_signals", {})

    # 1. Expert skill with 0 duration months (impossible)
    expert_zero = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    ]
    if len(expert_zero) >= 3:
        return True, f"Expert proficiency in {len(expert_zero)} skills with 0 months usage"

    # 2. YOE < sum of career duration / 12 (inflated experience)
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    stated_yoe = profile.get("years_of_experience", 0)
    if total_career_months > 0 and stated_yoe < (total_career_months / 12) * 0.5:
        return True, f"Stated YOE ({stated_yoe}) far below career months ({total_career_months})"

    # 3. Future start dates in career
    for role in career:
        start = parse_date(role.get("start_date"))
        if start and start > today:
            return True, f"Future start date: {start}"

    # 4. End date before start date
    for role in career:
        start = parse_date(role.get("start_date"))
        end = parse_date(role.get("end_date"))
        if start and end and end < start:
            return True, f"End date before start date at {role.get('company','?')}"

    # 5. Profile completeness 100 but signup today and no activity
    signup = parse_date(sigs.get("signup_date"))
    last_active = parse_date(sigs.get("last_active_date"))
    if signup and last_active and last_active < signup:
        return True, "Last active before signup date"

    # 6. Extreme skill count with all expert and no endorsements
    all_expert = [s for s in skills if s.get("proficiency") == "expert"]
    if len(all_expert) >= 12 and sigs.get("endorsements_received", 0) == 0:
        return True, f"{len(all_expert)} expert skills with 0 endorsements"

    # 7. Impossibly high response rate + offer acceptance for brand new user
    if signup:
        months_on_platform = (today - signup).days / 30
        if months_on_platform < 1:
            if sigs.get("recruiter_response_rate", 0) > 0.9 and \
               sigs.get("offer_acceptance_rate", -1) > 0.9:
                return True, "Perfect engagement scores on <1 month old account"

    return False, ""

# SCORING COMPONENTS

def score_title_fit(candidate: dict) -> float:
    """
    0.0-1.0. Is this person an AI/ML/Software engineer by role?
    The single most important disqualifier: an HR Manager with AI skills is not a fit.
    """
    profile = candidate["profile"]
    title = normalize(profile.get("current_title", ""))
    headline = normalize(profile.get("headline", ""))
    summary = normalize(profile.get("summary", ""))

    # Hard disqualifier: non-engineering title
    if text_contains_any(title, DISQUALIFYING_TITLE_TERMS):
        return 0.02  # Very low but not zero (might still rank above true garbage)

    # Strong positive: explicitly AI/ML title
    if text_contains_any(title, {"machine learning", "ml engineer", "ai engineer",
                                   "nlp engineer", "data scientist", "applied ml",
                                   "applied ai", "research engineer"}):
        return 1.0

    # Moderate positive: general engineering with ML signals in headline
    if text_contains_any(title, {"engineer", "developer", "architect", "scientist",
                                   "analyst", "data"}):
        if text_contains_any(headline + " " + summary, CORE_SKILLS):
            return 0.85
        return 0.55

    # Unknown title — check if career suggests engineering
    career_titles = " ".join(r.get("title", "") for r in candidate.get("career_history", []))
    if text_contains_any(normalize(career_titles), POSITIVE_TITLE_TERMS):
        return 0.6

    return 0.2


def score_skills(candidate: dict) -> float:
    """
    0.0-1.0. How well do skills match the JD's must-haves?
    Weighted by proficiency level and duration. Penalizes pure keyword stuffing.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    # Build a rich text blob of all profile text for contextual matching
    profile = candidate["profile"]
    career_text = " ".join(
        r.get("description", "") + " " + r.get("title", "") + " " + r.get("company", "")
        for r in candidate.get("career_history", [])
    )
    all_text = normalize(
        profile.get("summary", "") + " " +
        profile.get("headline", "") + " " +
        career_text
    )

    # Proficiency weights
    prof_weight = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}

    skill_score = 0.0
    max_possible = 0.0

    # Score each core skill
    for skill in skills:
        name = normalize(skill.get("name", ""))
        prof = prof_weight.get(skill.get("proficiency", "beginner"), 0.3)
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        # Check if this skill is in our core or bonus lists
        is_core = any(cs in name or name in cs for cs in CORE_SKILLS)
        is_bonus = any(bs in name or name in bs for bs in BONUS_SKILLS)

        if is_core or is_bonus:
            # Trust multiplier: duration + endorsements prevent keyword stuffing
            dur_factor = min(1.0, duration / 24) if duration > 0 else 0.4
            endorse_factor = min(1.0, 0.5 + endorsements / 20)
            trust = 0.5 * dur_factor + 0.5 * endorse_factor

            weight = 1.5 if is_core else 0.8
            skill_score += weight * prof * trust
            max_possible += weight

    # Also count core skills mentioned in career text / summary (contextual)
    context_matches = count_matches(all_text, CORE_SKILLS)
    context_bonus = min(0.3, context_matches * 0.03)

    # Assessment scores bonus
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    assess_bonus = 0.0
    for skill_name, score in assessments.items():
        sn = normalize(skill_name)
        if any(cs in sn or sn in cs for cs in CORE_SKILLS):
            assess_bonus += (score / 100) * 0.1
    assess_bonus = min(0.2, assess_bonus)

    if max_possible == 0:
        raw = context_bonus
    else:
        raw = (skill_score / max_possible) + context_bonus + assess_bonus

    return min(1.0, raw)


def score_career_quality(candidate: dict) -> float:
    """
    0.0-1.0. Quality of career trajectory.
    Rewards: product companies, AI/tech industry, progression, meaningful duration.
    Penalizes: pure consulting, CV/speech-only, no production experience.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.1

    profile = candidate["profile"]
    total_score = 0.0
    total_weight = 0.0
    consulting_months = 0
    total_months = 0
    wrong_domain_score = 0.0

    for role in career:
        company = normalize(role.get("company", ""))
        title = normalize(role.get("title", ""))
        industry = normalize(role.get("industry", ""))
        description = normalize(role.get("description", ""))
        duration = role.get("duration_months", 1)
        is_current = role.get("is_current", False)

        total_months += duration
        weight = duration * (1.5 if is_current else 1.0)

        role_score = 0.5  # baseline

        # Boost for AI/tech product company
        if text_contains_any(industry, PRODUCT_INDUSTRIES):
            role_score += 0.3
        if text_contains_any(title, {"machine learning", "ml", "ai", "nlp", "data scientist",
                                      "research engineer", "applied", "ranking", "search"}):
            role_score += 0.3
        if text_contains_any(description, CORE_SKILLS):
            role_score += 0.2

        # Penalize consulting
        if any(cc in company for cc in CONSULTING_COMPANIES):
            role_score -= 0.4
            consulting_months += duration

        # Wrong domain penalty
        if text_contains_any(title + " " + description, WRONG_DOMAIN_SKILLS):
            wrong_domain_score += duration

        role_score = max(0.0, min(1.0, role_score))
        total_score += role_score * weight
        total_weight += weight

    base = total_score / total_weight if total_weight > 0 else 0.1

    # Global consulting penalty: if >70% career in consulting, heavy penalty
    if total_months > 0:
        consulting_pct = consulting_months / total_months
        if consulting_pct > 0.9:
            base *= 0.3
        elif consulting_pct > 0.7:
            base *= 0.6

    # Wrong domain penalty
    if total_months > 0:
        wrong_pct = wrong_domain_score / total_months
        base *= max(0.5, 1.0 - wrong_pct * 0.6)

    return min(1.0, base)


def score_experience_years(candidate: dict) -> float:
    """
    0.0-1.0. JD wants 5-9 years (sweet spot: 6-8).
    Also verifies years in AI/ML specifically.
    """
    profile = candidate["profile"]
    yoe = profile.get("years_of_experience", 0)

    # Ideal range: 5-9 years total
    if 6 <= yoe <= 8:
        base = 1.0
    elif 5 <= yoe < 6 or 8 < yoe <= 9:
        base = 0.85
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        base = 0.65
    elif 3 <= yoe < 4 or 11 < yoe <= 14:
        base = 0.45
    else:
        base = 0.2

    # Check how much of that is in AI/ML roles
    career = candidate.get("career_history", [])
    ai_months = sum(
        r.get("duration_months", 0) for r in career
        if text_contains_any(
            normalize(r.get("title", "") + " " + r.get("industry", "") + " " + r.get("description", "")),
            {"machine learning", "ml", "ai", "nlp", "data science", "retrieval",
             "ranking", "recommendation", "search", "embeddings", "llm"}
        )
    )
    ai_years = ai_months / 12

    if ai_years >= 4:
        ai_bonus = 0.15
    elif ai_years >= 2:
        ai_bonus = 0.05
    else:
        ai_bonus = -0.1

    return min(1.0, max(0.0, base + ai_bonus))


def score_behavioral_availability(candidate: dict, today: date) -> float:
    """
    0.0-1.0. Behavioral engagement signals from Redrob platform.
    A perfect-on-paper candidate who's inactive is not hireable.
    """
    sigs = candidate.get("redrob_signals", {})

    score = 0.0

    # 1. Recency of activity
    last_active = parse_date(sigs.get("last_active_date"))
    days_inactive = days_since(last_active, today)
    if days_inactive <= 7:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.85
    elif days_inactive <= 90:
        recency = 0.6
    elif days_inactive <= 180:
        recency = 0.35
    else:
        recency = 0.1
    score += recency * 0.25

    # 2. Open to work flag
    if sigs.get("open_to_work_flag", False):
        score += 0.15

    # 3. Recruiter response rate
    response_rate = sigs.get("recruiter_response_rate", 0.0)
    score += response_rate * 0.20

    # 4. Profile completeness
    completeness = sigs.get("profile_completeness_score", 0) / 100
    score += completeness * 0.10

    # 5. Active applications
    apps = min(sigs.get("applications_submitted_30d", 0), 10)
    score += (apps / 10) * 0.08

    # 6. GitHub activity
    gh_score = sigs.get("github_activity_score", -1)
    if gh_score >= 0:
        score += (gh_score / 100) * 0.10

    # 7. Interview completion rate
    icr = sigs.get("interview_completion_rate", 0.5)
    score += icr * 0.08

    # 8. Saved by recruiters
    saved = min(sigs.get("saved_by_recruiters_30d", 0), 10)
    score += (saved / 10) * 0.04

    return min(1.0, score)


def score_location_logistics(candidate: dict) -> float:
    """
    0.0-1.0. Location fit and logistical readiness.
    """
    profile = candidate["profile"]
    sigs = candidate.get("redrob_signals", {})

    score = 0.5

    # Location check
    location = normalize(profile.get("location", "") + " " + profile.get("country", ""))
    if text_contains_any(location, {"pune", "noida"}):
        score += 0.3
    elif text_contains_any(location, {"bangalore", "bengaluru", "mumbai", "hyderabad",
                                       "delhi", "ncr", "gurgaon", "gurugram"}):
        score += 0.15
    elif "india" in location:
        score += 0.05
    else:
        score -= 0.2

    # Relocation willingness
    if sigs.get("willing_to_relocate", False):
        score += 0.1

    # Work mode preference
    mode = sigs.get("preferred_work_mode", "")
    if mode in ("hybrid", "flexible"):
        score += 0.1
    elif mode == "onsite":
        score += 0.05
    # remote gets no bonus

    # Notice period
    notice = sigs.get("notice_period_days", 90)
    if notice <= 30:
        score += 0.1
    elif notice <= 60:
        score += 0.0
    elif notice <= 90:
        score -= 0.05
    else:
        score -= 0.1

    return min(1.0, max(0.0, score))


def score_soft_disqualifiers(candidate: dict) -> float:
    """
    Returns a penalty multiplier (0.0-1.0).
    1.0 = no penalty, 0.0 = fully disqualified.
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    penalty = 1.0

    # 1. Pure research profile
    all_text = normalize(
        profile.get("summary", "") + " " +
        " ".join(r.get("description", "") + " " + r.get("title", "") for r in career)
    )
    research_signals = count_matches(all_text, {"research lab", "phd research",
                                                  "academic", "research scientist",
                                                  "paper", "arxiv", "published"})
    production_signals = count_matches(all_text, {"production", "deployed", "shipped",
                                                    "serving", "latency", "scale",
                                                    "real users", "a/b test", "api"})
    if research_signals >= 3 and production_signals == 0:
        penalty *= 0.4

    # 2. LangChain-only AI experience
    is_langchain_only = "langchain" in all_text and not any(
        t in all_text for t in ["vector", "embedding", "bm25", "ranking",
                                  "faiss", "retrieval", "recommendation"]
    )
    if is_langchain_only:
        penalty *= 0.6

    # 3. Title chaser pattern: many short stints in different companies
    if len(career) >= 4:
        short_stints = sum(1 for r in career if r.get("duration_months", 99) < 14)
        if short_stints >= 3:
            penalty *= 0.8

    # 4. Entirely wrong domain (CV/speech primary, no NLP/IR)
    skill_names = " ".join(normalize(s.get("name", "")) for s in skills)
    career_text = " ".join(normalize(r.get("title", "") + " " + r.get("description", "")) for r in career)
    wrong_signals = count_matches(skill_names + " " + career_text, WRONG_DOMAIN_SKILLS)
    right_signals = count_matches(skill_names + " " + career_text, CORE_SKILLS)
    if wrong_signals > right_signals and wrong_signals >= 3:
        penalty *= 0.65

    return penalty

# MAIN SCORING FUNCTION

WEIGHTS = {
    "title": 0.25,
    "skills": 0.25,
    "career": 0.20,
    "experience": 0.10,
    "behavioral": 0.15,
    "location": 0.05,
}

def score_candidate(candidate: dict, today: date) -> tuple[float, dict]:
    """
    Returns (final_score 0-1, component_scores dict).
    """
    # Honeypot check — if detected, return near-zero score
    is_hp, hp_reason = detect_honeypot(candidate, today)
    if is_hp:
        return 0.001, {"honeypot": True, "reason": hp_reason}

    # Component scores
    title_s = score_title_fit(candidate)
    skills_s = score_skills(candidate)
    career_s = score_career_quality(candidate)
    exp_s = score_experience_years(candidate)
    behavioral_s = score_behavioral_availability(candidate, today)
    location_s = score_location_logistics(candidate)

    # Weighted sum
    raw = (
        WEIGHTS["title"] * title_s +
        WEIGHTS["skills"] * skills_s +
        WEIGHTS["career"] * career_s +
        WEIGHTS["experience"] * exp_s +
        WEIGHTS["behavioral"] * behavioral_s +
        WEIGHTS["location"] * location_s
    )

    # Soft disqualifier multiplier
    penalty = score_soft_disqualifiers(candidate)
    final = raw * penalty

    components = {
        "title": round(title_s, 3),
        "skills": round(skills_s, 3),
        "career": round(career_s, 3),
        "experience": round(exp_s, 3),
        "behavioral": round(behavioral_s, 3),
        "location": round(location_s, 3),
        "penalty": round(penalty, 3),
        "raw": round(raw, 3),
        "final": round(final, 3),
    }

    return final, components

# REASONING GENERATOR

def generate_reasoning(candidate: dict, components: dict, rank: int) -> str:
    """
    Generates specific, fact-grounded 1-2 sentence reasoning per submission spec.
    References actual profile data, connects to JD requirements, acknowledges concerns.
    """
    profile = candidate["profile"]
    sigs = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    if components.get("honeypot"):
        return f"Flagged as honeypot: {components.get('reason', 'impossible profile data')}."

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "unknown title")
    company = profile.get("current_company", "unknown company")
    location = profile.get("location", "unknown location")
    notice = sigs.get("notice_period_days", 90)
    response_rate = sigs.get("recruiter_response_rate", 0)
    open_to_work = sigs.get("open_to_work_flag", False)
    last_active = sigs.get("last_active_date", "unknown")
    github = sigs.get("github_activity_score", -1)

    # Top core skills the candidate has
    core_skill_matches = [
        s["name"] for s in skills
        if any(cs in normalize(s.get("name", "")) or normalize(s.get("name", "")) in cs
               for cs in CORE_SKILLS)
        and s.get("proficiency") in ("advanced", "expert")
    ][:4]

    # Recent company + industry
    current_role = next((r for r in career if r.get("is_current")), None)
    current_industry = current_role.get("industry", "") if current_role else ""

    # Build reasoning pieces
    strengths = []
    concerns = []

    # Title/role fit
    if components["title"] >= 0.8:
        strengths.append(f"{title} role directly matches JD requirements")
    elif components["title"] <= 0.1:
        concerns.append(f"title ({title}) is outside engineering/ML — significant misfit")

    # Skills
    if core_skill_matches:
        strengths.append(f"advanced/expert skills in {', '.join(core_skill_matches[:3])}")
    if components["skills"] >= 0.7:
        strengths.append("strong core retrieval/ranking skill coverage")
    elif components["skills"] <= 0.2:
        concerns.append("limited match on must-have embedding/vector DB skills")

    # Experience
    if 5 <= yoe <= 9:
        strengths.append(f"{yoe:.1f} years experience squarely in JD's 5-9 yr range")
    elif yoe < 5:
        concerns.append(f"only {yoe:.1f} years total experience — below JD minimum")
    else:
        concerns.append(f"{yoe:.1f} years — above ideal range but not disqualifying")

    # Career quality
    if components["career"] >= 0.7:
        if current_industry:
            strengths.append(f"product-company background in {current_industry}")
    elif components["career"] <= 0.3:
        concerns.append("heavy consulting/services background — JD explicitly flags this")

    # Behavioral
    if open_to_work and response_rate >= 0.7:
        strengths.append(f"actively engaged (open to work, {response_rate:.0%} response rate)")
    elif response_rate <= 0.15:
        concerns.append(f"low recruiter response rate ({response_rate:.0%}) — availability uncertain")
    elif not open_to_work:
        concerns.append("not marked open to work")

    if notice <= 30:
        strengths.append(f"short notice period ({notice}d — matches JD preference)")
    elif notice >= 90:
        concerns.append(f"long notice period ({notice}d)")

    # GitHub
    if github >= 40:
        strengths.append(f"strong GitHub activity (score {github:.0f})")

    # Location
    loc_lower = location.lower()
    if any(city in loc_lower for city in ["pune", "noida"]):
        strengths.append(f"based in {location} (preferred JD location)")
    elif any(city in loc_lower for city in ["bangalore", "bengaluru", "mumbai", "hyderabad", "delhi"]):
        strengths.append(f"based in {location} (acceptable per JD)")

    # Build final sentence
    if rank <= 10:
        opener = f"{yoe:.1f}yr {title} at {company}"
        s_text = "; ".join(strengths[:3]) if strengths else "strong overall fit"
        c_text = f" Minor concerns: {'; '.join(concerns[:1])}." if concerns else "."
        return f"{opener} — {s_text}{c_text}"
    elif rank <= 50:
        s_text = "; ".join(strengths[:2]) if strengths else "moderate fit"
        c_text = f" Concerns: {'; '.join(concerns[:2])}." if concerns else "."
        return f"{yoe:.1f}yr {title}: {s_text}.{c_text}"
    else:
        c_text = "; ".join(concerns[:2]) if concerns else "marginal fit"
        s_text = f" Some strengths: {'; '.join(strengths[:1])}." if strengths else ""
        return f"{yoe:.1f}yr {title} — below cutoff due to: {c_text}.{s_text}"


# MAIN ENTRY POINT

def load_candidates(path: str):
    """Load from .jsonl or .jsonl.gz"""
    import gzip
    p = Path(path)
    if p.suffix == ".gz":
        open_fn = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        open_fn = lambda: open(path, "r", encoding="utf-8")

    candidates = []
    with open_fn() as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def rank_candidates(candidates_path: str, output_path: str, top_n: int = 100):
    today = date.today()
    print(f"Loading candidates from {candidates_path}...", flush=True)
    candidates = load_candidates(candidates_path)
    print(f"Loaded {len(candidates)} candidates. Scoring...", flush=True)

    scored = []
    for i, cand in enumerate(candidates):
        if i % 10000 == 0:
            print(f"  Scored {i}/{len(candidates)}...", flush=True)
        score, components = score_candidate(cand, today)
        scored.append((score, cand, components))

    # Sort descending by score, then by candidate_id ascending for tie-breaking
    # candidate_id format is CAND_XXXXXXX — sort numerically for correct tie-break
    scored.sort(key=lambda x: (-round(x[0], 4), x[1]["candidate_id"]))

    top = scored[:top_n]

    print(f"Writing top {top_n} to {output_path}...", flush=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cand, components) in enumerate(top, 1):
            cid = cand["candidate_id"]
            rounded_score = round(score, 4)
            reasoning = generate_reasoning(cand, components, rank)
            writer.writerow([cid, rank, rounded_score, reasoning])

    print("Done!")

    # Print top 10 summary
    print("\n=== TOP 10 CANDIDATES ===")
    for rank, (score, cand, components) in enumerate(top[:10], 1):
        profile = cand["profile"]
        print(f"#{rank} {cand['candidate_id']} — {profile['current_title']} | "
              f"YOE: {profile['years_of_experience']} | Score: {score:.4f}")
        print(f"     Components: {components}")
        print(f"     Reasoning: {generate_reasoning(cand, components, rank)}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="./submission.csv",
                        help="Output CSV path")
    parser.add_argument("--top", type=int, default=100,
                        help="Number of top candidates to output")
    args = parser.parse_args()

    rank_candidates(args.candidates, args.out, args.top)


if __name__ == "__main__":
    main()

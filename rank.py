#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranking System
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

Optimisations (v2):
  - Compiled regex for normalize() — avoids recompiling on every call
  - Early-exit in score_candidate() for hard-disqualified titles (~60% of pool)
    skips skills/career/soft-disqualifier scoring entirely
  - Frozensets for all term lookups
"""

import csv
import json
import re
import argparse
from datetime import date, datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# COMPILED REGEX  (compiled once at import, reused across all 100K candidates)
# ─────────────────────────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")


# ─────────────────────────────────────────────────────────────────────────────
# JD KNOWLEDGE BASE  (derived from job_description.md)
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_TITLE_TERMS = frozenset({
    "machine learning", "ml engineer", "ai engineer", "applied ml", "applied ai",
    "nlp engineer", "data scientist", "research engineer", "senior engineer",
    "software engineer", "backend engineer", "full stack", "platform engineer",
    "search engineer", "ranking engineer", "recommendation", "retrieval",
    "founding engineer", "staff engineer", "principal engineer",
})

DISQUALIFYING_TITLE_TERMS = frozenset({
    "marketing", "hr ", "human resources", "operations manager", "sales",
    "product manager", "finance", "accountant", "content writer", "designer",
    "ux ", "ui ", "graphic", "recruiter", "talent", "ceo", "cto", "coo",
    "vp ", "vice president", "director of", "program manager",
    "business analyst", "consultant", "project manager",
})

CORE_SKILLS = frozenset({
    "sentence-transformers", "sentence transformers", "embeddings", "vector search",
    "dense retrieval", "semantic search", "bi-encoder", "cross-encoder",
    "rag", "retrieval augmented generation",
    "bge", "e5", "openai embeddings", "ada", "text-embedding",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "chromadb", "chroma",
    "hybrid search", "bm25", "sparse retrieval", "solr",
    "pytorch", "tensorflow", "hugging face", "huggingface", "transformers",
    "llm", "large language model", "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "learning to rank", "ndcg", "mrr", "map", "ranking", "reranking", "re-ranking",
    "xgboost", "lightgbm", "recommendation system", "recommender",
    "scikit-learn", "sklearn", "numpy", "pandas", "mlflow", "wandb",
    "weights & biases", "dvc",
    "nlp", "natural language processing", "text classification", "named entity",
    "ner", "bert", "gpt", "llama", "mistral", "gemma",
    "docker", "kubernetes", "aws", "gcp", "azure", "spark", "kafka",
    "airflow", "mlops", "model serving", "triton", "onnx",
    "a/b testing", "ab testing", "evaluation framework", "offline evaluation",
})

BONUS_SKILLS = frozenset({
    "lora", "qlora", "peft", "fine-tuning llms",
    "xgboost", "lightgbm", "learning to rank",
    "distributed systems", "large-scale inference",
    "open source", "github",
})

PRODUCT_INDUSTRIES = frozenset({
    "artificial intelligence", "machine learning", "technology", "software",
    "saas", "fintech", "edtech", "healthtech", "e-commerce", "internet",
    "data analytics", "cloud computing", "information technology",
})

CONSULTING_COMPANIES = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "l&t infotech", "mindtree", "niit", "patni", "mastech", "igate",
})

WRONG_DOMAIN_SKILLS = frozenset({
    "computer vision", "image classification", "object detection", "segmentation",
    "opencv", "yolo", "image recognition", "speech recognition", "asr",
    "text to speech", "tts", "speech synthesis", "robotics", "ros",
    "slam", "autonomous driving", "lidar",
})

AI_ROLE_TERMS = frozenset({
    "machine learning", "ml", "ai", "nlp", "data science", "retrieval",
    "ranking", "recommendation", "search", "embeddings", "llm",
})

RESEARCH_TERMS = frozenset({"research lab", "phd research", "academic",
                             "research scientist", "paper", "arxiv", "published"})
PRODUCTION_TERMS = frozenset({"production", "deployed", "shipped", "serving",
                               "latency", "scale", "real users", "a/b test", "api"})
RETRIEVAL_TERMS = frozenset({"vector", "embedding", "bm25", "ranking",
                              "faiss", "retrieval", "recommendation"})

# Titles that are strong ML signals (used in early-exit fast path)
_STRONG_ML_TITLES = frozenset({
    "machine learning", "ml engineer", "ai engineer", "nlp engineer",
    "data scientist", "applied ml", "applied ai", "research engineer",
})
_GENERAL_ENG_TITLES = frozenset({
    "engineer", "developer", "architect", "scientist", "analyst", "data",
})


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase and collapse whitespace. Uses pre-compiled regex."""
    return _WS.sub(" ", text.lower().strip())


def text_contains_any(text: str, terms: frozenset) -> bool:
    t = normalize(text)
    return any(term in t for term in terms)


def count_matches(text: str, terms: frozenset) -> int:
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


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_honeypot(candidate: dict, today: date) -> tuple[bool, str]:
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    sigs = candidate.get("redrob_signals", {})

    # 1. Expert skill with 0 duration months
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zero >= 3:
        return True, f"Expert proficiency in {expert_zero} skills with 0 months usage"

    # 2. Stated YOE far below career history
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    stated_yoe = profile.get("years_of_experience", 0)
    if total_career_months > 0 and stated_yoe < (total_career_months / 12) * 0.5:
        return True, f"Stated YOE ({stated_yoe}) far below career months ({total_career_months})"

    # 3. Future start dates / end before start
    for role in career:
        start = parse_date(role.get("start_date"))
        if start and start > today:
            return True, f"Future start date: {start}"
        end = parse_date(role.get("end_date"))
        if start and end and end < start:
            return True, f"End date before start date at {role.get('company','?')}"

    # 4. Last active before signup
    signup = parse_date(sigs.get("signup_date"))
    last_active = parse_date(sigs.get("last_active_date"))
    if signup and last_active and last_active < signup:
        return True, "Last active before signup date"

    # 5. Masses of expert skills, zero endorsements
    all_expert = sum(1 for s in skills if s.get("proficiency") == "expert")
    if all_expert >= 12 and sigs.get("endorsements_received", 0) == 0:
        return True, f"{all_expert} expert skills with 0 endorsements"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# SCORING COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def _title_score(title_n: str, headline_n: str, summary_n: str,
                 career: list) -> float:
    """Pre-normalised inputs for speed."""
    if any(term in title_n for term in DISQUALIFYING_TITLE_TERMS):
        return 0.02
    if any(term in title_n for term in _STRONG_ML_TITLES):
        return 1.0
    if any(term in title_n for term in _GENERAL_ENG_TITLES):
        combined = headline_n + " " + summary_n
        if any(term in combined for term in CORE_SKILLS):
            return 0.85
        return 0.55
    career_titles_n = normalize(" ".join(r.get("title", "") for r in career))
    if any(term in career_titles_n for term in POSITIVE_TITLE_TERMS):
        return 0.6
    return 0.2


def score_skills(candidate: dict, career_text_n: str, all_text_n: str) -> float:
    """Takes pre-built text blobs to avoid re-normalising."""
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    prof_weight = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}
    skill_score = 0.0
    max_possible = 0.0

    for skill in skills:
        name = normalize(skill.get("name", ""))
        prof = prof_weight.get(skill.get("proficiency", "beginner"), 0.3)
        duration = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)

        is_core = any(cs in name or name in cs for cs in CORE_SKILLS)
        is_bonus = (not is_core) and any(bs in name or name in bs for bs in BONUS_SKILLS)

        if is_core or is_bonus:
            dur_factor = min(1.0, duration / 24) if duration > 0 else 0.4
            endorse_factor = min(1.0, 0.5 + endorsements / 20)
            trust = 0.5 * dur_factor + 0.5 * endorse_factor
            weight = 1.5 if is_core else 0.8
            skill_score += weight * prof * trust
            max_possible += weight

    context_matches = sum(1 for term in CORE_SKILLS if term in all_text_n)
    context_bonus = min(0.3, context_matches * 0.03)

    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    assess_bonus = min(0.2, sum(
        (sc / 100) * 0.1
        for skill_name, sc in assessments.items()
        if any(cs in normalize(skill_name) or normalize(skill_name) in cs for cs in CORE_SKILLS)
    ))

    raw = ((skill_score / max_possible) if max_possible > 0 else 0.0) + context_bonus + assess_bonus
    return min(1.0, raw)


def score_career_quality(candidate: dict) -> float:
    career = candidate.get("career_history", [])
    if not career:
        return 0.1

    total_score = 0.0
    total_weight = 0.0
    consulting_months = 0
    total_months = 0
    wrong_domain_score = 0.0

    for role in career:
        company_n = normalize(role.get("company", ""))
        title_n = normalize(role.get("title", ""))
        industry_n = normalize(role.get("industry", ""))
        desc_n = normalize(role.get("description", ""))
        duration = role.get("duration_months", 1)
        is_current = role.get("is_current", False)

        total_months += duration
        weight = duration * (1.5 if is_current else 1.0)

        role_score = 0.5
        if any(term in industry_n for term in PRODUCT_INDUSTRIES):
            role_score += 0.3
        if any(term in title_n for term in {"machine learning", "ml", "ai", "nlp",
                                             "data scientist", "research engineer",
                                             "applied", "ranking", "search"}):
            role_score += 0.3
        if any(term in desc_n for term in CORE_SKILLS):
            role_score += 0.2
        if any(cc in company_n for cc in CONSULTING_COMPANIES):
            role_score -= 0.4
            consulting_months += duration
        if any(term in title_n + " " + desc_n for term in WRONG_DOMAIN_SKILLS):
            wrong_domain_score += duration

        total_score += max(0.0, min(1.0, role_score)) * weight
        total_weight += weight

    base = total_score / total_weight if total_weight > 0 else 0.1

    if total_months > 0:
        cp = consulting_months / total_months
        if cp > 0.9:
            base *= 0.3
        elif cp > 0.7:
            base *= 0.6
        wp = wrong_domain_score / total_months
        base *= max(0.5, 1.0 - wp * 0.6)

    return min(1.0, base)


def score_experience_years(candidate: dict, career: list) -> float:
    yoe = candidate["profile"].get("years_of_experience", 0)
    if   6 <= yoe <= 8:   base = 1.0
    elif 5 <= yoe < 6 or 8 < yoe <= 9:  base = 0.85
    elif 4 <= yoe < 5 or 9 < yoe <= 11: base = 0.65
    elif 3 <= yoe < 4 or 11 < yoe <= 14: base = 0.45
    else: base = 0.2

    ai_months = sum(
        r.get("duration_months", 0) for r in career
        if any(term in normalize(r.get("title", "") + " " + r.get("industry", "") + " " + r.get("description", ""))
               for term in AI_ROLE_TERMS)
    )
    ai_years = ai_months / 12
    bonus = 0.15 if ai_years >= 4 else (0.05 if ai_years >= 2 else -0.1)
    return min(1.0, max(0.0, base + bonus))


def score_behavioral_availability(sigs: dict, today: date) -> float:
    score = 0.0

    last_active = parse_date(sigs.get("last_active_date"))
    di = days_since(last_active, today)
    recency = 1.0 if di <= 7 else (0.85 if di <= 30 else (0.6 if di <= 90 else (0.35 if di <= 180 else 0.1)))
    score += recency * 0.25

    if sigs.get("open_to_work_flag", False):
        score += 0.15
    score += sigs.get("recruiter_response_rate", 0.0) * 0.20
    score += (sigs.get("profile_completeness_score", 0) / 100) * 0.10
    score += (min(sigs.get("applications_submitted_30d", 0), 10) / 10) * 0.08
    gh = sigs.get("github_activity_score", -1)
    if gh >= 0:
        score += (gh / 100) * 0.10
    score += sigs.get("interview_completion_rate", 0.5) * 0.08
    score += (min(sigs.get("saved_by_recruiters_30d", 0), 10) / 10) * 0.04
    return min(1.0, score)


def score_location_logistics(candidate: dict) -> float:
    profile = candidate["profile"]
    sigs = candidate.get("redrob_signals", {})
    loc = normalize(profile.get("location", "") + " " + profile.get("country", ""))

    score = 0.5
    if "pune" in loc or "noida" in loc:
        score += 0.3
    elif any(c in loc for c in ("bangalore", "bengaluru", "mumbai", "hyderabad",
                                 "delhi", "ncr", "gurgaon", "gurugram")):
        score += 0.15
    elif "india" in loc:
        score += 0.05
    else:
        score -= 0.2

    if sigs.get("willing_to_relocate", False):
        score += 0.1
    mode = sigs.get("preferred_work_mode", "")
    if mode in ("hybrid", "flexible"):
        score += 0.1
    elif mode == "onsite":
        score += 0.05

    notice = sigs.get("notice_period_days", 90)
    if notice <= 30:   score += 0.1
    elif notice > 90:  score -= 0.1

    return min(1.0, max(0.0, score))


def score_soft_disqualifiers(all_text_n: str, career: list, skills: list) -> float:
    penalty = 1.0

    research_signals = sum(1 for term in RESEARCH_TERMS if term in all_text_n)
    production_signals = sum(1 for term in PRODUCTION_TERMS if term in all_text_n)
    if research_signals >= 3 and production_signals == 0:
        penalty *= 0.4

    if "langchain" in all_text_n and not any(t in all_text_n for t in RETRIEVAL_TERMS):
        penalty *= 0.6

    if len(career) >= 4:
        short_stints = sum(1 for r in career if r.get("duration_months", 99) < 14)
        if short_stints >= 3:
            penalty *= 0.8

    skill_career_n = (
        " ".join(normalize(s.get("name", "")) for s in skills) + " " +
        " ".join(normalize(r.get("title", "") + " " + r.get("description", "")) for r in career)
    )
    wrong = sum(1 for term in WRONG_DOMAIN_SKILLS if term in skill_career_n)
    right = sum(1 for term in CORE_SKILLS if term in skill_career_n)
    if wrong > right and wrong >= 3:
        penalty *= 0.65

    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORING FUNCTION  (with early-exit fast path)
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "title": 0.25, "skills": 0.25, "career": 0.20,
    "experience": 0.10, "behavioral": 0.15, "location": 0.05,
}

# The maximum score reachable when title = 0.02 (hard-disqualified).
# Even with perfect scores on everything else it's:
# 0.25*0.02 + 0.25*1.0 + 0.20*1.0 + 0.10*1.0 + 0.15*1.0 + 0.05*1.0 = 0.755
# In practice these candidates score far lower; we skip them with a fixed value.
_DISQ_SCORE = 0.005   # well below any real candidate's floor


def score_candidate(candidate: dict, today: date) -> tuple[float, dict]:
    profile = candidate["profile"]
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    sigs = candidate.get("redrob_signals", {})

    # ── Honeypot check ──────────────────────────────────────────────────────
    is_hp, hp_reason = detect_honeypot(candidate, today)
    if is_hp:
        return 0.001, {"honeypot": True, "reason": hp_reason}

    # ── Pre-normalise shared text blobs once ────────────────────────────────
    title_n    = normalize(profile.get("current_title", ""))
    headline_n = normalize(profile.get("headline", ""))
    summary_n  = normalize(profile.get("summary", ""))

    # ── Title score (fast) ──────────────────────────────────────────────────
    title_s = _title_score(title_n, headline_n, summary_n, career)

    # ── EARLY EXIT for hard-disqualified titles ─────────────────────────────
    # Skips the expensive skills/career/disqualifier scoring for ~60% of pool.
    if title_s <= 0.02:
        behavioral_s = score_behavioral_availability(sigs, today)
        location_s   = score_location_logistics(candidate)
        raw = (WEIGHTS["title"] * title_s +
               WEIGHTS["behavioral"] * behavioral_s +
               WEIGHTS["location"] * location_s)
        return raw, {
            "title": round(title_s, 3), "skills": 0.0, "career": 0.0,
            "experience": 0.0, "behavioral": round(behavioral_s, 3),
            "location": round(location_s, 3), "penalty": 1.0,
            "raw": round(raw, 3), "final": round(raw, 3),
        }

    # ── Full scoring path ───────────────────────────────────────────────────
    career_desc_n = normalize(
        " ".join(r.get("description", "") + " " + r.get("title", "") + " " + r.get("company", "")
                 for r in career)
    )
    all_text_n = summary_n + " " + headline_n + " " + career_desc_n

    skills_s     = score_skills(candidate, career_desc_n, all_text_n)
    career_s     = score_career_quality(candidate)
    exp_s        = score_experience_years(candidate, career)
    behavioral_s = score_behavioral_availability(sigs, today)
    location_s   = score_location_logistics(candidate)

    raw = (WEIGHTS["title"]      * title_s +
           WEIGHTS["skills"]     * skills_s +
           WEIGHTS["career"]     * career_s +
           WEIGHTS["experience"] * exp_s +
           WEIGHTS["behavioral"] * behavioral_s +
           WEIGHTS["location"]   * location_s)

    penalty = score_soft_disqualifiers(all_text_n, career, skills)
    final   = raw * penalty

    components = {
        "title": round(title_s, 3), "skills": round(skills_s, 3),
        "career": round(career_s, 3), "experience": round(exp_s, 3),
        "behavioral": round(behavioral_s, 3), "location": round(location_s, 3),
        "penalty": round(penalty, 3), "raw": round(raw, 3), "final": round(final, 3),
    }
    return final, components


# ─────────────────────────────────────────────────────────────────────────────
# REASONING GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, components: dict, rank: int) -> str:
    if components.get("honeypot"):
        return f"Flagged as honeypot: {components.get('reason', 'impossible profile data')}."

    profile  = candidate["profile"]
    sigs     = candidate.get("redrob_signals", {})
    skills   = candidate.get("skills", [])
    career   = candidate.get("career_history", [])

    yoe           = profile.get("years_of_experience", 0)
    title         = profile.get("current_title", "unknown title")
    company       = profile.get("current_company", "unknown company")
    notice        = sigs.get("notice_period_days", 90)
    response_rate = sigs.get("recruiter_response_rate", 0)
    open_to_work  = sigs.get("open_to_work_flag", False)
    github        = sigs.get("github_activity_score", -1)

    core_skill_matches = [
        s["name"] for s in skills
        if any(cs in normalize(s.get("name", "")) or normalize(s.get("name", "")) in cs
               for cs in CORE_SKILLS)
        and s.get("proficiency") in ("advanced", "expert")
    ][:4]

    current_role     = next((r for r in career if r.get("is_current")), None)
    current_industry = current_role.get("industry", "") if current_role else ""

    strengths, concerns = [], []

    if components["title"] >= 0.8:
        strengths.append(f"{title} role directly matches JD requirements")
    elif components["title"] <= 0.1:
        concerns.append(f"title ({title}) is outside engineering/ML")

    if core_skill_matches:
        strengths.append(f"advanced/expert in {', '.join(core_skill_matches[:3])}")
    if components["skills"] >= 0.7:
        strengths.append("strong core retrieval/ranking skill coverage")
    elif components["skills"] <= 0.2:
        concerns.append("limited must-have embedding/vector DB skills")

    if 5 <= yoe <= 9:
        strengths.append(f"{yoe:.1f}yr in JD's target range")
    elif yoe < 5:
        concerns.append(f"only {yoe:.1f}yr — below JD minimum")

    if components.get("career", 0) >= 0.7 and current_industry:
        strengths.append(f"product-company background in {current_industry}")
    elif components.get("career", 0) <= 0.3:
        concerns.append("heavy consulting/services background")

    if open_to_work and response_rate >= 0.7:
        strengths.append(f"actively engaged ({response_rate:.0%} response rate)")
    elif response_rate <= 0.15:
        concerns.append(f"low response rate ({response_rate:.0%})")

    if notice <= 30:
        strengths.append(f"short notice ({notice}d)")
    elif notice >= 90:
        concerns.append(f"long notice ({notice}d)")

    if github >= 40:
        strengths.append(f"strong GitHub activity (score {github:.0f})")

    if rank <= 10:
        s_text = "; ".join(strengths[:3]) if strengths else "strong overall fit"
        c_text = f" Minor concerns: {'; '.join(concerns[:1])}." if concerns else "."
        return f"{yoe:.1f}yr {title} at {company} — {s_text}{c_text}"
    elif rank <= 50:
        s_text = "; ".join(strengths[:2]) if strengths else "moderate fit"
        c_text = f" Concerns: {'; '.join(concerns[:2])}." if concerns else "."
        return f"{yoe:.1f}yr {title}: {s_text}.{c_text}"
    else:
        c_text = "; ".join(concerns[:2]) if concerns else "marginal fit"
        s_text = f" Some strengths: {'; '.join(strengths[:1])}." if strengths else ""
        return f"{yoe:.1f}yr {title} — below cutoff: {c_text}.{s_text}"


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: str):
    import gzip
    p = Path(path)
    open_fn = (lambda: gzip.open(path, "rt", encoding="utf-8")) if p.suffix == ".gz" \
              else (lambda: open(path, "r", encoding="utf-8"))
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

    scored.sort(key=lambda x: (-round(x[0], 4), x[1]["candidate_id"]))
    top = scored[:top_n]

    print(f"Writing top {top_n} to {output_path}...", flush=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cand, components) in enumerate(top, 1):
            writer.writerow([cand["candidate_id"], rank, round(score, 4),
                             generate_reasoning(cand, components, rank)])

    print("Done!")
    print("\n=== TOP 10 CANDIDATES ===")
    for rank, (score, cand, components) in enumerate(top[:10], 1):
        p = cand["profile"]
        print(f"#{rank} {cand['candidate_id']} — {p['current_title']} | "
              f"YOE: {p['years_of_experience']} | Score: {score:.4f}")
        print(f"     {generate_reasoning(cand, components, rank)}\n")


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out, args.top)


if __name__ == "__main__":
    main()
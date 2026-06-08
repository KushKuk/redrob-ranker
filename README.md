# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

## Reproduce Command

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runtime: ~3 minutes on CPU (tested on 8-core machine, 16 GB RAM). No GPU, no network calls, no external dependencies beyond Python 3.8+ stdlib.

---

## Architecture Overview

A **multi-signal, rule-driven scoring pipeline** with explicit JD knowledge encoding. No external ML models or API calls. Each candidate is scored against six components, combined via a weighted sum, then modulated by a soft-disqualifier penalty multiplier.

```
candidates.jsonl
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                   HONEYPOT DETECTION                        │
│  (flags impossible profiles → score ≈ 0, never in top 100) │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                  COMPONENT SCORING (6 signals)              │
│                                                             │
│  [0.25] Title Fit      — is this person an ML/AI engineer?  │
│  [0.25] Skills Match   — embeddings, vector DBs, Python     │
│  [0.20] Career Quality — product co. history, not services  │
│  [0.10] Experience Yrs — JD sweet spot 5-9 yrs              │
│  [0.15] Behavioral     — recency, response rate, GitHub     │
│  [0.05] Location/Ops   — Pune/Noida, notice period          │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│           SOFT DISQUALIFIER MULTIPLIER (0–1)                │
│  Penalizes: pure researchers, LangChain-only, consulting-   │
│  only, CV/speech domain, title-chaser career patterns       │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
   final_score = weighted_sum × penalty_multiplier
      │
      ▼
   Sort by score desc, candidate_id asc (tie-break)
   → Top 100 with per-candidate reasoning
```

---

## Methodology Deep Dive

### Why rule-based instead of embedding-based?

The compute constraint (5 min CPU, no GPU, no network) rules out running sentence-transformer models over 100K candidates at scoring time. A lightweight embedding model (e.g. `all-MiniLM-L6-v2`) would take ~20 minutes on CPU for 100K candidates.

More importantly, this JD has **explicit structured requirements** that map well to rule-based features:
- "Must have production experience with vector databases" → check skills list for Pinecone/Weaviate/FAISS with `duration_months > 0`
- "Pure consultants are disqualified" → check company names against known consulting firms
- "Title-chasers are a red flag" → count stints < 14 months

Rule-based systems are also easier to audit, explain, and defend in a Stage 5 interview.

### 1. Title/Role Fit (weight: 0.25)

The **highest-leverage single signal** to avoid keyword stuffers. The JD explicitly warns: "A candidate who has all the AI keywords listed as skills but whose title is 'Marketing Manager' is not a fit."

Logic:
- Hard disqualify (score 0.02): Marketing, HR, Operations, Sales, Content Writer, etc.
- Strong positive (1.0): Machine Learning Engineer, AI Engineer, NLP Engineer, Data Scientist, Applied ML
- Moderate positive (0.55–0.85): General "Engineer/Developer/Scientist" titles — gated by whether their headline/summary contains core ML skills
- Unknown title: fall back to career history scan

### 2. Core Skills Match (weight: 0.25)

Built a vocabulary of ~80 core skills derived directly from JD requirements: embedding models (BGE, E5, sentence-transformers), vector DBs (Pinecone, Weaviate, FAISS, Qdrant, Milvus), Python, evaluation frameworks (NDCG, MRR, MAP), LLM fine-tuning (LoRA, QLoRA, PEFT), and ranking/retrieval systems.

**Anti-keyword-stuffing measures:**
- Each skill is weighted by `proficiency_level × duration_trust × endorsement_trust`
- `duration_trust = min(1.0, duration_months / 24)` — a skill listed as "expert" with 0 months is worth very little
- `endorsement_trust = min(1.0, 0.5 + endorsements / 20)` — skills without external validation are discounted
- Contextual matches in career history and summary also contribute (but at 0.03 per match, capped at 0.3)
- Platform assessment scores add a small bonus (0.1 per relevant skill assessment)

### 3. Career Quality (weight: 0.20)

Evaluates the trajectory of each role in career history, weighted by duration:

Per-role score:
- +0.3 if in a tech/product industry (AI, SaaS, fintech, e-commerce, etc.)
- +0.3 if title contains ML/AI/NLP/ranking/search signals
- +0.2 if role description mentions core skills in context
- -0.4 if company name matches a known consulting firm (TCS, Infosys, Wipro, Accenture, etc.)

Global penalties:
- >90% career in consulting → final score × 0.3
- >70% career in consulting → × 0.6
- High proportion of CV/speech/robotics domain work → further penalty

### 4. Experience Years (weight: 0.10)

JD sweet spot is 5-9 years. Score:

| YOE Range | Score |
|-----------|-------|
| 6–8 years | 1.0 |
| 5–6 or 8–9 | 0.85 |
| 4–5 or 9–11 | 0.65 |
| 3–4 or 11–14 | 0.45 |
| <3 or >14 | 0.20 |

A bonus/penalty of ±0.05–0.15 adjusts based on how many of those years are specifically in AI/ML roles (calculated from career history).

### 5. Behavioral Availability (weight: 0.15)

The JD and signals doc both emphasize: "A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available."

Sub-components:
- **Recency of activity** (0.25 weight within behavioral): 1.0 if active in last 7 days, decays to 0.1 for >180 days inactive
- **Open-to-work flag** (0.15): direct binary signal
- **Recruiter response rate** (0.20): most predictive of actual reachability
- **Profile completeness** (0.10): indicates investment in job search
- **Applications submitted 30d** (0.08): active job-seeking signal
- **GitHub activity score** (0.10): engineering credibility
- **Interview completion rate** (0.08): reliability signal
- **Saved by recruiters 30d** (0.04): external recruiter validation

### 6. Location/Logistics (weight: 0.05)

- Pune/Noida: +0.3 (preferred JD locations)
- Bangalore, Mumbai, Hyderabad, Delhi NCR: +0.15 (explicitly listed in JD)
- Other India: +0.05
- Outside India: -0.2 (case-by-case, no visa sponsorship)
- Relocation willingness: +0.1
- Hybrid/flexible work mode: +0.1 (JD is hybrid)
- Notice period ≤30 days: +0.1 (JD prefers sub-30, can buy out up to 30)
- Notice period >90 days: -0.1

### Soft Disqualifier Multiplier

Applied as a final multiplier (0.0–1.0) to the weighted sum:

| Condition | Multiplier |
|-----------|------------|
| Pure research profile (3+ research signals, 0 production signals) | × 0.4 |
| LangChain-only AI experience (no vector/retrieval/ranking foundation) | × 0.6 |
| Title-chaser pattern (3+ stints < 14 months) | × 0.8 |
| Wrong domain primary (CV/speech > NLP/IR signals) | × 0.65 |

### Honeypot Detection

Checks run before any scoring:
1. ≥3 expert-proficiency skills with 0 months duration → flagged
2. Stated YOE < 50% of career history months → flagged
3. Any future start dates in career history → flagged
4. End date before start date → flagged
5. Last active date before signup date → flagged
6. ≥12 expert skills with 0 endorsements → flagged
7. Perfect engagement scores on <1 month old account → flagged

Honeypot candidates receive score ≈ 0.001 and naturally fall out of the top 100.

---

## Score Distribution

- Top score: ~0.98 (Senior ML Engineer, 7.2 yrs, Zomato, Pune-based, all signals strong)
- Score at rank 50: ~0.94
- Score at rank 100: ~0.92
- The tight range reflects a clean separation: top candidates are genuine AI engineers with production backgrounds and active engagement; the long tail starts well below 0.5 (keyword stuffers, wrong roles, inactive candidates)

---

## Key Design Decisions

**Why not use sentence-transformers?** Compute constraints. For a production system, we'd precompute candidate embeddings offline (allowed by the spec under "pre-computation"), then do fast dot-product similarity at ranking time. This would be the v2 architecture.

**Why is Title the highest-weight component?** Empirically, a strong title + skills combination is the most reliable indicator of fit, and title is the hardest thing to fake (it comes from career history context, not self-reported). This catches the "Marketing Manager with all AI keywords" trap.

**Why down-weight consulting so heavily?** The JD is explicit: "People who have only worked at consulting firms in their entire career." This isn't bias against individuals — it's respecting the JD's stated requirements for product-company production experience.

**Why include behavioral signals at 15%?** The JD explicitly instructs participants to use them: "A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is... not actually available. Down-weight them appropriately."

---

## File Structure

```
.
├── rank.py                          # Main ranker — run this
├── requirements.txt                 # No external dependencies
├── README.md                        # This file
├── submission.csv                   # Output: top 100 ranked candidates
└── submission_metadata.yaml         # Submission metadata
```

---

## Running the Ranker

**Requirements:** Python 3.8+, no external packages needed.

```bash
# Full 100K candidate pool
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Gzipped input is also supported
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

# Validate output format
python validate_submission.py ./submission.csv
```

**Expected runtime:** ~3 minutes on a standard laptop (8-core CPU, 16 GB RAM).

---

## Submission Metadata

See `submission_metadata.yaml` for team info and declarations.

import re
import datetime
import json

# =============================================================================
# SCORING DESIGN — READ THIS BEFORE CHANGING ANY NUMBERS
# =============================================================================
#
# All intermediate scores are on a 0–100 scale so that weighted combinations
# are mathematically meaningful.
#
# FINAL SCORE  = 0.70 × skill_score  +  0.30 × exp_score
#
# skill_score  (computed in p.py, passed in here)
#   = (matched + 0.4 × partial) / total_jd_skills × 100
#   Partial skills are penalised at 40 % — they show awareness but not depth.
#
# exp_score  (computed here, 0–100)
#   = weighted combination of:
#       • overall experience ratio   (40 % weight when skill-specific data exists)
#       • per-skill experience match (60 % weight when skill-specific data exists)
#     Falls back to overall-only when the JD has no skill-specific year requirements.
#
# Overqualification:
#   Candidates with MORE experience than required are not penalised — they get
#   100 on experience. Slight overqualification is a feature, not a bug.
#
# Experience score bands (0–100):
#   ratio ≥ 1.0  → 100   (meets or exceeds requirement)
#   ratio ≥ 0.85 → 85    (close — e.g. 4 yrs for 5 yrs required)
#   ratio ≥ 0.70 → 70    (moderate gap — e.g. 3.5 yrs for 5 yrs required)
#   ratio ≥ 0.55 → 55    (noticeable gap)
#   ratio ≥ 0.40 → 40    (significant gap)
#   below 0.40   → 20    (large gap — still non-zero to avoid harsh cliff)
#
# =============================================================================


# -----------------------------
# 1. SKILL DICTIONARY
# (used only by the rule-based extract_skills fallback)
# -----------------------------

technical_skills = {
    # ── Languages ──────────────────────────────────────────────────────────
    "python":       ("advanced",     "technical"),
    "javascript":   ("advanced",     "technical"),
    "typescript":   ("advanced",     "technical"),
    "java":         ("advanced",     "technical"),
    "c++":          ("advanced",     "technical"),
    "c#":           ("advanced",     "technical"),
    "go":           ("advanced",     "technical"),
    "rust":         ("advanced",     "technical"),
    "kotlin":       ("advanced",     "technical"),
    "swift":        ("advanced",     "technical"),
    # ── Frontend ───────────────────────────────────────────────────────────
    "react":        ("intermediate", "technical"),
    "vue":          ("intermediate", "technical"),
    "angular":      ("intermediate", "technical"),
    "next.js":      ("intermediate", "technical"),
    # ── Backend ────────────────────────────────────────────────────────────
    "django":       ("intermediate", "technical"),
    "fastapi":      ("intermediate", "technical"),
    "flask":        ("intermediate", "technical"),
    "spring":       ("intermediate", "technical"),
    "express":      ("intermediate", "technical"),
    # ── Databases ──────────────────────────────────────────────────────────
    "sql":          ("advanced",     "technical"),
    "postgresql":   ("advanced",     "technical"),
    "mysql":        ("advanced",     "technical"),
    "mongodb":      ("intermediate", "technical"),
    "redis":        ("intermediate", "technical"),
    # ── Cloud / Infra ──────────────────────────────────────────────────────
    "aws":          ("intermediate", "technical"),
    "gcp":          ("intermediate", "technical"),
    "azure":        ("intermediate", "technical"),
    "docker":       ("intermediate", "technical"),
    "kubernetes":   ("intermediate", "technical"),
    # ── APIs / Architecture ────────────────────────────────────────────────
    "rest api":     ("advanced",     "technical"),
    "graphql":      ("intermediate", "technical"),
    "microservices":("intermediate", "technical"),
    "system design":("intermediate", "technical"),
    # ── Testing ────────────────────────────────────────────────────────────
    "pytest":       ("intermediate", "technical"),
    "jest":         ("intermediate", "technical"),
    # ── DevOps ─────────────────────────────────────────────────────────────
    "git":          ("advanced",     "technical"),
    "ci/cd":        ("intermediate", "technical"),
    "linux":        ("advanced",     "technical"),
    "bash":         ("advanced",     "technical"),
}

ALL_SKILLS = set(technical_skills.keys())


# -----------------------------
# 2. SKILL ALIASES
# (minimal — the full alias table lives in p.py)
# -----------------------------

SKILL_ALIASES = {
    "js":       "javascript",
    "ts":       "typescript",
    "reactjs":  "react",
    "nodejs":   "node.js",
    "postgres": "postgresql",
}


def normalize_skill(skill: str) -> str:
    skill = skill.lower().strip()
    return SKILL_ALIASES.get(skill, skill)


# -----------------------------
# 3. EXTRACT SKILLS  (rule-based fallback only)
# -----------------------------

def extract_skills(text: str) -> set:
    """
    Lightweight regex-based skill extractor.
    Used ONLY as a fallback when the LLM extraction in p.py fails.
    For primary extraction, p.py uses the LLM-powered _extract_skills_llm().
    """
    text = text.lower()
    found = set()
    for skill in ALL_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'(s|\.js|js)?\b'
        if re.search(pattern, text):
            found.add(skill)
    return found


# -----------------------------
# 4. GROUP SKILLS
# -----------------------------

def group_skills(skills: set) -> dict:
    grouped = {"advanced": set(), "intermediate": set(), "beginner": set()}
    for skill in skills:
        if skill in technical_skills:
            level = technical_skills[skill][0]
            grouped[level].add(skill)
    return grouped


# =============================================================================
# 5. COMPUTE MATCH SCORE V2
# (kept for backward compatibility — NOT called by p.py's main flow)
# p.py performs its own three-layer classification and scoring.
# =============================================================================

async def compute_match_score_v2(resume_skills: set, jd_skills: set, llm=None) -> tuple:
    """
    Compute skill match score using LLM semantic matching.
    Falls back to normalised set intersection if LLM is unavailable or fails.

    NOTE: This function is retained for backward compatibility but is NOT used
    by p.py's main /process/jd_resume_match route. p.py uses its own
    three-layer classify_skills() pipeline which is more accurate.

    Returns:
        (score: float 0–10, matched: list, missing: list)
    """
    if not jd_skills:
        return 0.0, [], []

    if llm is not None:
        try:
            resume_list = sorted(set(s.lower().strip() for s in resume_skills if s))
            jd_list     = sorted(set(s.lower().strip() for s in jd_skills if s))

            prompt = f"""You are a skill-matching classifier. Your only job is to decide, for each JD skill, whether the candidate's resume skills cover it.

### Semantic equivalence rules
- Acronyms equal full names: "ml" = "machine learning", "aws" = "amazon web services", "k8s" = "kubernetes", "nlp" = "natural language processing"
- Aliases are equal: "react" = "reactjs" = "react.js", "node" = "node.js", "postgres" = "postgresql", "tf" = "tensorflow"
- A broader skill covers a narrower one: "python" covers "python scripting"; "llm fine-tuning" covers "model fine-tuning"
- Minor spelling/casing variants are equal: "rest api" = "restful apis", "css3" = "css"
- A general cloud platform covers its sub-services when no other match exists: "aws" covers "ec2", "s3", "lambda"

### Few-shot examples

Example 1
Resume Skills: ["python", "aws", "machine learning", "sql"]
JD Skills: ["ml", "amazon web services", "python scripting", "nosql"]
Output:
{{"matched": ["ml", "amazon web services", "python scripting"], "missing": ["nosql"]}}

Example 2
Resume Skills: ["react", "typescript", "node.js", "docker"]
JD Skills: ["reactjs", "ts", "express", "kubernetes"]
Output:
{{"matched": ["reactjs", "ts"], "missing": ["express", "kubernetes"]}}

Example 3
Resume Skills: ["java", "spring boot", "postgresql", "rest api"]
JD Skills: ["java", "spring", "mysql", "restful apis", "graphql"]
Output:
{{"matched": ["java", "spring", "restful apis"], "missing": ["mysql", "graphql"]}}

### Hard rules
- Every JD skill must appear in exactly one list — no omissions, no duplicates.
- Use the exact JD skill wording in both lists — never the resume wording.
- Do not add skills not in the JD list.
- Return ONLY the JSON object — no explanation, no markdown, no preamble.

### Output format
{{"matched": ["<jd_skill>", ...], "missing": ["<jd_skill>", ...]}}

### Input
Resume Skills: {resume_list}
JD Skills: {jd_list}"""

            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content="You are a precise skill-matching classifier. Return only valid JSON."),
                HumanMessage(content=prompt),
            ]
            deterministic_llm = llm.bind(temperature=0)
            raw     = await deterministic_llm.ainvoke(messages)
            content = raw.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$",           "", content)

            parsed  = json.loads(content)
            matched = [s for s in parsed.get("matched", []) if s in jd_skills]
            missing = [s for s in parsed.get("missing", []) if s in jd_skills]

            accounted = set(matched) | set(missing)
            for skill in jd_skills:
                if skill not in accounted:
                    missing.append(skill)

            # Returns 0–10 for backward compat (this function is not used in main flow)
            score = (len(matched) / len(jd_skills)) * 10 if jd_skills else 0.0
            print(f"[LLM match] matched={len(matched)}, missing={len(missing)}, score={round(score, 2)}")
            return round(score, 2), matched, missing

        except Exception as e:
            print(f"[WARN] LLM skill matching failed ({e}), falling back to set intersection")

    matched = list(resume_skills & jd_skills)
    missing = list(jd_skills - resume_skills)
    score   = (len(matched) / len(jd_skills)) * 10 if jd_skills else 0.0
    return round(score, 2), matched, missing


# =============================================================================
# 6. EXPERIENCE EXTRACTION
# =============================================================================

def extract_experience(text: str) -> int:
    """
    Extract total years of experience from text.
    Priority order:
      1. Explicit date ranges  (e.g. "2019 – present")
      2. Direct year mentions  (e.g. "5+ years of experience")
      3. Heuristic signals     (fallback only)
    """
    text_lower = text.lower()
    current_year = datetime.datetime.now().year

    # ── 1. Date ranges ────────────────────────────────────────────────────
    date_matches = re.findall(
        r'(20\d{2})\s*[-–]\s*(present|current|(20\d{2}))',
        text_lower,
    )
    durations = []
    for match in date_matches:
        start = int(match[0])
        end   = current_year if match[1] in ("present", "current") else int(match[2])
        if end >= start:
            durations.append(end - start)

    if durations:
        return max(durations)

    # ── 2. Direct year mentions ───────────────────────────────────────────
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*\+?\s*(years|yrs)', text_lower)
    if matches:
        return int(float(max(m[0] for m in matches)))

    # ── 3. Heuristic fallback ─────────────────────────────────────────────
    exp_score = 0

    if any(t in text_lower for t in ["principal", "staff engineer", "director",
                                      "vp ", "vice president", "partner", "head of"]):
        exp_score += 8
    elif any(t in text_lower for t in ["senior", "lead", "architect", "manager",
                                        "consultant", "specialist"]):
        exp_score += 5
    elif any(t in text_lower for t in ["engineer", "developer", "analyst",
                                        "associate", "executive"]):
        exp_score += 3

    project_count = text_lower.count("project name")
    exp_score += min(project_count * 1.5, 5)

    bullet_points = text_lower.count("·")
    if bullet_points > 20:
        exp_score += 2

    return int(min(exp_score, 15))


# =============================================================================
# 7. EXPERIENCE SCORE  (0–100, aligned with skill_score scale)
# =============================================================================

def _ratio_to_score(ratio: float) -> int:
    """
    Convert a resume/JD experience ratio to a 0–100 score.

    Bands are deliberately non-linear:
    - Small gaps (≥85 %) are treated nearly as full matches.
    - Large gaps (<40 %) still get a floor of 20 so a single weak area
      doesn't catastrophically tank an otherwise strong candidate.
    - Overqualification (ratio > 1) is capped at 100, never penalised.
    """
    if ratio >= 1.0:
        return 100
    elif ratio >= 0.85:
        return 85
    elif ratio >= 0.70:
        return 70
    elif ratio >= 0.55:
        return 55
    elif ratio >= 0.40:
        return 40
    else:
        return 20


def compute_experience_score(resume_exp: int, jd_exp: int) -> int:
    """
    Simple overall experience score (0–100).
    Called by p.py's main route.

    Returns an int on the 0–100 scale so it is directly compatible
    with skill_score (also 0–100) inside compute_final_score().
    """
    if jd_exp == 0:
        return 70   # JD has no stated requirement → give benefit of the doubt

    ratio = resume_exp / jd_exp
    return _ratio_to_score(ratio)


# =============================================================================
# 8. SKILL-SPECIFIC EXPERIENCE EXTRACTION
# =============================================================================

_SKILL_EXP_PATTERNS = [
    # "5+ years of experience with Python"  /  "5 years of Python experience"
    r'(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?(?:experience\s+)?(?:with|in|using|of)\s+'
    r'([a-z][a-z0-9 \.\+\#\-]{1,35}?)(?:\s+experience)?(?=\s*[,\.\;\n\(\)]|$)',
    # "Python experience of 5+ years"  /  "Python: 5 years"
    r'([a-z][a-z0-9 \.\+\#\-]{1,35}?)\s*(?:experience|expertise|background|proficiency)'
    r'\s+(?:of\s+)?(\d+(?:\.\d+)?)\+?\s*years?',
    # "5+ years Python"  (no connector word)
    r'(\d+(?:\.\d+)?)\+?\s*years?\s+([a-z][a-z0-9 \.\+\#\-]{2,30}?)(?=\s*[,\.\;\n\(\)]|$)',
]

_EXP_NOISE = {
    "experience", "in", "of", "with", "using", "working", "total", "relevant",
    "industry", "professional", "development", "software", "programming", "engineering",
}


def _clean_skill_token(token: str) -> str:
    token = token.strip().rstrip(".,;:()")
    token = re.sub(r'\s+', ' ', token)
    return token.lower()


def extract_skill_specific_experience(text: str) -> dict:
    """
    Extract skill-specific experience durations from text.
    Returns {skill: years}  e.g. {"python": 5, "react": 3, "aws": 2}.
    Only keeps entries where the skill token looks like a real skill name.
    """
    text_lower = text.lower()
    result: dict = {}

    for pattern in _SKILL_EXP_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            g1, g2 = match.group(1).strip(), match.group(2).strip()
            try:
                years = float(g1)
                skill = _clean_skill_token(g2)
            except ValueError:
                try:
                    years = float(g2)
                    skill = _clean_skill_token(g1)
                except ValueError:
                    continue

            if skill in _EXP_NOISE or len(skill) < 2:
                continue

            if skill not in result or years > result[skill]:
                result[skill] = int(years)

    return result


def _score_skill_years(resume_years: int, jd_years: int) -> int:
    """Per-skill year comparison → 0–100 score."""
    if jd_years == 0:
        return 100
    return _ratio_to_score(resume_years / jd_years)


# =============================================================================
# 9. EXPERIENCE SCORE V2  (full LLM-powered version, 0–100)
# =============================================================================

async def compute_experience_score_v2(
    resume_text: str,
    jd_text: str,
    llm=None,
) -> tuple:
    """
    Compute an experience score (0–100) by comparing:
      1. Overall years of experience (resume vs JD requirement).
      2. Skill-specific year requirements stated in the JD.

    Weighting when skill-specific data is available:
      final_exp_score = 0.40 × overall_score + 0.60 × avg(per_skill_scores)

    When no skill-specific requirements are found in the JD:
      final_exp_score = overall_score

    Uses LLM for skill-specific extraction when available;
    falls back to regex (_SKILL_EXP_PATTERNS).

    Returns:
        (score: int 0–100, breakdown: dict)

    breakdown keys:
        overall_resume_exp    – years extracted from resume
        overall_jd_exp        – years required by JD
        overall_score         – 0-100 based on overall years only
        skill_requirements    – list of per-skill dicts
        skill_specific_score  – avg of per-skill scores (None if not available)
        final_score           – weighted final (same as returned score)
    """
    # ── Step 1: Overall experience ────────────────────────────────────────
    resume_exp    = extract_experience(resume_text)
    jd_exp        = extract_experience(jd_text)
    overall_score = compute_experience_score(resume_exp, jd_exp)

    # ── Step 2: Skill-specific extraction ────────────────────────────────
    jd_skill_exp: dict     = {}
    resume_skill_exp: dict = {}

    if llm is not None:
        try:
            prompt = f"""Extract skill-specific experience durations from the two texts below.

Rules:
- Only extract entries where a number of years is explicitly tied to a specific skill/technology.
- Normalize skill names to lowercase canonical forms (e.g. "python", "aws", "react").
- If the same skill appears more than once, keep the highest value.
- If no skill-specific experience is found, return an empty object {{}}.
- Return ONLY valid JSON — no explanation, no markdown.

Output format:
{{
  "jd_skill_exp": {{"<skill>": <years>, ...}},
  "resume_skill_exp": {{"<skill>": <years>, ...}}
}}

--- JD ---
{jd_text}

--- RESUME ---
{resume_text}"""

            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content="You are a precise information-extraction engine. Return only valid JSON."),
                HumanMessage(content=prompt),
            ]
            raw     = await llm.bind(temperature=0).ainvoke(messages)
            content = raw.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$",           "", content)
            parsed           = json.loads(content)
            jd_skill_exp     = {k.lower().strip(): int(v) for k, v in parsed.get("jd_skill_exp",     {}).items()}
            resume_skill_exp = {k.lower().strip(): int(v) for k, v in parsed.get("resume_skill_exp", {}).items()}
            print(f"[EXP LLM] jd_skill_exp={jd_skill_exp}  resume_skill_exp={resume_skill_exp}")
        except Exception as e:
            print(f"[WARN] LLM skill-exp extraction failed ({e}), falling back to regex")

    if not jd_skill_exp:
        jd_skill_exp     = extract_skill_specific_experience(jd_text)
        resume_skill_exp = extract_skill_specific_experience(resume_text)

    # ── Step 3: Per-skill scoring ─────────────────────────────────────────
    skill_requirements: list[dict] = []
    skill_scores:       list[int]  = []

    for skill, jd_years in jd_skill_exp.items():
        resume_years = resume_skill_exp.get(skill)

        if resume_years is not None:
            s = _score_skill_years(resume_years, jd_years)
        else:
            # No explicit years found for this skill in the resume.
            # Use overall experience as a proxy but cap at 70 —
            # we have no direct evidence so we shouldn't give full credit.
            s = min(_ratio_to_score(resume_exp / jd_years if jd_years else 1), 70)

        skill_requirements.append({
            "skill":        skill,
            "jd_years":     jd_years,
            "resume_years": resume_years,
            "score":        s,
        })
        skill_scores.append(s)

    # ── Step 4: Weighted final experience score ───────────────────────────
    if skill_scores:
        skill_specific_score = round(sum(skill_scores) / len(skill_scores), 1)
        final_score          = round(0.40 * overall_score + 0.60 * skill_specific_score)
    else:
        skill_specific_score = None
        final_score          = overall_score

    # Clamp to 0–100
    final_score = max(0, min(100, final_score))

    print(f"[EXP v2] overall={overall_score}  skill_specific={skill_specific_score}  final={final_score}")

    breakdown = {
        "overall_resume_exp":  resume_exp,
        "overall_jd_exp":      jd_exp,
        "overall_score":       overall_score,
        "skill_requirements":  skill_requirements,
        "skill_specific_score": skill_specific_score,
        "final_score":         final_score,
    }
    return final_score, breakdown


# =============================================================================
# 10. FINAL COMBINED SCORE  (0–100)
# =============================================================================

def compute_final_score(skill_score: float, exp_score: float) -> float:
    """
    Combine skill coverage score and experience score into a single
    compatibility score.

    BOTH inputs must be on the 0–100 scale:
      skill_score : computed in p.py  → (matched + 0.4×partial) / total × 100
      exp_score   : computed here     → compute_experience_score() returns 0–100

    Weights: Skills 70 %, Experience 30 %.
    Skills dominate because they are the most direct signal of fit.
    Experience adjusts for seniority / depth but should not override strong skills.

    Examples:
      Perfect skills + perfect exp  → (0.7×100) + (0.3×100) = 100.0
      Perfect skills + weak exp     → (0.7×100) + (0.3×20)  =  76.0
      Weak skills   + perfect exp   → (0.7×20)  + (0.3×100) =  44.0
      Average both                  → (0.7×65)  + (0.3×70)  =  66.5
    """
    score = round((0.70 * skill_score) + (0.30 * exp_score), 1)
    return max(0.0, min(100.0, score))   # clamp to [0, 100]
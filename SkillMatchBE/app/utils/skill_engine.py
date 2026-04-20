import re
import datetime
import json

# -----------------------------
# 1. YOUR SKILL DICTIONARY
# -----------------------------

technical_skills = {
    "python": ("advanced", "technical"),
    "javascript": ("advanced", "technical"),
    "typescript": ("advanced", "technical"),
    "java": ("advanced", "technical"),
    "c++": ("advanced", "technical"),
    "c#": ("advanced", "technical"),
    "go": ("advanced", "technical"),
    "rust": ("advanced", "technical"),
    "kotlin": ("advanced", "technical"),
    "swift": ("advanced", "technical"),

    "react": ("intermediate", "technical"),
    "vue": ("intermediate", "technical"),
    "angular": ("intermediate", "technical"),
    "next.js": ("intermediate", "technical"),

    "django": ("intermediate", "technical"),
    "fastapi": ("intermediate", "technical"),
    "flask": ("intermediate", "technical"),
    "spring": ("intermediate", "technical"),
    "express": ("intermediate", "technical"),

    "sql": ("advanced", "technical"),
    "postgresql": ("advanced", "technical"),
    "mysql": ("advanced", "technical"),
    "mongodb": ("intermediate", "technical"),
    "redis": ("intermediate", "technical"),

    "aws": ("intermediate", "technical"),
    "gcp": ("intermediate", "technical"),
    "azure": ("intermediate", "technical"),
    "docker": ("intermediate", "technical"),
    "kubernetes": ("intermediate", "technical"),

    "rest api": ("advanced", "technical"),
    "graphql": ("intermediate", "technical"),

    "microservices": ("intermediate", "technical"),
    "system design": ("intermediate", "technical"),

    "pytest": ("intermediate", "technical"),
    "jest": ("intermediate", "technical"),

    "git": ("advanced", "technical"),
    "ci/cd": ("intermediate", "technical"),

    "linux": ("advanced", "technical"),
    "bash": ("advanced", "technical"),
}

ALL_SKILLS = set(technical_skills.keys())

# -----------------------------
# 2. SKILL ALIASES
# -----------------------------

SKILL_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "reactjs": "react",
    "nodejs": "node.js",
    "postgres": "postgresql",
}

def normalize_skill(skill):
    skill = skill.lower().strip()
    return SKILL_ALIASES.get(skill, skill)

# -----------------------------
# 3. EXTRACT SKILLS
# -----------------------------

def extract_skills(text):
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

def group_skills(skills):
    grouped = {
        "advanced": set(),
        "intermediate": set(),
        "beginner": set()
    }

    for skill in skills:
        if skill in technical_skills:
            level = technical_skills[skill][0]
            grouped[level].add(skill)

    return grouped

# -----------------------------
# 5. MATCH SCORE
# -----------------------------

async def compute_match_score_v2(resume_skills: set, jd_skills: set, llm=None) -> tuple:
    """
    Compute skill match score using LLM semantic matching.
    Falls back to normalized set intersection if LLM is unavailable or fails.

    Returns:
        (score: float, matched: list, missing: list)
    """
    if not jd_skills:
        return 0.0, [], []

    if llm is not None:
        try:
            # Canonical, stable inputs
            resume_list = sorted(set(s.lower().strip() for s in resume_skills if s))
            jd_list = sorted(set(s.lower().strip() for s in jd_skills if s))

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

                Think

                ### Output format
                {{"matched": ["<jd_skill>", ...], "missing": ["<jd_skill>", ...]}}

                ### Input
                Resume Skills: {resume_list}
                JD Skills: {jd_list}"""

            from langchain_core.messages import SystemMessage, HumanMessage

            print("SKILL COMPARISON PROMPT:", prompt)
            messages = [
                SystemMessage(content="You are a precise skill-matching classifier. Return only valid JSON."),
                HumanMessage(content=prompt),
            ]

            deterministic_llm = llm.bind(temperature=0)
            raw = await deterministic_llm.ainvoke(messages)
            content = raw.content.strip()

            # Strip markdown fences if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            parsed = json.loads(content)
            matched = [s for s in parsed.get("matched", []) if s in jd_skills]
            missing = [s for s in parsed.get("missing", []) if s in jd_skills]

            # Ensure every JD skill is accounted for
            accounted = set(matched) | set(missing)
            for skill in jd_skills:
                if skill not in accounted:
                    missing.append(skill)

            score = (len(matched) / len(jd_skills)) * 10 if jd_skills else 0.0
            print(f"[LLM match] matched={len(matched)}, missing={len(missing)}, score={round(score, 2)}")
            return round(score, 2), matched, missing

        except Exception as e:
            print(f"[WARN] LLM skill matching failed ({e}), falling back to set intersection")

    # --- Fallback: normalized set intersection ---
    matched = list(resume_skills & jd_skills)
    missing = list(jd_skills - resume_skills)
    score = (len(matched) / len(jd_skills)) * 10 if jd_skills else 0.0
    return round(score, 2), matched, missing

# async def compute_match_score_v2(resume_skills: set, jd_skills: set, llm=None) -> tuple:
#     """
#     Compute skill match score using LLM semantic matching.
#     Falls back to normalized set intersection if LLM is unavailable or fails.

#     Returns:
#         (score: float, matched: list, missing: list)
#     """
#     if not jd_skills:
#         return 0.0, [], []

#     if llm is not None:
#         try:
#             # Canonical, stable inputs
#             resume_list = sorted(set(s.lower().strip() for s in resume_skills if s))
#             jd_list = sorted(set(s.lower().strip() for s in jd_skills if s))

#             prompt = f"""You are a skill-matching classifier. Your task is to determine, for each Job Description (JD) skill, whether it is covered by the candidate’s resume skills.

#                 ** Mandatory Process (follow strictly in order)
#                     Normalize all Resume Skills and JD Skills into canonical forms.
#                     Perform semantic matching using the normalized forms.
#                     Return results using the original JD skill wording.

#                 ** Normalization Rules (STRICT)
#                     Convert all skills to lowercase.
#                     Remove punctuation, hyphens, and extra spaces.
#                     Standardize singular/plural forms (e.g., “apis” → “api”, “systems” → “system”).
#                     Expand or align acronyms and abbreviations with their full forms when clearly equivalent.
#                     Treat synonymous or closely related phrases as the same skill when they represent the same capability (e.g., different wording for the same concept).
#                     Treat minor wording variations as equivalent (e.g., “management” vs “managing”, “analysis” vs “analyzing”).
#                     If one skill is a broader category that clearly includes another, treat it as a match.
#                     If a JD skill is clearly implied by a resume skill, consider it covered.

#                 ** Matching Rules
#                     A JD skill is matched if its normalized form is equal to or semantically covered by any normalized resume skill.
#                     Otherwise, it is missing.
#                     Do NOT rely on exact string matching alone.
#                     Avoid overly strict interpretation when semantic equivalence is clear.
#                     Use semantic equivalence.


#                 ** Hard Constraints
#                     Every JD skill must appear in exactly one list: matched or missing.
#                     No duplicates.
#                     Use the exact JD skill wording in the output.
#                     Do not introduce or infer skills not present in the inputs.
#                     Output ONLY the JSON object.
                

#                 ### Output format
#                 {{"matched": ["<jd_skill>", ...], "missing": ["<jd_skill>", ...]}}

#                 ### Input
#                 Resume Skills: {resume_list}
#                 JD Skills: {jd_list}"""

#             from langchain_core.messages import SystemMessage, HumanMessage

#             print("SKILL COMPARISON PROMPT:", prompt)
#             messages = [
#                 SystemMessage(content="You are a precise skill-matching classifier. Return only valid JSON."),
#                 HumanMessage(content=prompt),
#             ]

#             deterministic_llm = llm.bind(temperature=0)
#             raw = await deterministic_llm.ainvoke(messages)
#             content = raw.content.strip()

#             print("Content: ", content)

#             # Strip markdown fences if present
#             if content.startswith("```"):
#                 content = re.sub(r"^```(?:json)?\s*", "", content)
#                 content = re.sub(r"\s*```$", "", content)

#             parsed = json.loads(content)
#             matched = [s for s in parsed.get("matched", []) if s in jd_skills]
#             missing = [s for s in parsed.get("missing", []) if s in jd_skills]

#             # Ensure every JD skill is accounted for
#             accounted = set(matched) | set(missing)
#             for skill in jd_skills:
#                 if skill not in accounted:
#                     missing.append(skill)

#             score = (len(matched) / len(jd_skills)) * 10 if jd_skills else 0.0
#             print(f"[LLM match] matched={len(matched)}, missing={len(missing)}, score={round(score, 2)}")
#             return round(score, 2), matched, missing

#         except Exception as e:
#             print(f"[WARN] LLM skill matching failed ({e}), falling back to set intersection")

#     # --- Fallback: normalized set intersection ---
#     matched = list(resume_skills & jd_skills)
#     missing = list(jd_skills - resume_skills)
#     score = (len(matched) / len(jd_skills)) * 10 if jd_skills else 0.0
#     return round(score, 2), matched, missing


# -----------------------------
# 6. EXPERIENCE EXTRACTION
# -----------------------------

def extract_experience(text):
    text_lower = text.lower()
    current_year = datetime.datetime.now().year

    # -----------------------------
    # 1. DATE RANGE (if present)
    # -----------------------------
    date_matches = re.findall(
        r'(20\d{2})\s*[-–]\s*(present|current|(20\d{2}))',
        text_lower
    )

    durations = []
    for match in date_matches:
        start = int(match[0])
        end = current_year if match[1] in ["present", "current"] else int(match[2])
        durations.append(end - start)

    if durations:
        return max(durations)

    # -----------------------------
    # 2. DIRECT YEARS (if present)
    # -----------------------------
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*\+?\s*(years|yrs)', text_lower)
    if matches:
        return int(float(max(m[0] for m in matches)))

    # -----------------------------
    # 3. 🔥 HEURISTIC (YOUR CASE)
    # -----------------------------

    exp_score = 0

    # Senior role indicator
    if "senior" in text_lower:
        exp_score += 3

    # Project count heuristic
    project_count = text_lower.count("project name")
    exp_score += min(project_count * 1.5, 5)

    # Responsibility density
    bullet_points = text_lower.count("·")
    if bullet_points > 20:
        exp_score += 2

    # Cap to realistic value
    return int(min(exp_score, 10))

# -----------------------------
# 7. EXPERIENCE SCORE
# -----------------------------

def compute_experience_score(resume_exp, jd_exp):
    if jd_exp == 0:
        return 5  # neutral

    ratio = resume_exp / jd_exp

    if ratio >= 1:
        return 10
    elif ratio >= 0.8:
        return 8
    elif ratio >= 0.6:
        return 6
    elif ratio >= 0.4:
        return 4
    else:
        return 2


# -----------------------------
# 8. SKILL-SPECIFIC EXPERIENCE EXTRACTION
# -----------------------------

# Patterns ordered from most-specific to least-specific to avoid greedy over-capture.
_SKILL_EXP_PATTERNS = [
    # "5+ years of experience with Python"  /  "5 years of Python experience"
    r'(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?(?:experience\s+)?(?:with|in|using|of)\s+([a-z][a-z0-9 \.\+\#\-]{1,35}?)(?:\s+experience)?(?=\s*[,\.\;\n\(\)]|$)',
    # "Python experience of 5+ years"  /  "Python: 5 years"
    r'([a-z][a-z0-9 \.\+\#\-]{1,35}?)\s*(?:experience|expertise|background|proficiency)\s+(?:of\s+)?(\d+(?:\.\d+)?)\+?\s*years?',
    # "5+ years Python"  (no connector word)
    r'(\d+(?:\.\d+)?)\+?\s*years?\s+([a-z][a-z0-9 \.\+\#\-]{2,30}?)(?=\s*[,\.\;\n\(\)]|$)',
]

# Noise words that appear after a skill count match but are not real skills.
_EXP_NOISE = {"experience", "in", "of", "with", "using", "working", "total", "relevant",
              "industry", "professional", "development", "software", "programming", "engineering"}


def _clean_skill_token(token: str) -> str:
    token = token.strip().rstrip(".,;:()")
    token = re.sub(r'\s+', ' ', token)
    return token.lower()


def extract_skill_specific_experience(text: str) -> dict:
    """
    Extract skill-specific experience mentions from text.
    Returns {skill: years} e.g. {"python": 5, "react": 3, "aws": 2}.
    Only keeps entries where the skill token looks like a real technology/skill name.
    """
    text_lower = text.lower()
    result: dict = {}

    for pattern in _SKILL_EXP_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            g1, g2 = match.group(1).strip(), match.group(2).strip()

            # Determine which group is the year and which is the skill.
            try:
                years = float(g1)
                skill = _clean_skill_token(g2)
            except ValueError:
                try:
                    years = float(g2)
                    skill = _clean_skill_token(g1)
                except ValueError:
                    continue

            # Skip if the captured "skill" is just noise.
            if skill in _EXP_NOISE or len(skill) < 2:
                continue

            # Keep the highest year value if skill appears multiple times.
            if skill not in result or years > result[skill]:
                result[skill] = int(years)

    return result


def _score_years(resume_years: int, jd_years: int) -> float:
    if jd_years == 0:
        return 10.0
    ratio = resume_years / jd_years
    if ratio >= 1.0:
        return 10.0
    elif ratio >= 0.8:
        return 8.0
    elif ratio >= 0.6:
        return 6.0
    elif ratio >= 0.4:
        return 4.0
    else:
        return 2.0


# -----------------------------
# 9. EXPERIENCE SCORE V2
# -----------------------------

async def compute_experience_score_v2(
    resume_text: str,
    jd_text: str,
    llm=None
) -> tuple:
    """
    Compute an experience score (0–10) by comparing:
      1. Overall years of experience (resume vs JD requirement).
      2. Skill-specific experience requirements stated in the JD.

    Uses LLM for skill-specific extraction when available; falls back to regex.

    Returns:
        (score: float, breakdown: dict)

    breakdown keys:
        overall_resume_exp   – years extracted from resume
        overall_jd_exp       – years required by JD
        overall_score        – 0-10 based on overall years only
        skill_requirements   – list of per-skill dicts
        final_score          – weighted final score (same as returned score)
    """
    # ── Step 1: overall experience ────────────────────────────────────────────
    resume_exp = extract_experience(resume_text)
    jd_exp     = extract_experience(jd_text)
    overall_score = float(compute_experience_score(resume_exp, jd_exp))

    # ── Step 2: skill-specific extraction ────────────────────────────────────
    jd_skill_exp: dict     = {}   # {skill: required_years}
    resume_skill_exp: dict = {}   # {skill: years_in_resume}

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
            raw = await llm.bind(temperature=0).ainvoke(messages)
            content = raw.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            parsed = json.loads(content)
            jd_skill_exp     = {k.lower().strip(): int(v) for k, v in parsed.get("jd_skill_exp", {}).items()}
            resume_skill_exp = {k.lower().strip(): int(v) for k, v in parsed.get("resume_skill_exp", {}).items()}
            print(f"[EXP LLM] jd_skill_exp={jd_skill_exp}, resume_skill_exp={resume_skill_exp}")
        except Exception as e:
            print(f"[WARN] LLM skill-exp extraction failed ({e}), falling back to regex")

    if not jd_skill_exp:
        jd_skill_exp     = extract_skill_specific_experience(jd_text)
        resume_skill_exp = extract_skill_specific_experience(resume_text)

    # ── Step 3: per-skill scoring ─────────────────────────────────────────────
    skill_requirements = []
    skill_scores: list[float] = []

    for skill, jd_years in jd_skill_exp.items():
        resume_years = resume_skill_exp.get(skill)

        if resume_years is not None:
            s = _score_years(resume_years, jd_years)
        else:
            # Skill not explicitly tied to years in resume — use overall exp as proxy
            # but cap the proxy score at 8 since we have no direct evidence.
            s = min(_score_years(resume_exp, jd_years), 8.0)

        skill_requirements.append({
            "skill": skill,
            "jd_years": jd_years,
            "resume_years": resume_years,
            "score": s,
        })
        skill_scores.append(s)

    # ── Step 4: weighted final score ─────────────────────────────────────────
    if skill_scores:
        skill_specific_score = sum(skill_scores) / len(skill_scores)
        final_score = round(0.4 * overall_score + 0.6 * skill_specific_score, 2)
    else:
        skill_specific_score = None
        final_score = round(overall_score, 2)

    print(f"[EXP v2] overall={overall_score}, skill_specific={skill_specific_score}, final={final_score}")

    breakdown = {
        "overall_resume_exp":  resume_exp,
        "overall_jd_exp":      jd_exp,
        "overall_score":       overall_score,
        "skill_requirements":  skill_requirements,
        "final_score":         final_score,
    }
    return final_score, breakdown


# -----------------------------
# 10. FINAL COMBINED SCORE
# -----------------------------

def compute_final_score(skill_score, exp_score):
    return round((0.7 * skill_score) + (0.3 * exp_score), 1)
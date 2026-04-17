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

                ### Important
                Create a plan first and then proceed for output.

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
# 8. FINAL COMBINED SCORE
# -----------------------------

def compute_final_score(skill_score, exp_score):
    return round((0.7 * skill_score) + (0.3 * exp_score), 1)
import re
import datetime

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

def compute_match_score_v2(resume_skills, jd_skills):
    if not jd_skills:
        return 0, [], []

    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    

    jd_grouped = group_skills(jd_skills)

    importance = {
        "advanced": 3,
        "intermediate": 2,
        "beginner": 1
    }

    total = 0
    max_total = 0

    for level, skills in jd_grouped.items():
        for skill in skills:
            max_total += importance[level]
            if skill in matched:
                total += importance[level]

    score = (total / max_total) * 10 if max_total else 0

    return round(score, 2), list(matched), list(missing)

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
    return round((0.7 * skill_score) + (0.3 * exp_score), 2)
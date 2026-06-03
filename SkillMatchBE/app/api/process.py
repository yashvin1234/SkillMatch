# scoring logic changed to /10 in this code
# latest refined llm usage code — all scoring fixes applied
# using llm to generate interview questions

from fastapi import APIRouter, HTTPException, Request, Depends
from app.utils.text_extract import (
    extract_client_names_advanced,
    extract_text,
    filter_spelling_errors,
    filter_grammar_errors,
)
from app.config import memory_store
from app.schemas.schemas import (
    ResumeAnalysisResponse,
    JDAnalysisResponse,
    ShrinkSummaryResponse,
    SkillExtractionResponse,
)
import json
import ast
import hashlib
import asyncio
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables.base import RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
import re

from app.utils.skill_engine import (
    extract_skills,
    compute_match_score_v2,
    extract_experience,
    compute_experience_score,
    compute_experience_score_v2,
    compute_final_score,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def get_question_suggester(request: Request):
    return request.app.state.question_suggester


# ---------------------------------------------------------------------------
# LLM + Embedder factories
# ---------------------------------------------------------------------------

def _make_llm() -> ChatOpenAI:
    """temperature=0 + seed=42 → maximum determinism across all chains."""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        model_kwargs={"seed": 42},
    )


def _make_embedder() -> OpenAIEmbeddings:
    """text-embedding-3-small: cheap, fast, accurate enough."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


# ---------------------------------------------------------------------------
# Content-hash cache
# ---------------------------------------------------------------------------

def _content_hash(resume_text: str, jd_text: str) -> str:
    return hashlib.md5(f"{resume_text}||{jd_text}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def normalize_suggested_questions(raw_content: str) -> list[str]:
    # Step 1: strip markdown fences
    cleaned = _clean_llm_json(raw_content)

    # Step 2: try JSON parse on cleaned content
    try:
        arr = json.loads(cleaned)
        if isinstance(arr, list):
            return [str(i).strip().strip('"').strip("'")
                    for i in arr if str(i).strip()]
    except Exception:
        pass

    # Step 3: try literal eval
    try:
        arr = ast.literal_eval(cleaned)
        if isinstance(arr, list):
            return [str(i).strip().strip('"').strip("'")
                    for i in arr if str(i).strip()]
    except Exception:
        pass

    # Step 4: last resort — line by line, skip fence lines
    results = []
    for line in cleaned.split('\n'):
        line = line.strip().strip('"').strip("'").strip(',').strip()
        if not line or line in ['[', ']', '```', '```json']:
            continue
        if line.startswith('```'):
            continue
        results.append(line)
    return results


def _clean_llm_json(raw_content: str) -> str:
    """Strip markdown fences and trailing commas before JSON parsing."""
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.DOTALL
    )
    content = re.sub(r',\s*([}\]])', r'\1', content)
    return content.strip()


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

async def _with_timeout(coro, timeout_seconds: float, fallback, label: str = ""):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print(f"[TIMEOUT] '{label}' exceeded {timeout_seconds}s — using fallback")
        return fallback
    except Exception as e:
        print(f"[ERROR] '{label}' raised {e} — using fallback")
        return fallback


# ---------------------------------------------------------------------------
# Minimal canonical normalisation
# ---------------------------------------------------------------------------

def _basic_normalize(skill: str) -> str:
    skill = skill.lower().strip()
    skill = re.sub(r'\(.*?\)', '', skill)
    skill = re.sub(r'\s+', ' ', skill).strip()
    return skill


def _word_boundary_re(skill: str) -> re.Pattern:
    escaped = re.escape(skill)
    return re.compile(
        r'(?<![a-z0-9])' + escaped + r'(?![a-z0-9])',
        re.IGNORECASE,
    )


def _exact_overlap(skill_a: str, skill_b: str) -> bool:
    a = _basic_normalize(skill_a)
    b = _basic_normalize(skill_b)
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 2:
        return False
    try:
        return bool(_word_boundary_re(shorter).search(longer))
    except re.error:
        return shorter in longer


# =============================================================================
# COMBINED Step 1+2+3+normalisation — single LLM call
# =============================================================================

async def _extract_and_normalise_combined(
    resume_text: str,
    jd_text: str,
    llm: ChatOpenAI,
) -> tuple[list[str], list[str], dict[str, str], dict[str, str]]:
    exp_section = _extract_experience_section(resume_text)

    prompt = f"""You are a skill extraction and normalisation engine. Complete ALL tasks below in one pass.

════════════════════════════════════════════════════════════
TASK 1 — Extract skills from RESUME
════════════════════════════════════════════════════════════
A) Explicit skills: everything directly stated in skills sections,
   certifications, tools, technologies, and methodologies.

B) Implied skills: read every work experience bullet and infer skills
   that are STRONGLY demonstrated even if not explicitly named.
   Valid inference examples:
   - "Managed P&L of $50M" → financial management, budgeting, forecasting
   - "Led team of 15 engineers" → team leadership, performance management
   - "Reduced infra costs 40% via cloud migration" → cloud, cost optimisation
   - "Built CI/CD pipeline" → cicd, devops, automation
   - "Conducted SEBI filings" → regulatory compliance, financial reporting
   - "Performed appendectomies" → surgical experience, clinical skills
   Only infer if you are CONFIDENT. Do not guess or hallucinate.

C) Deduplicate: merge all variants into one canonical lowercase form.
   - "ReactJS" + "React.js" + "React framework" → "react"
   - "Postgres" + "PostgreSQL" → "postgresql"
   - "CI/CD" + "GitLab CI/CD" → "cicd"
   - "ML" + "machine learning" → "machine learning"

════════════════════════════════════════════════════════════
TASK 2 — Extract skills from JOB DESCRIPTION
════════════════════════════════════════════════════════════
- Include must-have AND nice-to-have / preferred skills.
- Include soft skills, tools, methodologies, and domain knowledge.
- Deduplicate variants the same as Task 1C above.

════════════════════════════════════════════════════════════
TASK 3 — Build canonical mapping for BOTH lists
════════════════════════════════════════════════════════════
For every skill in both output lists, provide raw → canonical mapping.
Canonical = short (1-3 words), lowercase, standard industry term.
Every item in resume_skills and jd_skills MUST appear as a key here.

════════════════════════════════════════════════════════════
OUTPUT — STRICT JSON ONLY, no markdown, no explanation
════════════════════════════════════════════════════════════
{{
  "resume_skills": ["skill1", "skill2", ...],
  "jd_skills": ["skill1", "skill2", ...],
  "resume_mapping": {{"raw_skill": "canonical", ...}},
  "jd_mapping": {{"raw_skill": "canonical", ...}}
}}

Hard rules:
- resume_mapping MUST have one entry per item in resume_skills
- jd_mapping MUST have one entry per item in jd_skills
- All keys and values must be lowercase
- No markdown fences, no preamble, no trailing explanation

--- RESUME (full) ---
{resume_text}

--- RESUME EXPERIENCE SECTION (for implied inference) ---
{exp_section[:3000]}

--- JOB DESCRIPTION (full) ---
{jd_text}"""

    messages = [
        SystemMessage(
            content="You are a deterministic extraction and normalisation engine. "
                    "Return only valid JSON."
        ),
        HumanMessage(content=prompt),
    ]

    try:
        raw     = await llm.ainvoke(messages)
        content = _clean_llm_json(raw.content)
        data    = json.loads(content)

        resume_skills  = [str(s) for s in data.get("resume_skills", [])]
        jd_skills      = [str(s) for s in data.get("jd_skills",     [])]
        resume_mapping = {k.lower(): v.lower()
                          for k, v in data.get("resume_mapping", {}).items()}
        jd_mapping     = {k.lower(): v.lower()
                          for k, v in data.get("jd_mapping",     {}).items()}

        for s in resume_skills:
            if s.lower() not in resume_mapping:
                resume_mapping[s.lower()] = _basic_normalize(s)
        for s in jd_skills:
            if s.lower() not in jd_mapping:
                jd_mapping[s.lower()] = _basic_normalize(s)

        print(f"[Combined extract] Resume: {len(resume_skills)} skills, "
              f"JD: {len(jd_skills)} skills")
        print(f"[Combined extract] Resume mapping sample: "
              f"{list(resume_mapping.items())[:5]}")
        print(f"[Combined extract] JD mapping sample: "
              f"{list(jd_mapping.items())[:5]}")

        return resume_skills, jd_skills, resume_mapping, jd_mapping

    except Exception as e:
        print(f"[WARN] Combined extraction failed ({e}) — falling back to rule-based")
        resume_skills  = list(extract_skills(resume_text))
        jd_skills      = list(extract_skills(jd_text))
        resume_mapping = {s.lower(): _basic_normalize(s) for s in resume_skills}
        jd_mapping     = {s.lower(): _basic_normalize(s) for s in jd_skills}
        return resume_skills, jd_skills, resume_mapping, jd_mapping


def _apply_mapping(raw_skills: list[str], mapping: dict[str, str]) -> set[str]:
    result = set()
    for s in raw_skills:
        key = s.lower()
        result.add(mapping.get(key, _basic_normalize(s)))
    return result


# =============================================================================
# LAYER 1 — Exact match
# =============================================================================

def _l1_exact_match(jd_canonical: str, resume_canonicals: set[str]) -> bool:
    return jd_canonical in resume_canonicals or any(
        _exact_overlap(jd_canonical, rs) for rs in resume_canonicals
    )


# =============================================================================
# LAYER 2 — Embedding similarity
# =============================================================================

EMBEDDING_THRESHOLD  = 0.82
SOFT_SKILL_THRESHOLD = 0.78

_SOFT_SKILL_KEYWORDS = re.compile(
    r'\b(communication|collaboration|leadership|management|'
    r'stakeholder|influenc|negotiat|coaching|facilitat|'
    r'strategic|analytical|interpersonal|relationship|'
    r'presentation|persuasion|conflict|empathy|adaptability|'
    r'creativity|innovation|problem.solving|critical.thinking|'
    r'decision.making|time.management|prioriti)\b',
    re.IGNORECASE,
)


def _embedding_threshold_for(skill: str) -> float:
    return (
        SOFT_SKILL_THRESHOLD
        if _SOFT_SKILL_KEYWORDS.search(skill)
        else EMBEDDING_THRESHOLD
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


async def _embedding_match(
    unmatched_jd_skills: set[str],
    resume_skills: set[str],
    embedder: OpenAIEmbeddings,
) -> dict[str, tuple[str, float]]:
    if not unmatched_jd_skills or not resume_skills:
        return {}

    jd_list = sorted(unmatched_jd_skills)
    rs_list = sorted(resume_skills)

    print(f"[Embedding L2] {len(jd_list)} JD skills vs {len(rs_list)} resume skills")

    jd_embeddings, rs_embeddings = await asyncio.gather(
        embedder.aembed_documents(jd_list),
        embedder.aembed_documents(rs_list),
    )

    hits: dict[str, tuple[str, float]] = {}
    for jd_skill, jd_vec in zip(jd_list, jd_embeddings):
        threshold = _embedding_threshold_for(jd_skill)
        best_sim  = 0.0
        best_rs   = ""
        for rs_skill, rs_vec in zip(rs_list, rs_embeddings):
            sim = _cosine_similarity(jd_vec, rs_vec)
            if sim > best_sim:
                best_sim = sim
                best_rs  = rs_skill
        if best_sim >= threshold:
            print(f"  [embed hit]  '{jd_skill}' ↔ '{best_rs}'  "
                  f"sim={best_sim:.3f} (threshold={threshold})")
            hits[jd_skill] = (best_rs, best_sim)
        else:
            print(f"  [embed miss] '{jd_skill}' best='{best_rs}' "
                  f"sim={best_sim:.3f} (threshold={threshold})")

    return hits


# =============================================================================
# COMBINED LAYER 3 + PROFICIENCY — Single LLM call
# =============================================================================

def _extract_relevant_sentences(
    skill: str,
    resume_text: str,
    raw_aliases: list[str] | None = None,
    max_sentences: int = 8,
) -> list[str]:
    search_terms = [_basic_normalize(skill)]
    if raw_aliases:
        search_terms += [_basic_normalize(a) for a in raw_aliases]
    search_terms = list(dict.fromkeys(search_terms))

    sentences   = re.split(r'(?<=[.!?\n])\s+', resume_text)
    relevant:   list[str] = []
    seen_sents: set[str]  = set()

    for term in search_terms:
        try:
            pattern = _word_boundary_re(term)
        except re.error:
            pattern = None

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean or sent_clean in seen_sents:
                continue
            matched = (
                pattern.search(sent_clean.lower()) if pattern
                else term in sent_clean.lower()
            )
            if matched:
                relevant.append(sent_clean)
                seen_sents.add(sent_clean)
            if len(relevant) >= max_sentences:
                return relevant

    return relevant


async def _classify_and_assess_unmatched(
    unmatched_jd_skills: set[str],
    matched_jd_skills: set[str],
    resume_text: str,
    jd_text: str,
    llm: ChatOpenAI,
    jd_mapping: dict[str, str] | None = None,
    resume_mapping: dict[str, str] | None = None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    if not unmatched_jd_skills and not matched_jd_skills:
        return {}, {}

    canonical_to_raws: dict[str, list[str]] = {}
    if jd_mapping:
        for raw, canon in jd_mapping.items():
            canonical_to_raws.setdefault(canon, []).append(raw)

    unmatched_with_evidence = []
    for skill in sorted(unmatched_jd_skills):
        aliases  = canonical_to_raws.get(skill, [])
        evidence = _extract_relevant_sentences(
            skill, resume_text, raw_aliases=aliases, max_sentences=6
        )
        unmatched_with_evidence.append({
            "skill":              skill,
            "evidence_sentences": evidence,
        })

    matched_with_evidence = []
    for skill in sorted(matched_jd_skills):
        aliases  = canonical_to_raws.get(skill, [])
        evidence = _extract_relevant_sentences(
            skill, resume_text, raw_aliases=aliases, max_sentences=5
        )
        matched_with_evidence.append({
            "skill":           skill,
            "resume_evidence": evidence if evidence else [
                "(mentioned but no specific sentences found)"
            ],
        })

    prompt = f"""You are an expert recruiter completing two tasks in one pass.

--- JOB DESCRIPTION (first 800 chars) ---
{jd_text[:800]}

--- RESUME BROAD CONTEXT (first 1500 chars) ---
{resume_text[:1500]}

════════════════════════════════════════════════════════════
TASK A — Evidence reasoning for UNMATCHED skills
════════════════════════════════════════════════════════════
For each skill below, judge whether the resume DEMONSTRATES it through
actual work experience — even if the exact term is absent.

Look for:
- Direct mentions with any variation of the skill name
- Work experience that implies the skill
- Projects or achievements requiring the skill

Rules:
- Return matched=true ONLY if you have real evidence, not assumption
- Be conservative — if genuinely unsure, return false
- Only return matched=true if confidence >= 0.65

Skills to evaluate (with extracted evidence sentences):
{json.dumps(unmatched_with_evidence, indent=2)}

════════════════════════════════════════════════════════════
TASK B — Proficiency assessment for MATCHED skills
════════════════════════════════════════════════════════════
For each skill below, rate depth of experience from the evidence sentences
AND the broad resume context above.

Scoring guide:
- 5 = Expert / Architected / Led at scale / 5+ years explicit
- 4 = Proficient / Hands-on project experience / 3-5 years
- 3 = Solid working experience / 1-3 years / multiple projects
- 2 = Basic / Limited exposure / mentioned without real detail
- 1 = Skills list only / "familiar with" / "knowledge of" / no evidence

Level mapping:
- score 4-5 → "deep"
- score 3   → "adequate"
- score 1-2 → "shallow"

Skills to assess (with extracted evidence sentences):
{json.dumps(matched_with_evidence, indent=2)}

════════════════════════════════════════════════════════════
OUTPUT — STRICT JSON ONLY, no markdown, no explanation
════════════════════════════════════════════════════════════
{{
  "evidence_results": {{
    "skill_name": {{
      "matched": true,
      "confidence": 0.85,
      "reason": "one concise sentence"
    }}
  }},
  "proficiency_results": {{
    "skill_name": {{
      "score": 4,
      "level": "deep",
      "reason": "one concise sentence"
    }}
  }}
}}"""

    messages = [
        SystemMessage(
            content="You are a deterministic recruiter evaluator. "
                    "Return only valid JSON."
        ),
        HumanMessage(content=prompt),
    ]

    evidence_results:    dict[str, dict] = {}
    proficiency_results: dict[str, dict] = {}

    try:
        raw     = await llm.ainvoke(messages)
        content = _clean_llm_json(raw.content)
        data    = json.loads(content)

        raw_evidence = data.get("evidence_results", {})
        for skill in unmatched_jd_skills:
            skill_data = raw_evidence.get(skill, {})
            evidence_results[skill] = {
                "matched":    bool(skill_data.get("matched",    False)),
                "confidence": float(skill_data.get("confidence", 0.0)),
                "reason":     str(skill_data.get("reason",      "")),
            }
            r = evidence_results[skill]
            print(f"  [L3] '{skill}' matched={r['matched']} "
                  f"conf={r['confidence']:.2f} | {r['reason'][:80]}")

        raw_proficiency = data.get("proficiency_results", {})
        for skill in matched_jd_skills:
            skill_data = raw_proficiency.get(skill, {})
            proficiency_results[skill] = {
                "level":  str(skill_data.get("level",  "adequate")),
                "score":  int(skill_data.get("score",  3)),
                "reason": str(skill_data.get("reason", "")),
            }
            p = proficiency_results[skill]
            print(f"  [Prof] '{skill}' level={p['level']} "
                  f"score={p['score']} | {p['reason'][:70]}")

    except Exception as e:
        print(f"[WARN] Combined L3+proficiency failed ({e}) — using safe defaults")
        for skill in unmatched_jd_skills:
            evidence_results[skill] = {
                "matched": False, "confidence": 0.0, "reason": "LLM error"
            }
        for skill in matched_jd_skills:
            proficiency_results[skill] = {
                "level": "adequate", "score": 3, "reason": "assessment unavailable"
            }

    return evidence_results, proficiency_results


# =============================================================================
# OR-group handling
# =============================================================================

def _extract_or_groups(jd_text: str) -> list[set[str]]:
    groups: list[set[str]] = []

    comma_or_pattern = re.compile(
        r'(?:such as|like|including|e\.g\.?|:)?\s*'
        r'([A-Za-z0-9][A-Za-z0-9\.\+\#/\-]*'
        r'(?:\s*,\s*[A-Za-z0-9][A-Za-z0-9\.\+\#/\-]*)*'
        r'\s*,?\s*or\s+[A-Za-z0-9][A-Za-z0-9\.\+\#/\-]*)',
        re.IGNORECASE,
    )
    for m in comma_or_pattern.finditer(jd_text):
        raw_parts = re.split(r',\s*|\s+or\s+', m.group(0), flags=re.IGNORECASE)
        cleaned: set[str] = set()
        for p in raw_parts:
            p = re.sub(
                r'^(?:such as|like|including|e\.g\.?|:)\s*', '', p, flags=re.IGNORECASE
            )
            norm = _basic_normalize(p.strip())
            if norm and len(norm) > 1:
                cleaned.add(norm)
        if len(cleaned) > 1:
            groups.append(cleaned)

    slash_pattern = re.compile(
        r'\b([A-Za-z][A-Za-z0-9\.\+\#\-]{1,20})'
        r'(?:/([A-Za-z][A-Za-z0-9\.\+\#\-]{1,20}))+'
        r'\b'
    )
    for m in slash_pattern.finditer(jd_text):
        raw_parts = m.group(0).split('/')
        cleaned = {_basic_normalize(p.strip()) for p in raw_parts if len(p.strip()) > 1}
        if len(cleaned) > 1 and not any('/' in p for p in cleaned):
            groups.append(cleaned)

    either_pattern = re.compile(
        r'\beither\s+([A-Za-z][A-Za-z0-9\.\+\#\-]*)\s+or\s+([A-Za-z][A-Za-z0-9\.\+\#\-]*)\b',
        re.IGNORECASE,
    )
    for m in either_pattern.finditer(jd_text):
        a = _basic_normalize(m.group(1).strip())
        b = _basic_normalize(m.group(2).strip())
        if a and b and len(a) > 1 and len(b) > 1:
            groups.append({a, b})

    unique_groups: list[set[str]] = []
    for g in groups:
        if not any(g == existing or g.issubset(existing) for existing in unique_groups):
            unique_groups.append(g)

    print(f"[OR groups]: {unique_groups}")
    return unique_groups


def _resolve_or_groups(
    missing: set[str],
    partial: set[str],
    matched: set[str],
    or_groups: list[set[str]],
) -> tuple[set[str], set[str]]:
    missing = set(missing)
    partial = set(partial)
    for group in or_groups:
        satisfied_by = group & (matched | partial)
        if satisfied_by:
            unchosen = group - satisfied_by
            missing -= unchosen
            partial -= unchosen
            print(f"[OR resolve] satisfied_by={satisfied_by}  removed={unchosen}")
    return missing, partial


# =============================================================================
# Main classify_skills
# =============================================================================

async def classify_skills(
    resume_skills_raw: list[str],
    jd_skills_raw: list[str],
    resume_text: str,
    jd_text: str,
    embedder: OpenAIEmbeddings,
    llm: ChatOpenAI,
    resume_mapping: dict[str, str],
    jd_mapping: dict[str, str],
) -> tuple[set[str], set[str], set[str]]:
    resume_canonicals = _apply_mapping(resume_skills_raw, resume_mapping)
    jd_canonicals     = {
        jd_mapping.get(s.lower(), _basic_normalize(s)) for s in jd_skills_raw
    }

    canonical_to_jd: dict[str, str] = {}
    for raw in jd_skills_raw:
        canonical = jd_mapping.get(raw.lower(), _basic_normalize(raw))
        canonical_to_jd[canonical] = raw

    l1_matched:    set[str] = set()
    still_missing: set[str] = set()

    for jd_canonical in jd_canonicals:
        if _l1_exact_match(jd_canonical, resume_canonicals):
            print(f"  [L1 match] '{jd_canonical}'")
            l1_matched.add(jd_canonical)
        else:
            still_missing.add(jd_canonical)

    embed_hits = await _with_timeout(
        _embedding_match(still_missing, resume_canonicals, embedder),
        timeout_seconds=20,
        fallback={},
        label="L2 embedding match",
    )
    l2_matched:    set[str] = set(embed_hits.keys())
    still_missing -= l2_matched

    all_matched_so_far = l1_matched | l2_matched
    matched_originals  = {canonical_to_jd.get(c, c) for c in all_matched_so_far}
    l3_candidates      = {canonical_to_jd.get(c, c) for c in still_missing}

    evidence_results, proficiency_results = await _with_timeout(
        _classify_and_assess_unmatched(
            unmatched_jd_skills=l3_candidates,
            matched_jd_skills=matched_originals,
            resume_text=resume_text,
            jd_text=jd_text,
            llm=llm,
            jd_mapping=jd_mapping,
            resume_mapping=resume_mapping,
        ),
        timeout_seconds=30,
        fallback=(
            {skill: {"matched": False, "confidence": 0.0, "reason": "timeout"}
             for skill in l3_candidates},
            {skill: {"level": "adequate", "score": 3, "reason": "timeout fallback"}
             for skill in matched_originals},
        ),
        label="L3 evidence + proficiency",
    )

    l3_matched:    set[str] = set()
    truly_missing: set[str] = set()

    for canonical in still_missing:
        original = canonical_to_jd.get(canonical, canonical)
        result   = evidence_results.get(original, {"matched": False})
        if result["matched"] and result.get("confidence", 0) >= 0.65:
            print(f"  [L3 match] '{canonical}' "
                  f"(confidence={result.get('confidence', 0):.2f})")
            l3_matched.add(canonical)
        else:
            print(f"  [missing]  '{canonical}'")
            truly_missing.add(canonical)

    all_matched_canonicals = l1_matched | l2_matched | l3_matched

    final_matched: set[str] = set()
    final_partial: set[str] = set()

    for canonical in all_matched_canonicals:
        original = canonical_to_jd.get(canonical, canonical)
        prof     = proficiency_results.get(original, {"level": "adequate"})
        if prof["level"] == "shallow":
            print(f"  [shallow → partial] '{original}'")
            final_partial.add(original)
        else:
            final_matched.add(original)

    final_missing = {canonical_to_jd.get(c, c) for c in truly_missing}

    return final_matched, final_partial, final_missing


# =============================================================================
# Experience helpers
# =============================================================================

def _extract_experience_section(resume_text: str) -> str:
    patterns = [
        r'(?i)(work\s+experience|professional\s+experience|employment\s+history'
        r'|experience|career\s+history)',
    ]
    for pat in patterns:
        m = re.search(pat, resume_text)
        if m:
            return resume_text[m.start():]
    return resume_text


def _resolve_resume_experience(resume_text: str) -> int:
    exp = extract_experience(resume_text)
    if exp > 0:
        return exp
    text_lower = resume_text.lower()
    if any(t in text_lower for t in ["principal", "staff engineer", "director"]):
        return 10
    if any(t in text_lower for t in ["senior", "lead", "architect", "head of"]):
        return 6
    if "engineer" in text_lower or "developer" in text_lower or "analyst" in text_lower:
        return 3
    if len(resume_text.split()) > 800:
        return 3
    return 2


async def _extract_client_names_llm(resume_text: str, llm: ChatOpenAI) -> list[str]:
    messages = [
        SystemMessage(
            content="You are a strict information extraction engine. Follow rules exactly."
        ),
        HumanMessage(content=f"""
Task:
Extract ONLY real client/company names explicitly mentioned in the resume.

Rules:
1. Include ONLY real, non-anonymised company names (e.g., Google, Walmart, Accenture).
2. DO NOT infer, guess, or deduce from context.
3. EXCLUDE:
   - "Fortune 500 company"
   - "US-based client"
   - "Leading organization"
   - Any vague or anonymised descriptions
4. DO NOT include:
   - Employers (unless explicitly stated as client)
   - Tools/technologies (e.g., AWS, SAP, Snowflake)
   - Certifications or education institutes
5. If ANY doubt → DO NOT include it.
6. Prefer returning [] over incorrect output.

Return ONLY valid JSON array:
["Client1", "Client2"]

--- RESUME ---
{resume_text}
"""),
    ]

    try:
        raw       = await llm.bind(temperature=0).ainvoke(messages)
        content   = _clean_llm_json(raw.content)
        names     = json.loads(content)
        llm_names = (
            [str(n).strip() for n in names if isinstance(n, str) and n.strip()]
            if isinstance(names, list) else []
        )
    except Exception as e:
        print(f"[WARN] LLM client extraction failed: {e}")
        llm_names = []

    rule_based_names = extract_client_names_advanced(resume_text)

    final = set()
    for name in llm_names + rule_based_names:
        clean = name.strip()
        if len(clean) > 2:
            final.add(clean)
    return sorted(final)


# =============================================================================
# Gap deduplication
# =============================================================================

def _dedupe_gaps(gap_list: list[str], all_gaps: set[str]) -> list[str]:
    seen:   set[str] = set()
    result: list[str] = []
    for item in gap_list:
        key = next(
            (s for s in all_gaps if _exact_overlap(s, item)),
            item.lower()[:40],
        )
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# =============================================================================
# Analysis chain
# =============================================================================

def _build_analysis_chain(llm: ChatOpenAI):
    pydantic_parser = PydanticOutputParser(pydantic_object=ResumeAnalysisResponse)
    fixing_parser   = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert recruiter, resume strategist, and proofreader."),
        ("user", """Return JSON matching this schema:
{format_instructions}

═══════════════════════════════════════════════════════════
GROUND TRUTH — pre-computed, authoritative, use as-is.
═══════════════════════════════════════════════════════════

MATCHED SKILLS  → candidate has adequate, evidenced depth.
                  Populate Key_Matches with brief evidence description.

PARTIAL SKILLS  → candidate mentions skill but depth is shallow/basic.
                  These ARE gaps. List EVERY one in Key_Gaps.
                  Format: "Limited <skill> — JD expects deeper proficiency"

MISSING SKILLS  → skill completely absent from resume.
                  These ARE gaps. List EVERY one in Key_Gaps.
                  Format: "No experience with <skill>"

━━━ STRICT RULES ━━━
1. Key_Gaps MUST contain one entry for EVERY skill in Partial + Missing.
2. NEVER write "No major gaps" if Partial or Missing is non-empty.
3. NEVER put a Partial or Missing skill in Key_Matches.
4. NEVER put a Matched skill in Key_Gaps.
5. Recommendations MUST address every partial and missing skill.
6. Be specific — reference actual resume evidence, not generic filler.
7. Apply these rules equally to technical skills AND business/soft skills.
8. If a skill is in MATCHED SKILLS, do NOT mention it as a gap anywhere.
━━━━━━━━━━━━━━━━━━━━

--- JOB DESCRIPTION ---
{jd_text}

--- RESUME ---
{resume_text}

--- ALL JD SKILLS ---
{jd_skills}

--- RESUME SKILLS ---
{resume_skills}

--- MATCHED SKILLS (→ Key_Matches) ---
{matched_skills}

--- PARTIAL SKILLS (→ Key_Gaps) ---
{partial_skills}

--- MISSING SKILLS (→ Key_Gaps) ---
{missing_skills}
""")
    ]).partial(format_instructions=fixing_parser.get_format_instructions())

    return (
        RunnableMap({
            "jd_text":        lambda x: x["jd_text"],
            "resume_text":    lambda x: x["resume_text"],
            "jd_skills":      lambda x: x["jd_skills"],
            "resume_skills":  lambda x: x["resume_skills"],
            "matched_skills": lambda x: x["matched_skills"],
            "partial_skills": lambda x: x["partial_skills"],
            "missing_skills": lambda x: x["missing_skills"],
        })
        | prompt
        | llm
        | fixing_parser
    )


def _build_shrink_chain(llm: ChatOpenAI):
    pydantic_parser = PydanticOutputParser(pydantic_object=ShrinkSummaryResponse)
    fixing_parser   = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical recruiter and resume summarization assistant."),
        ("user", """Return JSON matching this schema:
{format_instructions_shrink}

Write 4–6 short, clear sentences covering key technologies, tools,
core skills, and relevant domains from the combined text.

--- COMBINED TEXT ---
{combined_text}
""")
    ]).partial(format_instructions_shrink=fixing_parser.get_format_instructions())

    return (
        RunnableMap({"combined_text": lambda x: x["combined_text"]})
        | prompt
        | llm
        | fixing_parser
    )


# =============================================================================
# Interview question generation — LLM-based, domain-agnostic
# NOTE: Defined at module level. Do NOT nest inside any route function.
# =============================================================================

async def _generate_interview_questions(
    resume_text: str,
    jd_text: str,
    matched_skills: set[str],
    partial_skills: set[str],
    missing_skills: set[str],
    llm: ChatOpenAI,
) -> list[str]:
    prompt = f"""You are a senior technical interviewer with 15+ years of hiring experience.
Generate highly targeted, specific interview questions for THIS candidate and THIS role only.

CANDIDATE CONTEXT:
- Resume experience: {resume_text[:2000]}

ROLE CONTEXT:
- Job Description: {jd_text[:1500]}

SKILL CONTEXT:
- Strong skills (probe depth): {sorted(matched_skills)}
- Weak skills (probe gaps): {sorted(partial_skills)}  
- Missing skills (probe transferability): {sorted(missing_skills)}

GENERATE exactly 12-14 questions across these categories:
1. TECHNICAL DEPTH (4-5 questions)
   - Pick the 3-4 most critical matched skills
   - Ask about architecture decisions, tradeoffs, failure scenarios
   - Example style: "You mentioned using Kubernetes in production — walk me through how you handled a pod crash loop and what your debugging process was?"

2. GAP PROBING (3-4 questions)
   - For partial skills: give a realistic scenario and ask how they'd handle it
   - For missing skills: ask if they've solved similar problems with different tools
   - Example style: "You haven't worked with GCP directly — if you had to migrate an AWS workload to GCP in 30 days, what would your approach be?"

3. BEHAVIORAL / SITUATIONAL (3-4 questions)
   - Tie directly to JD responsibilities
   - Use STAR triggers: "Tell me about a time...", "Describe a situation where..."
   - Must reference something specific from their resume achievements

4. DOMAIN / ROLE-SPECIFIC (2 questions)
   - Test business understanding beyond technical skills
   - Ask about industry trends, stakeholder management, or strategic thinking
STRICT RULES:
- Every question must reference something SPECIFIC from the resume or JD
- No generic questions that could apply to any candidate
- No yes/no questions
- No duplicate themes
- Questions must match the seniority level implied by the JD
- Each question must end with ?

Return ONLY a raw JSON array of question strings. No markdown, no explanation.
Example: ["Question 1?", "Question 2?"]
"""

    messages = [
        SystemMessage(
            content="You are a senior technical interviewer with deep hiring expertise. "
                    "Return only valid JSON."
        ),
        HumanMessage(content=prompt),
    ]

    try:
        raw      = await llm.ainvoke(messages)
        content  = _clean_llm_json(raw.content)
        questions = json.loads(content)
        if isinstance(questions, list):
            return [str(q).strip() for q in questions if str(q).strip().endswith("?")]
    except Exception as e:
        print(f"[WARN] Question generation failed: {e}")

    return []


# =============================================================================
# Hard validation — final correctness guarantee
# =============================================================================

def _apply_hard_validation(
    merged: dict,
    matched: set[str],
    partial: set[str],
    missing: set[str],
) -> dict:
    all_gaps = partial | missing

    merged["Key_Matches"] = [
        item for item in merged.get("Key_Matches", [])
        if any(
            _basic_normalize(s) in item.lower() or _exact_overlap(s, item)
            for s in matched
        )
    ]
    represented_m = {
        s for s in matched
        if any(
            _basic_normalize(s) in item.lower() or _exact_overlap(s, item)
            for item in merged["Key_Matches"]
        )
    }
    for skill in sorted(matched - represented_m):
        merged["Key_Matches"].append(f"{skill} — demonstrated in resume")

    merged["Key_Gaps"] = [
        item for item in merged.get("Key_Gaps", [])
        if any(
            _basic_normalize(s) in item.lower() or _exact_overlap(s, item)
            for s in all_gaps
        )
    ]
    merged["Key_Gaps"] = _dedupe_gaps(merged["Key_Gaps"], all_gaps)

    represented_g = {
        s for s in all_gaps
        if any(
            _basic_normalize(s) in item.lower() or _exact_overlap(s, item)
            for item in merged["Key_Gaps"]
        )
    }
    for skill in sorted(partial - represented_g):
        merged["Key_Gaps"].append(
            f"Limited {skill} experience — JD expects deeper proficiency"
        )
    for skill in sorted((missing - represented_g) - partial):
        merged["Key_Gaps"].append(f"No experience with {skill}")

    merged["Key_Gaps"] = _dedupe_gaps(merged["Key_Gaps"], all_gaps)

    if "Score_Explanation_Technical" in merged:
        explanation = merged["Score_Explanation_Technical"]
        for skill in matched:
            explanation = re.sub(
                rf"(?i)\b(no|lack of|lacks|missing|absent)\b[^.]*\b{re.escape(skill)}\b",
                f"experience present with {skill}",
                explanation,
            )
        for skill in all_gaps:
            if not _word_boundary_re(_basic_normalize(skill)).search(explanation.lower()):
                if skill in partial:
                    explanation += f" {skill.title()} is present but only at a basic level."
                else:
                    explanation += f" No experience found for {skill}."
        if all_gaps:
            for phrase in [
                "no major gaps", "no significant gaps", "no gaps detected",
                "no gaps found", "strong match overall", "excellent match",
            ]:
                if phrase in explanation.lower():
                    gap_list = ", ".join(sorted(all_gaps))
                    explanation = re.sub(
                        re.escape(phrase),
                        f"gaps exist in: {gap_list}",
                        explanation,
                        flags=re.IGNORECASE,
                    )
                    break
        merged["Score_Explanation_Technical"] = explanation

    return merged


# =============================================================================
# Route: /process/jd_resume_match
# =============================================================================

@router.get("/process/jd_resume_match")
async def process(suggester=Depends(get_question_suggester)):
    if "resume" not in memory_store or "jd" not in memory_store:
        raise HTTPException(
            status_code=400,
            detail="Both resume and job description files must be uploaded first.",
        )

    resume_info = memory_store["resume"]
    jd_info     = memory_store["jd"]["jd_resume_match"]

    resume_text = extract_text(resume_info["bytes"], resume_info["filename"])
    jd_text     = extract_text(jd_info["bytes"],     jd_info["filename"])

    print("**** RESUME (first 400) ****\n", resume_text[:400])
    print("**** JD     (first 400) ****\n", jd_text[:400])

    # ── Cache check ───────────────────────────────────────────────────────
    cache_key = _content_hash(resume_text, jd_text)
    cached    = memory_store.get("analysis_cache", {}).get(cache_key)
    if cached:
        print("[CACHE HIT] returning cached result")
        return cached

    # ── Shared instances ──────────────────────────────────────────────────
    llm      = _make_llm()
    embedder = _make_embedder()

    # =========================================================================
    # Step 1 + 2: Combined extraction + normalisation (single LLM call)
    # Run in parallel with OR-group detection (pure regex, no LLM needed).
    # =========================================================================
    (
        (resume_skills_raw, jd_skills_raw, resume_mapping, jd_mapping),
        or_groups,
    ) = await asyncio.gather(
        _with_timeout(
            _extract_and_normalise_combined(resume_text, jd_text, llm),
            timeout_seconds=35,
            fallback=(
                list(extract_skills(resume_text)),
                list(extract_skills(jd_text)),
                {},
                {},
            ),
            label="combined extraction + normalisation",
        ),
        asyncio.to_thread(_extract_or_groups, jd_text),
    )

    print("RESUME SKILLS (raw):", sorted(resume_skills_raw))
    print("JD SKILLS     (raw):", sorted(jd_skills_raw))

    # =========================================================================
    # Step 3: Three-phase skill classification
    # =========================================================================
    matched, partial, missing = await classify_skills(
        resume_skills_raw,
        jd_skills_raw,
        resume_text,
        jd_text,
        embedder,
        llm,
        resume_mapping,
        jd_mapping,
    )

    # ── Step 4: OR-group resolution ───────────────────────────────────────
    missing, partial = _resolve_or_groups(missing, partial, matched, or_groups)

    print("MATCHED :", sorted(matched))
    print("PARTIAL :", sorted(partial))
    print("MISSING :", sorted(missing))

    # =========================================================================
    # Step 5: Skill score only
    # =========================================================================
    resume_skills_canonical = sorted(_apply_mapping(resume_skills_raw, resume_mapping))
    jd_skills_canonical     = sorted({
        jd_mapping.get(s.lower(), _basic_normalize(s)) for s in jd_skills_raw
    })

    total       = len(jd_skills_canonical) if jd_skills_canonical else 1
    skill_score = round((len(matched) + 0.4 * len(partial)) / total * 100, 1)

    resume_exp = _resolve_resume_experience(resume_text)
    jd_exp     = extract_experience(jd_text)

    # =========================================================================
    # Step 6: Analysis + shrink + experience score + question generation
    #         ALL four tasks run in parallel via asyncio.gather
    # =========================================================================
    resp_task = _build_analysis_chain(llm).ainvoke({
        "jd_text":        jd_text,
        "resume_text":    resume_text,
        "resume_skills":  resume_skills_canonical,
        "jd_skills":      jd_skills_canonical,
        "matched_skills": sorted(matched),
        "partial_skills": sorted(partial),
        "missing_skills": sorted(missing),
    })
    shrink_task = _build_shrink_chain(llm).ainvoke({
        "combined_text": f"{jd_text}\n{resume_text}"
    })
    exp_task = compute_experience_score_v2(resume_text, jd_text, llm=llm)
    question_task = _with_timeout(
        _generate_interview_questions(
            resume_text=resume_text,
            jd_text=jd_text,
            matched_skills=matched,
            partial_skills=partial,
            missing_skills=missing,
            llm=llm,
        ),
        timeout_seconds=25,
        fallback=[],
        label="interview question generation",
    )

    resp, shrinked_output, (exp_score, exp_breakdown), questions = await asyncio.gather(
        resp_task, shrink_task, exp_task, question_task
    )

    print("Shrink sentences:", shrinked_output.sentences)
    print(f"[EXP] score={exp_score}  breakdown={exp_breakdown}")
    print(f"[QUESTIONS] {len(questions)} generated")

    # final_score is 0–10 (compute_final_score divides by 10 internally)
    final_score = compute_final_score(skill_score, exp_score)

    # ── Step 7: Build response ────────────────────────────────────────────
    response = resp.model_dump()
    merged   = {**response["Evaluation"], **response["Grammar_Check"]}

    merged["JD_MatchScore"]           = f"{final_score}/10"
    merged["Skill_Score"]             = skill_score
    merged["Skill_Coverage"]          = f"{len(matched)}/{len(jd_skills_canonical)}"
    merged["Experience_Score"]        = exp_score
    merged["Experience_Breakdown"]    = exp_breakdown
    merged["Resume_Experience"]       = resume_exp
    merged["JD_Required_Experience"]  = jd_exp
    merged["Matched_Skills"]          = sorted(matched)
    merged["Partial_Skills"]          = sorted(partial)
    merged["Missing_Skills"]          = sorted(missing)
    merged["Extracted_Resume_Skills"] = resume_skills_canonical
    merged["Extracted_JD_Skills"]     = jd_skills_canonical

    # Hard validation — final correctness guarantee
    merged = _apply_hard_validation(merged, matched, partial, missing)

    merged["Grammatical_Errors"] = filter_grammar_errors(
        merged.get("Grammatical_Errors", []), resume_text
    )
    merged["Spelling_Mistakes"] = filter_spelling_errors(
        merged.get("Spelling_Mistakes", []), resume_text
    )
    merged["Client_Names"]        = await _extract_client_names_llm(resume_text, llm)
    merged["Suggested_Questions"] = questions

    # ── Step 8: Course suggestions (per-skill targeted retrieval) ─────────
    seen_courses:  set[str] = set()
    suggest_course: list    = []

    for skill in sorted(partial | missing)[:8]:   # cap at 8 skills
        results = suggester.suggest_courses(skill, top_k=3, filter_value='resource')
        for course in results:
            key = (course.get("course") or "")[:60]
            if key and key not in seen_courses:
                seen_courses.add(key)
                suggest_course.append(course)

    if not suggest_course:
        suggest_course = suggester.suggest_courses(
            " ".join(sorted(partial | missing)[:5]),
            top_k=5,
            filter_value='resource',
        )

    merged["Suggest_course"]  = suggest_course
    merged["Resume_Filename"] = (
        resume_info.get("filename", "analysis-result").rsplit('.', 1)[0]
    )

    json_merged = json.dumps(merged, indent=2)
    print(json_merged)

    # Cache result
    memory_store.setdefault("analysis_cache", {})[cache_key] = json_merged
    return json_merged


# =============================================================================
# Route: /process/analyze_jd/
# =============================================================================

@router.get("/process/analyze_jd/")
async def analyzejd():
    jd_store = memory_store.get("jd", {})
    jd_info  = jd_store['analyze_jd']
    jd_text  = extract_text(jd_info["bytes"], jd_info["filename"])

    llm             = ChatGroq(model="openai/gpt-oss-20b")
    pydantic_parser = PydanticOutputParser(pydantic_object=JDAnalysisResponse)
    fixing_parser   = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    prompt = PromptTemplate(
        input_variables=["jd_text", "format_instructions"],
        template="""
You are an HR Analyst AI assistant. Given the Job Description below:

1. Sanitize: remove sensitive info (names, emails, phone numbers).
2. Extract:
   - Must-have skills (all)
   - Good-to-have skills (all)
   - Location
   - Duration
   - Experience: all requirements, semicolon-separated. "Not specified" if absent.

Return JSON using this schema:
{format_instructions}

--- JOB DESCRIPTION ---
{jd_text}
""",
    )

    chain = (
        prompt.partial(format_instructions=fixing_parser.get_format_instructions())
        | llm
        | fixing_parser
    )

    result = await chain.ainvoke({"jd_text": jd_text})
    print(result.dict())
    return result.dict()
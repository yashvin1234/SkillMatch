# Plan: New Endpoint `/process/jd_resume_match_llm`

The new function mirrors the existing `process` (`/process/jd_resume_match`) exactly,
except **skill extraction is delegated to the LLM** instead of the rule-based `extract_skills()`.
Everything downstream (scoring, questions, courses, grammar check) stays the same.

---

## Step 1 — Add a new Pydantic schema for LLM skill extraction

**File:** `SkillMatchBE/app/schemas/schemas.py`

Add a `SkillExtractionResponse` model:

```python
class SkillExtractionResponse(BaseModel):
    resume_skills: List[str]
    jd_skills: List[str]
```

This is the structured output target for the skill-extraction chain.

---

## Step 2 — Build a dedicated LLM skill-extraction chain

Before the main analysis chain runs, add a new chain that:

- Takes `resume_text` and `jd_text` as inputs
- Asks the LLM to return two lists: skills found in the resume, and skills required by the JD
- Uses `PydanticOutputParser(pydantic_object=SkillExtractionResponse)` + `OutputFixingParser` for reliability
- Runs via `chain.ainvoke(...)` and returns a `SkillExtractionResponse`

The prompt will instruct the LLM to:
- Extract only **explicit, concrete skills** (tools, frameworks, languages, platforms)
- Not infer skills that are not written in the text
- Return normalized lowercase skill names

---

## Step 3 — Convert LLM skill lists to `set` and apply `normalize_skills`

After getting `SkillExtractionResponse`, convert both lists to sets via the existing
`normalize_skills()` helper already present in `process.py`:

```python
resume_skills = normalize_skills(skill_extraction.resume_skills)
jd_skills = normalize_skills(skill_extraction.jd_skills)
```

This ensures `compute_match_score_v2` receives the same `set` type as the existing endpoint.

---

## Step 4 — Execution order

Run the skill extraction chain **sequentially first**, since the main analysis chain
needs the skill lists as input. Then run the main analysis chain and shrink chain
**in parallel** (as today with `asyncio.gather`).

```
1. await skill_extraction_chain   →  resume_skills, jd_skills (sets)
2. asyncio.gather(main_chain, shrink_chain)
3. rest of pipeline unchanged
```

---

## Step 5 — Feed LLM-extracted skills into the unchanged downstream pipeline

Replace the two `extract_skills()` calls in the existing flow:

```python
# BEFORE (rule-based)
resume_skills = extract_skills(resume_text)
jd_skills = extract_skills(jd_text)

# AFTER (LLM-based)
resume_skills = normalize_skills(skill_extraction.resume_skills)
jd_skills = normalize_skills(skill_extraction.jd_skills)
```

Everything from `compute_match_score_v2` onward remains **identical**.

---

## Step 6 — Register the new route

Add the new handler as:

```
GET /process/jd_resume_match_llm
```

No changes to memory store keys, upload flow, or any other existing endpoints.

---

## Summary of changes

| File | Change |
|---|---|
| `SkillMatchBE/app/schemas/schemas.py` | Add `SkillExtractionResponse` model |
| `SkillMatchBE/app/api/process.py` | Add new `jd_resume_match_llm` endpoint function |

No changes to `skill_engine.py`, `text_extract.py`, or any other file.

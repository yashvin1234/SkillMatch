from fastapi import APIRouter, HTTPException, Request, Depends
from app.utils.text_extract import (
    extract_client_names_advanced,
    extract_text,
    filter_spelling_errors,
    filter_grammar_errors,
    format_score
)
from app.config import memory_store
from app.schemas.schemas import ResumeAnalysisResponse, JDAnalysisResponse, ShrinkSummaryResponse, SkillExtractionResponse
import json
import ast
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables.base import RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re
from app.utils.skill_engine import (
    extract_skills,
    compute_match_score_v2,
    extract_experience,
    compute_experience_score,
    compute_final_score
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def get_question_suggester(request: Request):
    return request.app.state.question_suggester


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_suggested_questions(raw_content):
    try:
        arr = json.loads(raw_content)
        if isinstance(arr, list):
            return [str(i) for i in arr]
    except Exception:
        pass

    try:
        arr = ast.literal_eval(raw_content)
        if isinstance(arr, list):
            return [str(i) for i in arr]
    except Exception:
        pass

    return [
        s.strip().strip('"').strip("'")
        for s in raw_content.replace('[', '').replace(']', '').split('\n')
        if s.strip()
    ]


def normalize_skill(skill):
    skill = skill.lower()
    skill = re.sub(r'\(.*?\)', '', skill)        # remove brackets
    skill = skill.replace("programming", "").strip()
    skill = re.sub(r'[^a-z0-9+#\. ]', '', skill) # remove special chars
    return skill.strip()


def normalize_skills(skill_list):
    return set(normalize_skill(s) for s in skill_list if s)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

async def _extract_skills_llm(resume_text: str, jd_text: str, llm) -> tuple[set, set]:
    """Extract skills from resume and JD using the LLM. Falls back to rule-based extraction on parse failure."""
    pydantic_parser = PydanticOutputParser(pydantic_object=SkillExtractionResponse)
    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    user_content = f"""You are a skill extraction engine. Extract skills explicitly mentioned or clearly implied in the Resume and Job Description below.

### Rules
1. Extract ONLY skills present in the text — do NOT invent or infer skills not mentioned.
2. Each skill must be concise: 3 words maximum.
3. Normalize equivalent phrasings to a single canonical form:
   - "stakeholder communication" / "working with stakeholders" → "Stakeholder Management"
   - "ML" / "machine learning" → "Machine Learning"
4. Assign each skill to exactly one category — no duplicates across categories.
5. Populate both `resume_skills` and `jd_skills` independently from their respective texts.

### Categories (use these exact names)
- Technical Skills
- GenAI / AI Skills
- Data Skills
- Product / Business Skills
- Agile / Process Skills
- Soft Skills

### Output format (return ONLY this JSON — no explanation, no markdown, no preamble)
{fixing_parser.get_format_instructions()}

### Input
--- RESUME ---
{resume_text}

--- JOB DESCRIPTION ---
{jd_text}"""

    messages = [
        SystemMessage(content="You are a precise skill extraction engine. Return only valid JSON."),
        HumanMessage(content=user_content),
    ]

    print("*****************PROMPT*****************")
    print(user_content)

    raw = await llm.bind(temperature=0).ainvoke(messages)
    print("***************** RAW LLM SKILL RESPONSE ************")
    print(raw)

    try:
        parsed = fixing_parser.parse(raw.content)
        print("***************** PARSED RESUME SKILLS (pre-normalize) ************")
        print(parsed.resume_skills)
        print("***************** PARSED JD SKILLS (pre-normalize) ************")
        print(parsed.jd_skills)
    except Exception as e:
        print(f"[WARN] LLM skill extraction parse error ({e}), falling back to extract_skills()")
        return extract_skills(resume_text), extract_skills(jd_text)

    if not parsed.resume_skills and not parsed.jd_skills:
        print("[WARN] LLM returned empty skill lists, falling back to extract_skills()")
        return extract_skills(resume_text), extract_skills(jd_text)

    return normalize_skills(parsed.resume_skills), normalize_skills(parsed.jd_skills)


def _build_analysis_chain(llm):
    pydantic_parser = PydanticOutputParser(pydantic_object=ResumeAnalysisResponse)
    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert recruiter, resume strategist, and proofreader."),
        ("user", """You will return JSON matching this schema:
        {format_instructions}

        CRITICAL INSTRUCTIONS (MUST FOLLOW STRICTLY):
        1. You are NOT allowed to infer skills.
        2. You MUST ONLY use the provided skill lists.
        3. You MUST compute:
           Key_Matches = intersection of Resume Skills and JD Skills
           Key_Gaps    = JD Skills - Resume Skills
        4. DO NOT contradict the provided skills.
        5. DO NOT say a skill is missing if it exists in Resume Skills.
        6. Your explanation MUST align with the computed matches/gaps.

        --- JOB DESCRIPTION ---
        {jd_text}
        --- RESUME ---
        {resume_text}
        --- EXTRACTED JD SKILLS ---
        {jd_skills}
        --- EXTRACTED RESUME SKILLS ---
        {resume_skills}
        """)
    ]).partial(format_instructions=fixing_parser.get_format_instructions())

    chain = (
        RunnableMap({
            "jd_text": lambda x: x["jd_text"],
            "resume_text": lambda x: x["resume_text"],
            "jd_skills": lambda x: x["jd_skills"],
            "resume_skills": lambda x: x["resume_skills"],
        })
        | prompt
        | llm
        | fixing_parser
    )
    return chain


def _build_shrink_chain(llm):
    pydantic_parser = PydanticOutputParser(pydantic_object=ShrinkSummaryResponse)
    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical recruiter and resume summarization assistant."),
        ("user", """You will return JSON matching this schema:
        {format_instructions_shrink}

        Extract a concise, semantically rich summary from the combined text.
        Focus on: key technologies and tools, core technical and soft skills, relevant domains or frameworks.
        Write 4–6 short, clear sentences — one key aspect per sentence.

        --- COMBINED TEXT ---
        {combined_text}
        """)
    ]).partial(format_instructions_shrink=fixing_parser.get_format_instructions())

    chain = (
        RunnableMap({"combined_text": lambda x: x["combined_text"]})
        | prompt
        | llm
        | fixing_parser
    )
    return chain


def _build_question_reframe_chain(llm):
    prompt = PromptTemplate(
        input_variables=["suggested_questions"],
        template="""
        You are an expert recruiter, career strategist, and English language specialist.
        You are given a list of raw interview questions retrieved from a database. These questions may be incomplete, repetitive, unpolished, or poorly worded.

        Your task is to:
        - Analyze and refine the list.
        - Rephrase the questions using clear, professional, and grammatically correct language.
        - Remove duplicates or near-duplicates.
        - Ensure the final set is well-structured and suitable for sharing directly with a candidate.
        - Make sure the questions collectively cover all key skills and topics represented in the input.

        Return ONLY a valid JSON array of strings, for example:
        ["What is your experience with FastAPI?", "How do you secure REST APIs?", "Describe your approach to CI/CD pipelines."]

        Do NOT use Python list syntax.
        Do NOT include explanations, prefixes, or suffixes—only the JSON array.
        Do NOT include metadata, labels, or extra formatting.

        suggested_questions:
        {suggested_questions}
        """
    )
    return prompt | llm


def _resolve_resume_experience(resume_text: str) -> int:
    exp = extract_experience(resume_text)
    if exp == 0:
        text_lower = resume_text.lower()
        if "senior" in text_lower:
            exp = 5
        elif "engineer" in text_lower:
            exp = 3
        elif len(resume_text.split()) > 800:
            exp = 3
        else:
            exp = 2
    return exp


def extract_json_array(text: str):
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return []
    return []
async def _extract_client_names_llm(resume_text: str, llm) -> list[str]:
    # 5. If none are found, return an empty array: []
    """Extract client/company names from the resume using the LLM. Falls back to rule-based on failure."""
    messages = [
        SystemMessage(content="You are an expert resume analyst. Extract only company or client names that are explicitly mentioned in the resume text."),
        HumanMessage(content=f"""
            Analyze the resume and extract client/company names.
            
            STRICT RULES:
            - Return ONLY a valid JSON array.
            - Do NOT include explanations, plans, or markdown.
            - Do NOT wrap in ```json```
            - Do NOT include tools, technologies, or products (e.g., AWS, Oracle, Jira).
            - Only include actual company or client organizations.
            
            Example:
            ["Accenture", "JPMorgan Chase"]
            
            --- RESUME ---
            {resume_text}
            """)
                ]

    try:
        raw = await llm.ainvoke(messages)
        content = raw.content.strip()
        print("********************CLIENT NAME****************")
        print(content)
        # Strip markdown code fences if present
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL).strip()
        names = extract_json_array(content)
        if isinstance(names, list) and names:
            return [str(n).strip() for n in names if str(n).strip()]
        else:
            return []
    except Exception as e:
        print(f"[WARN] LLM client name extraction failed ({e}), falling back to rule-based")
        return []
    


def _apply_hard_validation(merged: dict, matched: set, missing: set) -> dict:
    if "Key_Matches" in merged:
        merged["Key_Matches"] = [
            item for item in merged["Key_Matches"]
            if any(skill.lower() in item.lower() for skill in matched)
        ]

    if "Key_Gaps" in merged:
        merged["Key_Gaps"] = [
            item for item in merged["Key_Gaps"]
            if any(skill.lower() in item.lower() for skill in missing)
        ]

    if "Score_Explanation_Technical" in merged:
        explanation = merged["Score_Explanation_Technical"]
        for skill in matched:
            explanation = re.sub(
                rf"(?i)(no|lack of|lacks).*{skill}",
                f"experience present with {skill}",
                explanation
            )
        for skill in missing:
            if skill.lower() not in explanation.lower():
                explanation += f" Missing exposure to {skill}."
        merged["Score_Explanation_Technical"] = explanation

    return merged


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/process/jd_resume_match")
async def process(suggester=Depends(get_question_suggester)):
    if "resume" not in memory_store or "jd" not in memory_store:
        raise HTTPException(status_code=400, detail="Both resume and job description files must be uploaded first.")

    resume_info = memory_store["resume"]
    jd_info = memory_store["jd"]["jd_resume_match"]

    resume_text = extract_text(resume_info["bytes"], resume_info["filename"])
    jd_text = extract_text(jd_info["bytes"], jd_info["filename"])
    print("***************** RESUME TEXT ***************")
    print(resume_text)
    print("***************** JD TEXT ***************")
    print(jd_text)

    # ---------------- SKILL EXTRACTION (LLM) ----------------
    # llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    resume_skills, jd_skills = await _extract_skills_llm(resume_text, jd_text, llm)
    print("***************** RESUME SKILLS ************")
    print(resume_skills)
    print("***************** JD SKILLS ************")
    print(jd_skills)

    # ---------------- MAIN ANALYSIS + SHRINK (parallel) ----------------
    resp_task = _build_analysis_chain(llm).ainvoke({
        "jd_text": jd_text,
        "resume_text": resume_text,
        "resume_skills": list(resume_skills),
        "jd_skills": list(jd_skills),
    })
    shrink_task = _build_shrink_chain(llm).ainvoke({
        "combined_text": f"{jd_text}\n{resume_text}"
    })
    resp, shrinked_output = await asyncio.gather(resp_task, shrink_task)
    print("shrinked output:", shrinked_output.sentences)

    # ---------------- QUESTION SUGGESTION + REFRAMING ----------------
    suggested_questions = list(set(
        q
        for query in shrinked_output.sentences
        for q in suggester.suggest_questions(query, top_k=20)
    ))
    print("suggested questions:", suggested_questions)

    reframed_raw = await _build_question_reframe_chain(llm).ainvoke({
        "suggested_questions": suggested_questions
    })
    questions = normalize_suggested_questions(reframed_raw.content) or suggested_questions[:10]

    # ---------------- SCORING ----------------
    skill_score, matched, missing = await compute_match_score_v2(resume_skills, jd_skills, llm)
    resume_exp = _resolve_resume_experience(resume_text)
    jd_exp = extract_experience(jd_text)
    exp_score = compute_experience_score(resume_exp, jd_exp)
    final_score = compute_final_score(skill_score, exp_score)

    # ---------------- BUILD RESPONSE ----------------
    response = resp.model_dump()
    merged = {**response["Evaluation"], **response["Grammar_Check"]}

    merged["JD_MatchScore"] = format_score(final_score)
    merged["Skill_Score"] = skill_score
    merged["Skill_Coverage"] = f"{len(matched)}/{len(jd_skills)}"
    merged["Experience_Score"] = exp_score
    merged["Resume_Experience"] = resume_exp
    merged["JD_Required_Experience"] = jd_exp
    merged["Matched_Skills"] = matched
    merged["Missing_Skills"] = missing

    # Fallbacks (only if LLM left fields empty)
    if not merged.get("Key_Matches"):
        merged["Key_Matches"] = matched
    if not merged.get("Key_Gaps"):
        merged["Key_Gaps"] = [f"Missing experience in {skill}" for skill in missing]
    if not merged.get("Recommendations"):
        merged["Recommendations"] = [f"Improve experience in {skill}" for skill in missing]

    merged = _apply_hard_validation(merged, matched, missing)

    merged["Extracted_Resume_Skills"] = list(resume_skills)
    merged["Extracted_JD_Skills"] = list(jd_skills)
    merged["Grammatical_Errors"] = filter_grammar_errors(merged.get("Grammatical_Errors", []), resume_text)
    merged["Spelling_Mistakes"] = filter_spelling_errors(merged.get("Spelling_Mistakes", []), resume_text)
    merged["Client_Names"] = await _extract_client_names_llm(resume_text, llm)
    merged["Suggested_Questions"] = questions

    # ---------------- COURSE SUGGESTIONS ----------------
    key_gaps_str = " ".join(merged.get("Key_Gaps") or [])
    suggest_course = suggester.suggest_courses(key_gaps_str, top_k=20, filter_value='resource')
    if not suggest_course:
        suggest_course = suggester.suggest_courses(" ".join(missing), top_k=5, filter_value='resource')
    merged["Suggest_course"] = suggest_course

    merged["Resume_Filename"] = resume_info.get("filename", "analysis-result").rsplit('.', 1)[0]

    json_merged = json.dumps(merged, indent=2)
    print(json_merged)
    return json_merged


@router.get("/process/analyze_jd/")
async def analyzejd():
    jd_store = memory_store.get("jd", {})
    jd_info = jd_store['analyze_jd']
    jd_text = extract_text(jd_info["bytes"], jd_info["filename"])

    llm = ChatGroq(model="openai/gpt-oss-20b")
    pydantic_parser = PydanticOutputParser(pydantic_object=JDAnalysisResponse)
    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    prompt = PromptTemplate(
        input_variables=["jd_text", "format_instructions"],
        template="""
            You are an HR Analyst AI assistant. Given the following Job Description (JD), perform the following tasks:

            1. Sanitize the JD: Remove any sensitive or customer-identifiable info (like client names, company names, emails, phone numbers).
            2. Extract:
            - Must-have skills (3-5)
            - Good-to-have skills (2-3)
            - Location
            - Duration
            - Experience: Extract and return all experience-related requirements or statements found in the JD, including ranges, role-specific requirements (e.g., "2+ years with LLMs" and "8+ years Python"). If more than one experience is mentioned, include each in a single text field separated by semicolons. If absent, return "Not specified".

            Respond in JSON using this schema:
            {format_instructions}

            --- JOB DESCRIPTION ---
            {jd_text}
            """
    )

    chain = (
        prompt.partial(format_instructions=fixing_parser.get_format_instructions())
        | llm
        | fixing_parser
    )

    result = await chain.ainvoke({"jd_text": jd_text})
    print(result.dict())
    return result.dict()

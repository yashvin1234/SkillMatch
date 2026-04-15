from fastapi import APIRouter, HTTPException, Request, Depends
from app.utils.text_extract import (
    extract_client_names_advanced,
    extract_text,
    filter_spelling_errors,
    filter_grammar_errors,
    format_score
)
from app.config import memory_store
from app.schemas.schemas import ResumeAnalysisResponse,JDAnalysisResponse,ShrinkSummaryResponse
import json
import ast
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables.base import RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import re
from app.utils.skill_engine import (
    extract_skills,
    compute_match_score_v2,
    extract_experience,
    compute_experience_score,
    compute_final_score
)

router = APIRouter()


def get_question_suggester(request: Request):
    return request.app.state.question_suggester

def normalize_suggested_questions(raw_content):
    """Ensure suggested questions are always a list of strings (from LLM or fallback)."""
    try:
        arr = json.loads(raw_content)
        if isinstance(arr, list):
            return [str(i) for i in arr]
    except Exception: pass

    try:
        arr = ast.literal_eval(raw_content)
        if isinstance(arr, list):
            return [str(i) for i in arr]
    except Exception: pass

    return [
        s.strip().strip('"').strip("'")
        for s in raw_content.replace('[', '').replace(']', '').split('\n')
        if s.strip()
    ]

def normalize_skill(skill):
    skill = skill.lower()

    # remove brackets
    skill = re.sub(r'\(.*?\)', '', skill)

    # remove extra words
    skill = skill.replace("programming", "").strip()

    # remove special chars
    skill = re.sub(r'[^a-z0-9+#\. ]', '', skill)

    return skill.strip()

def normalize_skills(skill_list):
    return set(normalize_skill(s) for s in skill_list if s)

@router.get("/process/jd_resume_match")
async def process(suggester=Depends(get_question_suggester)):
    if "resume" not in memory_store or "jd" not in memory_store:
        raise HTTPException(status_code=400, detail="Both resume and job description files must be uploaded first.")

    resume_info = memory_store["resume"]
    jd_store = memory_store.get("jd", {})
    jd_info = jd_store['jd_resume_match']

    resume_text = extract_text(resume_info["bytes"], resume_info["filename"])
    jd_text = extract_text(jd_info["bytes"], jd_info["filename"]) 

    # ---------------- SKILL EXTRACTION ----------------
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0.1)
    pydantic_parser = PydanticOutputParser(pydantic_object=ResumeAnalysisResponse)
    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert recruiter, resume strategist, and proofreader."),
        ("user", """You will return JSON matching this schema:
        {format_instructions}

        IMPORTANT:
        CRITICAL INSTRUCTIONS (MUST FOLLOW STRICTLY):

        1. You are NOT allowed to infer skills.
        2. You MUST ONLY use the provided skill lists.
        3. You MUST compute:

        Key_Matches = intersection of Resume Skills and JD Skills  
        Key_Gaps = JD Skills - Resume Skills  

        4. DO NOT contradict the provided skills.
        5. DO NOT say a skill is missing if it exists in Resume Skills.
        6. Your explanation MUST align with the computed matches/gaps.

        Then, analyze:
        --- JOB DESCRIPTION ---
        {jd_text}
        --- RESUME ---
        {resume_text}
        --- EXTRACTED JD SKILLS ---
        {jd_skills}
        --- EXTRACTED RESUME SKILLS ---
        {resume_skills}
        """)
         ])

    format_instructions = fixing_parser.get_format_instructions()

    prompt_with_instructions = prompt.partial(format_instructions=format_instructions)

    chain = (
        RunnableMap({
            "jd_text": lambda x: x["jd_text"],
            "resume_text": lambda x: x["resume_text"],
            "jd_skills": lambda x: x["jd_skills"],
            "resume_skills": lambda x: x["resume_skills"],
        })
        | prompt_with_instructions
        | llm
        | fixing_parser
    )

    resp_task = chain.ainvoke({
        "jd_text": jd_text,
        "resume_text": resume_text,
        "resume_skills": list(resume_skills),
        "jd_skills": list(jd_skills)
    })

    pydantic_parser_shrink = PydanticOutputParser(pydantic_object=ShrinkSummaryResponse)
    fixing_parser_shrink = OutputFixingParser.from_llm(parser=pydantic_parser_shrink, llm=llm)

    prompt_shrink = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical recruiter and resume summarization assistant."),
        ("user", """You will return JSON matching this schema:
    {format_instructions_shrink}

    Your task is to extract and generate a concise, semantically rich summary from a combined job description and resume.

    Focus only on:
    - Key technologies and tools
    - Core technical and soft skills
    - Relevant domains or frameworks

    Write **multiple short, clear sentences** (ideally 4–6) instead of long ones.
    Each sentence should describe one key aspect or capability derived from the text.

    --- COMBINED TEXT ---
    {combined_text}
    """)
    ])

    format_instructions_shrink = fixing_parser_shrink.get_format_instructions()
    prompt_with_instructions_shrink = prompt_shrink.partial(format_instructions_shrink=format_instructions_shrink)
    shrink_chain = (
        RunnableMap({
            "combined_text": lambda x: x["combined_text"],
        })
        | prompt_with_instructions_shrink
        | llm
        | fixing_parser_shrink
    )

    combined_text = f"{jd_text}\n{resume_text}"
    shrink_task = shrink_chain.ainvoke({
        "combined_text": combined_text
    })

    resp, shrinked_output = await asyncio.gather(resp_task, shrink_task)
    print('shrinked output:',shrinked_output.sentences)
    # suggested_questions = [
    # q
    # for query in shrinked_output.sentences
    # for q in suggester.suggest_questions(query, top_k=20)
    # ]
    suggested_questions = list(set([
        q
        for query in shrinked_output.sentences
        for q in suggester.suggest_questions(query, top_k=20)
    ]))
    print('suggested questions:',suggested_questions)


    question_reframming_prompt = PromptTemplate(
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

    question_reframming = question_reframming_prompt | llm
    question_reframming_task = question_reframming.ainvoke({
        "suggested_questions": suggested_questions
    })

    reframmed_questions = await asyncio.gather(question_reframming_task)

    response = resp.model_dump()
    merged = {**response["Evaluation"], **response["Grammar_Check"]}

    grammar = merged.get("Grammatical_Errors", [])
    spelling = merged.get("Spelling_Mistakes", [])
    
    skill_score, matched, missing = compute_match_score_v2(resume_skills, jd_skills)

    # ---------------- EXPERIENCE EXTRACTION ----------------

    resume_exp = extract_experience(resume_text)
    if resume_exp == 0:
        text_lower = resume_text.lower()

        if "senior" in text_lower:
            resume_exp = 5
        elif "engineer" in text_lower:
            resume_exp = 3
        elif len(resume_text.split()) > 800:
            resume_exp = 3
        else:
            resume_exp = 2 # default to 2 years if no clear experience indicators are found

    jd_exp = extract_experience(jd_text)

    exp_score = compute_experience_score(resume_exp, jd_exp)

    # ---------------- FINAL SCORE ----------------

    final_score = compute_final_score(skill_score, exp_score)

    merged["JD_MatchScore"] = format_score(final_score)
    merged["Skill_Score"] = skill_score
    merged["Skill_Coverage"] = f"{len(matched)}/{len(jd_skills)}"
    merged["Experience_Score"] = exp_score
    merged["Resume_Experience"] = resume_exp
    merged["JD_Required_Experience"] = jd_exp
    merged["Matched_Skills"] = matched
    merged["Missing_Skills"] = missing

    # ---------------- FALLBACKS (ONLY IF LLM FAILS) ----------------
    if not merged.get("Key_Matches"):
        merged["Key_Matches"] = list(matched)

    if not merged.get("Key_Gaps"):
        merged["Key_Gaps"] = [f"Missing experience in {skill}" for skill in missing]

    if not merged.get("Recommendations"):
        merged["Recommendations"] = [
            f"Improve experience in {skill}" for skill in missing
        ]

    # ---------------- HARD VALIDATION ----------------

    # Fix Key_Matches → must be subset of matched
    if "Key_Matches" in merged:
        merged["Key_Matches"] = [
            item for item in merged["Key_Matches"]
            if any(skill.lower() in item.lower() for skill in matched)
        ]

    # Fix Key_Gaps → must be subset of missing
    if "Key_Gaps" in merged:
        merged["Key_Gaps"] = [
            item for item in merged["Key_Gaps"]
            if any(skill.lower() in item.lower() for skill in missing)
        ]

    if "Score_Explanation_Technical" in merged:
        explanation = merged["Score_Explanation_Technical"]

        # Remove wrong negatives
        for skill in matched:
            explanation = re.sub(
                rf"(?i)(no|lack of|lacks).*{skill}",
                f"experience present with {skill}",
                explanation
            )

        # Add missing skills explicitly if not mentioned
        for skill in missing:
            if skill.lower() not in explanation.lower():
                explanation += f" Missing exposure to {skill}."

        merged["Score_Explanation_Technical"] = explanation
    merged["Extracted_Resume_Skills"] = list(resume_skills)
    merged["Extracted_JD_Skills"] = list(jd_skills)
    merged["Grammatical_Errors"] = filter_grammar_errors(grammar, resume_text)
    merged["Spelling_Mistakes"] = filter_spelling_errors(spelling, resume_text)
    merged["Client_Names"] = extract_client_names_advanced(resume_text)

    questions = normalize_suggested_questions(reframmed_questions[0].content)
    if not questions:
        questions = suggested_questions[:10]
    merged["Suggested_Questions"] = questions

    key_gaps_list = merged.get("Key_Gaps") or []
    key_gaps_str = " ".join(key_gaps_list)

    suggest_course = suggester.suggest_courses(
        key_gaps_str,
        top_k=20,
        filter_value='resource'
    )

    # fallback if empty
    if not suggest_course:
        suggest_course = suggester.suggest_courses(
            " ".join(missing),
            top_k=5,
            filter_value='resource'
        )

    merged['Suggest_course'] = suggest_course

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

    format_instructions = fixing_parser.get_format_instructions()

    # Combine prompt and model
    chain = (
        prompt.partial(format_instructions=format_instructions)
        | llm
        | fixing_parser
    )

    result = await chain.ainvoke({"jd_text": jd_text})

    print (result.dict())

    return result.dict()
from fastapi import APIRouter, HTTPException, Request, Depends
from app.utils.text_extract import extract_text,filter_spelling_errors,filter_grammar_errors
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

@router.get("/process/jd_resume_match")
async def process(suggester=Depends(get_question_suggester)):
    if "resume" not in memory_store or "jd" not in memory_store:
        raise HTTPException(status_code=400, detail="Both resume and job description files must be uploaded first.")

    resume_info = memory_store["resume"]
    jd_store = memory_store.get("jd", {})
    jd_info = jd_store['jd_resume_match']

    resume_text = extract_text(resume_info["bytes"], resume_info["filename"])
    jd_text = extract_text(jd_info["bytes"], jd_info["filename"]) 

    llm = ChatGroq(model="openai/gpt-oss-20b")
    pydantic_parser = PydanticOutputParser(pydantic_object=ResumeAnalysisResponse)
    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert recruiter, resume strategist, and proofreader."),
        ("user", """You will return JSON matching this schema:
        {format_instructions}

        Then, analyze:
        --- JOB DESCRIPTION ---
        {jd_text}
        --- RESUME ---
        {resume_text}
        """)
         ])

    format_instructions = fixing_parser.get_format_instructions()

    prompt_with_instructions = prompt.partial(format_instructions=format_instructions)

    chain = (
        RunnableMap({
            "jd_text": lambda x: x["jd_text"],
            "resume_text": lambda x: x["resume_text"],
        })
        | prompt_with_instructions
        | llm
        | fixing_parser
    )

    resp_task = chain.ainvoke({
        "jd_text": jd_text,
        "resume_text": resume_text
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

    combined_text = f"{jd_text}" 
    shrink_task = shrink_chain.ainvoke({
        "combined_text": combined_text
    })

    resp, shrinked_output = await asyncio.gather(resp_task, shrink_task)
    print('shrinked output:',shrinked_output.sentences)
    suggested_questions = [
    q
    for query in shrinked_output.sentences
    for q in suggester.suggest_questions(query, top_k=20)
    ]
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

    merged["Grammatical_Errors"] = filter_grammar_errors(grammar, resume_text)
    merged["Spelling_Mistakes"] = filter_spelling_errors(spelling, resume_text)

    merged["Suggested_Questions"] = normalize_suggested_questions(reframmed_questions[0].content)

    key_gaps_clean = [str(item).replace("\n", " ") for item in merged.get("Key_Gaps", [])]
    key_gaps_str = " ".join(key_gaps_clean)

    suggest_course = suggester.suggest_courses(key_gaps_str, top_k=20,filter_value = 'resource')
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
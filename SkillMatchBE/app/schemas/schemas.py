from pydantic import BaseModel, Field
from typing import List

class ResumeReport(BaseModel):
    result: str

class EvaluationSection(BaseModel):
    JD_MatchScore: str = Field(
        description="A numerical score representing the match between the job description (JD) and the candidate's resume. The score should be out of 10, with 10 indicating a perfect match and 1 indicating very poor match. The score must be based on a detailed evaluation of how well the candidate's skills, experience, and qualifications align with the requirements outlined in the JD. Just providing a number without context is not acceptable. Example of output: '7/10 - Good match with key skills but lacks experience in some required technologies.'"
    )
    Score_Explanation_NonTechnical: str = Field(
        description="A short explanation (2-3 lines) for non-technical stakeholders that summarizes why the candidate received the given JD match score. Focus on high-level alignment with the job role, such as years of experience, industry fit, or major skill areas."
    )
    Score_Explanation_Technical: str = Field(
        description="A detailed explanation of how the JD match score was determined, written for a technical reviewer. This should include specific technical skills, tools, technologies, certifications, and project experience that justify the score. Mention both strengths and areas for improvement."
    )
    Key_Matches: List[str] = Field(
        default_factory=list,
        description="A list of key skills, experiences, or technologies from the job description (JD) that the candidate's resume successfully matches. Each item should be described briefly but precisely, indicating where and how it appears in the resume. For example, 'Python programming' or '5+ years in project management'."
    )
    Key_Gaps: List[str] = Field(
        default_factory=list,
        description="A list of missing or underrepresented qualifications, skills, or experiences from the resume that are explicitly required or mentioned in the job description. Each gap should be described clearly, such as 'No experience with JavaScript frameworks' or 'Missing leadership experience'."
    )
    Recommendations: List[str] = Field(
        default_factory=list,
        description="Suggestions for improving the resume to better match the job description. These may include skill additions, phrasing adjustments, or structural improvements. For example, 'Add relevant certifications like AWS Certified Solutions Architect' or 'Rephrase the summary to focus more on leadership experience'. Recommendations should be practical and actionable."
    )

class GrammarCheckSection(BaseModel):
    Grammatical_Errors: List[str] = Field(
        default_factory=list,
        description="A list of grammatical errors found in the resume. Each error should be identified with its location and a suggestion for correction."
    )
    Spelling_Mistakes: List[str] = Field(
        default_factory=list,
        description="A list of spelling mistakes found in the resume. Each mistake should include the incorrect word along with the suggested correction. For example, 'Teh -> The'."
    )
    Client_Names: List[str] = Field(
        default_factory=list,
        description="A list of client or company names identified in the resume."
    )

class ResumeAnalysisResponse(BaseModel):
    Evaluation: EvaluationSection
    Grammar_Check: GrammarCheckSection

class JDAnalysisResponse(BaseModel):
    sanitized_jd: str = Field(
        description="Job description text with sensitive or customer-identifiable information removed."
    )
    must_have_skills: List[str] = Field(
        description="List of all essential skills required for the job."
    )
    good_to_have_skills: List[str] = Field(
        description="List of all the optional or nice-to-have skills."
    )
    location: str = Field(
        description="Job location as mentioned in the JD (e.g., Remote, On-site, City name)."
    )
    duration: str = Field(
        description="Job duration in Months or years. Return 'No duration specified' if not available."
    )
    experience: str = Field(
        description="Experience requirements for the role, as stated in the JD (e.g., '2-5 years', '5+ years'). Return 'Not specified' if missing."
    )

class ShrinkSummaryResponse(BaseModel):
    sentences: List[str] = Field(
        ...,
        description="List of concise sentences summarizing key technologies, tools, skills, and domains."
    )
from fastapi import APIRouter, HTTPException, Request, Depends
from app.utils.text_extract import (
    extract_client_names_advanced,
    extract_text,
    filter_spelling_errors,
    filter_grammar_errors,
    format_score,
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
    compute_final_score,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def get_question_suggester(request: Request):
    return request.app.state.question_suggester


# ---------------------------------------------------------------------------
# LLM + Embedder factories  (single source, frozen settings)
# ---------------------------------------------------------------------------

def _make_llm() -> ChatOpenAI:
    """temperature=0 + seed=42 → maximum determinism across all chains."""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        model_kwargs={"seed": 42},
    )


def _make_embedder() -> OpenAIEmbeddings:
    """text-embedding-3-small: cheap, fast, accurate enough at 0.85 threshold."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


# ---------------------------------------------------------------------------
# Content-hash cache  (same resume+JD → identical result every time)
# ---------------------------------------------------------------------------

def _content_hash(resume_text: str, jd_text: str) -> str:
    return hashlib.md5(f"{resume_text}||{jd_text}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def normalize_suggested_questions(raw_content: str) -> list[str]:
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


def _clean_llm_json(raw_content: str) -> str:
    """Strip markdown fences and trailing commas before JSON parsing."""
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw_content.strip(), flags=re.DOTALL
    )
    content = re.sub(r',\s*([}\]])', r'\1', content)
    return content.strip()


# =============================================================================
# LAYER 1 — Skill normalisation
# =============================================================================

SKILL_ALIASES: dict[str, str] = {
    # ── Languages ──────────────────────────────────────────────────────────
    "js":                               "javascript",
    "ts":                               "typescript",
    "java script":                      "javascript",
    "type script":                      "typescript",
    "py":                               "python",
    "golang":                           "go",
    "golang language":                  "go",
    "c sharp":                          "c#",
    "dotnet":                           ".net",
    "dot net":                          ".net",
    ".net core":                        ".net",
    "asp.net":                          ".net",
    "asp.net core":                     ".net",
    "net framework":                    ".net",

    # ── AI / ML ────────────────────────────────────────────────────────────
    "ml":                               "machine learning",
    "ai":                               "artificial intelligence",
    "dl":                               "deep learning",
    "nlp":                              "natural language processing",
    "natural language proc":            "natural language processing",
    "ai/ml":                            "ai/ml development",
    "aiml":                             "ai/ml development",
    "ai ml":                            "ai/ml development",
    "machine learning development":     "ai/ml development",
    "ai development":                   "ai/ml development",
    "ml development":                   "ai/ml development",
    "ai/ml application development":    "ai/ml development",
    "ai/ml applications":               "ai/ml development",
    "artificial intelligence development": "ai/ml development",

    # ── LLMs ───────────────────────────────────────────────────────────────
    "llm":                              "large language models",
    "llms":                             "large language models",
    "large language model":             "large language models",

    # ── Generative AI ──────────────────────────────────────────────────────
    "gen ai":                           "generative ai",
    "genai":                            "generative ai",

    # ── Cloud ──────────────────────────────────────────────────────────────
    "aws lambda":                       "aws",
    "amazon web services":              "aws",
    "aws basics":                       "aws",
    "aws services":                     "aws",
    "amazon aws":                       "aws",
    "gcp":                              "google cloud platform",
    "google cloud":                     "google cloud platform",
    "google cloud services":            "google cloud platform",
    "azure devops":                     "azure",
    "microsoft azure":                  "azure",
    "ms azure":                         "azure",

    # ── Containers / orchestration ─────────────────────────────────────────
    "k8s":                              "kubernetes",
    "kube":                             "kubernetes",

    # ── Microservices ──────────────────────────────────────────────────────
    "micro services":                   "microservices",
    "micro-services":                   "microservices",
    "microservice":                     "microservices",
    "micro service":                    "microservices",
    "micro service architecture":       "microservices",
    "microservices architecture":       "microservices",

    # ── Event-driven ───────────────────────────────────────────────────────
    "event driven architecture":        "event-driven architecture",
    "event-driven arch":                "event-driven architecture",
    "eda":                              "event-driven architecture",

    # ── Messaging ──────────────────────────────────────────────────────────
    "apache kafka":                     "kafka",
    "kafka connect":                    "kafka",
    "kafka topic":                      "kafka",
    "kafka sink connector":             "kafka",
    "kafka streams":                    "kafka",
    "rabbit mq":                        "rabbitmq",
    "rabbit-mq":                        "rabbitmq",

    # ── CI/CD ──────────────────────────────────────────────────────────────
    "ci/cd":                            "cicd",
    "ci cd":                            "cicd",
    "gitlab cicd":                      "cicd",
    "gitlab ci/cd":                     "cicd",
    "github actions":                   "cicd",
    "ci/cd pipelines":                  "cicd",
    "continuous integration":           "cicd",
    "continuous deployment":            "cicd",
    "continuous delivery":              "cicd",

    # ── Databases ──────────────────────────────────────────────────────────
    "mongo":                            "mongodb",
    "postgres":                         "postgresql",
    "postgres sql":                     "postgresql",
    "psql":                             "postgresql",
    "big query":                        "bigquery",
    "ms sql":                           "mssql",
    "sql server":                       "mssql",
    "microsoft sql server":             "mssql",
    "dynamo db":                        "dynamodb",
    "dynamo":                           "dynamodb",
    "elastic search":                   "elasticsearch",
    "open search":                      "opensearch",

    # ── ML libs ────────────────────────────────────────────────────────────
    "tf":                               "tensorflow",
    "sklearn":                          "scikit-learn",
    "scikit learn":                     "scikit-learn",
    "sci-kit learn":                    "scikit-learn",
    "hugging face":                     "huggingface",
    "huggingface transformers":         "huggingface",

    # ── Node ───────────────────────────────────────────────────────────────
    "node":                             "node.js",
    "nodejs":                           "node.js",

    # ── Frontend ───────────────────────────────────────────────────────────
    "react.js":                         "react",
    "reactjs":                          "react",
    "react native":                     "react native",
    "vue.js":                           "vue",
    "vuejs":                            "vue",
    "angular.js":                       "angular",
    "angularjs":                        "angular",
    "next.js":                          "next.js",
    "nextjs":                           "next.js",
    "nuxt.js":                          "nuxt.js",
    "nuxtjs":                           "nuxt.js",

    # ── REST ───────────────────────────────────────────────────────────────
    "rest api":                         "rest apis",
    "rest":                             "rest apis",
    "restful":                          "rest apis",
    "restful web services":             "rest apis",
    "restful apis":                     "rest apis",
    "restful api":                      "rest apis",
    "rest services":                    "rest apis",

    # ── Security ───────────────────────────────────────────────────────────
    "spring security":                  "security",
    "application security":             "security",
    "cyber security":                   "cybersecurity",
    "information security":             "cybersecurity",
    "infosec":                          "cybersecurity",

    # ── Monitoring ─────────────────────────────────────────────────────────
    "cloud watch":                      "cloudwatch",
    "aws cloudwatch":                   "cloudwatch",

    # ── OOP ────────────────────────────────────────────────────────────────
    "oop":                              "object oriented programming",
    "object oriented":                  "object oriented programming",
    "oops":                             "object oriented programming",

    # ── Spring ─────────────────────────────────────────────────────────────
    "spring boot":                      "spring boot",
    "springboot":                       "spring boot",
    "jpa":                              "jpa",
    "junit":                            "unit testing",
    "junit testing":                    "unit testing",
    "junit5":                           "unit testing",

    # ── Data engineering ───────────────────────────────────────────────────
    "apache spark":                     "spark",
    "pyspark":                          "spark",
    "apache airflow":                   "airflow",
    "apache flink":                     "flink",
    "apache hive":                      "hive",
    "apache hadoop":                    "hadoop",
    "data pipeline":                    "data pipelines",
    "etl pipeline":                     "etl",
    "extract transform load":           "etl",

    # ── Mobile ─────────────────────────────────────────────────────────────
    "ios development":                  "ios",
    "android development":              "android",
    "flutter development":              "flutter",
    "react-native":                     "react native",

    # ── DevOps / Infra ─────────────────────────────────────────────────────
    "infrastructure as code":           "iac",
    "infra as code":                    "iac",
    "terraform iac":                    "terraform",
    "ansible automation":               "ansible",

    # ── Version control ────────────────────────────────────────────────────
    "github":                           "git",
    "gitlab":                           "git",
    "bitbucket":                        "git",
    "source control":                   "git",
    "version control":                  "git",

    # ── Finance / Accounting ───────────────────────────────────────────────
    "p&l":                              "profit and loss",
    "pl management":                    "profit and loss",
    "profit & loss":                    "profit and loss",
    "p&l management":                   "profit and loss",
    "p&l ownership":                    "profit and loss",
    "p&l reporting":                    "profit and loss",
    "profit loss":                      "profit and loss",
    "fp&a":                             "financial planning and analysis",
    "financial planning & analysis":    "financial planning and analysis",
    "gaap":                             "accounting standards",
    "ifrs":                             "accounting standards",
    "accounts receivable":              "ar/ap",
    "accounts payable":                 "ar/ap",
    "kpis":                             "kpi management",
    "kpi":                              "kpi management",
    "roi analysis":                     "roi",
    "return on investment":             "roi",
    "financial modelling":              "financial modeling",
    "variance analysis":                "financial analysis",
    "management reporting":             "financial reporting",
    "mis reporting":                    "financial reporting",
    "board reporting":                  "financial reporting",
    "management accounts":              "financial reporting",
    "financial statements":             "financial reporting",
    "cash flow":                        "financial management",
    "cash flow management":             "financial management",
    "working capital":                  "financial management",
    "treasury":                         "financial management",
    "cost analysis":                    "financial analysis",
    "cost management":                  "financial analysis",
    "capex":                            "financial planning and analysis",
    "opex":                             "financial planning and analysis",
    "budget vs actuals":                "budgeting",
    "annual budgeting":                 "budgeting",
    "quarterly forecasting":            "forecasting",
    "balance sheet":                    "accounting",
    "income statement":                 "accounting",
    "bookkeeping":                      "accounting",
    "accounts":                         "accounting",
    "tally":                            "accounting software",
    "quickbooks":                       "accounting software",
    "zoho books":                       "accounting software",
    "xero":                             "accounting software",
    "sage":                             "accounting software",
    "sap fico":                         "sap",
    "sap fi":                           "sap",
    "sap co":                           "sap",
    "sap s/4hana":                      "sap",
    "tax compliance":                   "compliance",
    "gst":                              "tax compliance",
    "tds":                              "tax compliance",
    "statutory compliance":             "compliance",

    # ── HR / People ────────────────────────────────────────────────────────
    "talent acquisition":               "recruitment",
    "talent management":                "talent development",
    "performance management":           "performance reviews",
    "performance appraisal":            "performance reviews",
    "appraisal":                        "performance reviews",
    "goal setting":                     "performance management",
    "okrs":                             "performance management",
    "kras":                             "performance management",
    "360 feedback":                     "performance management",
    "360-degree feedback":              "performance management",
    "hris":                             "hr information systems",
    "human resource information system": "hr information systems",
    "workday":                          "hr information systems",
    "sap successfactors":               "hr information systems",
    "successfactors":                   "hr information systems",
    "zoho people":                      "hr information systems",
    "bamboohr":                         "hr information systems",
    "darwinbox":                        "hr information systems",
    "greythr":                          "hr information systems",
    "keka":                             "hr information systems",
    "peoplesoft":                       "hr information systems",
    "oracle hcm":                       "hr information systems",
    "dei":                              "diversity and inclusion",
    "diversity & inclusion":            "diversity and inclusion",
    "d&i":                              "diversity and inclusion",
    "equity and inclusion":             "diversity and inclusion",
    "l&d":                              "learning and development",
    "learning & development":           "learning and development",
    "training and development":         "learning and development",
    "training & development":           "learning and development",
    "onboarding":                       "employee onboarding",
    "induction":                        "employee onboarding",
    "joining formalities":              "employee onboarding",
    "new hire orientation":             "employee onboarding",
    "employee engagement":              "employee engagement",
    "culture building":                 "employee engagement",
    "employer branding":                "employee engagement",
    "esat":                             "employee engagement",
    "pulse surveys":                    "employee engagement",
    "stay interviews":                  "employee engagement",
    "workforce planning":               "workforce planning",
    "headcount planning":               "workforce planning",
    "manpower planning":                "workforce planning",
    "succession planning":              "succession planning",
    "compensation & benefits":          "compensation and benefits",
    "comp & ben":                       "compensation and benefits",
    "ctc structuring":                  "compensation and benefits",
    "salary benchmarking":              "compensation and benefits",
    "payroll":                          "payroll management",
    "payroll processing":               "payroll management",
    "payroll coordination":             "payroll management",
    "conflict resolution":              "employee relations",
    "employee relations":               "employee relations",
    "exit interviews":                  "offboarding",
    "attrition":                        "retention",
    "employee attrition":               "retention",
    "employee retention":               "retention",
    "talent pipeline":                  "recruitment",
    "sourcing":                         "recruitment",
    "screening":                        "recruitment",
    "talent sourcing":                  "recruitment",
    "background verification":          "recruitment",
    "bgv":                              "recruitment",
    "offer management":                 "recruitment",
    "full cycle recruiting":            "recruitment",
    "full-cycle recruiting":            "recruitment",
    "mass hiring":                      "recruitment",
    "bulk hiring":                      "recruitment",
    "campus hiring":                    "recruitment",
    "lateral hiring":                   "recruitment",
    "headhunting":                      "recruitment",
    "naukri":                           "recruitment",
    "naukri rms":                       "recruitment",
    "iimjobs":                          "recruitment",
    "linkedin recruiter":               "recruitment",
    "ats":                              "applicant tracking system",
    "applicant tracking":               "applicant tracking system",
    "employee lifecycle":               "hr operations",
    "hr policies":                      "hr operations",
    "policy implementation":            "hr operations",
    "hr policy":                        "hr operations",
    "employee handbook":                "hr operations",
    "hr operations":                    "hr operations",
    "hrbp":                             "hr business partner",
    "hr business partnering":           "hr business partner",
    "strategic hr":                     "hr strategies",
    "hr strategy":                      "hr strategies",
    "hr transformation":                "hr strategies",
    "people analytics":                 "hr analytics",
    "workforce analytics":              "hr analytics",
    "hr dashboard":                     "hr analytics",
    "hr reporting":                     "hr analytics",
    "hr metrics":                       "hr analytics",
    "organizational development":       "change management",
    "organisational development":       "change management",
    "od":                               "change management",
    "shrm":                             "hr certification",
    "shrm-scp":                         "hr certification",
    "shrm-cp":                          "hr certification",
    "cipd":                             "hr certification",
    "phr":                              "hr certification",
    "sphr":                             "hr certification",
    "chrp":                             "hr certification",
    "prosci":                           "change management certification",
    "change management certification":  "change management certification",

    # ── Sales / Marketing ──────────────────────────────────────────────────
    "b2b sales":                        "b2b",
    "b2c sales":                        "b2c",
    "crm tools":                        "crm",
    "salesforce crm":                   "salesforce",
    "hubspot":                          "crm",
    "hubspot crm":                      "crm",
    "zoho crm":                         "crm",
    "ms dynamics":                      "crm",
    "microsoft dynamics":               "crm",
    "dynamics 365":                     "crm",
    "go-to-market":                     "gtm strategy",
    "go to market":                     "gtm strategy",
    "gtm":                              "gtm strategy",
    "demand generation":                "demand gen",
    "lead generation":                  "demand gen",
    "lead nurturing":                   "crm",
    "pipeline management":              "sales",
    "quota attainment":                 "sales",
    "revenue generation":               "sales",
    "cold calling":                     "sales",
    "prospecting":                      "sales",
    "upselling":                        "account management",
    "cross-selling":                    "account management",
    "key account management":           "account management",
    "client retention":                 "account management",
    "seo/sem":                          "seo",
    "search engine optimisation":       "seo",
    "search engine optimization":       "seo",
    "google ads":                       "paid advertising",
    "facebook ads":                     "paid advertising",
    "meta ads":                         "paid advertising",
    "ppc":                              "paid advertising",
    "pay per click":                    "paid advertising",
    "paid media":                       "paid advertising",
    "google analytics":                 "marketing analytics",
    "account based marketing":          "abm",
    "content strategy":                 "content marketing",
    "content creation":                 "content marketing",
    "copywriting":                      "content marketing",
    "blog writing":                     "content marketing",
    "brand management":                 "brand strategy",
    "social media marketing":           "social media",
    "instagram marketing":              "social media",
    "linkedin marketing":               "social media",
    "email marketing":                  "email campaigns",
    "market research":                  "marketing",
    "competitive analysis":             "marketing",
    "product marketing":                "marketing",

    # ── Operations / Supply Chain ──────────────────────────────────────────
    "supply chain management":          "supply chain",
    "s&op":                             "sales and operations planning",
    "sales & operations planning":      "sales and operations planning",
    "lean manufacturing":               "lean",
    "lean methodology":                 "lean",
    "kaizen":                           "lean",
    "5s":                               "lean",
    "value stream mapping":             "lean",
    "six sigma":                        "six sigma",
    "6 sigma":                          "six sigma",
    "lean six sigma":                   "six sigma",
    "continuous improvement":           "process improvement",
    "process mapping":                  "process improvement",
    "sop creation":                     "process improvement",
    "standard operating procedures":    "process improvement",
    "sops":                             "process improvement",
    "bpm":                              "process improvement",
    "business process management":      "process improvement",
    "root cause analysis":              "process improvement",
    "rca":                              "process improvement",
    "ci":                               "process improvement",
    "erp systems":                      "erp",
    "sap erp":                          "sap",
    "oracle erp":                       "erp",
    "inventory management":             "inventory management",
    "vendor management":                "vendor management",
    "vendor negotiation":               "vendor management",
    "supplier management":              "vendor management",
    "contract manufacturing":           "vendor management",
    "procurement":                      "procurement",
    "category management":              "procurement",
    "logistics":                        "logistics",
    "warehouse management":             "logistics",
    "fleet management":                 "logistics",
    "last mile delivery":               "logistics",
    "3pl":                              "logistics",
    "demand planning":                  "supply chain",
    "capacity planning":                "supply chain",

    # ── Legal / Compliance ─────────────────────────────────────────────────
    "regulatory compliance":            "compliance",
    "corporate governance":             "compliance",
    "sebi compliance":                  "compliance",
    "rbi compliance":                   "compliance",
    "iso 27001":                        "compliance",
    "sox compliance":                   "compliance",
    "sarbanes oxley":                   "compliance",
    "gdpr":                             "data privacy",
    "hipaa":                            "data privacy",
    "ccpa":                             "data privacy",
    "data protection":                  "data privacy",
    "privacy policy":                   "data privacy",
    "pci dss":                          "data privacy",
    "contract management":              "contracts",
    "contract negotiation":             "contracts",
    "contract drafting":                "contracts",
    "contract review":                  "contracts",
    "legal drafting":                   "contracts",
    "mou":                              "contracts",
    "nda":                              "contracts",
    "term sheet":                       "contracts",
    "due diligence":                    "due diligence",
    "risk management":                  "risk",
    "enterprise risk":                  "risk",
    "internal audit":                   "audit",
    "internal controls":                "audit",
    "sox":                              "compliance",
    "legal research":                   "legal",
    "litigation support":               "legal",
    "intellectual property":            "legal",

    # ── Soft / Leadership ──────────────────────────────────────────────────
    "cross functional":                 "cross-functional collaboration",
    "cross-functional teams":           "cross-functional collaboration",
    "stakeholder mgmt":                 "stakeholder management",
    "c-suite":                          "executive communication",
    "c suite":                          "executive communication",
    "executive stakeholders":           "executive communication",
    "people management":                "team management",
    "people leadership":                "team management",
    "line management":                  "team management",
    "change management":                "change management",
    "organisational change":            "change management",
    "organizational change":            "change management",
    "strategic thinking":               "strategic planning",
    "strategy development":             "strategic planning",
    "business development":             "business development",
    "bd":                               "business development",
    "account management":               "account management",
    "client management":                "account management",
    "relationship management":          "relationship management",
    "project management":               "project management",
    "programme management":             "program management",
    "program management":               "program management",
    "pmo":                              "program management",
    "agile methodology":                "agile",
    "scrum methodology":                "scrum",
    "prince2":                          "project management",
    "pmp":                              "project management",
    "presentation skills":              "communication",
    "public speaking":                  "communication",
    "written communication":            "communication",
    "verbal communication":             "communication",
    "active listening":                 "interpersonal skills",
    "emotional intelligence":           "interpersonal skills",
    "empathy":                          "interpersonal skills",
    "team player":                      "collaboration",
    "team work":                        "collaboration",
    "teamwork":                         "collaboration",
    "multi-tasking":                    "time management",
    "multitasking":                     "time management",
    "deadline management":              "time management",
    "analytical thinking":              "analytical skills",
    "data driven":                      "analytical skills",
    "critical thinking":                "analytical skills",
    "problem solving":                  "analytical skills",
    "decision making":                  "analytical skills",
    "mentoring":                        "coaching",
    "mentorship":                       "coaching",
    "training delivery":                "learning and development",
    "facilitation":                     "learning and development",

    # ── Productivity / Office Tools ────────────────────────────────────────
    "ms office":                        "microsoft office",
    "ms excel":                         "excel",
    "microsoft excel":                  "excel",
    "advanced excel":                   "excel",
    "pivot tables":                     "excel",
    "vlookup":                          "excel",
    "ms powerpoint":                    "powerpoint",
    "microsoft powerpoint":             "powerpoint",
    "google workspace":                 "productivity tools",
    "gsuite":                           "productivity tools",
    "g suite":                          "productivity tools",
    "ms word":                          "microsoft office",
    "microsoft word":                   "microsoft office",

    # ── Analytics / BI Tools ───────────────────────────────────────────────
    "power bi":                         "data visualization",
    "tableau":                          "data visualization",
    "looker":                           "data visualization",
    "qlik":                             "data visualization",
    "metabase":                         "data visualization",
    "data studio":                      "data visualization",
    "google data studio":               "data visualization",
}

SKILL_ALIASES.update({

    # ───────────────────────────────────────
    # CLOUD SECURITY / DEVSECOPS
    # ───────────────────────────────────────
    "prisma": "prisma cloud",
    "prisma cloud security": "prisma cloud",
    "azure defender": "defender for cloud",
    "microsoft defender for cloud": "defender for cloud",
    "defender": "defender for cloud",

    "cve": "vulnerability management",
    "cves": "vulnerability management",
    "vulnerability scanning": "vulnerability management",
    "security scanning": "vulnerability management",
    "container scanning": "vulnerability management",
    "image scanning": "vulnerability management",

    "runtime protection": "container security",
    "kubernetes security": "container security",
    "docker security": "container security",

    "cspm": "cloud security",
    "cloud security posture management": "cloud security",

    "nist framework": "security frameworks",
    "cis": "cis benchmarks",
    "iso security": "security frameworks",

    # ───────────────────────────────────────
    # SOFT SKILLS NORMALIZATION
    # ───────────────────────────────────────
    "stakeholder communication": "communication",
    "executive communication": "communication",
    "verbal skills": "communication",
    "written skills": "communication",

    "relationship building": "interpersonal skills",
    "people skills": "interpersonal skills",
    "soft skills": "interpersonal skills",

    "problem solving skills": "analytical skills",
    "analytical thinking skills": "analytical skills",
    "data analysis": "analytical skills",
    "data analytics": "analytical skills",

    # ───────────────────────────────────────
    # AGILE / PROJECT / DELIVERY
    # ───────────────────────────────────────
    "scrum master": "scrum",
    "agile development": "agile",
    "agile framework": "agile",

    "jira tool": "jira",
    "atlassian jira": "jira",
    "confluence tool": "confluence",

    # ───────────────────────────────────────
    # HR / BUSINESS EDGE CASES
    # ───────────────────────────────────────
    "people management": "team management",
    "team leadership": "team management",

    "employee lifecycle management": "employee lifecycle",
    "exit process": "offboarding",

    "employee satisfaction": "employee engagement",
    "employee experience": "employee engagement",

    "hrbp role": "hr business partner",

    # ───────────────────────────────────────
    # CLOUD EDGE CASES
    # ───────────────────────────────────────
    "multi cloud": "cloud",
    "multi-cloud": "cloud",
    "hybrid cloud": "cloud",

    "aws ec2": "aws",
    "aws s3": "aws",
    "aws rds": "aws",

    "azure aks": "kubernetes",
    "azure kubernetes service": "kubernetes",

    # ───────────────────────────────────────
    # DATA / ANALYTICS EDGE CASES
    # ───────────────────────────────────────
    "dashboarding": "data visualization",
    "bi tools": "data visualization",

})

SKILL_ALIASES.update({

    # ───────────────────────────────────────
    # TECH LEAD / ARCHITECTURE
    # ───────────────────────────────────────
    "tech lead": "technical leadership",
    "technical lead": "technical leadership",
    "engineering lead": "technical leadership",
    "team lead": "technical leadership",

    "system architecture": "system design",
    "solution architecture": "system design",
    "application architecture": "system design",
    "architecture design": "system design",

    "scalable systems": "system design",
    "high availability": "system design",
    "distributed architecture": "distributed systems",

    "code review": "software development",
    "design reviews": "system design",

    # ───────────────────────────────────────
    # DATA / AI / ANALYTICS (ADVANCED)
    # ───────────────────────────────────────
    "machine learning model": "machine learning",
    "ml models": "machine learning",
    "model building": "machine learning",

    "feature engineering": "machine learning",
    "model deployment": "mlops",
    "ml pipeline": "mlops",
    "mlops pipeline": "mlops",

    "deep learning models": "deep learning",
    "neural networks": "deep learning",

    "data science": "data analysis",
    "data scientist": "data analysis",

    "bigquery": "data warehousing",
    "snowflake db": "data warehousing",

    # ───────────────────────────────────────
    # CYBERSECURITY (ADVANCED)
    # ───────────────────────────────────────
    "penetration testing": "security testing",
    "pen testing": "security testing",
    "ethical hacking": "security testing",

    "iam": "identity management",
    "identity access management": "identity management",

    "zero trust architecture": "security best practices",

    "soc": "security operations",
    "security operations center": "security operations",

    "siem": "security monitoring",
    "security monitoring tools": "security monitoring",

    # ───────────────────────────────────────
    # FINANCE (ADVANCED / EDGE CASES)
    # ───────────────────────────────────────
    "financial planning": "financial planning and analysis",
    "fpna": "financial planning and analysis",

    "business finance": "financial analysis",
    "corporate finance": "financial analysis",

    "cost optimization": "financial analysis",
    "profitability analysis": "financial analysis",

    "variance reporting": "financial analysis",

    "audit compliance": "audit",
    "internal controls testing": "audit",

    "working capital management": "financial management",

    # ───────────────────────────────────────
    # PRODUCT / BUSINESS / CONSULTING
    # ───────────────────────────────────────
    "product management": "product strategy",
    "product strategy": "product strategy",
    "roadmap planning": "product strategy",
    "product roadmap": "product strategy",

    "user research": "product strategy",
    "customer research": "product strategy",

    "stakeholder alignment": "stakeholder management",

    "business strategy": "strategic planning",
    "corporate strategy": "strategic planning",

    "management consulting": "business consulting",
    "consulting": "business consulting",

    # ───────────────────────────────────────
    # PROJECT / DELIVERY (ADVANCED)
    # ───────────────────────────────────────
    "delivery management": "project management",
    "project delivery": "project management",

    "risk mitigation": "risk management",
    "risk assessment": "risk management",

    "resource planning": "project management",

    # ───────────────────────────────────────
    # OPERATIONS (ADVANCED)
    # ───────────────────────────────────────
    "operations management": "operations",
    "business operations": "operations",

    "process optimization": "process improvement",

    "lean operations": "lean",

    # ───────────────────────────────────────
    # GENERAL BUSINESS SKILLS (IMPORTANT)
    # ───────────────────────────────────────
    "decision making skills": "analytical skills",
    "problem solving ability": "analytical skills",

    "organizational skills": "time management",
    "planning skills": "time management",

    "work management": "project management",

})


def normalize_skill(skill: str) -> str:
    """Lowercase, strip noise, apply alias map."""
    skill = skill.lower().strip()
    skill = re.sub(r'\(.*?\)', '', skill)                    # remove (parentheticals)
    skill = skill.replace("programming language", "").strip()
    skill = skill.replace("programming", "").strip()
    # FIX: preserve +, #, ., /, - which are meaningful in skill names (c++, c#, node.js)
    skill = re.sub(r'[^a-z0-9+#\./\- &]', ' ', skill)       # keep & for p&l, fp&a etc.
    skill = re.sub(r'\s+', ' ', skill).strip()
    return SKILL_ALIASES.get(skill, skill)


def normalize_skills(skill_list) -> set[str]:
    return {normalize_skill(s) for s in skill_list if s and str(s).strip()}


def _word_boundary_re(skill: str) -> re.Pattern:
    """
    Word boundary regex that correctly handles special chars in skill names
    like c#, c++, node.js — these chars are not word chars so \\b fails on them.
    """
    escaped = re.escape(skill)
    return re.compile(
        r'(?<![a-z0-9])' + escaped + r'(?![a-z0-9])',
        re.IGNORECASE,
    )


def _exact_overlap(skill_a: str, skill_b: str) -> bool:
    """
    Strict word-boundary string overlap.
    Prevents Java ↔ JavaScript false positives.
    Also handles skills with special chars like c#, c++, node.js.
    """
    a = normalize_skill(skill_a)
    b = normalize_skill(skill_b)
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    # Only match if shorter is at least 2 chars to avoid single-letter false positives
    if len(shorter) < 2:
        return False
    try:
        return bool(_word_boundary_re(shorter).search(longer))
    except re.error:
        # If regex fails due to special chars, fall back to plain substring
        return shorter in longer


# =============================================================================
# LAYER 2 — Implied-skill map
# Broad JD term → set of specific skills that satisfy it
# =============================================================================

IMPLIED_SKILL_MAP: dict[str, set[str]] = {
    # ── Databases ─────────────────────────────────────────────────────────
    "nosql":                    {"mongodb", "cassandra", "redis", "dynamodb",
                                 "couchdb", "firebase", "bigquery", "elasticsearch"},
    "sql":                      {"postgresql", "mysql", "sqlite", "mssql",
                                 "oracle", "bigquery"},
    "database":                 {"postgresql", "mysql", "mongodb", "sqlite",
                                 "mssql", "bigquery", "redis", "dynamodb"},
    "relational database":      {"postgresql", "mysql", "sqlite", "mssql", "oracle"},

    # ── Cloud ─────────────────────────────────────────────────────────────
    "cloud":                    {"aws", "azure", "google cloud platform"},
    "cloud deployment":         {"aws", "kubernetes", "docker"},
    "cloud platforms":          {"aws", "azure", "google cloud platform"},
    "cloud services":           {"aws", "azure", "google cloud platform"},

    # ── Containers ────────────────────────────────────────────────────────
    "containerization":         {"docker", "kubernetes", "podman"},
    "containerized environments": {"docker", "kubernetes"},
    "containers":               {"docker", "kubernetes"},

    # ── Microservices / architecture ──────────────────────────────────────
    "microservices":            {"spring boot", "fastapi", "flask",
                                 "express", "node.js"},
    "microservices architecture": {"microservices", "spring boot",
                                   "fastapi", "flask"},
    "distributed systems":      {"microservices", "kafka",
                                 "event-driven architecture"},
    "service oriented architecture": {"microservices", "rest apis", "spring boot"},
    "soa":                      {"microservices", "rest apis"},

    # ── Messaging / event-driven ──────────────────────────────────────────
    "messaging systems":        {"kafka", "activemq", "rabbitmq",
                                 "sqs", "kinesis", "pubsub"},
    "message queues":           {"kafka", "activemq", "rabbitmq", "sqs"},
    "event-driven architecture": {"kafka", "activemq", "kinesis",
                                  "rabbitmq", "event-driven architecture"},
    "event driven":             {"kafka", "activemq", "kinesis",
                                 "event-driven architecture"},

    # ── CI/CD ─────────────────────────────────────────────────────────────
    "cicd":                     {"jenkins", "gitlab cicd", "github actions",
                                 "circleci", "travis ci", "teamcity"},
    "ci/cd pipelines":          {"jenkins", "cicd", "github actions"},
    "devops":                   {"jenkins", "cicd", "docker", "kubernetes",
                                 "terraform", "ansible"},
    "build automation":         {"jenkins", "cicd", "maven", "gradle"},

    # ── REST / APIs ───────────────────────────────────────────────────────
    "rest apis":                {"spring boot", "fastapi", "flask",
                                 "express", "node.js"},
    "api development":          {"rest apis", "fastapi", "flask", "spring boot"},
    "api design":               {"rest apis", "graphql", "openapi"},
    "backend systems":          {"spring boot", "fastapi", "flask",
                                 "node.js", "express"},
    "backend development":      {"spring boot", "fastapi", "flask",
                                 "node.js", "express", "django"},
    "web services":             {"rest apis", "graphql", "spring boot", "fastapi"},

    # ── Auth / Security ───────────────────────────────────────────────────
    "authentication":           {"oauth2", "pkce", "jwt", "saml",
                                 "spring security", "openid"},
    "authorization":            {"oauth2", "jwt", "rbac", "saml"},
    "security":                 {"spring security", "oauth2", "jwt",
                                 "pkce", "saml"},
    "identity management":      {"oauth2", "saml", "openid", "ldap"},

    # ── Monitoring / observability ────────────────────────────────────────
    "monitoring":               {"prometheus", "grafana", "cloudwatch",
                                 "splunk", "datadog", "newrelic"},
    "observability":            {"prometheus", "grafana", "cloudwatch",
                                 "splunk", "datadog"},
    "application monitoring":   {"prometheus", "grafana", "cloudwatch",
                                 "splunk", "datadog"},
    "logging":                  {"elk stack", "splunk", "cloudwatch",
                                 "datadog", "loki"},

    # ── Version control ───────────────────────────────────────────────────
    "version control":          {"git"},
    "source control":           {"git"},

    # ── Language implications ─────────────────────────────────────────────
    "javascript":               {"typescript"},
    "jvm":                      {"java", "kotlin", "scala"},

    # ── System design ─────────────────────────────────────────────────────
    "system design":            {"system design", "distributed systems",
                                 "microservices"},
    "software architecture":    {"system design", "microservices",
                                 "distributed systems"},
    "design patterns":          {"system design", "object oriented programming"},

    # ── LLM / AI frameworks ───────────────────────────────────────────────
    "llm frameworks":           {"langchain", "llamaindex"},
    "llm apis":                 {"openai", "anthropic"},
    "ai frameworks":            {"langchain", "llamaindex", "huggingface",
                                 "tensorflow", "pytorch"},
    "generative ai":            {"large language models", "langchain",
                                 "openai", "huggingface"},

    # ── Testing ───────────────────────────────────────────────────────────
    "testing":                  {"unit testing", "junit", "pytest",
                                 "jest", "mocha", "selenium"},
    "test automation":          {"selenium", "cypress", "pytest",
                                 "jest", "unit testing"},
    "qa":                       {"unit testing", "selenium", "cypress",
                                 "test automation", "pytest"},

    # ── Data engineering ──────────────────────────────────────────────────
    "big data":                 {"spark", "hadoop", "hive", "bigquery",
                                 "kafka", "flink"},
    "data pipelines":           {"airflow", "spark", "kafka", "etl",
                                 "dbt", "flink"},
    "etl":                      {"spark", "airflow", "talend",
                                 "informatica", "dbt"},
    "data warehousing":         {"bigquery", "snowflake", "redshift",
                                 "databricks"},
    "data engineering":         {"spark", "airflow", "kafka", "etl",
                                 "bigquery", "snowflake"},

    # ── Frontend ──────────────────────────────────────────────────────────
    "frontend development":     {"react", "vue", "angular", "next.js",
                                 "typescript", "javascript"},
    "ui development":           {"react", "vue", "angular", "html", "css"},
    "web development":          {"react", "vue", "angular", "javascript",
                                 "typescript", "html", "css"},

    # ── Mobile ────────────────────────────────────────────────────────────
    "mobile development":       {"react native", "flutter", "ios",
                                 "android", "swift", "kotlin"},
    "cross platform":           {"react native", "flutter", "xamarin"},

    # ── Infrastructure / IaC ─────────────────────────────────────────────
    "infrastructure":           {"terraform", "ansible", "kubernetes",
                                 "docker", "aws"},
    "iac":                      {"terraform", "ansible", "pulumi",
                                 "cloudformation"},
    "configuration management": {"ansible", "chef", "puppet", "terraform"},

    # ── Finance / Accounting ──────────────────────────────────────────────
    "financial analysis":       {"profit and loss", "financial planning and analysis",
                                 "budgeting", "forecasting", "roi",
                                 "financial modeling", "financial reporting"},
    "financial management":     {"profit and loss", "budgeting", "forecasting",
                                 "financial planning and analysis", "cash flow management"},
    "accounting":               {"accounting standards", "ar/ap",
                                 "financial reporting", "audit",
                                 "accounting software", "tally", "quickbooks"},
    "accounting software":      {"tally", "quickbooks", "zoho books",
                                 "sap", "xero", "sage"},
    "budgeting":                {"financial planning and analysis",
                                 "profit and loss", "forecasting"},
    "forecasting":              {"financial planning and analysis",
                                 "profit and loss", "budgeting"},
    "financial reporting":      {"mis reporting", "financial statements",
                                 "management accounts", "board reporting"},
    "tax":                      {"tax compliance", "gst", "tds", "compliance"},
    "statutory compliance":     {"compliance", "tax compliance", "labor laws"},

    # ── HR / People ───────────────────────────────────────────────────────
    "hr":                       {"recruitment", "talent development",
                                 "employee onboarding", "learning and development",
                                 "diversity and inclusion", "performance reviews",
                                 "hr information systems", "hr operations",
                                 "employee engagement", "hr analytics"},
    "human resources":          {"recruitment", "talent development",
                                 "employee onboarding", "learning and development",
                                 "diversity and inclusion", "hr operations",
                                 "employee engagement", "performance reviews"},
    "hr information systems":   {"workday", "sap successfactors", "successfactors",
                                 "zoho people", "bamboohr", "darwinbox",
                                 "greythr", "keka", "peoplesoft", "oracle hcm"},
    "hris":                     {"workday", "sap successfactors", "successfactors",
                                 "zoho people", "bamboohr", "darwinbox",
                                 "greythr", "keka", "peoplesoft", "oracle hcm"},
    "hr tools":                 {"workday", "sap successfactors", "successfactors",
                                 "zoho people", "bamboohr", "darwinbox",
                                 "greythr", "keka"},
    "people management":        {"team management", "performance reviews",
                                 "talent development", "employee engagement",
                                 "coaching"},
    "talent":                   {"recruitment", "talent development",
                                 "succession planning", "workforce planning"},
    "retention":                {"employee engagement", "talent development",
                                 "employee onboarding", "recruitment",
                                 "compensation and benefits"},
    "employee retention":       {"employee engagement", "talent development",
                                 "compensation and benefits"},
    "labor laws":               {"compliance", "hr operations",
                                 "statutory compliance"},
    "labour laws":              {"compliance", "hr operations",
                                 "statutory compliance"},
    "training and development": {"learning and development", "employee onboarding",
                                 "coaching", "facilitation"},
    "interpersonal skills":     {"communication", "stakeholder management",
                                 "relationship management", "collaboration"},
    "hr strategies":            {"stakeholder management", "strategic planning",
                                 "hr analytics", "hr operations",
                                 "change management"},
    "hr strategy":              {"stakeholder management", "strategic planning",
                                 "hr analytics", "hr operations"},
    "offboarding":              {"recruitment", "employee onboarding",
                                 "hr operations", "employee relations"},
    "employee lifecycle":       {"recruitment", "employee onboarding",
                                 "performance reviews", "offboarding",
                                 "hr operations"},
    "compliance":               {"labor laws", "hr operations",
                                 "statutory compliance", "regulatory compliance"},
    "performance management":   {"performance reviews", "goal setting",
                                 "appraisal", "okrs", "kpi management"},
    "employee relations":       {"conflict resolution", "employee engagement",
                                 "hr operations"},
    "payroll":                  {"payroll management", "compensation and benefits",
                                 "hr operations"},
    "payroll management":       {"compensation and benefits", "hr operations",
                                 "hr information systems"},
    "recruitment":              {"talent acquisition", "sourcing", "screening",
                                 "applicant tracking system"},
    "hr analytics":             {"hr metrics", "people analytics",
                                 "workforce analytics", "data visualization",
                                 "excel"},
    "hr metrics":               {"hr analytics", "people analytics",
                                 "data visualization", "excel"},
    "change management":        {"change management", "stakeholder management",
                                 "cross-functional collaboration",
                                 "organizational development"},
    "organizational restructuring": {"change management", "stakeholder management",
                                     "strategic planning"},

    # ── Sales / Marketing ─────────────────────────────────────────────────
    "sales":                    {"b2b", "crm", "salesforce", "account management",
                                 "pipeline management", "negotiation",
                                 "business development"},
    "marketing":                {"gtm strategy", "demand gen", "seo",
                                 "content marketing", "paid advertising",
                                 "crm", "brand strategy", "social media"},
    "digital marketing":        {"seo", "paid advertising", "social media",
                                 "email campaigns", "demand gen", "crm",
                                 "marketing analytics"},
    "growth":                   {"demand gen", "gtm strategy", "seo",
                                 "paid advertising", "crm"},
    "crm":                      {"salesforce", "hubspot", "zoho crm",
                                 "ms dynamics", "dynamics 365"},
    "data visualization":       {"power bi", "tableau", "looker",
                                 "data studio", "excel"},
    "reporting":                {"excel", "data visualization",
                                 "mis reporting", "hr analytics"},
    "advanced analytics":       {"data visualization", "power bi", "tableau",
                                 "sql", "excel"},

    # ── Operations / Supply Chain ─────────────────────────────────────────
    "operations":               {"process improvement", "supply chain", "erp",
                                 "six sigma", "lean", "vendor management",
                                 "procurement"},
    "supply chain":             {"procurement", "logistics", "inventory management",
                                 "vendor management", "sales and operations planning"},
    "process improvement":      {"six sigma", "lean", "process improvement",
                                 "kaizen", "root cause analysis"},
    "vendor management":        {"procurement", "supplier management",
                                 "vendor negotiation"},

    # ── Legal / Compliance ────────────────────────────────────────────────
    "compliance":               {"data privacy", "risk", "contracts",
                                 "audit", "regulatory compliance",
                                 "labor laws", "statutory compliance"},
    "risk":                     {"risk", "compliance", "audit",
                                 "due diligence"},
    "legal":                    {"contracts", "compliance", "due diligence",
                                 "risk", "legal research"},
    "contracts":                {"contract drafting", "contract review",
                                 "contract management", "nda", "mou"},

    # ── Leadership / Soft ─────────────────────────────────────────────────
    "leadership":               {"team management", "stakeholder management",
                                 "performance reviews", "strategic planning",
                                 "executive communication"},
    "management":               {"team management", "project management",
                                 "performance reviews", "stakeholder management"},
    "strategy":                 {"strategic planning", "business development",
                                 "gtm strategy", "stakeholder management"},
    "communication":            {"stakeholder management", "executive communication",
                                 "cross-functional collaboration",
                                 "presentation skills", "interpersonal skills"},
    "collaboration":            {"cross-functional collaboration",
                                 "stakeholder management", "team management"},
    "project management":       {"agile", "scrum", "program management",
                                 "stakeholder management", "risk"},
    "program management":       {"project management", "stakeholder management",
                                 "strategic planning", "risk"},
    "business development":     {"b2b", "account management", "negotiation",
                                 "crm", "relationship management",
                                 "gtm strategy"},
    "team management":          {"stakeholder management", "employee engagement",
                                 "performance reviews", "coaching",
                                 "people management"},
    "analytical skills":        {"hr analytics", "financial analysis",
                                 "data visualization", "excel", "sql"},
    "microsoft office":         {"excel", "powerpoint", "microsoft office"},
    "productivity tools":       {"microsoft office", "excel",
                                 "google workspace", "productivity tools"},
}

IMPLIED_SKILL_MAP.update({

    # ───────────────────────────────────────
    # CLOUD SECURITY FIXES
    # ───────────────────────────────────────
    "vulnerability management": {
        "vulnerability scanning", "security scanning",
        "cve", "cves", "trivy", "hadolint",
        "defender for cloud", "prisma cloud",
        "image scanning", "container scanning"
    },

    "container security": {
        "kubernetes", "docker", "aks",
        "runtime protection", "image scanning",
        "container hardening", "kubernetes security"
    },

    "workload protection": {
        "runtime protection", "container security",
        "kubernetes security", "workload security"
    },

    "cloud security": {
        "defender for cloud", "prisma cloud", "cspm",
        "cloud security posture", "azure security",
        "aws security", "cloud security posture management"
    },

    "security best practices": {
        "secure configuration", "zero trust",
        "least privilege", "policy enforcement",
        "network security", "secure architecture"
    },

    "cloud terminology": {
        "aws", "azure", "cloud infrastructure",
        "cloud architecture", "multi-cloud"
    },

    # ───────────────────────────────────────
    # SECURITY FRAMEWORKS FIX
    # ───────────────────────────────────────
    "security frameworks": {
        "cis benchmarks", "nist", "iso 27001",
        "compliance standards"
    },

    "compliance standards": {
        "cis benchmarks", "regulatory compliance",
        "policy enforcement", "security frameworks"
    },

    "cis benchmarks": {
        "security frameworks", "compliance standards"
    },

    "nist": {
        "security frameworks", "compliance standards"
    },

    # ───────────────────────────────────────
    # ITIL / ITSM FIX (CRITICAL)
    # ───────────────────────────────────────
    "itil framework": {
        "incident management", "servicenow",
        "sla", "ola", "change management",
        "problem management"
    },

    "it service management": {
        "servicenow", "incident management",
        "change management", "itil"
    },

    # ───────────────────────────────────────
    # ROLE ABSTRACTION FIX
    # ───────────────────────────────────────
    "cloud developers": {
        "developers", "devops", "engineering team",
        "software engineers"
    },

    "security architects": {
        "architects", "cloud architects",
        "solution architects", "security engineers"
    },

    "cloud architects": {
        "architects", "solution architects",
        "infrastructure architects"
    },

    # ───────────────────────────────────────
    # METRICS / AUTOMATION
    # ───────────────────────────────────────
    "security metrics": {
        "monitoring", "reporting", "dashboards",
        "alerts", "log analytics"
    },

    "automated remediation": {
        "ci/cd", "pipelines", "automation",
        "terraform", "infrastructure as code"
    },

    # ───────────────────────────────────────
    # AGILE FIX
    # ───────────────────────────────────────
    "agile": {
        "scrum", "devops", "ci/cd",
        "sprints", "kanban"
    },

    "scrum": {
        "agile", "devops", "ci/cd"
    },

    # ───────────────────────────────────────
    # HR FIXES
    # ───────────────────────────────────────
    "retention": {
        "employee engagement", "attrition",
        "employee retention"
    },

    "employee retention": {
        "employee engagement", "attrition"
    },

    "interpersonal skills": {
        "communication", "stakeholder management",
        "relationship management", "collaboration"
    },

    "training and development": {
        "learning and development",
        "employee onboarding", "coaching",
        "facilitation"
    },

    "hr strategies": {
        "stakeholder management",
        "strategic planning",
        "hr analytics", "hr operations",
        "change management"
    },

    "offboarding": {
        "employee lifecycle", "exit interviews",
        "hr operations", "employee relations"
    },

    "hr metrics": {
        "hr analytics", "people analytics",
        "reporting", "dashboards"
    },

    "hr analytics": {
        "hr metrics", "people analytics",
        "workforce analytics"
    },

    "diversity and inclusion": {
        "dei", "culture", "employee engagement"
    },

    "labor laws": {
        "compliance", "hr policies",
        "statutory compliance"
    },
})

IMPLIED_SKILL_MAP.update({

    # ───────────────────────────────────────
    # SYSTEM DESIGN / ARCHITECTURE (CRITICAL)
    # ───────────────────────────────────────
    "scalability": {
        "distributed systems", "microservices", "load balancing"
    },

    "high availability": {
        "distributed systems", "cloud", "failover"
    },

    "fault tolerance": {
        "distributed systems", "microservices"
    },

    "system reliability": {
        "monitoring", "observability", "logging"
    },

    "design tradeoffs": {
        "system design", "architecture"
    },

    # ───────────────────────────────────────
    # GEN AI / MLOPS (IMPORTANT)
    # ───────────────────────────────────────
    "rag pipelines": {
        "langchain", "llamaindex", "vector databases",
        "embeddings", "retrieval"
    },

    "prompt engineering": {
        "large language models", "generative ai"
    },

    "embeddings": {
        "vector databases", "machine learning"
    },

    "vector databases": {
        "pinecone", "faiss", "weaviate"
    },

    "mlops": {
        "model deployment", "pipelines", "ci/cd"
    },

    # ───────────────────────────────────────
    # PRODUCT / STRATEGY (BIG GAP FIX)
    # ───────────────────────────────────────
    "product thinking": {
        "product strategy", "user research", "customer research"
    },

    "business impact": {
        "kpi management", "roi", "metrics"
    },

    "decision making": {
        "analytical skills", "problem solving"
    },

    # ───────────────────────────────────────
    # SECURITY (ADVANCED FIX)
    # ───────────────────────────────────────
    "identity access management": {
        "oauth2", "saml", "rbac", "iam"
    },

    "threat modeling": {
        "security", "risk", "architecture"
    },

    "zero trust": {
        "security best practices", "identity management"
    },

    # ───────────────────────────────────────
    # SOFT SKILLS (FIX FALSE GAPS)
    # ───────────────────────────────────────
    "problem solving": {
        "analytical skills", "decision making"
    },

    "critical thinking": {
        "analytical skills"
    },

    "decision making skills": {
        "analytical skills"
    },

})

IMPLIED_SKILL_MAP.update({

    # ───────────────────────────────────────
    # ADVANCED ARCHITECTURE / BACKEND
    # ───────────────────────────────────────
    "backend engineering": {
        "backend development", "apis", "microservices", "databases"
    },
    "api integration": {
        "rest apis", "graphql", "web services"
    },
    "performance optimization": {
        "scalability", "system design", "profiling"
    },
    "latency optimization": {
        "performance optimization", "system design"
    },

    # ───────────────────────────────────────
    # CLOUD (REAL-WORLD EDGE CASES)
    # ───────────────────────────────────────
    "cloud migration": {
        "aws", "azure", "cloud", "infrastructure"
    },
    "multi-cloud architecture": {
        "aws", "azure", "gcp", "cloud"
    },
    "cloud cost optimization": {
        "cloud", "financial analysis", "aws", "azure"
    },

    # ───────────────────────────────────────
    # SECURITY (DEEP EDGE CASES)
    # ───────────────────────────────────────
    "application security": {
        "security", "vulnerability management", "secure coding"
    },
    "devsecops": {
        "ci/cd", "security", "automation"
    },
    "security operations": {
        "incident management", "monitoring", "security"
    },
    "threat detection": {
        "monitoring", "security metrics"
    },

    # ───────────────────────────────────────
    # DATA / AI / GENAI EDGE CASES
    # ───────────────────────────────────────
    "nlp": {
        "natural language processing", "machine learning"
    },
    "computer vision": {
        "deep learning", "machine learning"
    },
    "data pipelines orchestration": {
        "airflow", "etl", "data pipelines"
    },
    "real-time data processing": {
        "kafka", "stream processing", "big data"
    },

    # ───────────────────────────────────────
    # PRODUCT / BUSINESS EDGE CASES
    # ───────────────────────────────────────
    "go to market": {
        "gtm strategy", "marketing", "sales"
    },
    "customer success": {
        "crm", "stakeholder management", "support"
    },
    "user experience": {
        "ui development", "frontend development"
    },
    "a/b testing": {
        "experimentation", "data analysis"
    },

    # ───────────────────────────────────────
    # FINANCE EDGE CASES
    # ───────────────────────────────────────
    "financial modeling": {
        "financial analysis", "excel"
    },
    "cash flow": {
        "financial management", "forecasting"
    },
    "cost analysis": {
        "financial analysis", "budgeting"
    },

    # ───────────────────────────────────────
    # HR EDGE CASES 
    # ───────────────────────────────────────
    "employee experience": {
        "employee engagement", "retention"
    },
    "talent acquisition": {
        "recruitment", "sourcing"
    },
    "workforce planning": {
        "hr strategies", "talent management"
    },
    "organizational development": {
        "change management", "hr strategies"
    },

    # ───────────────────────────────────────
    # PROJECT / DELIVERY EDGE CASES
    # ───────────────────────────────────────
    "program delivery": {
        "program management", "project management"
    },
    "stakeholder engagement": {
        "stakeholder management", "communication"
    },
    "execution": {
        "project management", "delivery"
    },

    # ───────────────────────────────────────
    # OPERATIONS EDGE CASES
    # ───────────────────────────────────────
    "process automation": {
        "automation", "process improvement"
    },
    "business process management": {
        "process improvement", "operations"
    },
    "supply planning": {
        "supply chain", "operations"
    },

    # ───────────────────────────────────────
    # SOFT SKILLS EDGE CASES 
    # ───────────────────────────────────────
    "problem solving ability": {
        "analytical skills"
    },
    "decision making ability": {
        "analytical skills"
    },
    "time management": {
        "productivity", "project management"
    },
    "adaptability": {
        "learning", "collaboration"
    },
    "ownership": {
        "accountability", "responsibility"
    },

})


def _layer2_match(jd_skill_norm: str, resume_skills: set[str]) -> bool:
    """
    Check if any resume skill satisfies the JD skill via the implied map.
    Also checks the reverse: if a specific resume skill implies a broad JD term.
    """
    implied = IMPLIED_SKILL_MAP.get(jd_skill_norm, set())
    if implied & resume_skills:
        return True

    # Reverse: resume has a broad term that contains the JD skill
    for rs in resume_skills:
        rs_implied = IMPLIED_SKILL_MAP.get(rs, set())
        if jd_skill_norm in rs_implied:
            return True

    return False


# =============================================================================
# LAYER 3 — Embedding semantic fallback
# Only called for JD skills that failed Layers 1 + 2
# =============================================================================

EMBEDDING_THRESHOLD = 0.85          # default for tech / hard skills
SOFT_SKILL_EMBEDDING_THRESHOLD = 0.80  # lower threshold for soft / behavioural skills

# Detect soft / behavioural skills that warrant a lower threshold
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
    """Return a lower threshold for soft / behavioural skills."""
    return (
        SOFT_SKILL_EMBEDDING_THRESHOLD
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
    threshold: float = EMBEDDING_THRESHOLD,
) -> set[str]:
    """
    For each unmatched JD skill, embed it and all resume skills,
    then check cosine similarity. Return JD skills that find a
    resume match above the threshold.

    Uses per-skill dynamic threshold via _embedding_threshold_for().
    Only called for skills that failed Layers 1 + 2 — keeps cost low.
    """
    if not unmatched_jd_skills or not resume_skills:
        return set()

    jd_list = sorted(unmatched_jd_skills)
    rs_list = sorted(resume_skills)

    print(f"[Embedding Layer 3] embedding {len(jd_list)} JD skills + {len(rs_list)} resume skills")

    # Embed in parallel
    jd_embeddings, rs_embeddings = await asyncio.gather(
        embedder.aembed_documents(jd_list),
        embedder.aembed_documents(rs_list),
    )

    semantic_matches: set[str] = set()
    for jd_skill, jd_vec in zip(jd_list, jd_embeddings):
        skill_threshold = _embedding_threshold_for(jd_skill)   # dynamic per-skill threshold
        best_sim = 0.0
        best_rs = ""
        for rs_skill, rs_vec in zip(rs_list, rs_embeddings):
            sim = _cosine_similarity(jd_vec, rs_vec)
            if sim > best_sim:
                best_sim = sim
                best_rs = rs_skill
        if best_sim >= skill_threshold:
            print(f"  [embed match] '{jd_skill}' ↔ '{best_rs}'  sim={best_sim:.3f}  threshold={skill_threshold}")
            semantic_matches.add(jd_skill)
        else:
            print(f"  [embed no-match] '{jd_skill}' best='{best_rs}' sim={best_sim:.3f}  threshold={skill_threshold}")

    return semantic_matches


# =============================================================================
# Depth detection — pure Python, fully deterministic, NO LLM
# =============================================================================

# Phrases that OVERRIDE shallow qualifiers — checked FIRST before any shallow logic.
# If any of these appear near a skill, the skill is NEVER shallow.
_STRONG_DEPTH = re.compile(
    r"\b("
    # Explicit years of experience (various forms)
    r"\d+\s*\+?\s*years?\s+(?:of\s+)?(?:experience|exp)\b"
    r"|\d+\s*\+?\s*yrs?\s+(?:of\s+)?(?:experience|exp)\b"
    r"|\d+\s*\+?\s*years?\s+in\b"
    # Strong knowledge qualifiers (NOT the same as bare "knowledge of")
    r"|strong\s+knowledge\b"
    r"|deep\s+knowledge\b"
    r"|expert\s+knowledge\b"
    r"|in.depth\s+knowledge\b"
    r"|solid\s+knowledge\b"
    r"|extensive\s+knowledge\b"
    # Proficiency / expertise markers
    r"|proficien\w*"
    r"|expertise\b"
    r"|expert\s+in\b"
    r"|specialist\b"
    r"|specialised\b"
    r"|specialized\b"
    # Hands-on / practical proof
    r"|hands.on\s+experience\b"
    r"|hands.on\s+exposure\b"
    r"|solid\s+experience\b"
    r"|proven\s+experience\b"
    r"|extensive\s+experience\b"
    r"|significant\s+experience\b"
    r"|rich\s+experience\b"
    r"|direct\s+experience\b"
    r"|practical\s+experience\b"
    r"|real.world\s+experience\b"
    # Seniority / ownership signals
    r"|led\b|architected\b|owned\b|spearheaded\b"
    r"|production\b|at\s+scale\b|enterprise\b"
    r"|team\s+of\b|mentored\b|principal\b|senior\b"
    r"|end.to.end\b|from\s+scratch\b"
    r"|p&l\s+responsibility\b|board.level\b|cross.functional\b"
    r"|company.wide\b|org.wide\b|enterprise.wide\b"
    r"|managed\s+a\s+team\b|team\s+of\s+\d+\b"
    r"|budget\s+of\b|revenue\s+of\b|\$[\d]+[mk]\b"
    r"|c.suite\b|vp.level\b|director.level\b"
    r"|full.cycle\b|end.to.end\b|portfolio\s+of\b"
    r"|signed\b|closed\b|won\b|delivered\b"
    r"|certified\b|certification\b"
    r")\b",
    re.IGNORECASE,
)

# Qualifier words that signal shallow depth near a skill mention.
# IMPORTANT: "knowledge of" alone is shallow, but "strong knowledge of" is NOT
# (handled by _STRONG_DEPTH check running first).
_DEPTH_QUALIFIERS = [
    r"\bbasics?\b",
    r"\bfundamentals?\b",
    r"\bintro(?:duction)?\b",
    r"\bawareness\b",
    r"\bfamiliar(?:ity)?\b",           # covers "familiar with" and "familiarity"
    r"\bsome experience\b",
    r"\bexposure\b",
    r"\blimited\b",
    r"\bentry.?level\b",
    r"\bbeginners?\b",
    r"\bworking knowledge\b",          # "working knowledge" = shallow (not "strong knowledge")
    r"\bbasic knowledge\b",
    r"\bknowledge of\b",               # bare "knowledge of X" without strong modifier = shallow
    r"\bunderstanding of\b",           # bare "understanding of X" = shallow
    r"\baware(?:ness)?\b",
    r"\bnovice\b",
    r"\blearning\b",
    r"\bexploring\b",
    r"\btheoretical\b",
]
_DEPTH_RE = re.compile("|".join(_DEPTH_QUALIFIERS), re.IGNORECASE)

# Action verbs that prove real hands-on work
# Extended to cover non-tech business / operational verbs
_EVIDENCE_VERBS = re.compile(
    r"\b(developed|implementing|implemented|built|designed|deployed|managed|"
    r"created|integrated|optimised|optimized|architected|led|migrated|"
    r"maintained|configured|automated|wrote|established|reduced|improved|"
    r"worked|used|utilised|utilized|delivered|shipped|set.?up|"
    r"build|builds|building|handles|handling|involved|contributed|"
    r"responsible|owns|owned|spearheaded|enhanced|resolved|debugged|"
    r"tested|reviewed|refactored|scaled|orchestrated|provisioned|"
    r"monitored|secured|analysed|analyzed|modelled|modeled|"
    r"trained|fine.tuned|published|released|launched|"
    r"negotiated|facilitated|presented|budgeted|forecasted|hired|"
    r"onboarded|coached|counselled|counseled|advised|consulted|"
    r"partnered|liaised|coordinated|oversaw|supervised|directed|"
    r"pitched|closed|generated|grew|expanded|retained|"
    r"audited|assessed|evaluated|approved|"
    r"authored|drafted|filed|"
    r"restructured|transformed|streamlined|standardized|standardised|"
    r"fundraised|allocated|reconciled|reported|"
    r"prioritised|prioritized|influenced|aligned|secured|"
    r"exceeded|surpassed|achieved|attained|"
    r"recruited|sourced|screened|interviewed|"
    r"signed|renewed|renegotiated|executed|"
    r"rolled.out|championed|"
    r"identified|proposed|recommended|formulated|"
    r"managed\s+\w+\s+team|led\s+\w+\s+team|"
    # Non-tech / HR / finance specific verbs
    r"administered|processed|coordinated|ensured|conducted|"
    r"assisted|supported|handled|performed|prepared|"
    r"organized|organised|maintained|tracked|monitored|"
    r"resolved|addressed|provided|delivered|shared|"
    r"assisted\s+in|helped|participated|contributed)\b",
    re.IGNORECASE,
)


def _skill_contexts(skill: str, resume_text: str, window: int = 400) -> list[str]:
    """Return ALL text snippets around every occurrence of the skill."""
    text_lower = resume_text.lower()
    norm = normalize_skill(skill)
    patterns = []
    try:
        patterns.append(_word_boundary_re(norm))
    except re.error:
        pass
    if norm != skill.lower():
        try:
            patterns.append(_word_boundary_re(skill.lower()))
        except re.error:
            pass

    contexts = []
    seen_positions: set[int] = set()
    for pat in patterns:
        for m in pat.finditer(text_lower):
            bucket = m.start() // 100
            if bucket in seen_positions:
                continue
            seen_positions.add(bucket)
            start = max(0, m.start() - window)
            end = min(len(text_lower), m.end() + window)
            contexts.append(text_lower[start:end])
    return contexts


def _has_experience_evidence(skill: str, resume_text: str) -> bool:
    """
    True if ANY occurrence of the skill appears alongside a real action verb.
    Checks ALL occurrences so a skills-list entry can't hide a strong
    experience-section entry.
    """
    for ctx in _skill_contexts(skill, resume_text, window=400):
        if _EVIDENCE_VERBS.search(ctx):
            return True
    return False


def _has_strong_depth(skill: str, resume_text: str) -> bool:
    """
    True if any context around the skill contains strong depth signals
    (years of experience, proficiency, expertise, hands-on experience,
    strong/deep knowledge, led, certified, production, P&L, etc.).
    These override any shallow qualifier words.
    """
    for ctx in _skill_contexts(skill, resume_text, window=300):
        if _STRONG_DEPTH.search(ctx):
            return True
    return False


def _skill_in_competency_section(skill: str, resume_text: str) -> bool:
    """
    Returns True if the skill appears in a dedicated skills/competencies section.
    Skills listed in Core Competencies, Technical Skills, Key Skills etc.
    are real skills — they should NOT be marked shallow just because no
    action verb appears in that bullet.

    Strategy: if the skill's surrounding context looks like a skills-list
    (very short snippet, comma/pipe/bullet separated items, no verb), we still
    treat it as a genuine skill rather than shallow.
    """
    text_lower = resume_text.lower()

    # Common section headers for skills blocks
    skills_section_re = re.compile(
        r'\b(core competenc|key competenc|technical skills?|key skills?|'
        r'skills? summary|areas of expertise|competenc|proficienc|'
        r'tools?\s*(?:&|and)\s*technolog|tools?\s*used|'
        r'hr tools?|software\s+skills?|professional\s+skills?)\b',
        re.IGNORECASE,
    )

    # Find section headers in the text
    for header_match in skills_section_re.finditer(text_lower):
        # Get the next 600 chars after the header
        section_start = header_match.start()
        section_text = text_lower[section_start: section_start + 600]

        norm = normalize_skill(skill)
        try:
            if _word_boundary_re(norm).search(section_text):
                return True
        except re.error:
            if norm in section_text:
                return True

    return False


def _is_shallow_mention(skill: str, resume_text: str) -> bool:
    """
    Returns True ONLY when ALL of these hold:
    1. No strong depth signals anywhere (years exp, proficiency, strong knowledge,
       led, production, certified, hands-on experience, etc.)
    2. No action-verb evidence anywhere in the resume.
    3. The skill is NOT listed in a dedicated competency/skills section.
    4. Every mention is near a depth-qualifier word.

    If the candidate has even one real usage sentence → never shallow.
    If the skill appears in a skills/competencies block → never shallow.
    """
    # Strong depth signal overrides everything
    if _has_strong_depth(skill, resume_text):
        return False

    # Real usage found anywhere → not shallow
    if _has_experience_evidence(skill, resume_text):
        return False

    # Skill listed in a competency / skills section → treat as genuine
    if _skill_in_competency_section(skill, resume_text):
        return False

    # No real usage — check if ALL contexts have qualifier words
    contexts = _skill_contexts(skill, resume_text)
    if not contexts:
        return False

    return all(_DEPTH_RE.search(ctx) for ctx in contexts)


# =============================================================================
# OR-group handling  ("AWS, Azure, or GCP" / "AWS/Azure/GCP" → needs only ONE)
# =============================================================================

def _extract_or_groups(jd_text: str) -> list[set[str]]:
    """
    Detect skill alternatives in the JD. Handles:
    - Comma-separated lists ending with 'or X'   ("AWS, Azure, or GCP")
    - Slash-separated alternatives                ("AWS/Azure/GCP")
    - Either...or patterns                        ("either Kafka or RabbitMQ")
    Returns each group as a set of normalised skill names.
    """
    groups: list[set[str]] = []

    # Pattern 1: comma/or lists  ("AWS, Azure, or GCP")
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
                r'^(?:such as|like|including|e\.g\.?|:)\s*', '',
                p, flags=re.IGNORECASE,
            )
            norm = normalize_skill(p.strip())
            if norm and len(norm) > 1:
                cleaned.add(norm)
        if len(cleaned) > 1:
            groups.append(cleaned)

    # Pattern 2: slash-separated tech alternatives ("AWS/Azure/GCP", "Kafka/RabbitMQ")
    # Only match known-tech-looking tokens (uppercase start or known patterns)
    slash_pattern = re.compile(
        r'\b([A-Za-z][A-Za-z0-9\.\+\#\-]{1,20})'
        r'(?:/([A-Za-z][A-Za-z0-9\.\+\#\-]{1,20}))+'
        r'\b'
    )
    for m in slash_pattern.finditer(jd_text):
        raw_parts = m.group(0).split('/')
        cleaned = {normalize_skill(p.strip()) for p in raw_parts if len(p.strip()) > 1}
        # Only treat as OR-group if all parts look like tech skills (not path segments)
        if len(cleaned) > 1 and not any('/' in p for p in cleaned):
            groups.append(cleaned)

    # Pattern 3: "either X or Y"
    either_pattern = re.compile(
        r'\beither\s+([A-Za-z][A-Za-z0-9\.\+\#\-]*)\s+or\s+([A-Za-z][A-Za-z0-9\.\+\#\-]*)\b',
        re.IGNORECASE,
    )
    for m in either_pattern.finditer(jd_text):
        a = normalize_skill(m.group(1).strip())
        b = normalize_skill(m.group(2).strip())
        if a and b and len(a) > 1 and len(b) > 1:
            groups.append({a, b})

    # Deduplicate groups that are subsets of each other
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
    """
    For each OR-group where the candidate satisfies at least one member
    (matched or partial), remove the unchosen alternatives from gaps.
    """
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
# Main classify_skills — combines all three layers + depth detection
# =============================================================================

async def classify_skills(
    resume_skills: set[str],
    jd_skills: set[str],
    resume_text: str,
    embedder: OpenAIEmbeddings,
) -> tuple[set[str], set[str], set[str]]:
    """
    Three-way classification for every JD skill:
      matched  — covered with adequate depth
      partial  — covered but shallow / basic only
      missing  — not found by any of the three layers

    Pipeline per JD skill:
      Layer 1 → string normalisation + alias overlap
      Layer 2 → implied-skill map
      Layer 3 → embedding cosine similarity (only for skills still unresolved)
      Depth   → pure Python action-verb + strong-depth + competency-section check
    """
    matched: set[str] = set()
    partial: set[str] = set()
    still_missing: set[str] = set()

    for jd_skill in jd_skills:
        norm = normalize_skill(jd_skill)

        # Layer 1: exact / variant string match
        found_l1 = any(_exact_overlap(norm, rs) for rs in resume_skills)

        # Layer 2: implied-skill map
        found_l2 = (not found_l1) and _layer2_match(norm, resume_skills)

        if found_l1 or found_l2:
            layer = "L1" if found_l1 else "L2"
            if _is_shallow_mention(jd_skill, resume_text):
                print(f"  [{layer} shallow] {jd_skill}")
                partial.add(jd_skill)
            else:
                print(f"  [{layer} matched] {jd_skill}")
                matched.add(jd_skill)
        else:
            still_missing.add(jd_skill)

    # Layer 3: embedding fallback for everything still unresolved
    if still_missing:
        semantic_hits = await _embedding_match(still_missing, resume_skills, embedder)
        for jd_skill in still_missing:
            if jd_skill in semantic_hits:
                if _is_shallow_mention(jd_skill, resume_text):
                    print(f"  [L3 shallow] {jd_skill}")
                    partial.add(jd_skill)
                else:
                    print(f"  [L3 matched] {jd_skill}")
                    matched.add(jd_skill)
            else:
                print(f"  [missing]    {jd_skill}")

    missing = still_missing - matched - partial
    return matched, partial, missing


# =============================================================================
# Gap deduplication
# =============================================================================

def _dedupe_gaps(gap_list: list[str], all_gaps: set[str]) -> list[str]:
    """Keep only the first occurrence for each underlying skill."""
    seen: set[str] = set()
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
# Skill extraction (LLM)
# =============================================================================

async def _extract_skills_llm(
    resume_text: str,
    jd_text: str,
    llm: ChatOpenAI,
) -> tuple[set[str], set[str]]:
    """
    Extract skills from both documents using the LLM.
    Normalises immediately. Falls back to rule-based on any failure.
    """
    pydantic_parser = PydanticOutputParser(pydantic_object=SkillExtractionResponse)
    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    prompt_text = f"""
You are a skill extraction engine. You work across ALL role types:
backend, frontend, fullstack, data engineering, ML/AI, mobile, devops,
QA, security, cloud, product management, finance, accounting, HR, people operations,
sales, marketing, operations, supply chain, legal, compliance, strategy,
consulting, and any other domain.

### Objective
Extract ALL skills from both the Resume and the Job Description.

### Rules
- Parse the ENTIRE document — every section, every bullet, every tech stack line.
- Extract explicitly mentioned AND strongly implied skills.
- CRITICAL: Extract skills from ALL resume sections including:
    * "Core Competencies", "Key Skills", "Skills Summary", "Areas of Expertise"
    * "Technical Skills", "Tools", "Certifications"
    * "Professional Experience" bullets
    * "Achievements" and "Projects" sections
    * Professional summary / objective paragraph
- Include skills from "Nice-to-Have" / "Good-to-Have" / "Preferred" sections of the JD.
- Do NOT invent skills not present in the text.
- Do NOT skip soft skills, leadership competencies, or business skills if they are listed.
- Treat business/functional skills with the same rigour as technical skills.
- Certifications count as skills (e.g. CHRP → "hr certification", PMP → "project management").
- Tool names in a "Tools" section ARE skills (Workday → "hr information systems").

### Normalisation (apply before returning)
- Canonical lowercase, max 3 words per skill.
- Expand acronyms:
    "ML" → "machine learning", "CI/CD" → "cicd", "K8s" → "kubernetes",
    "NLP" → "natural language processing", "IaC" → "iac",
    "OOP" → "object oriented programming",
    "FP&A" → "financial planning and analysis", "P&L" → "profit and loss",
    "GTM" → "gtm strategy", "DEI" → "diversity and inclusion",
    "L&D" → "learning and development", "S&OP" → "sales and operations planning",
    "CRM" → "crm", "ERP" → "erp", "KPI" → "kpi management",
    "HRIS" → "hr information systems", "HRBP" → "hr business partner",
    "OD" → "change management", "BGV" → "recruitment",
    "ATS" → "applicant tracking system", "MIS" → "financial reporting",
    "SHRM" → "hr certification", "CIPD" → "hr certification",
    "CHRP" → "hr certification", "PHR" → "hr certification",
    "BPM" → "process improvement", "RCA" → "process improvement",
    "SOP" → "process improvement", "OKRs" → "performance management",
    "KRAs" → "performance management", "360" → "performance management".

- Collapse variants to one canonical form:
    "Workday" / "SAP SuccessFactors" / "Zoho People" / "BambooHR" / "Darwinbox"
        → "hr information systems"
    "HR Operations & Compliance" → "hr operations" AND "compliance"
    "Labor Laws" / "Labour Laws" → "labor laws"
    "Talent Acquisition" → "recruitment"
    "Performance Appraisal" / "Appraisal" → "performance reviews"
    "Conflict Resolution" → "employee relations"
    "Exit Interviews" → "offboarding"
    "Employee Retention" / "Attrition" → "retention"
    "Payroll Coordination" / "Payroll Processing" → "payroll management"
    "Headcount Planning" / "Manpower Planning" → "workforce planning"
    "Organizational Development" / "OD" → "change management"
    "MIS Reporting" / "Management Reporting" → "financial reporting"
    "Cost Analysis" / "Cost Management" → "financial analysis"
    "Tally" / "QuickBooks" / "Zoho Books" / "Xero" → "accounting software"
    "Naukri" / "Naukri RMS" / "LinkedIn Recruiter" → "recruitment"
    "HR Policies" / "Policy Implementation" → "hr operations"
    "Employee Handbook" → "hr operations"
    "Culture Building" / "Employer Branding" → "employee engagement"
    "Pulse Surveys" / "Stay Interviews" / "ESAT" → "employee engagement"
    "Goal Setting" / "OKRs" / "KRAs" → "performance management"
    "360 Feedback" → "performance management"
    "People Analytics" / "Workforce Analytics" → "hr analytics"
    "Induction" / "Joining Formalities" → "employee onboarding"
    "Training Delivery" / "Facilitation" → "learning and development"
    "CTC Structuring" / "Salary Benchmarking" → "compensation and benefits"
    "MS Excel" / "Advanced Excel" → "excel"
    "MS Office" / "Microsoft Office" → "microsoft office"
    "Google Workspace" / "G Suite" → "productivity tools"
    "Power BI" / "Tableau" / "Looker" → "data visualization"
    "HubSpot" / "Zoho CRM" / "MS Dynamics" → "crm"
    "SAP FICO" / "SAP FI" → "sap"
    "Prosci" → "change management certification"
    "Node.js" / "NodeJS" → "node.js"
    "React.js" / "ReactJS" → "react"
    "Postgres SQL" / "Postgres" → "postgresql"
    "Kafka Connect" / "Kafka topic" → "kafka"
    "Spring Security" → "security"
    "JUnit" → "unit testing"
    "GitHub Actions" → "cicd"
    "GitLab CI/CD" → "cicd"
    "Apache Spark" / "PySpark" → "spark"
    "Golang" / "Go language" → "go"
    ".NET Core" / "ASP.NET" → ".net"

- Deduplicate — return each skill exactly once.
- Strip depth qualifiers from skill names:
    "AWS basics" → "aws"
    "working knowledge of CI/CD" → "cicd"
    "familiarity with Docker" → "docker"
    "understanding of Kubernetes" → "kubernetes"
    "exposure to SAP" → "sap"
    "basic P&L understanding" → "profit and loss"
    "knowledge of labor laws" → "labor laws"
    "strong knowledge of compliance" → "compliance"

### Few-shot examples (non-tech roles)
  Input                                    → Output
  "HR Operations & Compliance"             → "hr operations", "compliance"
  "Ensuring compliance with labor laws"    → "compliance", "labor laws"
  "Improved employee retention by 15%"     → "retention"
  "Onboarding & Training"                  → "employee onboarding", "learning and development"
  "Exit interviews"                        → "offboarding"
  "Build relationships, resolve conflicts" → "interpersonal skills", "employee relations"
  "Drive organizational growth via HR strategies" → "hr strategies"
  "Workday, SAP SuccessFactors, Zoho People" → "hr information systems"
  "Payroll Coordination & Attendance"      → "payroll management"
  "Proficiency in HRIS tools like Workday" → "hr information systems"
  "MIS Reporting"                          → "financial reporting"
  "SAP FICO"                               → "sap"
  "Tally ERP"                              → "accounting software"
  "P&L Management"                         → "profit and loss"
  "FP&A"                                   → "financial planning and analysis"
  "Stakeholder Management"                 → "stakeholder management"
  "Cross-functional collaboration"         → "cross-functional collaboration"
  "Salesforce CRM"                         → "salesforce"
  "GTM Strategy"                           → "gtm strategy"
  "SAP ERP"                                → "sap"
  "Six Sigma / Lean"                       → "six sigma", "lean"
  "GDPR compliance"                        → "data privacy"
  "Diversity & Inclusion"                  → "diversity and inclusion"
  "People Management"                      → "team management"
  "Change Management"                      → "change management"
  "Programme Management"                   → "program management"
  "Certified Human Resource Professional"  → "hr certification"
  "SHRM-SCP"                               → "hr certification"
  "Strong knowledge of labor laws"         → "labor laws"
  "Hands-on experience with HR analytics"  → "hr analytics"
  "Managed end-to-end recruitment"         → "recruitment"
  "Performance review cycles"              → "performance reviews"
  "Employee engagement activities"         → "employee engagement"
  "ATS Platforms: Naukri RMS"              → "applicant tracking system", "recruitment"

### Output — STRICT JSON ONLY (no markdown, no preamble)
{{
  "resume_skills": ["skill1", "skill2"],
  "jd_skills":     ["skill1", "skill2"]
}}

{fixing_parser.get_format_instructions()}

--- RESUME ---
{resume_text}

--- JOB DESCRIPTION ---
{jd_text}"""

    messages = [
        SystemMessage(content=(
            "You are a deterministic information extraction engine. "
            "Always return valid JSON. Never vary output for the same input."
        )),
        HumanMessage(content=prompt_text),
    ]

    raw = await llm.ainvoke(messages)
    print("[Skill extraction] raw response (first 600 chars):\n", raw.content[:600])

    try:
        parsed = fixing_parser.parse(raw.content)
    except Exception as e:
        print(f"[WARN] Skill parse error ({e}) — rule-based fallback")
        return (
            normalize_skills(extract_skills(resume_text)),
            normalize_skills(extract_skills(jd_text)),
        )

    if not parsed.resume_skills and not parsed.jd_skills:
        print("[WARN] Empty skill lists — rule-based fallback")
        return (
            normalize_skills(extract_skills(resume_text)),
            normalize_skills(extract_skills(jd_text)),
        )

    return normalize_skills(parsed.resume_skills), normalize_skills(parsed.jd_skills)


# =============================================================================
# Analysis chain
# =============================================================================

def _build_analysis_chain(llm: ChatOpenAI):
    pydantic_parser = PydanticOutputParser(pydantic_object=ResumeAnalysisResponse)
    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

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
7. Apply these rules equally to technical skills AND business/soft skills
   (e.g. stakeholder management, P&L, forecasting, recruitment, labor laws).
8. If a skill is in MATCHED SKILLS, do NOT mention it as a gap anywhere,
   including in Score_Explanation_Technical.
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
    fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

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


def _build_question_reframe_chain(llm: ChatOpenAI):
    return PromptTemplate(
        input_variables=["suggested_questions"],
        template="""
You are an expert recruiter. Refine these raw interview questions:
rephrase clearly, remove near-duplicates, keep all distinct topics.

Return ONLY a valid JSON array of strings. No explanation, no markdown.

suggested_questions:
{suggested_questions}
""",
    ) | llm


# =============================================================================
# Miscellaneous helpers
# =============================================================================

def _resolve_resume_experience(resume_text: str) -> int:
    """
    Extract years of experience from resume.
    Prefer explicit date ranges or year mentions.
    Only fall back to title/length heuristics if nothing found.
    """
    exp = extract_experience(resume_text)
    if exp > 0:
        return exp

    # Heuristic fallback — only used when no dates/years found at all
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
    """
    Strict client extraction using LLM + rule-based fallback.
    High precision + stable output.
    """

    messages = [
        SystemMessage(content=(
            "You are a strict information extraction engine. Follow rules exactly."
        )),
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
        raw = await llm.bind(temperature=0).ainvoke(messages)
        print("******** CLIENT EXTRACTION RAW ********")
        print(raw)

        # Clean JSON
        content = _clean_llm_json(raw.content)
        names = json.loads(content)

        if isinstance(names, list):
            llm_names = [
                str(n).strip()
                for n in names
                if isinstance(n, str) and n.strip()
            ]
        else:
            llm_names = []

    except Exception as e:
        print(f"[WARN] LLM client extraction failed: {e}")
        llm_names = []

    # ---------------- FALLBACK ----------------
    rule_based_names = extract_client_names_advanced(resume_text)

    # ---------------- MERGE + CLEAN ----------------
    final = set()

    for name in llm_names + rule_based_names:
        clean = name.strip()
        if len(clean) > 2:
            final.add(clean)

    return sorted(final)


# =============================================================================
# Hard validation  (safety net after LLM analysis chain)
# =============================================================================

def _apply_hard_validation(
    merged: dict,
    matched: set[str],
    partial: set[str],
    missing: set[str],
) -> dict:
    """
    Guarantees correctness of Key_Matches and Key_Gaps regardless of
    what the LLM analysis chain produced.
    """
    all_gaps = partial | missing

    # ── Key_Matches ───────────────────────────────────────────────────────
    # Keep only LLM entries grounded in actual matched skills
    merged["Key_Matches"] = [
        item for item in merged.get("Key_Matches", [])
        if any(
            normalize_skill(s) in item.lower() or _exact_overlap(s, item)
            for s in matched
        )
    ]
    # Ensure every matched skill is represented
    represented_m = {
        s for s in matched
        if any(
            normalize_skill(s) in item.lower() or _exact_overlap(s, item)
            for item in merged["Key_Matches"]
        )
    }
    for skill in sorted(matched - represented_m):
        merged["Key_Matches"].append(f"{skill} — demonstrated in resume")

    # ── Key_Gaps ──────────────────────────────────────────────────────────
    # Keep only LLM entries grounded in actual gaps
    merged["Key_Gaps"] = [
        item for item in merged.get("Key_Gaps", [])
        if any(
            normalize_skill(s) in item.lower() or _exact_overlap(s, item)
            for s in all_gaps
        )
    ]
    # Deduplicate
    merged["Key_Gaps"] = _dedupe_gaps(merged["Key_Gaps"], all_gaps)

    # Add any gap the LLM missed
    represented_g = {
        s for s in all_gaps
        if any(
            normalize_skill(s) in item.lower() or _exact_overlap(s, item)
            for item in merged["Key_Gaps"]
        )
    }
    for skill in sorted(partial - represented_g):
        merged["Key_Gaps"].append(
            f"Limited {skill} experience — JD expects deeper proficiency"
        )
    for skill in sorted((missing - represented_g) - partial):
        merged["Key_Gaps"].append(f"No experience with {skill}")

    # Final dedup
    merged["Key_Gaps"] = _dedupe_gaps(merged["Key_Gaps"], all_gaps)

    # ── Score explanation cleanup ─────────────────────────────────────────
    if "Score_Explanation_Technical" in merged:
        explanation = merged["Score_Explanation_Technical"]
        for skill in matched:
            explanation = re.sub(
                rf"(?i)\b(no|lack of|lacks|missing|absent)\b[^.]*\b{re.escape(skill)}\b",
                f"experience present with {skill}",
                explanation,
            )
        for skill in all_gaps:
            if not _word_boundary_re(normalize_skill(skill)).search(explanation.lower()):
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
    cached = memory_store.get("analysis_cache", {}).get(cache_key)
    if cached:
        print("[CACHE HIT] returning cached result")
        return cached

    # ── Shared instances ──────────────────────────────────────────────────
    llm      = _make_llm()
    embedder = _make_embedder()

    # ── Step 1: Skill extraction ──────────────────────────────────────────
    resume_skills, jd_skills = await _extract_skills_llm(resume_text, jd_text, llm)
    print("RESUME SKILLS:", sorted(resume_skills))
    print("JD SKILLS    :", sorted(jd_skills))

    # ── Step 2: OR-group detection ────────────────────────────────────────
    or_groups = _extract_or_groups(jd_text)

    # ── Step 3: Three-layer skill classification ──────────────────────────
    matched, partial, missing = await classify_skills(
        resume_skills, jd_skills, resume_text, embedder
    )

    # ── Step 4: OR-group resolution ───────────────────────────────────────
    missing, partial = _resolve_or_groups(missing, partial, matched, or_groups)

    print("MATCHED :", sorted(matched))
    print("PARTIAL :", sorted(partial))
    print("MISSING :", sorted(missing))

    # ── Step 5: Scoring ───────────────────────────────────────────────────
    # Partial skills penalised at 0.6 weight (raised from 0.4 — partial is close
    # to matched; over-penalising was a major source of score deflation).
    total         = len(jd_skills) if jd_skills else 1
    skill_score   = round((len(matched) + 0.6 * len(partial)) / total * 100, 1)

    resume_exp    = _resolve_resume_experience(resume_text)
    jd_exp        = extract_experience(jd_text)
    exp_score     = compute_experience_score(resume_exp, jd_exp)
    final_score   = compute_final_score(skill_score, exp_score)

    # ── Step 6: Analysis + shrink (parallel) ─────────────────────────────
    resp_task = _build_analysis_chain(llm).ainvoke({
        "jd_text":        jd_text,
        "resume_text":    resume_text,
        "resume_skills":  sorted(resume_skills),
        "jd_skills":      sorted(jd_skills),
        "matched_skills": sorted(matched),
        "partial_skills": sorted(partial),
        "missing_skills": sorted(missing),
    })
    shrink_task = _build_shrink_chain(llm).ainvoke({
        "combined_text": f"{jd_text}\n{resume_text}"
    })
    resp, shrinked_output = await asyncio.gather(resp_task, shrink_task)
    print("Shrink sentences:", shrinked_output.sentences)

    # ── Step 7: Question suggestion + reframing ───────────────────────────
    suggested_questions = list(set(
        q
        for query in shrinked_output.sentences
        for q in suggester.suggest_questions(query, top_k=20)
    ))
    reframed_raw = await _build_question_reframe_chain(llm).ainvoke({
        "suggested_questions": suggested_questions
    })
    questions = (
        normalize_suggested_questions(reframed_raw.content)
        or suggested_questions[:10]
    )

    # ── Step 8: Build response ────────────────────────────────────────────
    response = resp.model_dump()
    merged   = {**response["Evaluation"], **response["Grammar_Check"]}

    merged["JD_MatchScore"]           = format_score(final_score)
    merged["Skill_Score"]             = skill_score
    merged["Skill_Coverage"]          = f"{len(matched)}/{len(jd_skills)}"
    merged["Experience_Score"]        = exp_score
    merged["Resume_Experience"]       = resume_exp
    merged["JD_Required_Experience"]  = jd_exp
    merged["Matched_Skills"]          = sorted(matched)
    merged["Partial_Skills"]          = sorted(partial)
    merged["Missing_Skills"]          = sorted(missing)
    merged["Extracted_Resume_Skills"] = sorted(resume_skills)
    merged["Extracted_JD_Skills"]     = sorted(jd_skills)

    # Hard validation — final correctness guarantee
    merged = _apply_hard_validation(merged, matched, partial, missing)

    merged["Grammatical_Errors"] = filter_grammar_errors(
        merged.get("Grammatical_Errors", []), resume_text
    )
    merged["Spelling_Mistakes"]  = filter_spelling_errors(
        merged.get("Spelling_Mistakes", []), resume_text
    )
    merged["Client_Names"]        = await _extract_client_names_llm(resume_text, llm)
    merged["Suggested_Questions"] = questions

    # ── Step 9: Course suggestions ────────────────────────────────────────
    all_gap_skills = sorted(partial | missing)
    key_gaps_str   = " ".join(merged.get("Key_Gaps") or [])
    suggest_course = suggester.suggest_courses(
        key_gaps_str, top_k=20, filter_value='resource'
    )
    if not suggest_course:
        suggest_course = suggester.suggest_courses(
            " ".join(all_gap_skills), top_k=5, filter_value='resource'
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

    llm            = ChatGroq(model="openai/gpt-oss-20b")
    pydantic_parser = PydanticOutputParser(pydantic_object=JDAnalysisResponse)
    fixing_parser  = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm)

    prompt = PromptTemplate(
        input_variables=["jd_text", "format_instructions"],
        template="""
You are an HR Analyst AI assistant. Given the Job Description below:

1. Sanitize: remove sensitive info (names, emails, phone numbers).
2. Extract:
   - Must-have skills (3-5)
   - Good-to-have skills (2-3)
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
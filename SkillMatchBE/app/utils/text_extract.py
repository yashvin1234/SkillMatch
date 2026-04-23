import io
import re
from docx import Document
import pdfplumber

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extracts all available text from a DOCX file, including paragraphs and table cells.
    
    Args:
        file_bytes (bytes): The raw binary content of a DOCX file.

    Returns:
        str: All extracted text, separated by newlines.
    """
    document = Document(io.BytesIO(file_bytes))
    text_parts = []

    # Extract all paragraph text
    for para in document.paragraphs:
        if para.text.strip():
            text_parts.append(para.text.strip())

    # Extract text from tables
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    text_parts.append(cell_text)

    return "\n".join(text_parts)

def extract_text(file_bytes: bytes, name: str) -> str:
    ext = name.lower().split('.')[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == "docx":
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file extension")

def filter_spelling_errors(errors, text):
    filtered = []
    text_lower = text.lower()

    for err in errors:
        try:
            wrong_word = err.split("->")[0].strip().lower()
            if wrong_word and wrong_word in text_lower:
                filtered.append(err)
        except Exception:
            continue

    return filtered


def filter_grammar_errors(errors, text):
    filtered = []
    text_lower = text.lower()

    for err in errors:
        # Extract quoted parts
        matches = re.findall(r"'([^']+)'", err)

        for match in matches:
            if match.lower() in text_lower:
                filtered.append(err)
                break

    return filtered


def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def extract_from_client_section(lines):
    clients = []

    for i in range(len(lines)):
        line = lines[i].lower()

        # Case 1: exact "Client"
        if line == "client":
            j = i + 1
            while j < len(lines) and lines[j] == "":
                j += 1
            if j < len(lines):
                clients.append(lines[j])

        # Case 2: "Client: XYZ"
        elif line.startswith("client"):
            parts = lines[i].split(":", 1)
            if len(parts) > 1:
                clients.append(parts[1].strip())

    return clients


def extract_using_patterns(text):
    clients = []

    # Pattern 1: "Client - XYZ", "Client for XYZ"
    patterns = [
        r'client\s*[:\-]\s*([A-Z][A-Za-z0-9&,\.\' ]+)',
        r'worked with\s+([A-Z][A-Za-z0-9&,\.\' ]+)',
        r'project for\s+([A-Z][A-Za-z0-9&,\.\' ]+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        clients.extend(matches)

    return clients


def clean_clients(clients, resume_text):
    final = []
    resume_lower = resume_text.lower()

    for c in clients:
        c = normalize_text(c)

        # remove very short / noisy
        if len(c) < 3:
            continue

        # must roughly exist in resume
        if any(word in resume_lower for word in c.lower().split()):
            final.append(c)

    return list(set(final))


def extract_client_names_advanced(resume_text):
    lines = [line.strip() for line in resume_text.splitlines()]

    # Step 1: structured extraction
    clients = extract_from_client_section(lines)

    # Step 2: pattern fallback
    if not clients:
        clients += extract_using_patterns(resume_text)

    # Step 3: cleanup + validation
    clients = clean_clients(clients, resume_text)

    return clients


def compute_match_score(resume_skills, jd_skills):
    if not jd_skills:
        return 0, [], []

    matched = resume_skills.intersection(jd_skills)
    missing = jd_skills - resume_skills

    score = (len(matched) / len(jd_skills)) * 10

    return round(score, 1), list(matched), list(missing)


def format_score(score: float) -> str:
    """
    Format a 0–100 scale score into a human-readable string.
    score is produced by compute_final_score which blends
    skill_score (0-100) and exp_score (0-100).
    """
    score = round(score, 1)
    if score >= 75:
        return f"{score}/100 - Strong match"
    elif score >= 50:
        return f"{score}/100 - Moderate match"
    else:
        return f"{score}/100 - Weak match"
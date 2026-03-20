from fastapi import APIRouter, File, UploadFile, HTTPException
from app.utils.text_extract import extract_text
from app.config import ALLOWED_EXTENSIONS, memory_store

router = APIRouter()

# Allowed process types
VALID_PROCESS_TYPES = {"analyze_jd", "jd_resume_match"}

@router.post("/upload/{process_type}")
async def upload_jd(process_type: str,
    file: UploadFile = File(...)):

    # Validate process type
    if process_type not in VALID_PROCESS_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid process type. Choose from: {', '.join(VALID_PROCESS_TYPES)}."
        )
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file sent")
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Only .docx and .pdf files are allowed")

    if "jd" not in memory_store:
        memory_store["jd"] = {}

    # Save file to memory under the given process type
    memory_store["jd"][process_type] = {
        "bytes": await file.read(),
        "filename": file.filename
    }
    return {"info": f"Uploaded Job Description '{file.filename}' successfully."}

def allowed_file(filename: str) -> bool:
    ext = filename.split(".")[-1].lower()
    return ext in ALLOWED_EXTENSIONS

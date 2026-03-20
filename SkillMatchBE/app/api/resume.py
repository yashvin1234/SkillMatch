from fastapi import APIRouter, File, UploadFile, HTTPException
from app.utils.text_extract import extract_text
from app.config import ALLOWED_EXTENSIONS, memory_store

router = APIRouter()

@router.post("/upload/")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file sent")
    
    print(file.filename.split(".")[-1].lower())
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Only .docx and .pdf files are allowed")

    # Save Resume to memory
    memory_store["resume"] = {"bytes": await file.read(), "filename": file.filename}
    return {"info": f"Uploaded Resume '{file.filename}' successfully."}

def allowed_file(filename: str) -> bool:
    ext = filename.split(".")[-1].lower()
    return ext in ALLOWED_EXTENSIONS

from dotenv import load_dotenv
from pathlib import Path

PROJECT_PATH = Path.cwd()
APP_PATH = PROJECT_PATH / "app"
DATA_PATH = PROJECT_PATH / "app" / "data" 

load_dotenv()

# Allowed file extensions
ALLOWED_EXTENSIONS = {"pdf", "docx"}

# In-memory store for files (you can use a proper database in production)
memory_store = {}

# backend/config.py
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

# ==========================================
# 1) Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
INTERNAL_PROJECTS_FILE = DATA_DIR / "internal_projects.json"

# ==========================================
# 2) LLM Settings
# ==========================================
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "900"))

# ==========================================
# 3) Retrieval Settings 
# ==========================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.3"))

# Access control
INTERNAL_ALLOW_C2 = os.getenv("INTERNAL_ALLOW_C2", "false").lower() in ("1", "true", "yes", "y")

# ==========================================
# 4) Web Search Settings
# ==========================================
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() in ("1", "true", "yes", "y")
WEB_TOP_K = int(os.getenv("WEB_TOP_K", "3")) 

# ==========================================
# 5) API Keys
# ==========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables (.env).")

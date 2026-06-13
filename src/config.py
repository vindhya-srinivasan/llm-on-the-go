import os
from dotenv import load_dotenv

load_dotenv()

# Model
OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Docs
DOCS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
MAX_FILES = 5

# Retrieval
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 4

# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- All Available Models ---
AVAILABLE_MODELS = {
    # ── Local Models (Ollama) ──
    "LLaMA 3.2 1B (Local)": {
        "type": "ollama",
        "model": "llama3.2:1b",
        "icon": "🦙",
        "description": "Meta · Fastest · Offline",
        "speed": "⚡⚡⚡"
    },
    "LLaMA 3.2 3B (Local)": {
        "type": "ollama",
        "model": "llama3.2:3b",
        "icon": "🦙",
        "description": "Meta · Balanced · Offline",
        "speed": "⚡⚡"
    },
    "Qwen 2.5 3B (Local)": {
        "type": "ollama",
        "model": "qwen2.5:3b",
        "icon": "🐉",
        "description": "Alibaba · Strong reasoning · Offline",
        "speed": "⚡⚡"
    },
    "Phi-3 Mini (Local)": {
        "type": "ollama",
        "model": "phi3:mini",
        "icon": "🔷",
        "description": "Microsoft · Efficient · Offline",
        "speed": "⚡⚡"
    },
    "Gemma 2B (Local)": {
        "type": "ollama",
        "model": "gemma:2b",
        "icon": "💎",
        "description": "Google · Lightweight · Offline",
        "speed": "⚡⚡⚡"
    },
    # ── Cloud Models ──
    "Gemini 2.0 Flash (Cloud)": {
        "type": "gemini",
        "model": "gemini-2.0-flash",
        "icon": "✨",
        "description": "Google · Most capable · Requires API key",
        "speed": "⚡⚡"
    },
   "LLaMA 3.3 70B via Groq (Cloud)": {
        "type": "groq",
        "model": "llama-3.3-70b-versatile",
        "icon": "⚡",
        "description": "Meta · Ultra fast · Requires API key",
        "speed": "⚡⚡⚡"
    },
    "LLaMA 3.1 8B via Groq (Cloud)": {
        "type": "groq",
        "model": "llama-3.1-8b-instant",
        "icon": "🚀",
        "description": "Groq · Fastest · Requires API key",
        "speed": "⚡⚡⚡"
    },
    "Qwen3 32B via Groq (Cloud)": {
        "type": "groq",
        "model": "qwen/qwen3-32b",
        "icon": "🌀",
        "description": "Alibaba · Strong reasoning · Requires API key",
        "speed": "⚡⚡"
    },
}

DEFAULT_MODEL = "LLaMA 3.2 3B (Local)"

# --- Embedding Model ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- ChromaDB ---
CHROMA_DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "documents"

# --- Retrieval ---
TOP_K_RESULTS = 5

# --- Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
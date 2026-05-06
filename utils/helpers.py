# utils/helpers.py
import os
import re
import streamlit as st
from config import DEFAULT_MODEL


# ════════════════════════════════════════
# TEXT UTILITIES
# ════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text.
    Removes extra whitespace, blank lines, null characters.
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.replace('\x00', '')
    return text.strip()


def format_chunk_preview(chunk: str, max_length: int = 150) -> str:
    """
    Shortens a chunk for preview display in UI.
    """
    if len(chunk) <= max_length:
        return chunk
    return chunk[:max_length] + "..."


def validate_pdf(file_path: str) -> dict:
    """
    Checks if a file is a valid readable PDF before processing.
    """
    if not os.path.exists(file_path):
        return {"valid": False, "message": "File does not exist"}
    if not file_path.lower().endswith(".pdf"):
        return {"valid": False, "message": "File is not a PDF"}
    if os.path.getsize(file_path) == 0:
        return {"valid": False, "message": "File is empty"}
    return {"valid": True, "message": "PDF is valid"}


def sanitize_filename(filename: str) -> str:
    """
    Removes special characters from filename.
    Used when creating chunk IDs in ChromaDB.
    """
    filename = filename.replace(" ", "_")
    filename = re.sub(r'[^\w\-.]', '', filename)
    return filename


def get_file_size_mb(file_path: str) -> float:
    """Returns file size in MB."""
    size_bytes = os.path.getsize(file_path)
    return round(size_bytes / (1024 * 1024), 2)


# ════════════════════════════════════════
# SESSION STATE MANAGEMENT
# ════════════════════════════════════════

def init_session_state():
    """
    Initializes all Streamlit session state variables.
    Call this once at the top of app.py.
    Keeps app.py clean — no scattered if-not-in checks.
    """
    defaults = {
        "model_histories": {},   # per-model chat memory for LLM context
        "global_timeline": [],   # all messages across all models for display
        "pdf_loaded": False,
        "pdf_name": "",
        "selected_model": DEFAULT_MODEL,
        "model_index": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_current_history() -> list:
    """
    Returns chat history for the currently selected model.
    Used to pass conversation memory to the LLM.
    If model hasn't been used yet, creates an empty list.
    """
    model = st.session_state.selected_model
    if model not in st.session_state.model_histories:
        st.session_state.model_histories[model] = []
    return st.session_state.model_histories[model]


def add_to_history(role: str, content: str, context: list = []):
    """
    Saves a message to two places:
    1. model_histories[current_model] → LLM memory for follow-up questions
    2. global_timeline               → display all messages on screen
    Both stay in sync always.
    """
    model = st.session_state.selected_model
    if model not in st.session_state.model_histories:
        st.session_state.model_histories[model] = []

    entry = {
        "role": role,
        "content": content,
        "context": context,
        "model": model
    }

    st.session_state.model_histories[model].append(entry)
    st.session_state.global_timeline.append(entry)


def clear_current_model_history():
    """
    Clears ONLY the current model's messages.
    Removes from both model_histories and global_timeline.
    Other models are completely untouched.
    """
    model = st.session_state.selected_model
    st.session_state.model_histories[model] = []
    st.session_state.global_timeline = [
        m for m in st.session_state.global_timeline
        if m["model"] != model
    ]


def clear_all_history():
    """
    Clears everything — all models, all messages.
    Called when a new PDF is uploaded or document is cleared.
    """
    st.session_state.model_histories = {}
    st.session_state.global_timeline = []


def on_model_change(models: dict):
    """
    Called when user switches model in the selectbox.
    Updates selected_model and model_index in session state.
    Does NOT clear any history.
    """
    new_model = st.session_state._model_selector
    if new_model not in st.session_state.model_histories:
        st.session_state.model_histories[new_model] = []
    st.session_state.selected_model = new_model
    st.session_state.model_index = list(models.keys()).index(new_model)


def get_total_message_count() -> int:
    """Returns total messages across ALL models."""
    return sum(len(h) for h in st.session_state.model_histories.values())
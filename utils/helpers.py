# utils/helpers.py
import os
import re
import uuid
import tempfile
import streamlit as st
from config import DEFAULT_MODEL


# ════════════════════════════════════════
# TEXT UTILITIES
# ════════════════════════════════════════

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.replace('\x00', '')
    return text.strip()


def format_chunk_preview(chunk: str, max_length: int = 150) -> str:
    if len(chunk) <= max_length:
        return chunk
    return chunk[:max_length] + "..."


def validate_pdf(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {"valid": False, "message": "File does not exist"}
    if not file_path.lower().endswith(".pdf"):
        return {"valid": False, "message": "File is not a PDF"}
    if os.path.getsize(file_path) == 0:
        return {"valid": False, "message": "File is empty"}
    return {"valid": True, "message": "PDF is valid"}


def sanitize_filename(filename: str) -> str:
    filename = filename.replace(" ", "_")
    filename = re.sub(r'[^\w\-.]', '', filename)
    return filename


def get_file_size_mb(file_path: str) -> float:
    return round(os.path.getsize(file_path) / (1024 * 1024), 2)


# ════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════

def init_session_state():
    """
    Initializes every session variable once.
    Only place where defaults are defined.
    """
    defaults = {
        "chats":             [],     # list of all chat dicts
        "active_chat_id":    None,   # currently open chat
        "renaming_chat_id":  None,   # chat being renamed
        "open_menu_id":      None,   # chat showing ⋯ menu
        "show_pdf_uploader": False,  # toggle attach panel
        "model_histories":   {},     # per-model LLM memory
        "global_timeline":   [],     # messages on screen
        "pdf_loaded":        False,
        "pdf_name":          "",
        "selected_model":    DEFAULT_MODEL,
        "model_index":       0,
        "waiting_for_response": False,  # NEW: prevents duplicate responses
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ════════════════════════════════════════
# MESSAGE HISTORY
# ════════════════════════════════════════

def get_current_history() -> list:
    """Returns current model's message list for LLM context."""
    model = st.session_state.selected_model
    if model not in st.session_state.model_histories:
        st.session_state.model_histories[model] = []
    return st.session_state.model_histories[model]


def add_to_history(role: str, content: str, context: list = []):
    """
    Adds one message to:
    1. model_histories → LLM sees this as conversation memory
    2. global_timeline → shown on screen
    """
    model = st.session_state.selected_model
    if model not in st.session_state.model_histories:
        st.session_state.model_histories[model] = []

    entry = {"role": role, "content": content, "context": context, "model": model}
    st.session_state.model_histories[model].append(entry)
    st.session_state.global_timeline.append(entry)


def clear_current_model_history():
    """Clears only current model's messages."""
    model = st.session_state.selected_model
    st.session_state.model_histories[model] = []
    st.session_state.global_timeline = [
        m for m in st.session_state.global_timeline
        if m["model"] != model
    ]


def clear_all_history():
    """Wipes all messages. Called on new chat or document removal."""
    st.session_state.model_histories = {}
    st.session_state.global_timeline = []


def on_model_change(models: dict):
    """Selectbox on_change handler — updates selected model."""
    new = st.session_state._model_selector
    if new not in st.session_state.model_histories:
        st.session_state.model_histories[new] = []
    st.session_state.selected_model = new
    st.session_state.model_index    = list(models.keys()).index(new)


def get_total_message_count() -> int:
    return sum(len(v) for v in st.session_state.model_histories.values())


# ════════════════════════════════════════
# CHAT SESSION MANAGEMENT
# ════════════════════════════════════════

def get_active_chat() -> dict | None:
    """Returns the currently active chat dict, or None."""
    for c in st.session_state.chats:
        if c["id"] == st.session_state.active_chat_id:
            return c
    return None


def save_current_messages():
    """
    Snapshots global_timeline + model_histories into the active chat dict.
    Called before any switch, creation, or clear so nothing is lost.
    """
    cid = st.session_state.active_chat_id
    if not cid:
        return
    for chat in st.session_state.chats:
        if chat["id"] == cid:
            chat["messages"]        = list(st.session_state.global_timeline)
            chat["model_histories"] = {
                k: list(v)
                for k, v in st.session_state.model_histories.items()
            }
            break


def load_chat_messages(chat: dict):
    """
    Restores a saved chat's messages + model memory into session state.
    Called when the user clicks a chat in the sidebar.
    """
    st.session_state.global_timeline  = list(chat.get("messages", []))
    st.session_state.model_histories  = {
        k: list(v)
        for k, v in chat.get("model_histories", {}).items()
    }


def create_chat(pdf_name: str, chunks: int) -> str:
    """
    Saves current chat, then creates a fresh one.
    Returns the new chat_id.
    """
    save_current_messages()

    chat_id = str(uuid.uuid4())
    st.session_state.chats.insert(0, {
        "id":              chat_id,
        "name":            pdf_name,
        "pdf_name":        pdf_name,
        "chunks":          chunks,
        "messages":        [],
        "model_histories": {},
    })
    st.session_state.active_chat_id = chat_id
    clear_all_history()
    return chat_id


def delete_chat(chat_id: str):
    """
    Removes a chat permanently.
    If it was active, loads the next available chat or resets to landing.
    """
    st.session_state.chats = [
        c for c in st.session_state.chats if c["id"] != chat_id
    ]
    if st.session_state.active_chat_id == chat_id:
        if st.session_state.chats:
            nxt = st.session_state.chats[0]
            st.session_state.active_chat_id = nxt["id"]
            st.session_state.pdf_loaded     = True
            st.session_state.pdf_name       = nxt["pdf_name"]
            load_chat_messages(nxt)
        else:
            st.session_state.active_chat_id = None
            st.session_state.pdf_loaded     = False
            st.session_state.pdf_name       = ""
            clear_all_history()
    st.session_state.open_menu_id = None


def process_pdf_upload(uploaded_file) -> dict:
    """
    Writes uploaded file to a temp path, runs the ingest pipeline,
    cleans up, and returns the result dict.
    Import is done inline to avoid circular imports.
    """
    from src.rag_pipeline import ingest_pdf

    with st.spinner("Processing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        result = ingest_pdf(tmp_path, uploaded_file.name)
        os.unlink(tmp_path)
    return result


def switch_to_chat(chat: dict):
    """
    Saves current chat then loads the selected one.
    Called when clicking a chat in the sidebar.
    """
    save_current_messages()
    st.session_state.active_chat_id = chat["id"]
    st.session_state.pdf_loaded     = True
    st.session_state.pdf_name       = chat["pdf_name"]
    st.session_state.open_menu_id   = None
    load_chat_messages(chat)


def start_new_chat():
    """
    Saves current chat, resets everything to landing state.
    Called by the New Chat button.
    """
    save_current_messages()
    st.session_state.active_chat_id   = None
    st.session_state.pdf_loaded       = False
    st.session_state.pdf_name         = ""
    st.session_state.open_menu_id     = None
    st.session_state.renaming_chat_id = None
    st.session_state.show_pdf_uploader = False
    clear_all_history()


def rename_chat(chat: dict, new_name: str):
    """Renames a chat. Trims whitespace, falls back to old name if empty."""
    chat["name"] = new_name.strip() or chat["name"]
    st.session_state.renaming_chat_id = None


def save_after_message():
    """Convenience wrapper — saves after every message exchange."""
    save_current_messages()
# app.py
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import answer_question
from src.llm_handler  import get_available_models, check_model_available
from utils.helpers import (
    init_session_state,
    get_current_history,
    add_to_history,
    on_model_change,
    get_active_chat,
    create_chat,
    delete_chat,
    process_pdf_upload,
    switch_to_chat,
    start_new_chat,
    rename_chat,
    save_after_message,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDIS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background: #212121 !important;
    color: #ececec !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] > section > div:first-child { padding-top: 0 !important; }
[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
.block-container { padding-top: 0 !important; padding-bottom: 0 !important; }
[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer { visibility: hidden !important; }

[data-testid="stSidebar"] {
    background: #171717 !important;
    border-right: 1px solid #2f2f2f !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="collapsedControl"] { color: #ececec !important; background: transparent !important; }

[data-testid="stVerticalBlockBorderWrapper"] {
    border: none !important; box-shadow: none !important;
    padding: 0 !important; margin-top: 0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 0 !important; }
div[data-testid="element-container"]:has(> div[data-testid="stVerticalBlockBorderWrapper"]) {
    padding: 0 !important; margin: 0 !important;
}
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }
.stMainBlockContainer > div > div > div { gap: 0 !important; }

.stButton > button {
    background: transparent !important; border: none !important;
    color: #ececec !important; font-family: 'Inter', sans-serif !important;
    cursor: pointer !important; transition: background 0.15s !important;
    padding: 6px 10px !important; border-radius: 6px !important;
}
.stButton > button:hover { background: #2f2f2f !important; }

[data-testid="stChatInput"] {
    background: #2f2f2f !important;
    border: 1px solid #3f3f3f !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #ececec !important; font-family: 'Inter', sans-serif !important;
    background: transparent !important;
}
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important; padding: 8px 0 !important;
}
[data-testid="stExpander"] {
    background: #2a2a2a !important; border: 1px solid #3f3f3f !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #2f2f2f !important; border: 1px solid #3f3f3f !important;
    color: #ececec !important; border-radius: 12px !important;
    font-size: 13px !important; font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextInput"] > div > div > input {
    background: #2a2a2a !important; border: 1px solid #6366f1 !important;
    border-radius: 6px !important; color: #ececec !important;
    font-size: 13px !important; padding: 5px 8px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stFileUploader"] {
    background: #2a2a2a !important; border: 1px dashed #3f3f3f !important;
    border-radius: 12px !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3f3f3f; border-radius: 4px; }
hr { border-color: #2f2f2f !important; }

.input-row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    display: flex !important; align-items: flex-end !important; padding-bottom: 0 !important;
}
.input-row .stButton > button {
    height: 45px !important; width: 45px !important; padding: 0 !important;
    font-size: 20px !important; border: 1px solid #3f3f3f !important;
    border-radius: 12px !important; background: #2f2f2f !important;
    display: flex !important; align-items: center !important;
    justify-content: center !important; margin-bottom: 1px !important;
}
.input-row .stButton > button:hover {
    border-color: #6366f1 !important; background: #3a3a3a !important;
}
.input-row [data-testid="stSelectbox"] label { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Boot ───────────────────────────────────────────────────────────────────────
init_session_state()
models = get_available_models()

active_chat_exists = any(
    c["id"] == st.session_state.active_chat_id
    for c in st.session_state.chats
)

# Show/hide floating input bar
if not active_chat_exists:
    st.markdown("""
    <style>
    [data-testid="stBottom"], [data-testid="stBottom"] > *,
    div[class*="stChatFloatingInputContainer"],
    section[class*="stChatFloatingInputContainer"] {
        display: none !important; height: 0 !important;
        min-height: 0 !important; overflow: hidden !important;
        padding: 0 !important; margin: 0 !important;
    }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    [data-testid="stBottom"] { display: flex !important; height: auto !important; }
    </style>""", unsafe_allow_html=True)


# ── UI helpers (pure rendering, no logic) ─────────────────────────────────────
def render_message(message: dict):
    """Renders one message bubble. No logic — display only."""
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            m_info = models.get(message.get("model", ""), {})
            st.markdown(
                f"<span style='font-size:11px;color:#6b6b6b;"
                f"font-family:JetBrains Mono,monospace;'>"
                f"{m_info.get('icon','🤖')} {message.get('model','')}</span>",
                unsafe_allow_html=True
            )
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("context"):
            with st.expander("📚 Source Chunks"):
                for i, chunk in enumerate(message["context"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.markdown(
                        f"<small style='color:#8e8ea0;font-family:JetBrains Mono,"
                        f"monospace;font-size:12px;line-height:1.6;'>{chunk}</small>",
                        unsafe_allow_html=True
                    )
                    if i < len(message["context"]) - 1:
                        st.divider()


# ════════════════════════════════════════════════════════
# SIDEBAR — navigation only
# ════════════════════════════════════════════════════════
with st.sidebar:

    # Logo
    st.markdown("""
    <div style="padding:18px 16px 14px;border-bottom:1px solid #2f2f2f;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:30px;height:30px;
                        background:linear-gradient(135deg,#6366f1,#10b981);
                        border-radius:7px;display:flex;align-items:center;
                        justify-content:center;font-size:15px;">🧠</div>
            <span style="font-weight:600;font-size:15px;color:#ececec;">EDIS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New Chat
    st.markdown("<div style='padding:10px 10px 6px;'>", unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True, key="new_chat_btn"):
        start_new_chat()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat list
    if st.session_state.chats:
        st.markdown(
            "<p style='font-size:10px;color:#6b6b6b;padding:4px 16px 4px;"
            "letter-spacing:0.08em;text-transform:uppercase;margin:0;'>Recent</p>",
            unsafe_allow_html=True
        )

        for chat in st.session_state.chats:
            is_active = chat["id"] == st.session_state.active_chat_id
            menu_open = st.session_state.open_menu_id == chat["id"]
            renaming  = st.session_state.renaming_chat_id == chat["id"]
            row_bg    = "background:#2a2a2a;border-radius:8px;" if is_active else ""

            st.markdown(f"<div style='margin:1px 8px;{row_bg}'>", unsafe_allow_html=True)

            if renaming:
                c1, c2, c3 = st.columns([5, 1, 1])
                with c1:
                    new_name = st.text_input(
                        "rename", value=chat["name"],
                        label_visibility="collapsed",
                        key=f"rename_input_{chat['id']}",
                    )
                with c2:
                    if st.button("✓", key=f"ok_{chat['id']}"):
                        rename_chat(chat, new_name)
                        st.rerun()
                with c3:
                    if st.button("✕", key=f"cancel_{chat['id']}"):
                        st.session_state.renaming_chat_id = None
                        st.rerun()

            elif menu_open:
                c1, c2 = st.columns([5, 1])
                with c1:
                    display = chat["name"][:22] + "…" if len(chat["name"]) > 22 else chat["name"]
                    st.markdown(
                        f"<p style='font-size:13px;color:#ececec;padding:6px 4px 2px 8px;"
                        f"margin:0;white-space:nowrap;overflow:hidden;"
                        f"text-overflow:ellipsis;'>{display}</p>",
                        unsafe_allow_html=True
                    )
                with c2:
                    if st.button("✕", key=f"close_menu_{chat['id']}"):
                        st.session_state.open_menu_id = None
                        st.rerun()

                st.markdown("""
                <div style="background:#1e1e1e;border:1px solid #3a3a3a;
                            border-radius:8px;margin:2px 4px 4px;overflow:hidden;">
                """, unsafe_allow_html=True)
                cr, cd = st.columns(2)
                with cr:
                    if st.button("✏️  Rename", key=f"rename_btn_{chat['id']}", use_container_width=True):
                        st.session_state.renaming_chat_id = chat["id"]
                        st.session_state.open_menu_id     = None
                        st.rerun()
                with cd:
                    if st.button("🗑  Delete", key=f"delete_btn_{chat['id']}", use_container_width=True):
                        delete_chat(chat["id"])
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                c1, c2 = st.columns([8, 1])
                with c1:
                    display = chat["name"][:24] + "…" if len(chat["name"]) > 24 else chat["name"]
                    label   = ("▌ " if is_active else "") + display
                    if st.button(label, key=f"select_{chat['id']}", use_container_width=True):
                        if not is_active:
                            switch_to_chat(chat)
                            st.rerun()
                with c2:
                    if st.button("⋯", key=f"dots_{chat['id']}"):
                        st.session_state.open_menu_id = chat["id"]
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            "<p style='font-size:13px;color:#4b4b4b;padding:12px 16px;'>No chats yet</p>",
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════
# MAIN — pure rendering
# ════════════════════════════════════════════════════════
active_chat = get_active_chat()


# ── LANDING PAGE ──────────────────────────────────────────────────────────────
if active_chat is None:
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("<div style='height:80px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;margin-bottom:36px;">
            <div style="width:54px;height:54px;
                        background:linear-gradient(135deg,#6366f1,#10b981);
                        border-radius:14px;display:flex;align-items:center;
                        justify-content:center;font-size:26px;
                        margin:0 auto 14px;">🧠</div>
            <h1 style="font-size:26px;font-weight:600;color:#ececec;margin:0 0 8px;">
                Enterprise Document Intelligence
            </h1>
            <p style="font-size:14px;color:#8e8ea0;margin:0;">
                Upload a PDF to start chatting with your document
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload PDF", type="pdf",
            label_visibility="collapsed",
            key="landing_uploader",
        )

        if uploaded_file is not None:
            st.markdown(
                f"<p style='font-size:13px;color:#10b981;margin:10px 0 12px;'>"
                f"✓ &nbsp;{uploaded_file.name} &nbsp;·&nbsp;"
                f"{round(uploaded_file.size/1024,1)} KB</p>",
                unsafe_allow_html=True
            )
            if st.button("Process Document", type="primary",
                         use_container_width=True, key="process_landing"):
                result = process_pdf_upload(uploaded_file)
                if result["success"]:
                    create_chat(uploaded_file.name, result["chunks"])
                    st.session_state.pdf_loaded = True
                    st.session_state.pdf_name   = uploaded_file.name
                    st.rerun()
                else:
                    st.error(result["message"])

        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        for col, (icon, title, desc) in zip([c1, c2, c3], [
            ("🔍", "Semantic Search", "Finds answers by meaning, not just keywords"),
            ("🤖", "Multi-Model",     "Switch between LLaMA, Gemini, Groq"),
            ("💬", "Memory",          "Each model keeps its own conversation memory"),
        ]):
            with col:
                st.markdown(f"""
                <div style="background:#2a2a2a;border:1px solid #3f3f3f;
                            border-radius:10px;padding:16px 14px;text-align:center;">
                    <div style="font-size:20px;margin-bottom:8px;">{icon}</div>
                    <div style="font-size:13px;font-weight:500;
                                color:#ececec;margin-bottom:4px;">{title}</div>
                    <div style="font-size:12px;color:#8e8ea0;line-height:1.5;">{desc}</div>
                </div>""", unsafe_allow_html=True)


# ── CHAT PAGE ─────────────────────────────────────────────────────────────────
else:
    current_history = get_current_history()

    # Top bar
    st.markdown(f"""
    <div style="padding:16px 24px 8px;">
        <h2 style="font-size:18px;font-weight:600;color:#ececec;margin:0;">
            {active_chat['name']}
        </h2>
        <p style="font-size:12px;color:#8e8ea0;margin:4px 0 0;
                  font-family:'JetBrains Mono',monospace;">
            📄 {active_chat['pdf_name']} &nbsp;·&nbsp; {active_chat['chunks']} chunks
        </p>
    </div>
    <hr style="margin:0 24px 0;border-color:#2f2f2f;">
    """, unsafe_allow_html=True)

    # Messages container
    msg_container = st.container(height=550, border=False)
    with msg_container:
        if not st.session_state.global_timeline:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        min-height:300px;">
                <p style="font-size:15px;color:#3a3a3a;text-align:center;">
                    Ask anything about your document
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.global_timeline:
                render_message(msg)
        
        # Show thinking indicator if we're waiting for response
        if st.session_state.get("waiting_for_response", False):
            with st.chat_message("assistant"):
                st.markdown("🤔 **Thinking...**")
                st.markdown("<span style='font-size:12px;color:#8e8ea0;'>Generating response...</span>", unsafe_allow_html=True)

    # Optional PDF attach panel
    if st.session_state.show_pdf_uploader:
        st.markdown("""
        <div style="background:#2a2a2a;border:1px solid #3f3f3f;border-radius:10px;
                    padding:16px 16px 12px;margin:0 24px 10px;">
            <p style="font-size:12px;color:#8e8ea0;margin:0 0 10px;">
                Upload a new PDF — creates a new chat
            </p>""", unsafe_allow_html=True)
        new_pdf = st.file_uploader(
            "New PDF", type="pdf",
            label_visibility="collapsed", key="attach_uploader"
        )
        if new_pdf is not None:
            ci, cb = st.columns([3, 1])
            with ci:
                st.markdown(
                    f"<p style='font-size:13px;color:#10b981;margin:0;'>"
                    f"✓ {new_pdf.name} &nbsp;·&nbsp; {round(new_pdf.size/1024,1)} KB</p>",
                    unsafe_allow_html=True
                )
            with cb:
                if st.button("Process", key="process_attach"):
                    result = process_pdf_upload(new_pdf)
                    if result["success"]:
                        create_chat(new_pdf.name, result["chunks"])
                        st.session_state.pdf_loaded        = True
                        st.session_state.pdf_name          = new_pdf.name
                        st.session_state.show_pdf_uploader = False
                        st.rerun()
                    else:
                        st.error(result["message"])
        st.markdown("</div>", unsafe_allow_html=True)

    # Input row
    st.markdown('<div class="input-row" style="padding:0 24px;">', unsafe_allow_html=True)
    cp, cc, cm = st.columns([1, 9, 3])
    with cp:
        if st.button("＋", key="attach_btn", help="Upload a new PDF"):
            st.session_state.show_pdf_uploader = not st.session_state.show_pdf_uploader
            st.rerun()
    with cc:
        question = st.chat_input(f"Ask about {active_chat['pdf_name']}...", disabled=st.session_state.get("waiting_for_response", False))
    with cm:
        st.selectbox(
            "Model", options=list(models.keys()),
            index=st.session_state.model_index,
            label_visibility="collapsed",
            key="_model_selector",
            on_change=lambda: on_model_change(models),
            disabled=st.session_state.get("waiting_for_response", False),
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Handle new question
    if question and not st.session_state.get("waiting_for_response", False):
        # Add user message to history and save
        add_to_history("user", question)
        save_after_message()
        
        # Set waiting flag and rerun to show the thinking indicator
        st.session_state.waiting_for_response = True
        st.rerun()

    # Generate assistant response
    if (st.session_state.get("waiting_for_response", False) and
        st.session_state.global_timeline and 
        st.session_state.global_timeline[-1]["role"] == "user"):
        
        # Get the last user message
        last_user_msg = st.session_state.global_timeline[-1]
        user_question = last_user_msg["content"]
        
        # Generate response
        result = answer_question(
            user_question,
            st.session_state.selected_model,
            get_current_history()[:-1],  # Exclude the just-added user message
        )
        
        # Remove the thinking indicator by clearing and re-adding messages
        # We need to remove the temporary thinking message from display
        # The easiest way is to clear and rebuild the conversation
        
        # Add assistant response to history
        add_to_history("assistant", result["answer"], result.get("context", []))
        save_after_message()
        
        # Clear waiting flag
        st.session_state.waiting_for_response = False
        
        # Rerun to show the final response
        st.rerun()
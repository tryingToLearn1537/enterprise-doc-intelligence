# app.py
import streamlit as st
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import ingest_pdf, answer_question
from src.llm_handler import get_available_models, check_model_available
from src.vector_store import get_collection_count
from config import DEFAULT_MODEL
from utils.helpers import (
    init_session_state,
    get_current_history,
    add_to_history,
    clear_current_model_history,
    clear_all_history,
    on_model_change,
    get_total_message_count,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="EDIS — Enterprise Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f;
    color: #e2e8f0;
    font-family: 'Syne', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
}
[data-testid="stSidebar"] {
    background-color: #0f0f1a !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}
.edis-header {
    display: flex; align-items: center; gap: 16px;
    padding: 32px 0 8px 0;
    border-bottom: 1px solid rgba(99, 102, 241, 0.25);
    margin-bottom: 8px;
}
.edis-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #6366f1, #10b981);
    border-radius: 12px; display: flex; align-items: center;
    justify-content: center; font-size: 24px; flex-shrink: 0;
}
.edis-title {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 28px;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #a5b4fc, #6ee7b7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.1;
}
.edis-subtitle {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: #64748b; letter-spacing: 2px; text-transform: uppercase;
    margin: 4px 0 0 0;
}
.sidebar-label {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    letter-spacing: 2px; text-transform: uppercase;
    color: #475569; margin: 16px 0 8px 0;
}
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.05) !important;
    border: 1px dashed rgba(99,102,241,0.3) !important;
    border-radius: 10px !important; padding: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.5px !important;
    padding: 10px 20px !important; transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #818cf8, #6366f1) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
}
[data-testid="stChatMessage"] {
    background: rgba(15,15,26,0.8) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important; margin-bottom: 12px !important;
    padding: 4px !important;
}
[data-testid="stChatInput"] {
    background: #0f0f1a !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Syne', sans-serif !important;
    color: #e2e8f0 !important; background: transparent !important;
}
[data-testid="stExpander"] {
    background: rgba(99,102,241,0.05) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 8px !important;
}
[data-testid="stMetric"] {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important; padding: 12px 16px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #a5b4fc !important; font-size: 28px !important;
}
.feature-card {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 12px; padding: 24px 20px; height: 100%;
}
.feature-icon { font-size: 28px; margin-bottom: 12px; }
.feature-title {
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 15px; color: #a5b4fc; margin-bottom: 8px;
}
.feature-desc { font-size: 13px; color: #64748b; line-height: 1.6; }
.doc-card {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
}
.doc-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: #6ee7b7; word-break: break-all;
}
.model-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: #a5b4fc; background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px; padding: 2px 10px; margin-bottom: 6px;
}
hr { border-color: rgba(99,102,241,0.15) !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Initialize all session state in one clean call ──
init_session_state()

models = get_available_models()


# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div style="padding: 20px 0 8px 0;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            <div style="width:36px;height:36px;
                        background:linear-gradient(135deg,#6366f1,#10b981);
                        border-radius:9px;display:flex;align-items:center;
                        justify-content:center;font-size:18px;flex-shrink:0;">🧠</div>
            <span style="font-family:'Syne',sans-serif;font-weight:800;font-size:20px;
                         background:linear-gradient(90deg,#a5b4fc,#6ee7b7);
                         -webkit-background-clip:text;
                         -webkit-text-fill-color:transparent;">EDIS</span>
        </div>
        <p style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#475569;
                  letter-spacing:2px;text-transform:uppercase;margin:0 0 0 48px;">
            Document Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Model Selector ──
    st.markdown('<p class="sidebar-label">Select Model</p>', unsafe_allow_html=True)

    st.selectbox(
        "Model",
        options=list(models.keys()),
        index=st.session_state.model_index,
        label_visibility="collapsed",
        key="_model_selector",
        on_change=lambda: on_model_change(models),
    )

    # Model Info Card
    m = models[st.session_state.selected_model]
    is_ready = check_model_available(st.session_state.selected_model)
    current_history = get_current_history()
    total_msgs = get_total_message_count()
    ready_html = "<span style='color:#10b981;'>✅ Ready</span>" if is_ready \
                 else "<span style='color:#ef4444;'>❌ Not Available</span>"

    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.06);
                border:1px solid rgba(99,102,241,0.18);
                border-radius:8px;padding:10px 12px;margin:4px 0 8px 0;">
        <div style="font-size:20px;margin-bottom:4px;">{m['icon']}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    color:#a5b4fc;margin-bottom:2px;">{m['description']}</div>
        <div style="font-size:11px;color:#475569;margin-bottom:4px;">Speed: {m['speed']}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;">{ready_html}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    color:#475569;margin-top:4px;">
            💬 {len(current_history)} this model · {total_msgs} total
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(current_history) > 0:
        if st.button("🗑️ Clear This Model's Chat", use_container_width=True):
            clear_current_model_history()
            st.rerun()

    st.divider()

    # ── Upload Section ──
    st.markdown('<p class="sidebar-label">Document Upload</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop PDF here",
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        st.caption(f"📄 {uploaded_file.name}")
        st.caption(f"📦 {round(uploaded_file.size / 1024, 1)} KB")

        if st.button("⚡  Process Document", type="primary", use_container_width=True):
            with st.spinner("Parsing · Chunking · Indexing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                result = ingest_pdf(tmp_path, uploaded_file.name)
                os.unlink(tmp_path)

                if result["success"]:
                    st.session_state.pdf_loaded = True
                    st.session_state.pdf_name = uploaded_file.name
                    clear_all_history()  # ← clean function call
                    st.success(f"✅ Indexed {result['chunks']} chunks")
                else:
                    st.error(result["message"])

    if st.session_state.pdf_loaded:
        st.divider()
        st.markdown('<p class="sidebar-label">Active Document</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-name">📄 {st.session_state.pdf_name}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", get_collection_count())
        with col2:
            st.metric("Total Chats", total_msgs)

        if st.button("🗑️  Clear Document + All Chats", use_container_width=True):
            from src.vector_store import clear_collection
            clear_collection()
            st.session_state.pdf_loaded = False
            st.session_state.pdf_name = ""
            clear_all_history()  # ← clean function call
            st.rerun()

    st.divider()
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                color:#334155;line-height:1.8;">
        LLaMA · Gemini · Groq · ChromaDB<br>
        sentence-transformers · LangChain<br>
        BE AI&ML · Final Year Project
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════

st.markdown("""
<div class="edis-header">
    <div class="edis-logo">🧠</div>
    <div>
        <p class="edis-title">Enterprise Document Intelligence</p>
        <p class="edis-subtitle">RAG · Vector Search · Multi-Model · Local + Cloud</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Empty State ──
if not st.session_state.pdf_loaded:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-family:'JetBrains Mono',monospace;font-size:12px;
              color:#475569;letter-spacing:1px;text-align:center;margin-bottom:32px;">
        UPLOAD A DOCUMENT FROM THE SIDEBAR TO BEGIN
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("📄", "PDF Ingestion",
         "Extracts and chunks text from any PDF using PyMuPDF with smart overlap preservation"),
        ("🔍", "Semantic Search",
         "Finds relevant content by meaning using sentence-transformers + ChromaDB vector store"),
        ("🤖", "Multi-Model AI",
         "Switch between models freely — each keeps its own memory, all shown in one timeline"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-family:'JetBrains Mono',monospace;font-size:10px;
              color:#334155;text-align:center;letter-spacing:1px;">
        AVAILABLE MODELS
    </p>
    """, unsafe_allow_html=True)

    model_cols = st.columns(len(models))
    for col, (name, info) in zip(model_cols, models.items()):
        with col:
            ready = check_model_available(name)
            st.markdown(f"""
            <div style="text-align:center;padding:8px 4px;
                        background:rgba(99,102,241,0.04);
                        border:1px solid rgba(99,102,241,0.12);
                        border-radius:8px;">
                <div style="font-size:16px;">{info['icon']}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:8px;
                            color:#64748b;margin-top:2px;line-height:1.4;">
                    {name.split('(')[0].strip()}
                </div>
                <div style="font-size:9px;margin-top:2px;">
                    {"✅" if ready else "⚪"}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align:center;font-family:'JetBrains Mono',monospace;
              font-size:10px;color:#334155;">
        PyMuPDF · sentence-transformers · ChromaDB · LangChain · Streamlit
    </p>
    """, unsafe_allow_html=True)

# ── Chat State ──
else:
    active_model = models[st.session_state.selected_model]
    current_history = get_current_history()

    # Active model banner
    st.markdown(f"""
    <div style="display:inline-flex;align-items:center;gap:8px;
                background:rgba(99,102,241,0.08);
                border:1px solid rgba(99,102,241,0.2);
                border-radius:100px;padding:4px 14px;margin-bottom:16px;">
        <span style="font-size:14px;">{active_model['icon']}</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#a5b4fc;">
            {st.session_state.selected_model}
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#475569;">
            · {len(current_history)} messages · {get_total_message_count()} total
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Render global timeline — never clears on model switch ──
    for message in st.session_state.global_timeline:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                msg_model = models.get(message.get("model", ""), {})
                st.markdown(
                    f'<div class="model-badge">'
                    f'{msg_model.get("icon","🤖")}&nbsp;{message.get("model","")}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("context"):
                with st.expander("📚 Source Chunks"):
                    for i, chunk in enumerate(message["context"]):
                        st.markdown(f"**Chunk {i+1}**")
                        st.markdown(
                            f"<small style='color:#94a3b8;"
                            f"font-family:JetBrains Mono,monospace;"
                            f"line-height:1.6;'>{chunk}</small>",
                            unsafe_allow_html=True
                        )
                        if i < len(message["context"]) - 1:
                            st.divider()

    # ── Chat Input ──
    if question := st.chat_input(
        f"Ask anything about {st.session_state.pdf_name}..."
    ):
        add_to_history("user", question)

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                f"Retrieving · Reasoning via {st.session_state.selected_model}..."
            ):
                result = answer_question(
                    question,
                    st.session_state.selected_model,
                    current_history[:-1]
                )

            st.markdown(
                f'<div class="model-badge">'
                f'{active_model.get("icon","🤖")}&nbsp;{st.session_state.selected_model}'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(result["answer"])

            if result["success"] and result["context"]:
                with st.expander("📚 Source Chunks"):
                    for i, chunk in enumerate(result["context"]):
                        st.markdown(f"**Chunk {i+1}**")
                        st.markdown(
                            f"<small style='color:#94a3b8;"
                            f"font-family:JetBrains Mono,monospace;"
                            f"line-height:1.6;'>{chunk}</small>",
                            unsafe_allow_html=True
                        )
                        if i < len(result["context"]) - 1:
                            st.divider()

        add_to_history("assistant", result["answer"], result.get("context", []))
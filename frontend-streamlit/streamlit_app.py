import streamlit as st
import requests
import os
from typing import List, Dict, Any

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="EduQuery AI - Academic RAG Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint configuration
API_HOST = os.getenv("API_HOST", "http://127.0.0.1:8000")

# Custom premium CSS styling for dark mode glassmorphism
CUSTOM_CSS = """
<style>
    /* Import premium modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* General styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #0c0f1d;
        color: #e2e8f0;
    }
    
    /* Custom Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #080a13 !important;
        border-right: 1px solid #1f293d;
    }
    
    /* Header titles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Glassmorphic cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 242, 254, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.05);
    }
    
    /* Premium file uploader */
    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.6);
        border: 2px dashed rgba(79, 172, 254, 0.5);
        border-radius: 12px;
        padding: 10px;
    }
    
    /* Button custom styles */
    .stButton>button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: #000000 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4) !important;
        opacity: 0.95;
    }
    
    /* Delete buttons (red gradient) */
    .delete-btn-container .stButton>button {
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%) !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        padding: 4px 12px !important;
        box-shadow: 0 2px 10px rgba(239, 68, 68, 0.2) !important;
        font-size: 0.8rem !important;
    }
    
    .delete-btn-container .stButton>button:hover {
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4) !important;
    }
    
    /* Source pill styling */
    .source-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(79, 172, 254, 0.1);
        border: 1px solid rgba(79, 172, 254, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.85rem;
        color: #00f2fe;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .source-pill:hover {
        background: rgba(79, 172, 254, 0.2);
        transform: scale(1.02);
    }
    
    /* Prompt suggestion chips */
    .suggestion-chip {
        display: inline-block;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        padding: 8px 18px;
        margin: 5px;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        text-align: center;
        color: #e2e8f0;
    }
    
    .suggestion-chip:hover {
        background: rgba(0, 242, 254, 0.1);
        border-color: #00f2fe;
        color: #00f2fe;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "status" not in st.session_state:
    st.session_state.status = {"active_db": "Loading...", "document_count": 0, "status": "offline"}
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""

# Function to fetch library documents
def fetch_documents():
    try:
        response = requests.get(f"{API_HOST}/api/documents", timeout=5)
        if response.status_code == 200:
            st.session_state.documents = response.json().get("documents", [])
    except Exception as e:
        print(f"Error fetching documents: {e}")

# Function to fetch system status
def fetch_status():
    try:
        response = requests.get(f"{API_HOST}/api/status", timeout=5)
        if response.status_code == 200:
            st.session_state.status = response.json()
    except Exception as e:
        st.session_state.status = {"active_db": "None", "document_count": 0, "status": "offline"}

# Trigger setup fetches on load
fetch_documents()
fetch_status()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown('<div class="glass-card" style="padding: 15px; margin-bottom: 20px;">', unsafe_allow_html=True)
    st.markdown("### 🎓 EduQuery AI Workspace")
    
    # Render active database and LLM status
    db_type = st.session_state.status.get("active_db", "offline")
    status_label = "🟢 Connected" if st.session_state.status.get("status") == "healthy" else "🔴 API Offline"
    
    db_badge = "ChromaDB" if db_type == "chromadb" else "SQLite (Fallback Engine)" if db_type == "sqlite" else "Disconnected"
    db_color = "#38bdf8" if db_type == "chromadb" else "#fbbf24" if db_type == "sqlite" else "#ef4444"
    
    st.markdown(f"**System Status**: `{status_label}`")
    st.markdown(f"**Database**: <span style='color: {db_color}; font-weight: bold;'>{db_badge}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 1. File Upload Zone
    st.markdown("### 📥 Add Study Materials")
    uploaded_files = st.file_uploader(
        "Upload PDFs, PowerPoint Slides (.pptx) or Text files:",
        type=["pdf", "pptx", "ppt", "txt", "md"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        upload_btn = st.button("🚀 Process & Index Materials")
        if upload_btn:
            for file in uploaded_files:
                with st.spinner(f"Parsing & embedding '{file.name}'..."):
                    try:
                        files = {"file": (file.name, file.getvalue(), file.type)}
                        response = requests.post(f"{API_HOST}/api/upload", files=files, timeout=120)
                        if response.status_code == 200:
                            data = response.json()
                            st.toast(f"✅ Indexed {file.name} ({data.get('chunks_count', 0)} chunks)", icon="👉")
                        else:
                            st.error(f"Failed to process {file.name}: {response.json().get('detail')}")
                    except Exception as e:
                        st.error(f"Connection error uploading {file.name}: {e}")
            # Refresh doc list and status
            fetch_documents()
            fetch_status()
            st.rerun()

    st.markdown("---")
    
    # 2. Document Library list
    st.markdown("### 📚 Study Materials Library")
    if not st.session_state.documents:
        st.info("No study materials uploaded yet. Drag and drop documents above to start learning!")
    else:
        for doc in st.session_state.documents:
            file_name = doc["source"]
            doc_type = doc["type"].upper()
            
            # Icon styling
            icon = "📖" if doc_type == "PDF" else "📊" if doc_type in ["PPTX", "PPT"] else "📄"
            
            # Info label details
            info_str = ""
            if doc_type == "PDF" and doc.get("total_pages", 0) > 0:
                info_str = f" ({doc['total_pages']} pages)"
            elif doc_type in ["PPTX", "PPT"] and doc.get("slide_count", 0) > 0:
                info_str = f" ({doc['slide_count']} slides)"
                
            col1, col2 = st.columns([4, 1.2])
            with col1:
                st.markdown(f"**{icon} {file_name}**\n<span style='font-size:0.8rem; color:#94a3b8;'>{doc_type}{info_str}</span>", unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="delete-btn-container">', unsafe_allow_html=True)
                # Unique delete key
                if st.button("🗑️", key=f"del_{file_name}"):
                    with st.spinner("Deleting..."):
                        try:
                            res = requests.delete(f"{API_HOST}/api/documents/{file_name}", timeout=10)
                            if res.status_code == 200:
                                st.toast(f"Deleted {file_name}")
                                fetch_documents()
                                fetch_status()
                                st.rerun()
                            else:
                                st.error("Delete failed.")
                        except Exception as e:
                            st.error(f"Error deleting: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<hr style='margin: 8px 0; opacity: 0.15;' />", unsafe_allow_html=True)


# ----------------- MAIN LAYOUT -----------------

# Beautiful animated top headline
st.markdown("""
    <div style='text-align: center; margin-top: 20px; margin-bottom: 25px;'>
        <h1 style='font-size: 3.2rem; margin-bottom: 0px;'>🎓 EduQuery AI</h1>
        <p style='color: #94a3b8; font-size: 1.2rem; max-width: 700px; margin: 10px auto;'>
            Your expert academic tutor. Ask questions, clarify topics, and summarize chapters based <strong>strictly</strong> on your uploaded course slides and notes.
        </p>
    </div>
""", unsafe_allow_html=True)

# 1. Check if materials exist. If not, prompt to upload.
if not st.session_state.documents:
    st.markdown('<div class="glass-card" style="text-align: center; padding: 40px;">', unsafe_allow_html=True)
    st.markdown("""
        <h3>👋 Welcome to EduQuery AI!</h3>
        <p style='color: #94a3b8; margin: 15px 0;'>
            To get started, please upload your textbooks, syllabus, lecture notes, or slides in the sidebar on the left.<br>
            Once uploaded, the AI will parse your materials and index them, enabling it to answer queries using only your specific educational documents.
        </p>
    """, unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?auto=format&fit=crop&q=80&w=400", width=250)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # 2. Main Q&A Interface
    
    # Horizontal suggestion chips
    st.markdown("<p style='font-size: 0.95rem; color: #94a3b8; margin-bottom: 5px; font-weight: 500;'>💡 Quick Prompts:</p>", unsafe_allow_html=True)
    
    quick_queries = [
        "Explain Unit in detail",
        "Give important exam questions",
        "Summarize this study material",
    ]
    
    # We use a row of columns to hold buttons that set session state and rerun
    cols = st.columns(len(quick_queries))
    for idx, query in enumerate(quick_queries):
        with cols[idx]:
            if st.button(query, key=f"quick_{idx}"):
                st.session_state.prompt_input = query
                st.session_state.user_query = query
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Message log rendering
    st.markdown('<div class="glass-card" style="min-height: 350px;">', unsafe_allow_html=True)
    st.markdown("### 💬 Classroom Chat Thread")
    
    # Display clear instructions if chat is empty
    if not st.session_state.messages:
        st.markdown("""
            <div style='color: #64748b; font-style: italic; text-align: center; margin-top: 80px;'>
                Classroom chat initialized. Enter a question below, or select a Quick Prompt chip to begin.
            </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            
            with st.chat_message(role):
                st.write(content)
                
                # If there are sources, render beautiful reference pills
                if role == "assistant" and msg.get("sources"):
                    st.markdown("<br><strong style='font-size:0.85rem; color:#94a3b8;'>📖 Citations & Context Sources:</strong>", unsafe_allow_html=True)
                    
                    for src in msg["sources"]:
                        doc_name = src["document"]
                        ref = src["reference"]
                        doc_type = src["type"]
                        title = src.get("title", "")
                        
                        # Style pill depending on type
                        type_icon = "📊" if doc_type == "pptx" else "📖"
                        tooltip_text = f"{doc_name} ({ref})"
                        if title:
                            tooltip_text += f" - {title}"
                            
                        st.markdown(
                            f'<div class="source-pill" title="{tooltip_text}">'
                            f'{type_icon} {doc_name} • {ref}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if a prompt was set by suggestion chips
    initial_value = st.session_state.prompt_input
    
    # User message input bar
    user_input = st.chat_input("Ask a question about your studies...", key="chat_input_bar")
    
    # Process user query
    active_query = user_input or (st.session_state.prompt_input if st.session_state.prompt_input else None)
    
    if active_query:
        # Clear prompt_input to avoid double-processing on rerun
        st.session_state.prompt_input = ""
        
        # Append User Message to session state log
        st.session_state.messages.append({"role": "user", "content": active_query})
        
        # Immediate visual feedback
        st.rerun()
        
    # Check if the last message in the thread is from the user (requires a response)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        latest_user_msg = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing study materials & generating tutor response..."):
                try:
                    payload = {"query": latest_user_msg, "n_results": 5}
                    res = requests.post(f"{API_HOST}/api/query", json=payload, timeout=90)
                    
                    if res.status_code == 200:
                        data = res.json()
                        ans = data.get("answer", "")
                        sources = data.get("sources", [])
                        
                        # Render Assistant answer
                        st.write(ans)
                        
                        # Render citations if sources are returned
                        if sources:
                            st.markdown("<br><strong style='font-size:0.85rem; color:#94a3b8;'>📖 Citations & Context Sources:</strong>", unsafe_allow_html=True)
                            for src in sources:
                                doc_name = src["document"]
                                ref = src["reference"]
                                doc_type = src["type"]
                                type_icon = "📊" if doc_type == "pptx" else "📖"
                                st.markdown(
                                    f'<div class="source-pill" title="{doc_name} ({ref})">'
                                    f'{type_icon} {doc_name} • {ref}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                                
                        # Save response details in session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ans,
                            "sources": sources
                        })
                    else:
                        err_detail = res.json().get("detail", "Unknown server error.")
                        err_text = f"⚠️ **Connection/Server Error (FastAPI)**: {err_detail}"
                        st.error(err_text)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": err_text,
                            "sources": []
                        })
                except Exception as e:
                    err_text = f"⚠️ **Connection Error**: Could not connect to the backend server at `{API_HOST}`. Please check if the FastAPI server is running."
                    st.error(err_text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_text,
                        "sources": []
                    })
        # Rerun to cleanly update the thread
        st.rerun()

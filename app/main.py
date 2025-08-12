# main.py
import json
import os
import io
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components  

from qa_engine import (
    ask_question, load_documents, build_faiss_index,
    documents, embeddings
)
import qa_engine  # to refresh state after uploads


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Page config ----------
st.set_page_config(
    page_title="Immigration Case Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
:root{
  --card-bg:#14171a;
  --soft:#1b1f23;
  --accent:#4C8BF5;
  --accent-2:#C5D7FE;
  --success:#22c55e;
  --muted:#9aa4ad;
  --border:#2a2f36;
  --radius:14px;
}
section.main > div {padding-top: 1.0rem;}
/* Headings */
h1, h2, h3 { letter-spacing: .2px; }
/* Cards */
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 18px 14px 18px;
}
.card + .card { margin-top: 12px; }
/* Primary button */
.stButton > button {
  background: var(--accent);
  color: white;
  border-radius: 12px;
  border: 1px solid transparent;
  padding: 0.55rem 1.0rem;
  font-weight: 600;
}
.stButton > button:hover { filter: brightness(1.08); }
/* Secondary button look (used for copy) */
button[kind="secondary"] {
  background: #20252b !important;
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
/* Answer blocks */
.kp ul { margin: .25rem 0 .25rem 1.1rem; }
.kp li { margin: .2rem 0; }
.conf { color: var(--muted); font-size: 0.92rem; }
hr.soft { border: none; border-top: 1px solid var(--border); margin: 10px 0; }

/* ---------- Sidebar (chips list) ---------- */
section[data-testid="stSidebar"] > div { padding-top: .5rem; }
.files-head { font-weight:700; margin: .25rem 0 .5rem 0; }
.file-list { 
  margin: .25rem 0 1rem 0; padding: 0; list-style: none;
  display:flex; flex-direction:column; gap:.25rem;
}
.chip {
  display:flex; align-items:center; gap:.5rem;
  padding:.35rem .55rem; border-radius: 999px;
  background: #0f1114; border:1px solid #1f252c;
  font-size:.88rem; color:#d1d5db; width: fit-content; max-width: 100%;
}
.chip .dot { width:.42rem;height:.42rem;border-radius:50%;background:#16a34a;flex:0 0 auto; }
.chip .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
/* Uploader subtle */
div[data-testid="stFileUploader"] > div:first-child {
  border: 1px dashed rgba(255,255,255,.14);
  background: rgba(255,255,255,.03);
  border-radius: .65rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar: files + upload ----------
with st.sidebar:
    st.markdown('<div class="files-head">Files</div>', unsafe_allow_html=True)

    Path(DATA_DIR).mkdir(exist_ok=True)
    files = sorted([f for f in Path(DATA_DIR).iterdir()
                    if f.is_file() and not f.name.startswith('.')])

    if files:
        chips = "\n".join(
            [f'<li><div class="chip"><span class="dot"></span>'
             f'<span class="name">{f.name}</span></div></li>' for f in files]
        )
        st.markdown(f'<ul class="file-list">{chips}</ul>', unsafe_allow_html=True)
        st.caption(f"{len(files)} file{'s' if len(files)!=1 else ''} indexed")
    else:
        st.caption("No documents uploaded yet.")

    st.markdown("#### Add PDF or DOCX")


    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    col_a, col_b = st.columns([1, 1])
    with col_a:
        clear = st.button("Clear uploads")
    with col_b:
        refresh = st.button("Re-index")

    if clear:
        for f in Path(DATA_DIR).glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass
        st.session_state.uploader_key += 1
        st.rerun()

    uploads = st.file_uploader(
        "Drag and drop file here",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed"
    )

# ---------- Handle new uploads & (re)index ----------
new_files_saved = False
if uploads:
    Path(DATA_DIR).mkdir(exist_ok=True)
    for up in uploads:
        path = Path(DATA_DIR) / up.name
        with open(path, "wb") as f:
            f.write(up.getbuffer())
        new_files_saved = True

if new_files_saved or refresh:
    qa_engine.documents.clear()
    qa_engine.embeddings.clear()
    qa_engine.load_documents()
    qa_engine.faiss_index = qa_engine.build_faiss_index()
    st.success("Files uploaded & indexed.")


# ---------- Title & subtitle ----------
st.title("Immigration Case Brain")
st.caption(
    "Ask pointed questions about your RFEs, denials, approval memos, and support letters. "
    "You'll get a direct answer plus key supporting snippets with citations."
)

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Question form (Enter submits) ----------
with st.form("qa_form", clear_on_submit=False):
    q = st.text_input(
        "Your question",
        placeholder="e.g., What regulation was cited in the denial?",
        label_visibility="visible"
    )
    submit = st.form_submit_button("Answer")

# ---------- Answer area ----------
if submit and q.strip():
    if not embeddings:
        st.error("No documents indexed yet. Upload a PDF/DOCX in the left panel.")
    else:
        with st.spinner("Thinking..."):
            ans_md = ask_question(q.strip())


      
        st.markdown("Answer!!")   
        st.markdown(ans_md)  


       
        col1, col2, _ = st.columns([0.15, 0.12, 0.73])
        with col1:
          
            dt = datetime.now().strftime("%Y-%m-%d_%H%M")
            md_bytes = io.BytesIO(ans_md.encode("utf-8"))
            st.download_button(
                "Download",
                md_bytes,
                file_name=f"answer_{dt}.md",
                mime="text/markdown"
            )

        def copy_button(text: str, label="Copy"):
           
            js_text = json.dumps(text)  

            components.html(f"""
            <button id="copybtn" style="
            background:#20252b;border:1px solid #2a2f36;border-radius:10px;
            padding:.45rem .9rem;color:#cbd5e1;cursor:pointer;font-weight:600;">
          {label}
        </button>
        <script>
            const btn = document.getElementById('copybtn');
            if (btn) {{
                btn.onclick = async () => {{
                    try {{
                        await navigator.clipboard.writeText({js_text});
                        btn.innerText = 'Copied!';
                        setTimeout(() => btn.innerText = '{label}', 1200);
                    }} catch (e) {{
                        btn.innerText = 'Copy failed';
                        setTimeout(() => btn.innerText = '{label}', 1200);
                    }}
                }};
            }}
        </script>
        """, height=48)


        with col2:
            copy_button(ans_md, "Copy")

        # Save to simple session history
        st.session_state.history.insert(0, {"q": q.strip(), "a": "answer shown above"})

# ---------- Recent Q&A ----------
if st.session_state.history:
    st.markdown("### Recent (this session)")
    for item in st.session_state.history[:5]:
        st.markdown(f"**Q**  {item['q']}")
        st.markdown(f"<span class='conf'>**A**  {item['a']}</span>", unsafe_allow_html=True)

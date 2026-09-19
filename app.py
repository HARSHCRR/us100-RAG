import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Support both local (.env) and Streamlit Cloud (st.secrets)
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

from rag_chain import build_rag_chain

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US100 Macro Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

    /* Title */
    .hero-title {
        font-size: 2.2rem; font-weight: 700; color: #58a6ff;
        letter-spacing: -0.5px; margin-bottom: 0;
    }
    .hero-sub {
        font-size: 1rem; color: #8b949e; margin-top: 4px; margin-bottom: 28px;
    }

    /* Answer box */
    .answer-box {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 20px 24px; margin-top: 16px;
        line-height: 1.75;
    }

    /* Source card */
    .source-card {
        background: #0d1117; border: 1px solid #21262d;
        border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
        font-size: 0.85rem; color: #8b949e;
    }
    .source-card .date { color: #58a6ff; font-weight: 600; }
    .source-card .bull { color: #3fb950; }
    .source-card .bear { color: #f85149; }

    /* Stats in sidebar */
    .stat-item { padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 0.88rem; }
    .stat-label { color: #8b949e; }
    .stat-value { color: #e6edf3; font-weight: 600; float: right; }

    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important; border-radius: 8px !important;
    }
    .stButton > button {
        background: #1f6feb; color: white; border: none;
        border-radius: 8px; font-weight: 600; padding: 0.5rem 1.5rem;
        width: 100%;
    }
    .stButton > button:hover { background: #388bfd; }

    div[data-testid="stSpinner"] > div { color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ── Load chain (cached so it only runs once) ───────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG chain... (~10 seconds)")
def get_chain():
    return build_rag_chain()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 System Info")
    st.markdown("""
    <div class='stat-item'><span class='stat-label'>Dataset</span><span class='stat-value'>NQ Futures</span></div>
    <div class='stat-item'><span class='stat-label'>Period</span><span class='stat-value'>2016 – 2026</span></div>
    <div class='stat-item'><span class='stat-label'>Documents</span><span class='stat-value'>2,517</span></div>
    <div class='stat-item'><span class='stat-label'>Embeddings</span><span class='stat-value'>BAAI/bge-small-en</span></div>
    <div class='stat-item'><span class='stat-label'>Vector DB</span><span class='stat-value'>FAISS (local)</span></div>
    <div class='stat-item'><span class='stat-label'>LLM</span><span class='stat-value'>Gemini 3.6 Flash</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 💬 Example Queries")
    examples = [
        "How did NQ react to hot CPI prints?",
        "What happened on FOMC days in 2022?",
        "Show NFP days where NQ was bearish",
        "What is the latest CPI data?",
        "Average NQ range on jobless claims beats?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["query_input"] = ex

# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>📈 US100 Macro Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Ask questions about 10 years of NQ/NASDAQ-100 macro events — powered by RAG + Gemini</div>", unsafe_allow_html=True)

query = st.text_input(
    label="query",
    label_visibility="collapsed",
    placeholder="e.g. How did NQ react to hot CPI prints historically?",
    key="query_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_clicked = st.button("🔍 Ask", use_container_width=True)

if ask_clicked and query.strip():
    chain = get_chain()
    with st.spinner("Retrieving context and generating answer..."):
        result = chain({"query": query})

    answer = result["result"]
    source_docs = result["source_documents"]

    # Answer
    st.markdown("### 🤖 Answer")
    st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

    # Source documents
    st.markdown("<br>### 📂 Retrieved Source Documents", unsafe_allow_html=True)
    for doc in source_docs:
        m = doc.metadata
        bias = m.get("overall_bias", "N/A")
        bias_class = "bull" if bias == "BULLISH" else "bear"
        cpi   = "✅" if m.get("has_cpi_event") else "—"
        nfp   = "✅" if m.get("has_nfp_event") else "—"
        fomc  = "✅" if m.get("has_fed_event") else "—"

        with st.expander(f"📅 {m.get('date', '?')}  |  {bias}  |  CPI {cpi}  NFP {nfp}  FOMC {fomc}"):
            st.markdown(doc.page_content)

elif ask_clicked and not query.strip():
    st.warning("Please enter a question first.")

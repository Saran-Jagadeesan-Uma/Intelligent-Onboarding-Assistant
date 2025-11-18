import streamlit as st
import os
import sys
from pathlib import Path
import warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.generation.rag_pipeline import UniversalRAGPipeline
from src.retrieval.vector_store import VectorStore

st.set_page_config(
    page_title="GitLab Onboarding Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0f1117;
    }
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: #ffffff;
        border: 2px solid #4a4a4a;
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    .answer-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .source-box {
        background-color: #1e1e1e;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4a4a4a;
        text-align: center;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        color: #888;
        font-size: 0.9em;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'system_stats' not in st.session_state:
    st.session_state.system_stats = None

def initialize_pipeline():
    return UniversalRAGPipeline(provider="gemini")

def get_system_stats():
    try:
        vector_store = VectorStore(
            collection_name="gitlab_onboarding",
            persist_directory="models/vector_store"
        )
        doc_count = vector_store.collection.count()
        
        embeddings_path = Path("models/embeddings/embeddings.npy")
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            embedding_dim = embeddings.shape[1]
        else:
            embedding_dim = "Unknown"
        
        return {
            'documents': doc_count,
            'dimensions': embedding_dim,
            'cost': 'FREE'
        }
    except Exception as e:
        return {
            'documents': 'Error',
            'dimensions': 'Error',
            'cost': 'FREE'
        }

with st.sidebar:
    st.image("https://about.gitlab.com/images/press/logo/png/gitlab-logo-gray-rgb.png", width=200)
    
    st.markdown("## 🚀 GitLab Onboarding Assistant")
    st.markdown("---")
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if api_key:
        st.success("✅ Gemini API Connected")
        st.caption("Free tier - unlimited queries!")
    else:
        st.warning("⚠️ No API Key")
        st.caption("Retrieval-only mode")
        
        with st.expander("🔑 How to enable full RAG"):
            st.markdown("""
            1. Get free key: [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Create `.env` file in project root
            3. Add: `GOOGLE_API_KEY=your-key-here`
            4. Restart Streamlit
            """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.total_queries}</div>
            <div class="metric-label">Queries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.chat_history)}</div>
            <div class="metric-label">History</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Example Questions")
    examples = [
        "How does GitLab conduct user research?",
        "What is the CI/CD catalog?",
        "How do teams handle design reviews?",
        "What are GitLab's collaboration practices?",
        "How is AI being integrated at GitLab?"
    ]
    
    for example in examples:
        if st.button(example, key=f"example_{example}", use_container_width=True):
            st.session_state.current_query = example
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.total_queries = 0
        st.rerun()
    
    st.markdown("---")
    st.caption("Built with ❤️ by Team 13")
    st.caption("Powered by Gemini 2.0 Flash")

st.title("🚀 GitLab Onboarding Assistant")
st.markdown("### Ask me anything about GitLab's policies, processes, and culture!")

if st.session_state.rag_pipeline is None:
    with st.spinner("🔄 Loading RAG pipeline..."):
        st.session_state.rag_pipeline = initialize_pipeline()
        st.session_state.system_stats = get_system_stats()
    st.success("✅ Pipeline loaded! Ready for questions.")

if 'current_query' not in st.session_state:
    st.session_state.current_query = None

query_input = st.text_input(
    "Your Question:",
    value=st.session_state.current_query if st.session_state.current_query else "",
    placeholder="e.g., How does GitLab conduct user research?",
    key="query_input",
    label_visibility="collapsed"
)

if st.session_state.current_query:
    st.session_state.current_query = None

col1, col2, col3 = st.columns([3, 1, 3])

with col2:
    search_button = st.button("🔍 Ask", use_container_width=True)

if search_button and query_input:
    st.session_state.total_queries += 1
    
    with st.spinner("🤔 Searching knowledge base..."):
        result = st.session_state.rag_pipeline.generate_answer(query_input, k=5)
        
        st.session_state.chat_history.insert(0, result)
    
    st.markdown("---")
    
    st.markdown("### 💡 Answer")
    st.markdown(f"""
    <div class="answer-box">
        {result['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📚 Sources")
    
    for i, doc in enumerate(result['sources'], 1):
        title = doc['metadata'].get('title', 'Untitled')
        score = doc.get('rerank_score', doc.get('similarity', doc.get('distance', 0)))
        text = doc.get('document', doc.get('text', ''))
        
        similarity_pct = (1 - doc.get('distance', 1)) * 100 if 'distance' in doc else score * 100
        
        with st.expander(f"📄 Source {i}: {title} (Relevance: {score:.2f})"):
            st.markdown(f"**Similarity:** {similarity_pct:.1f}%")
            
            if 'rerank_score' in doc:
                st.markdown(f"**Rerank Score:** {doc['rerank_score']:.4f}")
            
            if doc.get('rank_before_rerank'):
                st.markdown(f"**Original Rank:** {doc['rank_before_rerank']}")
            
            st.markdown("**Content Preview:**")
            preview = text[:500] + "..." if len(text) > 500 else text
            st.text_area("Source Content", preview, height=200, key=f"source_{i}_{st.session_state.total_queries}", disabled=True, label_visibility="collapsed")

elif search_button:
    st.warning("⚠️ Please enter a question!")

if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📜 Recent Questions")
    
    for i, item in enumerate(st.session_state.chat_history[:5], 1):
        with st.expander(f"{i}. {item['query']}", expanded=(i == 1)):
            st.markdown(f"**Answer:** {item['answer'][:300]}{'...' if len(item['answer']) > 300 else ''}")
            st.caption(f"Sources: {item['num_sources']} | Provider: {item.get('provider', 'N/A')} | Model: {item.get('model', 'N/A')}")

st.markdown("---")

if st.session_state.system_stats is None:
    st.session_state.system_stats = get_system_stats()

stats = st.session_state.system_stats

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{stats['documents']}</div>
        <div class="metric-label">Documents Indexed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{stats['dimensions']}</div>
        <div class="metric-label">Embedding Dimensions</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{stats['cost']}</div>
        <div class="metric-label">Cost per Query</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8em;">
    <p>💡 Tip: Click on sources to see the full context used to generate answers</p>
    <p>🔄 System automatically uses advanced retrieval with reranking for better accuracy</p>
</div>
""", unsafe_allow_html=True)
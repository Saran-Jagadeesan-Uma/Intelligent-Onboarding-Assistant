"""
Streamlit UI for Intelligent Onboarding Assistant
Beautiful, modern interface for RAG-powered Q&A
"""

import streamlit as st
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.generation.rag_pipeline import UniversalRAGPipeline

# Page configuration
st.set_page_config(
    page_title="GitLab Onboarding Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
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

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0

def initialize_pipeline():
    """Initialize RAG pipeline with Gemini"""
    return UniversalRAGPipeline(provider="gemini")

# Sidebar
with st.sidebar:
    st.image("https://about.gitlab.com/images/press/logo/png/gitlab-logo-gray-rgb.png", width=200)
    
    st.markdown("## 🚀 GitLab Onboarding Assistant")
    st.markdown("---")
    
    # API Key status
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
    
    # Stats
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
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    examples = [
        "What is GitLab's sustainability approach?",
        "How does risk management work?",
        "Tell me about legal compliance",
        "What is the Privacy Team's role?",
        "Explain corporate sustainability"
    ]
    
    for example in examples:
        if st.button(example, key=f"example_{example}", use_container_width=True):
            st.session_state.current_query = example
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.total_queries = 0
        st.rerun()
    
    st.markdown("---")
    st.caption("Built with ❤️ by Team 13")
    st.caption("Powered by Gemini 2.0")

# Main content
st.title("🚀 GitLab Onboarding Assistant")
st.markdown("### Ask me anything about GitLab's policies, processes, and culture!")

# Initialize pipeline on first run
if st.session_state.rag_pipeline is None:
    with st.spinner("🔄 Loading RAG pipeline..."):
        st.session_state.rag_pipeline = initialize_pipeline()
    st.success("✅ Pipeline loaded! Ready for questions.")

# Query input
query_input = st.text_input(
    "Your Question:",
    placeholder="e.g., What is GitLab's approach to sustainability?",
    key="query_input",
    label_visibility="collapsed"
)

# Check if there's a query from example buttons
if 'current_query' in st.session_state and st.session_state.current_query:
    query_input = st.session_state.current_query
    st.session_state.current_query = None

# Search button
col1, col2, col3 = st.columns([3, 1, 3])

with col2:
    search_button = st.button("🔍 Ask", use_container_width=True)

# Process query
if search_button and query_input:
    st.session_state.total_queries += 1
    
    with st.spinner("🤔 Thinking..."):
        # Generate answer
        result = st.session_state.rag_pipeline.generate_answer(query_input, k=3)
        
        # Add to history
        st.session_state.chat_history.insert(0, result)
    
    # Display result
    st.markdown("---")
    
    # Answer
    st.markdown("### 💡 Answer")
    st.markdown(f"""
    <div class="answer-box">
        {result['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Sources
    st.markdown("### 📚 Sources")
    
    for i, doc in enumerate(result['sources'], 1):
        title = doc['metadata'].get('title', 'Untitled')
        score = doc.get('rerank_score', doc.get('similarity', 0))
        text = doc['document']
        
        with st.expander(f"📄 Source {i}: {title} (Relevance: {score:.2f})"):
            st.markdown(f"**Rerank Score:** {score:.4f}")
            
            if doc.get('rank_before_rerank'):
                st.markdown(f"**Rank Before Rerank:** {doc['rank_before_rerank']}")
            
            st.markdown("**Content:**")
            st.text_area("Source Content", text, height=200, key=f"source_{i}_{st.session_state.total_queries}", disabled=True, label_visibility="collapsed")

elif search_button:
    st.warning("⚠️ Please enter a question!")

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📜 Recent Questions")
    
    for i, item in enumerate(st.session_state.chat_history[:5], 1):
        with st.expander(f"{i}. {item['query']}", expanded=(i == 1)):
            st.markdown(f"**Answer:** {item['answer']}")
            st.caption(f"Sources: {item['num_sources']} | Provider: {item.get('provider', 'N/A')} | Model: {item.get('model', 'N/A')}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">19</div>
        <div class="metric-label">Documents Indexed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">384</div>
        <div class="metric-label">Embedding Dimensions</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">FREE</div>
        <div class="metric-label">Cost per Query</div>
    </div>
    """, unsafe_allow_html=True)
# Intelligent Onboarding Assistant - Model Development Pipeline

## 📋 Overview

A production-ready RAG (Retrieval-Augmented Generation) system for GitLab onboarding that uses semantic search to retrieve relevant information from company handbooks and meeting transcripts.

**Project Status:** ✅ Model Development Pipeline Complete

### Key Features
- 🔍 **Semantic Retrieval**: Vector-based search using sentence-transformers
- 📊 **Comprehensive Evaluation**: Precision@K, Recall@K, MRR, NDCG metrics
- 🔬 **Experiment Tracking**: MLflow integration for reproducibility
- ⚖️ **Bias Detection**: Fairness analysis across data slices
- 💾 **Vector Database**: ChromaDB for efficient similarity search

## 🏗️ Architecture
```
Data Pipeline → Embeddings → Vector Store → Retrieval → [Generation]
     ↓              ↓            ↓             ↓
   (JSON)      (384-dim)    (ChromaDB)    (Top-K Results)
```

### Components
- **Data Loading**: Local JSON files from processed pipeline
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector Store**: ChromaDB with cosine similarity
- **Retrieval**: Baseline semantic search (Top-K)
- **Evaluation**: Multiple metrics with ground truth labels
- **Tracking**: MLflow for experiment management

## 📁 Project Structure
```
Intelligent-Onboarding-Assistant/
├── data/                          # Raw and processed data
│   ├── debiased_data/             # Processed handbook data
│   └── meeting_transcripts/       # Processed meeting transcripts
├── src/                           # Source code
│   ├── data/                      # Data loading modules
│   │   ├── load_from_gcs.py       # GCS data loader
│   │   └── load_local_data.py     # Local data loader
│   ├── embeddings/                # Embedding generation
│   │   └── generate_embeddings.py
│   ├── retrieval/                 # Retrieval system
│   │   ├── vector_store.py        # ChromaDB interface
│   │   └── retriever.py           # Baseline retriever
│   ├── generation/                # RAG pipeline
│   │   └── rag_pipeline.py        # Full RAG system
│   ├── evaluation/                # Evaluation & metrics
│   │   ├── metrics.py             # Retrieval metrics
│   │   └── bias_detection.py      # Bias analysis
│   └── experiments/               # Experiment tracking
│       └── mlflow_tracking.py     # MLflow integration
├── models/                        # Saved models
│   ├── embeddings/                # Generated embeddings
│   └── vector_store/              # ChromaDB database
├── experiments/                   # Experiment results
│   ├── mlruns/                    # MLflow tracking data
│   ├── retrieval_evaluation.json  # Evaluation results
│   └── bias_report.json           # Bias analysis report
├── tests/                         # Unit tests
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 2GB RAM minimum
- 1GB disk space

### Installation

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Intelligent-Onboarding-Assistant.git
cd Intelligent-Onboarding-Assistant
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Verify installation
```bash
python -m src.data.load_local_data
```

## 📊 Running the Pipeline

### Step 1: Generate Embeddings

Generate vector embeddings for all documents:
```bash
python -m src.embeddings.generate_embeddings
```

Output:
- models/embeddings/embeddings.npy - Vector embeddings (19 docs, 384 dims)
- models/embeddings/texts.json - Original texts
- models/embeddings/metadata.json - Document metadata

### Step 2: Build Vector Store

Index embeddings in ChromaDB:
```bash
python -m src.retrieval.vector_store
```

Output:
- models/vector_store/ - ChromaDB database with 19 indexed documents

### Step 3: Test Retrieval

Test the retrieval system:
```bash
python -m src.retrieval.retriever
```

Expected Results:
- Interactive query testing
- Top-K document retrieval
- Similarity scores for each result

### Step 4: Run Evaluation

Evaluate retrieval performance:
```bash
python -m src.evaluation.metrics
```

Metrics Calculated:
- Precision@K (K=1,3,5)
- Recall@K
- F1@K
- NDCG@K
- Mean Reciprocal Rank (MRR)

### Step 5: Check for Bias

Analyze fairness across data slices:
```bash
python -m src.evaluation.bias_detection
```

Analysis Includes:
- Source distribution
- Title/content type bias
- Performance disparity detection

### Step 6: Track Experiments

Log experiments to MLflow:
```bash
python -m src.experiments.mlflow_tracking
```

View Results:
```bash
mlflow ui --backend-store-uri file:./experiments/mlruns
```
Then open http://localhost:5000 in browser

## 📈 Performance Metrics

### Current Results (Baseline Retriever)

| Metric | K=1 | K=3 | K=5 |
|--------|-----|-----|-----|
| Precision@K | 0.667 | 0.222 | 0.133 |
| Recall@K | 0.667 | 0.667 | 0.667 |
| F1@K | 0.667 | 0.333 | 0.222 |
| NDCG@K | 0.667 | 0.667 | 0.667 |

Mean Reciprocal Rank (MRR): 0.667

### Bias Analysis
- No significant performance disparities detected
- Balanced retrieval across sources
- Fair representation of document types

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
python -m src.data.load_local_data
python -m src.embeddings.generate_embeddings
python -m src.retrieval.retriever
```

## 🔧 Configuration

### Embedding Model
Default: all-MiniLM-L6-v2 (384 dimensions, fast)

To change the model, edit src/embeddings/generate_embeddings.py:
```python
embedding_gen = EmbeddingGenerator(model_name="all-mpnet-base-v2")
```

### Retrieval Parameters
Adjust top-K results in src/retrieval/retriever.py:
```python
results = retriever.retrieve(query, k=10)
```

### Vector Store
ChromaDB settings in src/retrieval/vector_store.py:
```python
VectorStore(
    collection_name="gitlab_onboarding",
    persist_directory="models/vector_store"
)
```

## 📚 Dataset Information

Source: GitLab public documentation (handbook + meeting transcripts)

Statistics:
- Total documents: 19 chunks
- Sources: Handbook pages + Meeting transcripts
- Content types: Legal, Sustainability, CI/CD, Operations

Data Location: data/
- debiased_data/ - Processed handbook content
- meeting_transcripts/ - Meeting transcriptions

## 🤖 Optional: RAG with LLM Generation

To enable full answer generation (requires OpenAI API key):

1. Get API key: https://platform.openai.com/api-keys

2. Set environment variable:
```bash
setx OPENAI_API_KEY "your-key-here"
```

3. Run RAG pipeline:
```bash
python -m src.generation.rag_pipeline
```

Note: Currently works in retrieval-only mode without API key.

## 📊 MLflow Experiment Tracking

### View Experiments

Start MLflow UI:
```bash
mlflow ui --backend-store-uri file:./experiments/mlruns
```

Access at: http://localhost:5000

### Tracked Metrics
- All retrieval metrics (Precision, Recall, F1, NDCG, MRR)
- Model parameters (embedding model, vector store)
- Experiment metadata (timestamps, tags)
- Evaluation artifacts

## 🔍 Troubleshooting

### Issue: ModuleNotFoundError
Solution: Install dependencies
```bash
pip install sentence-transformers chromadb
```

### Issue: No embeddings found
Solution: Generate embeddings first
```bash
python -m src.embeddings.generate_embeddings
```

### Issue: ChromaDB collection not found
Solution: Build vector store
```bash
python -m src.retrieval.vector_store
```

### Issue: MLflow UI not starting
Solution: Check backend store path
```bash
mlflow ui --backend-store-uri file:./experiments/mlruns --port 5001
```

## 👥 Team

Team 13:
- Akshaj Nevgi
- Lakshmi Vandhanie Ganesh
- Mithun Dineshkumar
- Saran Jagadeesan Uma
- Zankhana Pratik Mehta

## 📝 Assignment Compliance

### Model Development Requirements

- Data Loading: Loads processed data from pipeline
- Model Selection: Using pre-trained all-MiniLM-L6-v2
- Model Validation: Comprehensive retrieval metrics
- Bias Detection: Slice-based fairness analysis
- Experiment Tracking: MLflow integration
- Code Quality: Modular, documented, tested
- Reproducibility: Complete setup instructions

### Evaluation Metrics Implemented
- Precision@K, Recall@K, F1@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Bias analysis across data slices

## 🚀 Future Enhancements

- Hybrid retrieval (BM25 + vector search)
- Cross-encoder reranking
- Query expansion
- Multi-language support
- Larger dataset (1000+ documents)
- CI/CD automation with GitHub Actions
- Production deployment (FastAPI + Docker)

## 📄 License

This project is part of an academic assignment for MLOps coursework.

## 🔗 Links

- GitHub Repository: https://github.com/LakshmiVadhanie/Intelligent-Onboarding-Assistant
- MLflow Dashboard: http://localhost:5000 (when running)

Last Updated: November 12, 2025
"""
End-to-End Test Script for Model Development Pipeline
Tests all components: Data → Embeddings → Vector Store → Retrieval → Evaluation
"""

import sys
from pathlib import Path
import logging
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def test_data_loading():
    """Test 1: Data Loading"""
    print_header("TEST 1: DATA LOADING")
    
    try:
        from src.data.load_local_data import LocalDataLoader
        
        loader = LocalDataLoader(data_dir="data")
        chunks = loader.load_all_data()
        
        assert len(chunks) > 0, "No data loaded!"
        
        print(f"✅ PASSED: Loaded {len(chunks)} chunks")
        print(f"   Sample keys: {list(chunks[0].keys())}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_embeddings():
    """Test 2: Embeddings"""
    print_header("TEST 2: EMBEDDINGS")
    
    try:
        import numpy as np
        from pathlib import Path
        
        embeddings_path = Path("models/embeddings/embeddings.npy")
        
        assert embeddings_path.exists(), "Embeddings not found! Run: python -m src.embeddings.generate_embeddings"
        
        embeddings = np.load(embeddings_path)
        
        print(f"✅ PASSED: Embeddings loaded")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dtype: {embeddings.dtype}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_vector_store():
    """Test 3: Vector Store"""
    print_header("TEST 3: VECTOR STORE")
    
    try:
        from src.retrieval.vector_store import VectorStore
        
        vector_store = VectorStore(
            collection_name="gitlab_onboarding",
            persist_directory="models/vector_store"
        )
        
        count = vector_store.collection.count()
        
        assert count > 0, "Vector store is empty! Run: python -m src.retrieval.vector_store"
        
        print(f"✅ PASSED: Vector store initialized")
        print(f"   Documents in collection: {count}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_retrieval():
    """Test 4: Retrieval System"""
    print_header("TEST 4: RETRIEVAL SYSTEM")
    
    try:
        from src.retrieval.retriever import BaselineRetriever
        
        retriever = BaselineRetriever()
        
        test_query = "What is GitLab's approach to sustainability?"
        results = retriever.retrieve(test_query, k=3)
        
        assert len(results) == 3, "Should retrieve 3 documents"
        assert all('document' in doc for doc in results), "Missing document text"
        assert all('similarity' in doc for doc in results), "Missing similarity scores"
        
        print(f"✅ PASSED: Retrieval working")
        print(f"   Query: {test_query}")
        print(f"   Top result similarity: {results[0]['similarity']:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_evaluation():
    """Test 5: Evaluation Metrics"""
    print_header("TEST 5: EVALUATION METRICS")
    
    try:
        from src.retrieval.retriever import BaselineRetriever
        from src.evaluation.metrics import RetrievalEvaluator
        
        retriever = BaselineRetriever()
        evaluator = RetrievalEvaluator(retriever)
        
        test_queries = [
            {'query': 'sustainability', 'relevant_ids': ['doc_8']}
        ]
        
        results = evaluator.evaluate(test_queries, k_values=[1, 3])
        
        assert 'mrr' in results, "Missing MRR metric"
        assert 'metrics_by_k' in results, "Missing K-based metrics"
        
        print(f"✅ PASSED: Evaluation working")
        print(f"   MRR: {results['mrr']:.4f}")
        print(f"   Precision@1: {results['metrics_by_k']['k=1']['precision']:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_bias_detection():
    """Test 6: Bias Detection"""
    print_header("TEST 6: BIAS DETECTION")
    
    try:
        from src.retrieval.retriever import BaselineRetriever
        from src.evaluation.bias_detection import BiasDetector
        
        retriever = BaselineRetriever()
        detector = BiasDetector(retriever)
        
        test_queries = [
            {'query': 'sustainability', 'relevant_ids': ['doc_8']},
            {'query': 'risk management', 'relevant_ids': ['doc_12']}
        ]
        
        report = detector.generate_bias_report(test_queries, k=3)
        
        assert 'source_bias' in report, "Missing source bias analysis"
        assert 'has_significant_bias' in report, "Missing bias flag"
        
        print(f"✅ PASSED: Bias detection working")
        print(f"   Significant bias detected: {report['has_significant_bias']}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_mlflow():
    """Test 7: MLflow Tracking"""
    print_header("TEST 7: MLFLOW TRACKING")
    
    try:
        import mlflow
        from pathlib import Path
        
        mlruns_path = Path("experiments/mlruns")
        
        assert mlruns_path.exists(), "MLflow runs directory not found"
        
        mlflow.set_tracking_uri("file:./experiments/mlruns")
        experiments = mlflow.search_experiments()
        
        assert len(experiments) > 0, "No experiments found"
        
        print(f"✅ PASSED: MLflow tracking working")
        print(f"   Experiments found: {len(experiments)}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("  🧪 RUNNING END-TO-END PIPELINE TESTS")
    print("=" * 80)
    
    start_time = time.time()
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Embeddings", test_embeddings),
        ("Vector Store", test_vector_store),
        ("Retrieval System", test_retrieval),
        ("Evaluation Metrics", test_evaluation),
        ("Bias Detection", test_bias_detection),
        ("MLflow Tracking", test_mlflow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("  📊 TEST SUMMARY")
    print("=" * 80 + "\n")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print("\n" + "-" * 80)
    print(f"  Total: {passed_count}/{total_count} tests passed")
    print(f"  Time: {elapsed_time:.2f} seconds")
    print("=" * 80)
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Pipeline is ready for submission! 🎉\n")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please fix before submission.\n")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
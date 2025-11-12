import numpy as np
from typing import List, Dict, Tuple
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalMetrics:
    """Metrics for evaluating retrieval systems"""
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], 
                       relevant_ids: List[str], 
                       k: int) -> float:
        """
        Calculate Precision@K
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0-1)
        """
        if k <= 0 or not retrieved_ids:
            return 0.0
        
        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_ids))
        precision = relevant_retrieved / k
        
        return precision
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], 
                    relevant_ids: List[str], 
                    k: int) -> float:
        """
        Calculate Recall@K
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@K score (0-1)
        """
        if not relevant_ids or not retrieved_ids:
            return 0.0
        
        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_ids))
        recall = relevant_retrieved / len(relevant_ids)
        
        return recall
    
    @staticmethod
    def f1_at_k(retrieved_ids: List[str],
                relevant_ids: List[str],
                k: int) -> float:
        """
        Calculate F1@K (harmonic mean of Precision@K and Recall@K)
        """
        precision = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
        recall = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids_list: List[List[str]], 
                            relevant_ids_list: List[List[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            retrieved_ids_list: List of retrieved doc ID lists
            relevant_ids_list: List of relevant doc ID lists
            
        Returns:
            MRR score (0-1)
        """
        reciprocal_ranks = []
        
        for retrieved_ids, relevant_ids in zip(retrieved_ids_list, relevant_ids_list):
            rank = None
            for i, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_ids:
                    rank = i
                    break
            
            if rank:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        return mrr
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str],
                  relevant_ids: List[str],
                  k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            NDCG@K score (0-1)
        """
        retrieved_k = retrieved_ids[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k, 1):
            relevance = 1 if doc_id in relevant_ids else 0
            dcg += relevance / np.log2(i + 1)
        
        # Calculate IDCG (Ideal DCG)
        ideal_relevances = [1] * min(len(relevant_ids), k)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg


class RetrievalEvaluator:
    """Evaluate retrieval system performance"""
    
    def __init__(self, retriever):
        """
        Initialize evaluator
        
        Args:
            retriever: Retriever instance to evaluate
        """
        self.retriever = retriever
        self.metrics = RetrievalMetrics()
        logger.info("✅ RetrievalEvaluator initialized")
    
    def evaluate(self, 
                test_queries: List[Dict],
                k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Evaluate retrieval system on test queries
        
        Args:
            test_queries: List of dicts with 'query' and 'relevant_ids'
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=" * 80)
        logger.info(f"EVALUATING RETRIEVAL SYSTEM ON {len(test_queries)} QUERIES")
        logger.info("=" * 80)
        
        all_retrieved = []
        all_relevant = []
        
        metrics_by_k = {k: {'precision': [], 'recall': [], 'f1': [], 'ndcg': []} 
                       for k in k_values}
        
        # Evaluate each query
        for i, item in enumerate(test_queries, 1):
            query = item['query']
            relevant_ids = item['relevant_ids']
            
            logger.info(f"\n[{i}/{len(test_queries)}] Query: {query[:60]}...")
            
            # Retrieve documents
            max_k = max(k_values)
            results = self.retriever.retrieve(query, k=max_k)
            retrieved_ids = [doc['id'] for doc in results]
            
            all_retrieved.append(retrieved_ids)
            all_relevant.append(relevant_ids)
            
            # Calculate metrics for each K
            for k in k_values:
                precision = self.metrics.precision_at_k(retrieved_ids, relevant_ids, k)
                recall = self.metrics.recall_at_k(retrieved_ids, relevant_ids, k)
                f1 = self.metrics.f1_at_k(retrieved_ids, relevant_ids, k)
                ndcg = self.metrics.ndcg_at_k(retrieved_ids, relevant_ids, k)
                
                metrics_by_k[k]['precision'].append(precision)
                metrics_by_k[k]['recall'].append(recall)
                metrics_by_k[k]['f1'].append(f1)
                metrics_by_k[k]['ndcg'].append(ndcg)
                
                logger.info(f"  K={k}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, NDCG={ndcg:.3f}")
        
        # Calculate MRR
        mrr = self.metrics.mean_reciprocal_rank(all_retrieved, all_relevant)
        
        # Aggregate results
        results = {
            'num_queries': len(test_queries),
            'mrr': mrr,
            'metrics_by_k': {}
        }
        
        for k in k_values:
            results['metrics_by_k'][f'k={k}'] = {
                'precision': np.mean(metrics_by_k[k]['precision']),
                'recall': np.mean(metrics_by_k[k]['recall']),
                'f1': np.mean(metrics_by_k[k]['f1']),
                'ndcg': np.mean(metrics_by_k[k]['ndcg'])
            }
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        
        return results
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results"""
        print("\n" + "=" * 80)
        print("📊 RETRIEVAL EVALUATION RESULTS")
        print("=" * 80)
        
        print(f"\n📈 Overall Metrics:")
        print(f"  • Number of test queries: {results['num_queries']}")
        print(f"  • Mean Reciprocal Rank (MRR): {results['mrr']:.4f}")
        
        print(f"\n📊 Metrics by K:")
        for k_label, metrics in results['metrics_by_k'].items():
            print(f"\n  {k_label.upper()}:")
            print(f"    • Precision@K: {metrics['precision']:.4f}")
            print(f"    • Recall@K:    {metrics['recall']:.4f}")
            print(f"    • F1@K:        {metrics['f1']:.4f}")
            print(f"    • NDCG@K:      {metrics['ndcg']:.4f}")
        
        print("\n" + "=" * 80)
    
    def save_results(self, results: Dict, output_path: str = "experiments/retrieval_evaluation.json"):
        """Save evaluation results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {output_path}")


# Test with sample data
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from retrieval.retriever import BaselineRetriever
    
    print("\n" + "=" * 80)
    print("🧪 TESTING RETRIEVAL EVALUATION")
    print("=" * 80)
    
    # Initialize retriever
    retriever = BaselineRetriever()
    evaluator = RetrievalEvaluator(retriever)
    
    # Create test queries
    # NOTE: You should create a proper test set with ground truth
    test_queries = [
        {
            'query': 'What is GitLab\'s approach to sustainability?',
            'relevant_ids': ['doc_8']  # Based on our earlier test
        },
        {
            'query': 'How does risk management work at GitLab?',
            'relevant_ids': ['doc_12']
        },
        {
            'query': 'Tell me about CI/CD processes',
            'relevant_ids': ['doc_3', 'doc_15']  # Meeting transcripts about CI/CD
        }
    ]
    
    print("\n⚠️  NOTE: Using sample test queries for demonstration")
    print("You should create a comprehensive test set with ground truth labels!\n")
    
    # Evaluate
    results = evaluator.evaluate(test_queries, k_values=[1, 3, 5])
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION TEST COMPLETE!")
    print("=" * 80)
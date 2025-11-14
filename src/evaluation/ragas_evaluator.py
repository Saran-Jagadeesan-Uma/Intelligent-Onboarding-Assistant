from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity
)
from datasets import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGASEvaluator:
    
    def __init__(self):
        """Initialize RAGAS evaluator"""
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        logger.info(" RAGAS Evaluator initialized")
        logger.info(f"   Metrics: {[m.name for m in self.metrics]}")
    
    def evaluate_retrieval_only(self, 
                                retriever,
                                test_data: List[Dict]) -> Dict:
        
        logger.info("\n" + "="*80)
        logger.info(" RAGAS EVALUATION (RETRIEVAL-ONLY MODE)")
        logger.info("="*80)
        
        results_data = []
        
        for item in test_data:
            question = item['question']
            ground_truth = item.get('ground_truth', '')
            
            logger.info(f"\nEvaluating: {question[:60]}...")
            
            retrieved_docs = retriever.retrieve(question, k=5)
            contexts = [doc['document'] for doc in retrieved_docs]
            
            synthetic_answer = f"Based on the retrieved documents: {contexts[0][:200]}..."
            
            results_data.append({
                'question': question,
                'answer': synthetic_answer,
                'contexts': contexts,
                'ground_truth': ground_truth
            })
        
        dataset = Dataset.from_pandas(pd.DataFrame(results_data))
        
        logger.info("\n Running RAGAS evaluation...")
        
        context_metrics = [context_precision, context_recall]
        
        try:
            evaluation_results = evaluate(
                dataset,
                metrics=context_metrics
            )
            
            logger.info(" RAGAS evaluation complete!")
            
            results_dict = {
                'num_queries': len(test_data),
                'metrics': {}
            }
            
            for metric_name, value in evaluation_results.items():
                results_dict['metrics'][metric_name] = float(value)
                logger.info(f"  {metric_name}: {value:.4f}")
            
            return results_dict
            
        except Exception as e:
            logger.error(f" RAGAS evaluation failed: {e}")
            logger.warning("  RAGAS requires LLM for full evaluation")
            logger.warning("   Falling back to custom metrics...")
            
            return self._custom_context_metrics(results_data)
    
    def _custom_context_metrics(self, results_data: List[Dict]) -> Dict:

        logger.info("📊 Computing custom context metrics...")
        
        metrics = {
            'num_queries': len(results_data),
            'avg_contexts_retrieved': np.mean([len(item['contexts']) for item in results_data]),
            'context_coverage': self._calculate_context_coverage(results_data),
            'context_diversity': self._calculate_context_diversity(results_data)
        }
        
        logger.info(f"\n Custom Metrics:")
        logger.info(f"  Avg contexts retrieved: {metrics['avg_contexts_retrieved']:.2f}")
        logger.info(f"  Context coverage: {metrics['context_coverage']:.4f}")
        logger.info(f"  Context diversity: {metrics['context_diversity']:.4f}")
        
        return {'metrics': metrics}
    
    def _calculate_context_coverage(self, results_data: List[Dict]) -> float:
        import numpy as np
        
        coverages = []
        for item in results_data:
            question_tokens = set(item['question'].lower().split())
            context_tokens = set()
            
            for context in item['contexts']:
                context_tokens.update(context.lower().split())
            
            coverage = len(question_tokens & context_tokens) / len(question_tokens) if question_tokens else 0
            coverages.append(coverage)
        
        return np.mean(coverages) if coverages else 0.0
    
    def _calculate_context_diversity(self, results_data: List[Dict]) -> float:
        import numpy as np
        
        diversities = []
        for item in results_data:
            contexts = item['contexts']
            if len(contexts) < 2:
                diversities.append(1.0)
                continue
            
            all_words = []
            for context in contexts:
                all_words.extend(context.lower().split())
            
            unique_ratio = len(set(all_words)) / len(all_words) if all_words else 0
            diversities.append(unique_ratio)
        
        return np.mean(diversities) if diversities else 0.0
    
    def save_results(self, results: Dict, output_path: str = "experiments/ragas_evaluation.json"):
        """Save RAGAS results"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n RAGAS results saved: {output_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from retrieval.advanced_retriever import AdvancedRetriever
    
    print("\n" + "="*80)
    print(" TESTING RAGAS EVALUATION FRAMEWORK")
    print("="*80)
    
    retriever = AdvancedRetriever()
    evaluator = RAGASEvaluator()
    
    test_data = [
        {
            'question': 'What is GitLab\'s approach to sustainability?',
            'ground_truth': 'GitLab focuses on corporate sustainability through environmental and social responsibility.'
        },
        {
            'question': 'How does risk management work at GitLab?',
            'ground_truth': 'GitLab has a Risk Management and Dispute Resolution division that handles risk strategies.'
        },
        {
            'question': 'What are the legal compliance requirements?',
            'ground_truth': 'GitLab maintains legal compliance through their Legal and Corporate Affairs team.'
        }
    ]
    
    print("\n  NOTE: RAGAS requires OpenAI API for full evaluation")
    print("Without API key, we'll use custom context-based metrics instead.\n")
    
    results = evaluator.evaluate_retrieval_only(retriever, test_data)
    
    print("\n" + "="*80)
    print(" RAGAS EVALUATION RESULTS")
    print("="*80)
    
    if 'metrics' in results:
        for metric_name, value in results['metrics'].items():
            print(f"  • {metric_name}: {value:.4f}")
    
    evaluator.save_results(results)
    
    print("\n" + "="*80)
    print(" RAGAS EVALUATION COMPLETE!")
    print("="*80)
    print("\n  For full RAGAS metrics (faithfulness, answer_relevancy),")
    print("   set OPENAI_API_KEY environment variable.")
    print("="*80)
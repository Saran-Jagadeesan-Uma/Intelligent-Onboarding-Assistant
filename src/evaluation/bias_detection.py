import numpy as np
from typing import Dict, List
import logging
import json
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiasDetector:
    
    def __init__(self, retriever):

        self.retriever = retriever
        logger.info(" BiasDetector initialized")
    
    def analyze_source_bias(self, test_queries: List[Dict], k: int = 5) -> Dict:

        logger.info("\n" + "=" * 80)
        logger.info(" ANALYZING SOURCE BIAS")
        logger.info("=" * 80)
        
        source_stats = defaultdict(lambda: {
            'retrieved_count': 0,
            'total_retrievals': 0,
            'avg_rank': [],
            'avg_similarity': []
        })
        
        for query_data in test_queries:
            query = query_data['query']
            results = self.retriever.retrieve(query, k=k)
            
            for rank, doc in enumerate(results, 1):
                source = doc['metadata'].get('source', 'unknown')
                source_stats[source]['retrieved_count'] += 1
                source_stats[source]['total_retrievals'] += 1
                source_stats[source]['avg_rank'].append(rank)
                source_stats[source]['avg_similarity'].append(doc['similarity'])
        
        bias_report = {}
        for source, stats in source_stats.items():
            bias_report[source] = {
                'total_retrieved': stats['retrieved_count'],
                'retrieval_rate': stats['retrieved_count'] / (len(test_queries) * k),
                'avg_rank': np.mean(stats['avg_rank']) if stats['avg_rank'] else 0,
                'avg_similarity': np.mean(stats['avg_similarity']) if stats['avg_similarity'] else 0
            }
        
        logger.info("\n Source Distribution:")
        for source, metrics in bias_report.items():
            logger.info(f"\n  {source}:")
            logger.info(f"    • Retrieved: {metrics['total_retrieved']} times")
            logger.info(f"    • Retrieval rate: {metrics['retrieval_rate']:.2%}")
            logger.info(f"    • Avg rank: {metrics['avg_rank']:.2f}")
            logger.info(f"    • Avg similarity: {metrics['avg_similarity']:.4f}")
        
        return bias_report
    
    def analyze_title_bias(self, test_queries: List[Dict], k: int = 5) -> Dict:

        logger.info("\n" + "=" * 80)
        logger.info(" ANALYZING TITLE/CONTENT TYPE BIAS")
        logger.info("=" * 80)
        
        title_stats = defaultdict(lambda: {
            'count': 0,
            'avg_similarity': []
        })
        
        for query_data in test_queries:
            query = query_data['query']
            results = self.retriever.retrieve(query, k=k)
            
            for doc in results:
                title = doc['metadata'].get('title', 'Untitled')
                title_stats[title]['count'] += 1
                title_stats[title]['avg_similarity'].append(doc['similarity'])
        
        sorted_titles = sorted(title_stats.items(), 
                              key=lambda x: x[1]['count'], 
                              reverse=True)
        
        logger.info(f"\n Top Retrieved Titles:")
        for title, stats in sorted_titles[:10]:
            avg_sim = np.mean(stats['avg_similarity'])
            logger.info(f"  • {title[:50]:50s} | Count: {stats['count']:2d} | Avg Sim: {avg_sim:.4f}")
        
        return dict(sorted_titles)
    
    def detect_performance_disparity(self, 
                                    bias_report: Dict, 
                                    threshold: float = 0.2) -> Dict:

        logger.info("\n" + "=" * 80)
        logger.info("  CHECKING FOR PERFORMANCE DISPARITIES")
        logger.info("=" * 80)
        
        if not bias_report:
            logger.info("No bias data to analyze")
            return {}
        
        all_similarities = [metrics['avg_similarity'] 
                           for metrics in bias_report.values()]
        overall_avg = np.mean(all_similarities)
        
        disparities = {}
        for source, metrics in bias_report.items():
            diff = abs(metrics['avg_similarity'] - overall_avg)
            
            if diff > threshold:
                disparities[source] = {
                    'avg_similarity': metrics['avg_similarity'],
                    'overall_avg': overall_avg,
                    'difference': diff,
                    'percentage_diff': (diff / overall_avg) * 100
                }
                
                logger.warning(f"\n    DISPARITY DETECTED: {source}")
                logger.warning(f"      Source avg: {metrics['avg_similarity']:.4f}")
                logger.warning(f"      Overall avg: {overall_avg:.4f}")
                logger.warning(f"      Difference: {diff:.4f} ({disparities[source]['percentage_diff']:.1f}%)")
        
        if not disparities:
            logger.info("\n   No significant disparities detected!")
        
        return disparities
    
    def generate_bias_report(self, test_queries: List[Dict], k: int = 5) -> Dict:

        logger.info("\n" + "=" * 80)
        logger.info(" GENERATING COMPREHENSIVE BIAS REPORT")
        logger.info("=" * 80)
        
        source_bias = self.analyze_source_bias(test_queries, k=k)
        title_bias = self.analyze_title_bias(test_queries, k=k)
        disparities = self.detect_performance_disparity(source_bias)
        
        report = {
            'num_queries': len(test_queries),
            'k': k,
            'source_bias': source_bias,
            'title_distribution': {k: v for k, v in list(title_bias.items())[:10]},
            'disparities': disparities,
            'has_significant_bias': len(disparities) > 0
        }
        
        logger.info("\n" + "=" * 80)
        logger.info(" BIAS REPORT COMPLETE")
        logger.info("=" * 80)
        
        return report
    
    def save_report(self, report: Dict, output_path: str = "experiments/bias_report.json"):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n Bias report saved to: {output_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from retrieval.retriever import BaselineRetriever
    
    print("\n" + "=" * 80)
    print(" TESTING BIAS DETECTION")
    print("=" * 80)
    
    retriever = BaselineRetriever()
    detector = BiasDetector(retriever)
    
    test_queries = [
        {'query': 'What is GitLab\'s approach to sustainability?', 'relevant_ids': ['doc_8']},
        {'query': 'How does risk management work at GitLab?', 'relevant_ids': ['doc_12']},
        {'query': 'Tell me about CI/CD processes', 'relevant_ids': ['doc_3']},
        {'query': 'What are GitLab\'s company values?', 'relevant_ids': []},
        {'query': 'How does legal compliance work?', 'relevant_ids': []},
    ]
    
    report = detector.generate_bias_report(test_queries, k=5)
    
    detector.save_report(report)
    
    print("\n" + "=" * 80)
    print(" BIAS DETECTION SUMMARY")
    print("=" * 80)
    print(f"  • Queries analyzed: {report['num_queries']}")
    print(f"  • Sources found: {len(report['source_bias'])}")
    print(f"  • Significant bias detected: {'YES ' if report['has_significant_bias'] else 'NO '}")
    
    if report['disparities']:
        print(f"\n    Disparities found in {len(report['disparities'])} source(s)")
    
    print("\n" + "=" * 80)
    print(" BIAS DETECTION TEST COMPLETE!")
    print("=" * 80)
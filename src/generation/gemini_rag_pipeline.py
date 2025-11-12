"""
RAG Pipeline with Google Gemini (FREE!)
Alternative to OpenAI for answer generation
"""

import google.generativeai as genai
import os
from typing import List, Dict, Optional
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.advanced_retriever import AdvancedRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiRAGPipeline:
    """RAG Pipeline using Google Gemini (Free!)"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gemini-2.0-flash",
                 temperature: float = 0.3):
        """
        Initialize Gemini RAG pipeline
        
        Args:
            api_key: Google AI API key (or set GOOGLE_API_KEY env variable)
            model: Gemini model name
            temperature: Generation temperature
        """
        logger.info("Initializing Gemini RAG Pipeline...")
        
        # Initialize retriever with advanced reranking
        logger.info("Setting up advanced retriever...")
        self.retriever = AdvancedRetriever()
        
        # Initialize Gemini
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key is None:
                logger.warning("⚠️  No Google API key found!")
                logger.warning("Set GOOGLE_API_KEY environment variable")
                logger.warning("Get free key at: https://aistudio.google.com/app/apikey")
                logger.warning("Pipeline will only retrieve documents, not generate answers")
                self.model = None
            else:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model)
                logger.info(f"✅ Gemini initialized with model: {model}")
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            logger.info(f"✅ Gemini initialized with model: {model}")
        
        self.model_name = model
        self.temperature = temperature
        
        logger.info("✅ Gemini RAG Pipeline initialized!")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents"""
        logger.info(f"Retrieving top-{k} documents with reranking...")
        return self.retriever.retrieve(query, k=k)
    
    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Build prompt with retrieved context"""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc['metadata'].get('title', 'Untitled')
            text = doc['document']
            context_parts.append(f"[Source {i}: {title}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are an intelligent onboarding assistant for GitLab. Answer the question based ONLY on the provided context from GitLab's handbook and meeting transcripts.

If the answer is not in the context, say "I don't have enough information to answer this based on the available documents."

Be concise, accurate, and cite which source(s) you're using (e.g., "According to Source 1...").

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(self, query: str, k: int = 5) -> Dict:
        """
        Generate answer using Gemini RAG
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Response dictionary
        """
        logger.info("=" * 80)
        logger.info(f"Processing query: {query}")
        logger.info("=" * 80)
        
        # Retrieve documents
        retrieved_docs = self.retrieve_context(query, k=k)
        logger.info(f"✅ Retrieved {len(retrieved_docs)} documents with reranking")
        
        # If no Gemini model, return retrieval only
        if self.model is None:
            logger.warning("⚠️  No Gemini model - returning retrieval results only")
            return {
                'query': query,
                'answer': "[Generation not available - no Google API key]",
                'sources': retrieved_docs,
                'num_sources': len(retrieved_docs),
                'model': None
            }
        
        # Build prompt
        prompt = self.build_prompt(query, retrieved_docs)
        logger.info("✅ Built prompt with context")
        
        # Generate with Gemini
        logger.info(f"Generating answer using {self.model_name}...")
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=500,
                )
            )
            
            answer = response.text
            logger.info("✅ Answer generated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            answer = f"[Error generating answer: {str(e)}]"
        
        # Format response
        result = {
            'query': query,
            'answer': answer,
            'sources': retrieved_docs,
            'num_sources': len(retrieved_docs),
            'model': self.model_name
        }
        
        return result
    
    def print_response(self, result: Dict, show_full_sources: bool = False):
        """Pretty print response"""
        print("\n" + "=" * 80)
        print("💬 GEMINI RAG RESPONSE")
        print("=" * 80)
        
        print(f"\n❓ QUERY:\n{result['query']}")
        
        print(f"\n💡 ANSWER:\n{result['answer']}")
        
        print(f"\n📚 SOURCES ({result['num_sources']}):")
        print("-" * 80)
        
        for doc in result['sources']:
            print(f"\n🔹 Source {doc['rank']}")
            print(f"   Title: {doc['metadata'].get('title', 'N/A')}")
            print(f"   Rerank Score: {doc.get('rerank_score', 0):.4f}")
            
            if show_full_sources:
                print(f"   Text: {doc['document']}")
            else:
                preview = doc['document'][:200] + "..." if len(doc['document']) > 200 else doc['document']
                print(f"   Preview: {preview}")
            
            print("-" * 80)


# Test Gemini RAG
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 TESTING GEMINI RAG PIPELINE (FREE!)")
    print("=" * 80)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("\n⚠️  No Google API key found!")
        print("\n📝 TO GET FREE API KEY:")
        print("1. Go to: https://aistudio.google.com/app/apikey")
        print("2. Click 'Create API Key'")
        print("3. Copy the key")
        print("4. Set environment variable:")
        print("   PowerShell: $env:GOOGLE_API_KEY='your-key'")
        print("\n" + "=" * 80)
        
        response = input("\nDo you have an API key to test now? (yes/no): ")
        if response.lower() == 'yes':
            api_key = input("Paste your Gemini API key: ").strip()
    
    # Initialize
    rag = GeminiRAGPipeline(api_key=api_key)
    
    # Test queries
    test_queries = [
        "What is GitLab's approach to sustainability?",
        "How does risk management work at GitLab?",
        "Explain the CI/CD process at GitLab"
    ]
    
    for query in test_queries:
        result = rag.generate_answer(query, k=3)
        rag.print_response(result, show_full_sources=False)
        
        if rag.model:  # Only pause if generating answers
            input("\n⏸️  Press Enter for next query...")
    
    print("\n" + "=" * 80)
    print("✅ GEMINI RAG TEST COMPLETE!")
    print("=" * 80)
    
    if rag.model:
        print("\n🎉 Full RAG working with FREE Gemini API!")
        print("💰 Cost: $0.00 (completely free!)")
    else:
        print("\n⚠️  Get your free key to test full generation!")
    
    print("=" * 80)
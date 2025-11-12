from openai import OpenAI
import os
from typing import List, Dict, Optional
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.retriever import BaselineRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.3):
        """
        Initialize RAG pipeline with retriever and LLM
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
            model: OpenAI model name
            temperature: Generation temperature (0-1)
        """
        logger.info("Initializing RAG Pipeline...")
        
        # Initialize retriever
        logger.info("Setting up retriever...")
        self.retriever = BaselineRetriever()
        
        # Initialize OpenAI client
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                logger.warning("⚠️  No OpenAI API key found!")
                logger.warning("Set OPENAI_API_KEY environment variable or pass api_key parameter")
                logger.warning("Pipeline will only retrieve documents, not generate answers")
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key)
                logger.info(f"✅ OpenAI client initialized with model: {model}")
        else:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"✅ OpenAI client initialized with model: {model}")
        
        self.model = model
        self.temperature = temperature
        
        logger.info("✅ RAG Pipeline initialized successfully!")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving top-{k} documents for query...")
        return self.retriever.retrieve(query, k=k)
    
    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Build prompt with retrieved context
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc['metadata'].get('title', 'Untitled')
            text = doc['document']
            context_parts.append(f"[Source {i} - {title}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Build full prompt
        prompt = f"""You are an intelligent onboarding assistant for GitLab. Answer the question based on the provided context from GitLab's handbook and meeting transcripts.

If the answer is not in the context, say "I don't have enough information to answer this question based on the available documents."

Be concise, accurate, and cite which source(s) you're using.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(self, query: str, k: int = 5) -> Dict:
        """
        Generate answer using RAG (Retrieval + Generation)
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info("=" * 80)
        logger.info(f"Processing query: {query}")
        logger.info("=" * 80)
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve_context(query, k=k)
        logger.info(f"✅ Retrieved {len(retrieved_docs)} documents")
        
        # If no OpenAI client, return retrieval only
        if self.client is None:
            logger.warning("⚠️  No OpenAI client - returning retrieval results only")
            return {
                'query': query,
                'answer': "[Generation not available - no OpenAI API key]",
                'sources': retrieved_docs,
                'num_sources': len(retrieved_docs),
                'model': None
            }
        
        # Step 2: Build prompt with context
        prompt = self.build_prompt(query, retrieved_docs)
        logger.info("✅ Built prompt with context")
        
        # Step 3: Generate answer with LLM
        logger.info(f"Generating answer using {self.model}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful GitLab onboarding assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            logger.info("✅ Answer generated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            answer = f"[Error generating answer: {str(e)}]"
        
        # Step 4: Format response
        result = {
            'query': query,
            'answer': answer,
            'sources': retrieved_docs,
            'num_sources': len(retrieved_docs),
            'model': self.model
        }
        
        return result
    
    def print_response(self, result: Dict, show_full_sources: bool = False):
        """
        Pretty print RAG response
        
        Args:
            result: Result dictionary from generate_answer
            show_full_sources: Whether to show full source text
        """
        print("\n" + "=" * 80)
        print("💬 RAG RESPONSE")
        print("=" * 80)
        
        print(f"\n❓ QUERY:\n{result['query']}")
        
        print(f"\n💡 ANSWER:\n{result['answer']}")
        
        print(f"\n📚 SOURCES ({result['num_sources']}):")
        print("-" * 80)
        
        for doc in result['sources']:
            print(f"\n🔹 Source {doc['rank']}")
            print(f"   Title: {doc['metadata'].get('title', 'N/A')}")
            print(f"   Similarity: {doc['similarity']:.4f}")
            
            if show_full_sources:
                print(f"   Text: {doc['document']}")
            else:
                preview = doc['document'][:150] + "..." if len(doc['document']) > 150 else doc['document']
                print(f"   Preview: {preview}")
        
        print("\n" + "=" * 80)


# Test the RAG pipeline
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 TESTING RAG PIPELINE")
    print("=" * 80)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: No OPENAI_API_KEY found in environment variables")
        print("The pipeline will work but won't generate answers (retrieval only)")
        print("\nTo enable answer generation:")
        print("1. Get an API key from https://platform.openai.com/api-keys")
        print("2. Set it as an environment variable:")
        print("   Windows: setx OPENAI_API_KEY \"your-key-here\"")
        print("   Or pass it directly: RAGPipeline(api_key='your-key-here')")
        print("\n" + "=" * 80)
        input("\nPress Enter to continue with retrieval-only mode...")
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Test queries
    test_queries = [
        "What is GitLab's approach to sustainability?",
        "How does risk management work at GitLab?",
        "Tell me about the CI/CD process"
    ]
    
    for query in test_queries:
        result = rag.generate_answer(query, k=3)
        rag.print_response(result, show_full_sources=False)
        
        input("\n⏸️  Press Enter to continue to next query...")
    
    print("\n" + "=" * 80)
    print("✅ RAG PIPELINE TEST COMPLETE!")
    print("=" * 80)
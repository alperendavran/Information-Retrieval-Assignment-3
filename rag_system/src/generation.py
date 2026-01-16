# -*- coding: utf-8-sig -*-
"""
Answer Generation Module for RAG System.
Generates answers using GPT-4o based on retrieved passages.

University of Antwerp - Information Retrieval Assignment 3
References:
- Assignment: "Use GPT-4o to generate answers"
- Lecture slides: "Use frozen LLMs with zero-shot/few-shot learning"
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    OPENAI_API_KEY, 
    OPENAI_MODEL, 
    SYSTEM_PROMPT,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_K
)
from src.retrieval import Retriever, RetrievalResult
from src.cost_tracking import estimate_cost_usd, log_event
from config import MODEL_PRICING_USD_PER_1M


class AnswerGenerator:
    """
    Generates answers using GPT-4o with retrieved context.
    
    Pipeline:
    1. Accept user query
    2. Retrieve top-k passages using Retriever
    3. Format passages into prompt context
    4. Generate answer using GPT-4o
    5. Optionally include source citations
    """
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        api_key: Optional[str] = None,
        model: str = OPENAI_MODEL
    ):
        """
        Initialize the answer generator.
        
        Args:
            retriever: Retriever instance for fetching passages
            api_key: OpenAI API key (defaults to env var)
            model: OpenAI model to use (default: gpt-4o)
        """
        self.retriever = retriever
        self.model = model
        self.api_key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key:
            self._init_client()
    
    def _init_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            print(f"✅ OpenAI client initialized with model: {self.model}")
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    def set_api_key(self, api_key: str) -> None:
        """Set or update the API key."""
        self.api_key = api_key
        self._init_client()
    
    def format_prompt(
        self, 
        query: str, 
        context: str,
        include_instructions: bool = True
    ) -> List[Dict[str, str]]:
        """
        Format the prompt for the LLM.
        
        Args:
            query: User query
            context: Retrieved passages formatted as context
            include_instructions: Whether to include system instructions
            
        Returns:
            List of message dictionaries for OpenAI API
        """
        messages = []
        
        if include_instructions:
            messages.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
        
        user_content = f"""Context information from the University of Antwerp CS Masters program:

{context}

---

Based on the above context, please answer the following question:

Question: {query}

Answer:"""
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        k: int = TOP_K,
        include_sources: bool = True,
        retrieved_results: Optional[List[RetrievalResult]] = None,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE
    ) -> Dict[str, Any]:
        """
        Generate an answer for the query.
        
        Args:
            query: User query
            context: Pre-formatted context (if None, uses retriever)
            k: Number of passages to retrieve
            include_sources: Whether to include source citations
            max_tokens: Maximum tokens in response
            temperature: Generation temperature
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Set API key first.")
        
        # Retrieve passages if no context provided
        # If context is provided, you may optionally pass retrieved_results so
        # we can attach sources in the output (used by LangGraph agentic pipeline).
        effective_retrieved_results: List[RetrievalResult] = []
        if context is None:
            if self.retriever is None:
                raise ValueError("No context provided and retriever not set")

            # Dynamic k for list-like / study-programme questions (improves completeness)
            ql = query.lower()
            k_eff = int(k)
            if any(x in ql for x in ["list", "which compulsory", "compulsory courses", "study programme", "study program"]):
                k_eff = max(k_eff, 10)
            if any(x in ql for x in ["1st semester", "2nd semester", "exam contract not possible"]):
                k_eff = max(k_eff, 10)
            k_eff = min(k_eff, 12)

            context, effective_retrieved_results = self.retriever.retrieve_with_context(query, k=k_eff)
        else:
            effective_retrieved_results = retrieved_results or []
        
        # Format prompt
        messages = self.format_prompt(query, context)
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)
            total_tokens = getattr(response.usage, "total_tokens", None)
            cost_est = None
            if prompt_tokens is not None and completion_tokens is not None:
                cost_est = estimate_cost_usd(
                    model=self.model,
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    pricing_table=MODEL_PRICING_USD_PER_1M,
                )
            
            # Build result
            result = {
                "query": query,
                "answer": answer,
                "model": self.model,
                "retrieved_passages": [r.to_dict() for r in effective_retrieved_results],
                "num_passages": len(effective_retrieved_results),
                "tokens_used": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens
                }
            }

            if cost_est is not None:
                result["cost_estimate_usd"] = float(cost_est)

            # Log (timestamped) for reproducible cost documentation
            log_event(
                {
                    "operation": "answer_generation",
                    "mode": "rag" if effective_retrieved_results else "no_retrieval_context",
                    "model": self.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_estimate_usd": cost_est,
                    "k": k,
                    "k_effective": len(effective_retrieved_results),
                }
            )
            
            # Add source citations if requested
            if include_sources and effective_retrieved_results:
                sources = []
                for r in effective_retrieved_results:
                    source = r.source_title
                    if r.source_section:
                        source += f" - {r.source_section}"
                    sources.append({
                        "title": source,
                        "chunk_id": r.chunk_id,
                        "relevance_score": r.similarity_score
                    })
                result["sources"] = sources
            
            return result
            
        except Exception as e:
            return {
                "query": query,
                "answer": f"Error generating answer: {str(e)}",
                "error": str(e),
                "retrieved_passages": [r.to_dict() for r in effective_retrieved_results]
            }
    
    def generate_without_retrieval(
        self,
        query: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE
    ) -> Dict[str, Any]:
        """
        Generate an answer WITHOUT retrieval (baseline for comparison).
        
        Args:
            query: User query
            max_tokens: Maximum tokens in response
            temperature: Generation temperature
            
        Returns:
            Dictionary with answer and metadata
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Set API key first.")
        
        # Simple prompt without context
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions about the University of Antwerp Computer Science Masters program to the best of your knowledge."
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nAnswer:"
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)
            total_tokens = getattr(response.usage, "total_tokens", None)
            cost_est = None
            if prompt_tokens is not None and completion_tokens is not None:
                cost_est = estimate_cost_usd(
                    model=self.model,
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    pricing_table=MODEL_PRICING_USD_PER_1M,
                )
            
            result = {
                "query": query,
                "answer": response.choices[0].message.content,
                "model": self.model,
                "retrieval_used": False,
                "tokens_used": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens
                }
            }

            if cost_est is not None:
                result["cost_estimate_usd"] = float(cost_est)

            log_event(
                {
                    "operation": "answer_generation",
                    "mode": "baseline_no_retrieval",
                    "model": self.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_estimate_usd": cost_est,
                }
            )

            return result
            
        except Exception as e:
            return {
                "query": query,
                "answer": f"Error: {str(e)}",
                "error": str(e),
                "retrieval_used": False
            }
    
    def compare_with_without_retrieval(
        self,
        query: str,
        k: int = TOP_K
    ) -> Dict[str, Any]:
        """
        Compare answers with and without retrieval.
        
        Args:
            query: User query
            k: Number of passages to retrieve
            
        Returns:
            Dictionary with both answers for comparison
        """
        # With retrieval
        with_rag = self.generate(query, k=k)
        
        # Without retrieval (baseline)
        without_rag = self.generate_without_retrieval(query)
        
        return {
            "query": query,
            "with_retrieval": with_rag,
            "without_retrieval": without_rag
        }


class RAGPipeline:
    """
    Complete RAG pipeline combining all components.
    """
    
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator
    ):
        self.retriever = retriever
        self.generator = generator
        self.generator.retriever = retriever
    
    def answer(
        self, 
        query: str, 
        k: int = TOP_K,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a query using the full RAG pipeline.
        
        Args:
            query: User query
            k: Number of passages to retrieve
            include_sources: Whether to include source citations
            
        Returns:
            Complete answer with sources and metadata
        """
        return self.generator.generate(
            query=query,
            k=k,
            include_sources=include_sources
        )
    
    def interactive_mode(self) -> None:
        """Run an interactive Q&A session."""
        print("\n" + "="*60)
        print("🎓 University of Antwerp CS Masters RAG System")
        print("="*60)
        print("Ask questions about courses, programs, and requirements.")
        print("Type 'quit' or 'exit' to end the session.")
        print("Type 'compare' before a query to compare with/without RAG.")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("🔍 Your question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                compare_mode = query.lower().startswith('compare ')
                if compare_mode:
                    query = query[8:].strip()
                    print("\n⏳ Comparing with and without retrieval...")
                    result = self.generator.compare_with_without_retrieval(query)
                    
                    print("\n📚 WITH RETRIEVAL:")
                    print("-" * 40)
                    print(result["with_retrieval"]["answer"])
                    
                    print("\n🚫 WITHOUT RETRIEVAL (baseline):")
                    print("-" * 40)
                    print(result["without_retrieval"]["answer"])
                    
                    if "sources" in result["with_retrieval"]:
                        print("\n📖 Sources used:")
                        for src in result["with_retrieval"]["sources"]:
                            print(f"  - {src['title']} (score: {src['relevance_score']:.4f})")
                else:
                    print("\n⏳ Generating answer...")
                    result = self.answer(query)
                    
                    print("\n📝 Answer:")
                    print("-" * 40)
                    print(result["answer"])
                    
                    if "sources" in result:
                        print("\n📖 Sources:")
                        for src in result["sources"]:
                            print(f"  - {src['title']} (score: {src['relevance_score']:.4f})")
                
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    print("Answer Generator Module")
    print("This module requires a valid OpenAI API key.")
    print("Set the OPENAI_API_KEY environment variable to use.")

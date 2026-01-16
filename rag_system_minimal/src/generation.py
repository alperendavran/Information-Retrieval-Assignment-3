# -*- coding: utf-8-sig -*-
"""
Answer Generation Module (Assignment 4.4 - 30%)

This module generates answers using GPT-4o based on retrieved passages.

Requirements:
- Use GPT-4o (via provided API key)
- Accept a natural language query
- Retrieve top-k passages
- Insert retrieved passages into a prompt
- Generate an answer using GPT-4o
- Optionally include retrieved passages in output

References:
- Assignment 3: "Use GPT-4o to generate answers"
- Lewis et al. (2020): Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI

from src.retrieval import RetrievalResult


class AnswerGenerator:
    """
    Generates answers using GPT-4o with retrieved context.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize answer generator.
        
        Args:
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"
    
    def _format_context(self, passages: List[RetrievalResult]) -> str:
        """
        Format retrieved passages into context string for the prompt.
        
        Args:
            passages: List of retrieved passages
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, passage in enumerate(passages, 1):
            source_info = f"[Source {i}: {passage.source_title}"
            if passage.source_section:
                source_info += f" - {passage.source_section}"
            source_info += "]"
            
            context_parts.append(f"{source_info}\n{passage.text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate(
        self,
        query: str,
        retrieved_passages: List[RetrievalResult],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an answer using GPT-4o with retrieved context.
        
        Args:
            query: User's question
            retrieved_passages: Top-k retrieved passages
            include_sources: Whether to include source information in output
            
        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        # Format context
        context = self._format_context(retrieved_passages)
        
        # System prompt (instructs model to answer only from context)
        system_prompt = """You are a helpful assistant for the University of Antwerp Computer Science Masters program.
Your role is to answer questions about courses, programs, requirements, and other academic information.

IMPORTANT INSTRUCTIONS:
1. Answer questions based ONLY on the provided context below.
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
3. Be specific and cite course names or details when relevant.
4. Keep your answers concise but complete.
5. Do not make up information that is not in the context."""

        # User prompt with context
        user_prompt = f"""Context:
{context}

Question: {query}

Answer the question based on the context provided above. If the context doesn't contain enough information, say so."""

        # Call GPT-4o
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for factual responses
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content
        
        result: Dict[str, Any] = {
            "answer": answer,
            "model": self.model
        }
        
        if include_sources:
            result["sources"] = [
                {
                    "title": p.source_title,
                    "section": p.source_section,
                    "relevance_score": p.similarity_score,
                    "rank": p.rank
                }
                for p in retrieved_passages
            ]
        
        return result
    
    def generate_without_retrieval(self, query: str) -> Dict[str, Any]:
        """
        Generate an answer WITHOUT retrieval (baseline for comparison).
        
        This is used for evaluation to compare RAG vs. no-RAG performance.
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with 'answer' and 'model'
        """
        system_prompt = """You are a helpful assistant for the University of Antwerp Computer Science Masters program.
Answer questions about courses, programs, requirements, and other academic information.
If you don't know the answer, say so."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "model": self.model
        }

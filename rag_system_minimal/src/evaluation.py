# -*- coding: utf-8-sig -*-
"""
Evaluation Module (Assignment 4.5 - 20%)

This module evaluates the RAG system along three dimensions:
1. Retrieval quality: Recall@k, manual inspection
2. Answer quality: Compare with/without retrieval, correctness, completeness, hallucination
3. Error analysis: ≥3 retrieval failures, ≥3 hallucinations

References:
- Assignment 3: "Report Recall@k. Manually inspect relevance. Compare with/without retrieval. Evaluate correctness, completeness, hallucination. Provide ≥3 retrieval failures and ≥3 hallucinations."
- Manning et al. (2008): Introduction to Information Retrieval, Chapter 8 (Evaluation)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class EvaluationQuestion:
    """A test question with optional ground truth."""
    question: str
    expected_answer: Optional[str] = None
    relevant_chunk_ids: Optional[List[str]] = None  # Ground truth relevant chunk IDs


class RAGEvaluator:
    """
    Evaluates RAG system performance.
    
    Metrics:
    1. Retrieval Quality:
       - Recall@k: How many relevant docs are in top-k
       - Manual inspection: User labels relevance of top-k passages
       
    2. Answer Quality:
       - Comparison: RAG vs. baseline (no retrieval)
       - Correctness: Is the answer factually correct?
       - Completeness: Does the answer fully address the question?
       - Hallucination: Does the answer contain information not in context?
       
    3. Error Analysis:
       - Retrieval failures: Cases where relevant passages weren't retrieved
       - Hallucinations: Cases where GPT-4o produced incorrect/hallucinated answers
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def recall_at_k(
        self,
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@k.
        
        Recall@k = |relevant ∩ retrieved@k| / |relevant|
        
        Args:
            retrieved_chunk_ids: List of retrieved chunk IDs (in order, top-k)
            relevant_chunk_ids: List of ground truth relevant chunk IDs
            k: Number of top results to consider
            
        Returns:
            Recall score between 0 and 1
        """
        if not relevant_chunk_ids:
            return 0.0
        
        retrieved_set = set(retrieved_chunk_ids[:k])
        relevant_set = set(relevant_chunk_ids)
        
        intersection = retrieved_set & relevant_set
        recall = len(intersection) / len(relevant_set)
        
        return recall
    
    def evaluate_retrieval_quality(
        self,
        questions: List[EvaluationQuestion],
        retrieval_results: Dict[str, List[str]]  # question -> list of retrieved chunk_ids
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality using Recall@k.
        
        Args:
            questions: List of evaluation questions with ground truth
            retrieval_results: Dictionary mapping questions to retrieved chunk IDs
            
        Returns:
            Dictionary with recall statistics
        """
        recalls = []
        
        for q in questions:
            if q.relevant_chunk_ids:
                retrieved = retrieval_results.get(q.question, [])
                recall = self.recall_at_k(retrieved, q.relevant_chunk_ids, k=5)
                recalls.append(recall)
        
        if not recalls:
            return {"mean_recall_at_k": None, "n_questions": 0}
        
        mean_recall = sum(recalls) / len(recalls)
        
        return {
            "mean_recall_at_k": mean_recall,
            "n_questions": len(recalls),
            "individual_recalls": recalls
        }
    
    def compare_rag_vs_baseline(
        self,
        questions: List[str],
        rag_answers: Dict[str, str],  # question -> answer
        baseline_answers: Dict[str, str]  # question -> answer
    ) -> Dict[str, Any]:
        """
        Compare RAG answers vs. baseline (no retrieval) answers.
        
        This is a qualitative comparison. For quantitative evaluation,
        manual inspection or LLM-as-judge can be used.
        
        Args:
            questions: List of questions
            rag_answers: Answers generated with retrieval
            baseline_answers: Answers generated without retrieval
            
        Returns:
            Dictionary with comparison statistics
        """
        comparisons = []
        
        for q in questions:
            rag = rag_answers.get(q, "")
            baseline = baseline_answers.get(q, "")
            
            comparisons.append({
                "question": q,
                "rag_answer": rag,
                "baseline_answer": baseline
            })
        
        return {
            "comparisons": comparisons,
            "n_questions": len(comparisons)
        }
    
    def find_retrieval_failures(
        self,
        questions: List[EvaluationQuestion],
        retrieval_results: Dict[str, List[str]],
        min_recall_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find cases where retrieval failed (low Recall@k).
        
        Args:
            questions: List of evaluation questions with ground truth
            retrieval_results: Dictionary mapping questions to retrieved chunk IDs
            min_recall_threshold: Minimum recall to consider a failure (default: 0.3)
            
        Returns:
            List of failure cases with explanations
        """
        failures = []
        
        for q in questions:
            if not q.relevant_chunk_ids:
                continue
            
            retrieved = retrieval_results.get(q.question, [])
            recall = self.recall_at_k(retrieved, q.relevant_chunk_ids, k=5)
            
            if recall < min_recall_threshold:
                failures.append({
                    "question": q.question,
                    "recall_at_k": recall,
                    "retrieved_chunk_ids": retrieved[:5],
                    "relevant_chunk_ids": q.relevant_chunk_ids,
                    "explanation": f"Retrieval failed: only {recall:.2%} of relevant chunks retrieved in top-5"
                })
        
        return failures
    
    def find_hallucination_cases(
        self,
        questions: List[str],
        rag_answers: Dict[str, str],
        retrieved_contexts: Dict[str, List[str]]  # question -> list of context texts
    ) -> List[Dict[str, Any]]:
        """
        Find potential hallucination cases (answers that may contain information not in context).
        
        This is a simple heuristic. For proper evaluation, manual inspection or
        LLM-as-judge should be used.
        
        Args:
            questions: List of questions
            rag_answers: Answers generated with retrieval
            retrieved_contexts: Dictionary mapping questions to retrieved context texts
            
        Returns:
            List of potential hallucination cases
        """
        hallucinations = []
        
        for q in questions:
            answer = rag_answers.get(q, "")
            contexts = retrieved_contexts.get(q, [])
            context_text = " ".join(contexts)
            
            # Simple heuristic: if answer mentions entities/numbers not in context
            # This is a placeholder - proper hallucination detection requires
            # more sophisticated methods (e.g., LLM-as-judge, fact-checking)
            
            hallucinations.append({
                "question": q,
                "answer": answer,
                "context_preview": context_text[:200] + "..." if len(context_text) > 200 else context_text,
                "note": "Manual inspection required to confirm hallucination"
            })
        
        return hallucinations
    
    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Save evaluation results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved evaluation results to {output_path}")

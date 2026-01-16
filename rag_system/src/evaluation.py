# -*- coding: utf-8-sig -*-
"""
Evaluation Module for RAG System.
Evaluates retrieval quality and answer quality.

University of Antwerp - Information Retrieval Assignment 3
References:
- Assignment: "Evaluate Recall@k, compare with/without retrieval"
- Es et al. (2024): RAGAS evaluation framework
- Zheng et al. (2024): LLM-as-a-judge
- Manning et al. (2008): Introduction to Information Retrieval, Chapter 8
- IR Assignment 2: Comprehensive evaluation metrics (MAP, nDCG, MRR)
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import re
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    TOP_K, 
    EVALUATION_DIR, 
    OPENAI_API_KEY,
    OPENAI_MODEL
)
from config import MODEL_PRICING_USD_PER_1M
from src.retrieval import Retriever, RetrievalResult
from src.generation import AnswerGenerator
from src.cost_tracking import estimate_cost_usd, log_event


@dataclass
class EvaluationQuestion:
    """A test question with optional ground truth."""
    question: str
    expected_answer: Optional[str] = None
    relevant_chunks: Optional[List[str]] = None  # Ground truth chunk IDs
    category: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "relevant_chunks": self.relevant_chunks,
            "category": self.category
        }


class RAGEvaluator:
    """
    Evaluates RAG system performance.
    
    Metrics:
    1. Retrieval Quality:
       - Recall@k: How many relevant docs are in top-k
       - MRR: Mean Reciprocal Rank
       
    2. Answer Quality (using LLM-as-judge):
       - Faithfulness: Is answer grounded in retrieved context?
       - Answer Relevance: Does answer address the question?
       - Completeness: Is the answer complete?
       
    3. Error Analysis:
       - Retrieval failures
       - Hallucination cases
    """
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[AnswerGenerator] = None,
        api_key: Optional[str] = None
    ):
        self.retriever = retriever
        self.generator = generator
        self.api_key = api_key or OPENAI_API_KEY
        self.client = None
        
        if self.api_key:
            self._init_openai()
    
    def _init_openai(self) -> None:
        """Initialize OpenAI client for LLM-as-judge."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            print("⚠️ OpenAI not available for LLM-as-judge evaluation")
    
    # =========================================================================
    # RETRIEVAL METRICS (Based on Manning et al. Chapter 8 + IR HW2)
    # =========================================================================
    
    def recall_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int = TOP_K
    ) -> float:
        """
        Calculate Recall@k.
        
        Recall@k = |relevant ∩ retrieved@k| / |relevant|
        Reference: Manning et al., Section 8.4
        
        Args:
            retrieved_ids: List of retrieved chunk IDs (in order)
            relevant_ids: List of ground truth relevant chunk IDs
            k: Number of top results to consider
            
        Returns:
            Recall score between 0 and 1
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        intersection = retrieved_set & relevant_set
        return len(intersection) / len(relevant_set)
    
    def mrr(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR = 1 / rank of first relevant result
        Reference: Manning et al., Section 8.4
        
        Args:
            retrieved_ids: List of retrieved chunk IDs (in order)
            relevant_ids: List of ground truth relevant chunk IDs
            
        Returns:
            MRR score between 0 and 1
        """
        relevant_set = set(relevant_ids)
        
        for rank, chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    def precision_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int = TOP_K
    ) -> float:
        """
        Calculate Precision@k.
        
        Precision@k = |relevant ∩ retrieved@k| / k
        Reference: Manning et al., Section 8.4
        
        Args:
            retrieved_ids: List of retrieved chunk IDs
            relevant_ids: List of ground truth relevant chunk IDs
            k: Number of top results
            
        Returns:
            Precision score between 0 and 1
        """
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        intersection = retrieved_set & relevant_set
        return len(intersection) / k
    
    def f1_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = TOP_K
    ) -> float:
        """
        F1@k: Harmonic mean of Precision@k and Recall@k.
        Reference: Manning et al., Section 8.3 (from IR HW2)
        """
        p = self.precision_at_k(retrieved_ids, relevant_ids, k)
        r = self.recall_at_k(retrieved_ids, relevant_ids, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    def average_precision(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Average Precision (AP): Mean precision at each relevant position.
        
        Formula: AP = (1/R) * Σ P(k) * rel(k)
        Reference: Manning et al., Section 8.4 (from IR HW2)
        """
        relevant_set = set(relevant_ids)
        precisions = []
        hits = 0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                hits += 1
                precisions.append(hits / (i + 1))
        
        if not precisions:
            return 0.0
        return np.mean(precisions)
    
    def dcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = TOP_K
    ) -> float:
        """
        Discounted Cumulative Gain at k.
        
        Formula: DCG@k = Σ rel_i / log2(i + 2)
        Reference: Manning et al., Section 8.4 (from IR HW2)
        """
        relevant_set = set(relevant_ids)
        dcg = sum([
            1 / np.log2(i + 2) 
            for i, doc_id in enumerate(retrieved_ids[:k]) 
            if doc_id in relevant_set
        ])
        return dcg
    
    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = TOP_K
    ) -> float:
        """
        Normalized Discounted Cumulative Gain at k.
        
        Formula: nDCG@k = DCG@k / IDCG@k
        Reference: Manning et al., Section 8.4 (from IR HW2)
        """
        dcg = self.dcg_at_k(retrieved_ids, relevant_ids, k)
        # Ideal DCG: all relevant at top
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_ids), k))])
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def compute_precision_recall_curve(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        max_k: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Precision-Recall curve by varying k.
        Reference: Manning et al., Section 8.4, Figure 8.1 (from IR HW2)
        """
        if max_k is None:
            max_k = len(retrieved_ids)
        
        relevant_set = set(relevant_ids)
        R_total = len(relevant_set)
        
        recalls = []
        precisions = []
        hits = 0
        
        for k in range(1, max_k + 1):
            if retrieved_ids[k-1] in relevant_set:
                hits += 1
            recalls.append(hits / R_total if R_total > 0 else 0)
            precisions.append(hits / k)
        
        return np.array(recalls), np.array(precisions)
    
    # =========================================================================
    # ANSWER QUALITY METRICS (LLM-as-Judge)
    # =========================================================================

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
        """
        Best-effort JSON extraction.
        Some models may wrap JSON with prose or markdown; this keeps evaluation robust.
        """
        text = (text or "").strip()
        if not text:
            return None
        # Fast path: full JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Fallback: find first {...} block
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None
    
    def evaluate_faithfulness(
        self, 
        answer: str, 
        context: str
    ) -> Dict[str, Any]:
        """
        Evaluate if the answer is faithful to the retrieved context.
        Uses LLM-as-judge approach.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Dict with score and explanation
        """
        if not self.client:
            return {"score": None, "error": "OpenAI client not initialized"}
        
        prompt = f"""You are evaluating the faithfulness of an answer to its source context.

CONTEXT:
{context}

ANSWER:
{answer}

Evaluate whether the answer is faithful to the context. An answer is faithful if:
1. All claims in the answer can be verified from the context
2. The answer does not contain information not present in the context
3. The answer does not contradict the context

Rate the faithfulness on a scale of 0-100:
- 0-30: Low faithfulness (contains hallucinations or unsupported claims)
- 31-60: Medium faithfulness (mostly supported but some gaps)
- 61-90: High faithfulness (well-supported by context)
- 91-100: Perfect faithfulness (completely grounded in context)

Respond with JSON format:
{{"score": <0-100>, "explanation": "<brief explanation>"}} 

Return ONLY valid JSON. Do not add any extra text or markdown."""

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=220,
            )
            
            content = response.choices[0].message.content or ""
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)
            total_tokens = getattr(response.usage, "total_tokens", None)
            cost_est = None
            if prompt_tokens is not None and completion_tokens is not None:
                cost_est = estimate_cost_usd(
                    model=OPENAI_MODEL,
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    pricing_table=MODEL_PRICING_USD_PER_1M,
                )

            log_event(
                {
                    "operation": "llm_judge_faithfulness",
                    "model": OPENAI_MODEL,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_estimate_usd": cost_est,
                }
            )

            obj = self._extract_first_json_object(content)
            if not isinstance(obj, dict):
                return {
                    "score": None,
                    "error": "Failed to parse JSON judge response",
                    "raw_preview": content[:300],
                    "tokens_used": {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens},
                    "cost_estimate_usd": float(cost_est) if cost_est is not None else None,
                }

            obj["tokens_used"] = {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens}
            if cost_est is not None:
                obj["cost_estimate_usd"] = float(cost_est)
            return obj
            
        except Exception as e:
            return {"score": None, "error": str(e)}
    
    def evaluate_answer_relevance(
        self, 
        question: str, 
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate if the answer is relevant to the question.
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            Dict with score and explanation
        """
        if not self.client:
            return {"score": None, "error": "OpenAI client not initialized"}
        
        prompt = f"""You are evaluating the relevance of an answer to a question.

QUESTION:
{question}

ANSWER:
{answer}

Evaluate whether the answer is relevant and addresses the question. Consider:
1. Does the answer directly address the question asked?
2. Is the answer complete and informative?
3. Is the answer specific enough for the question?

Rate the relevance on a scale of 0-100:
- 0-30: Low relevance (off-topic or doesn't answer the question)
- 31-60: Medium relevance (partially answers the question)
- 61-90: High relevance (good answer with minor gaps)
- 91-100: Perfect relevance (complete and focused answer)

Respond with JSON format:
{{"score": <0-100>, "explanation": "<brief explanation>"}} 

Return ONLY valid JSON. Do not add any extra text or markdown."""

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=220,
            )
            
            content = response.choices[0].message.content or ""
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)
            total_tokens = getattr(response.usage, "total_tokens", None)
            cost_est = None
            if prompt_tokens is not None and completion_tokens is not None:
                cost_est = estimate_cost_usd(
                    model=OPENAI_MODEL,
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    pricing_table=MODEL_PRICING_USD_PER_1M,
                )

            log_event(
                {
                    "operation": "llm_judge_answer_relevance",
                    "model": OPENAI_MODEL,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_estimate_usd": cost_est,
                }
            )

            obj = self._extract_first_json_object(content)
            if not isinstance(obj, dict):
                return {
                    "score": None,
                    "error": "Failed to parse JSON judge response",
                    "raw_preview": content[:300],
                    "tokens_used": {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens},
                    "cost_estimate_usd": float(cost_est) if cost_est is not None else None,
                }

            obj["tokens_used"] = {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens}
            if cost_est is not None:
                obj["cost_estimate_usd"] = float(cost_est)
            return obj
            
        except Exception as e:
            return {"score": None, "error": str(e)}
    
    def compare_answers(
        self, 
        question: str, 
        answer_with_rag: str, 
        answer_without_rag: str
    ) -> Dict[str, Any]:
        """
        Compare answers with and without RAG (baseline comparison).
        
        Args:
            question: Original question
            answer_with_rag: Answer generated with retrieval
            answer_without_rag: Answer generated without retrieval
            
        Returns:
            Dict with comparison results
        """
        if not self.client:
            return {"error": "OpenAI client not initialized"}
        
        prompt = f"""Compare these two answers to the same question.

QUESTION:
{question}

ANSWER A (with retrieval):
{answer_with_rag}

ANSWER B (without retrieval):
{answer_without_rag}

Compare the answers on these criteria:
1. Correctness: Which answer is more factually accurate?
2. Completeness: Which answer is more comprehensive?
3. Specificity: Which answer provides more specific details?

For each criterion, respond with "A", "B", or "tie".
Also identify any potential hallucinations in Answer B (without retrieval).

Respond with JSON format:
{{
  "correctness": "A" | "B" | "tie",
  "completeness": "A" | "B" | "tie",
  "specificity": "A" | "B" | "tie",
  "overall_winner": "A" | "B" | "tie",
  "hallucinations_in_B": ["list of hallucinated claims if any"],
  "explanation": "<brief explanation>"
}}

Return ONLY valid JSON. Do not add any extra text or markdown."""

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=350,
            )
            
            content = response.choices[0].message.content or ""
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)
            total_tokens = getattr(response.usage, "total_tokens", None)
            cost_est = None
            if prompt_tokens is not None and completion_tokens is not None:
                cost_est = estimate_cost_usd(
                    model=OPENAI_MODEL,
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    pricing_table=MODEL_PRICING_USD_PER_1M,
                )

            log_event(
                {
                    "operation": "llm_judge_compare_answers",
                    "model": OPENAI_MODEL,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_estimate_usd": cost_est,
                }
            )

            obj = self._extract_first_json_object(content)
            if not isinstance(obj, dict):
                return {
                    "error": "Failed to parse JSON judge response",
                    "raw_preview": content[:300],
                    "tokens_used": {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens},
                    "cost_estimate_usd": float(cost_est) if cost_est is not None else None,
                }

            obj["tokens_used"] = {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens}
            if cost_est is not None:
                obj["cost_estimate_usd"] = float(cost_est)
            return obj
            
        except Exception as e:
            return {"error": str(e)}
    
    # =========================================================================
    # FULL EVALUATION PIPELINE
    # =========================================================================
    
    def evaluate_single_query(
        self, 
        question: EvaluationQuestion,
        k: int = TOP_K
    ) -> Dict[str, Any]:
        """
        Evaluate a single query end-to-end.
        
        Args:
            question: EvaluationQuestion object
            k: Number of passages to retrieve
            
        Returns:
            Complete evaluation results
        """
        result = {
            "question": question.question,
            "category": question.category,
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 1: Retrieval
        if self.retriever:
            context, retrieved_results = self.retriever.retrieve_with_context(
                question.question, k=k
            )
            retrieved_ids = [r.chunk_id for r in retrieved_results]
            
            result["retrieval"] = {
                "num_retrieved": len(retrieved_results),
                "top_scores": [r.similarity_score for r in retrieved_results],
                # For manual inspection in the report (assignment requirement)
                "top_passages": [
                    {
                        "rank": r.rank,
                        "chunk_id": r.chunk_id,
                        "source_title": r.source_title,
                        "source_section": r.source_section,
                        "similarity_score": r.similarity_score,
                        "text_preview": (r.text[:400] + "...") if len(r.text) > 400 else r.text,
                    }
                    for r in retrieved_results
                ],
            }
            
            # Retrieval metrics if ground truth available
            if question.relevant_chunks:
                relevant_ids = question.relevant_chunks
                result["retrieval"]["relevant_chunks"] = relevant_ids

                # Core required metric
                result["retrieval"]["recall_at_k"] = self.recall_at_k(retrieved_ids, relevant_ids, k)

                # Extra IR metrics (useful for analysis + report)
                result["retrieval"]["precision_at_k"] = self.precision_at_k(retrieved_ids, relevant_ids, k)
                result["retrieval"]["f1_at_k"] = self.f1_at_k(retrieved_ids, relevant_ids, k)
                result["retrieval"]["mrr"] = self.mrr(retrieved_ids, relevant_ids)
                result["retrieval"]["ndcg_at_k"] = self.ndcg_at_k(retrieved_ids, relevant_ids, k)
                result["retrieval"]["average_precision"] = self.average_precision(retrieved_ids, relevant_ids)
        
        # Step 2: Generation
        if self.generator:
            # With RAG
            rag_result = self.generator.generate(question.question, k=k)
            result["answer_with_rag"] = rag_result["answer"]
            
            # Without RAG (baseline)
            baseline_result = self.generator.generate_without_retrieval(question.question)
            result["answer_without_rag"] = baseline_result["answer"]
            
            # Step 3: Answer quality evaluation
            if self.client:
                result["faithfulness"] = self.evaluate_faithfulness(
                    rag_result["answer"], context
                )
                result["answer_relevance"] = self.evaluate_answer_relevance(
                    question.question, rag_result["answer"]
                )
                result["comparison"] = self.compare_answers(
                    question.question,
                    rag_result["answer"],
                    baseline_result["answer"]
                )
        
        return result
    
    def evaluate_batch(
        self, 
        questions: List[EvaluationQuestion],
        k: int = TOP_K,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of questions.
        
        Args:
            questions: List of EvaluationQuestion objects
            k: Number of passages to retrieve
            save_results: Whether to save results to file
            
        Returns:
            Aggregated evaluation results
        """
        print(f"\n🔍 Evaluating {len(questions)} questions...")
        
        all_results = []
        for i, question in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {question.question[:50]}...")
            result = self.evaluate_single_query(question, k=k)
            all_results.append(result)
        
        # Aggregate metrics
        aggregated = self._aggregate_results(all_results)
        
        full_results = {
            "summary": aggregated,
            "individual_results": all_results,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "k": k,
                "num_questions": len(questions)
            }
        }
        
        if save_results:
            EVALUATION_DIR.mkdir(exist_ok=True)
            output_path = EVALUATION_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w', encoding='utf-8-sig') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to {output_path}")
        
        return full_results
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate individual results into summary metrics.
        Enhanced with comprehensive IR metrics from IR HW2.
        """
        summary = {}
        
        # =====================================================================
        # RETRIEVAL METRICS (Enhanced from IR HW2)
        # =====================================================================
        retrieval_metrics = {
            "recall_at_k": [],
            "precision_at_k": [],
            "f1_at_k": [],
            "mrr": [],
            "ndcg_at_k": [],
            "average_precision": []
        }
        
        for r in results:
            if "retrieval" in r:
                ret = r["retrieval"]
                if "recall_at_k" in ret:
                    retrieval_metrics["recall_at_k"].append(ret["recall_at_k"])
                if "precision_at_k" in ret:
                    retrieval_metrics["precision_at_k"].append(ret["precision_at_k"])
                if "f1_at_k" in ret:
                    retrieval_metrics["f1_at_k"].append(ret["f1_at_k"])
                if "mrr" in ret:
                    retrieval_metrics["mrr"].append(ret["mrr"])
                if "ndcg_at_k" in ret:
                    retrieval_metrics["ndcg_at_k"].append(ret["ndcg_at_k"])
                if "average_precision" in ret:
                    retrieval_metrics["average_precision"].append(ret["average_precision"])
        
        # Compute mean for each metric
        for metric_name, values in retrieval_metrics.items():
            if values:
                summary[f"mean_{metric_name}"] = float(np.mean(values))
                summary[f"std_{metric_name}"] = float(np.std(values))
        
        # Compute MAP (Mean Average Precision)
        if retrieval_metrics["average_precision"]:
            summary["MAP"] = float(np.mean(retrieval_metrics["average_precision"]))
        
        # =====================================================================
        # ANSWER QUALITY METRICS
        # =====================================================================
        faithfulness_scores = [r["faithfulness"]["score"] 
                              for r in results 
                              if "faithfulness" in r and r["faithfulness"].get("score")]
        if faithfulness_scores:
            summary["avg_faithfulness"] = float(np.mean(faithfulness_scores))
            summary["std_faithfulness"] = float(np.std(faithfulness_scores))
        
        relevance_scores = [r["answer_relevance"]["score"] 
                           for r in results 
                           if "answer_relevance" in r and r["answer_relevance"].get("score")]
        if relevance_scores:
            summary["avg_relevance"] = float(np.mean(relevance_scores))
            summary["std_relevance"] = float(np.std(relevance_scores))
        
        # =====================================================================
        # RAG VS BASELINE COMPARISON
        # =====================================================================
        comparisons = [r["comparison"] for r in results if "comparison" in r and "overall_winner" in r["comparison"]]
        if comparisons:
            rag_wins = sum(1 for c in comparisons if c["overall_winner"] == "A")
            baseline_wins = sum(1 for c in comparisons if c["overall_winner"] == "B")
            ties = sum(1 for c in comparisons if c["overall_winner"] == "tie")
            summary["rag_vs_baseline"] = {
                "rag_wins": rag_wins,
                "baseline_wins": baseline_wins,
                "ties": ties,
                "rag_win_rate": float(rag_wins / len(comparisons)) if comparisons else 0,
                "total_comparisons": len(comparisons)
            }
        
        return summary
    
    # =========================================================================
    # ERROR ANALYSIS
    # =========================================================================
    
    def find_retrieval_failures(
        self, 
        results: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find cases where retrieval failed (low recall).
        
        Args:
            results: List of evaluation results
            threshold: Recall threshold below which is considered failure
            
        Returns:
            List of failed cases
        """
        failures = []
        for r in results:
            if "retrieval" in r and r["retrieval"].get("recall_at_k", 1.0) < threshold:
                failures.append({
                    "question": r["question"],
                    "recall": r["retrieval"]["recall_at_k"],
                    "top_scores": r["retrieval"].get("top_scores", [])
                })
        return failures
    
    def find_hallucination_cases(
        self, 
        results: List[Dict[str, Any]],
        threshold: float = 50
    ) -> List[Dict[str, Any]]:
        """
        Find cases where answers may contain hallucinations.
        
        Args:
            results: List of evaluation results
            threshold: Faithfulness score below which suggests hallucination
            
        Returns:
            List of potential hallucination cases
        """
        hallucinations = []
        for r in results:
            if "faithfulness" in r and r["faithfulness"].get("score", 100) < threshold:
                hallucinations.append({
                    "question": r["question"],
                    "answer": r.get("answer_with_rag", ""),
                    "faithfulness_score": r["faithfulness"]["score"],
                    "explanation": r["faithfulness"].get("explanation", "")
                })
        return hallucinations
    
    # =========================================================================
    # SIMILARITY THRESHOLD ANALYSIS (from IR HW2 feedback)
    # =========================================================================
    
    def analyze_similarity_distribution(
        self,
        results: List[Dict[str, Any]],
        k: int = TOP_K
    ) -> Dict[str, Any]:
        """
        Analyze the distribution of similarity scores for the k-th neighbor.
        
        IR HW2 FEEDBACK: "What is the distribution of the similarity of the 
        10th neighbor? Then tweak parameters such that top-k neighbors are 
        selected with high probability."
        
        Args:
            results: List of evaluation results
            k: Position of neighbor to analyze
            
        Returns:
            Distribution statistics
        """
        kth_scores = []
        all_scores = []
        
        for r in results:
            if "retrieval" in r and "top_scores" in r["retrieval"]:
                scores = r["retrieval"]["top_scores"]
                all_scores.extend(scores)
                if len(scores) >= k:
                    kth_scores.append(scores[k-1])
        
        if not kth_scores:
            return {"error": "No k-th neighbor scores available"}
        
        analysis = {
            f"k{k}_neighbor_stats": {
                "mean": float(np.mean(kth_scores)),
                "std": float(np.std(kth_scores)),
                "min": float(np.min(kth_scores)),
                "max": float(np.max(kth_scores)),
                "median": float(np.median(kth_scores)),
                "percentile_25": float(np.percentile(kth_scores, 25)),
                "percentile_75": float(np.percentile(kth_scores, 75))
            },
            "all_scores_stats": {
                "mean": float(np.mean(all_scores)),
                "std": float(np.std(all_scores)),
                "min": float(np.min(all_scores)),
                "max": float(np.max(all_scores))
            },
            "recommended_threshold": float(np.percentile(kth_scores, 10)),
            "explanation": f"Recommended threshold is 10th percentile of k={k} neighbor scores"
        }
        
        return analysis
    
    def compute_efficiency_metrics(
        self,
        n_candidates: int,
        n_total: int,
        query_time: float = None,
        baseline_time: float = None
    ) -> Dict[str, float]:
        """
        Compute efficiency metrics for retrieval.
        From IR HW2 evaluation module.
        
        Args:
            n_candidates: Number of candidate documents examined
            n_total: Total number of documents in corpus
            query_time: Time for the query method
            baseline_time: Time for baseline (brute-force)
            
        Returns:
            Efficiency metrics
        """
        metrics = {
            "candidate_ratio": n_candidates / n_total,
            "documents_examined": n_candidates,
            "corpus_size": n_total
        }
        
        if query_time is not None:
            metrics["query_time_ms"] = query_time * 1000
        
        if query_time is not None and baseline_time is not None and query_time > 0:
            metrics["speedup"] = baseline_time / query_time
        
        return metrics


def create_test_questions() -> List[EvaluationQuestion]:
    """
    Create a balanced and dataset-driven set of test questions for evaluation.

    IMPORTANT (data-driven design):
    - The provided corpus includes:
      - `course-pages.json`: 33 course pages with stable section headers
        (Prerequisites, Learning Outcomes, Assessment method and criteria, etc.)
      - `website-scraped.json`: includes 3 long "Study programme" pages (DS/SE/CN)
        that list compulsory courses in a structured markdown table-like format.

    This question set is designed to:
    - Cover single-hop (easy) + multi-hop (hard) queries
    - Stress "study programme" retrieval (multi-course lists, filtering by semester/lecturer)
    - Include some website-level queries (admission, job opportunities)
    - Include boundary/unanswerable questions to test abstention / hallucination control
    """
    return [
        # =====================================================================
        # COURSE PAGE — SINGLE-HOP (stable sections / details)
        # =====================================================================
        EvaluationQuestion(
            question="What are the prerequisites for the Internet of Things course?",
            category="course_single_hop"
        ),
        EvaluationQuestion(
            question="What are the learning outcomes of the Internet of Things course?",
            category="course_single_hop"
        ),
        EvaluationQuestion(
            question="How is the Internet of Things course assessed (assessment method and criteria)?",
            category="course_single_hop"
        ),
        EvaluationQuestion(
            question="What is the exam period for the Internet of Things course?",
            category="course_single_hop"
        ),
        EvaluationQuestion(
            question="Who teaches Database Systems and in which semester is it offered?",
            category="course_single_hop"
        ),
        EvaluationQuestion(
            question="What are the prerequisites for Advanced Wireless and 5G Networks?",
            category="course_single_hop"
        ),

        # =====================================================================
        # STUDY PROGRAMME PAGES — MULTI-HOP (structured course lists)
        # (These are designed to stress DS/SE/CN 'Study programme' pages.)
        # =====================================================================
        EvaluationQuestion(
            question="In the Computer Networks study programme (2025-2026), list the compulsory courses in the 1st semester (names + course codes).",
            category="study_programme_multi_hop"
        ),
        EvaluationQuestion(
            question="In the Data Science study programme (2025-2026), which compulsory courses are in the 2nd semester?",
            category="study_programme_multi_hop"
        ),
        EvaluationQuestion(
            question="In the Software Engineering study programme (2025-2026), which compulsory courses have 'Exam contract not possible' as contract restrictions?",
            category="study_programme_multi_hop"
        ),
        EvaluationQuestion(
            question="Which specialization (Data Science, Software Engineering, or Computer Networks) includes 'Future Internet' as a compulsory course, and what is its course code?",
            category="study_programme_multi_hop"
        ),
        EvaluationQuestion(
            question="Which compulsory course in the Software Engineering study programme is taught by Hans Vangheluwe?",
            category="study_programme_multi_hop"
        ),
        EvaluationQuestion(
            question="Which compulsory course in the Computer Networks study programme is taught by Juan Felipe Botero, and what is its exam period?",
            category="study_programme_multi_hop"
        ),

        # =====================================================================
        # CROSS-PAGE MULTI-HOP (study programme → course page sections)
        # =====================================================================
        EvaluationQuestion(
            question="For the course 'Model Driven Engineering' (listed in the Software Engineering study programme), what are its prerequisites and learning outcomes?",
            category="cross_page_multi_hop"
        ),
        EvaluationQuestion(
            question="For the course 'Future Internet' (listed in the Computer Networks study programme), what are its course contents and assessment method?",
            category="cross_page_multi_hop"
        ),
        EvaluationQuestion(
            question="For the course 'Data Mining' (listed in the Data Science study programme), what are its prerequisites and course contents?",
            category="cross_page_multi_hop"
        ),

        # =====================================================================
        # WEBSITE PAGES (admission / job opportunities)
        # =====================================================================
        EvaluationQuestion(
            question="What career opportunities are mentioned for the Data Science specialization?",
            category="website_single_hop"
        ),
        EvaluationQuestion(
            question="What are the admission and enrolment requirements for the Master of Computer Science: Software Engineering programme?",
            category="website_single_hop"
        ),

        # =====================================================================
        # BOUNDARY / UNANSWERABLE (to test abstention + hallucination control)
        # =====================================================================
        EvaluationQuestion(
            question="What are the tuition fees for the Computer Science Masters?",
            category="boundary"
        ),
        EvaluationQuestion(
            question="What is the thesis submission deadline?",
            category="boundary"
        ),
        EvaluationQuestion(
            question="How can I get a scholarship for this programme?",
            category="boundary"
        ),
    ]


def create_parameter_sweep_config() -> Dict[str, List[Any]]:
    """
    Create parameter configurations for hyperparameter tuning.
    Based on IR HW2 feedback to explore parameter space.
    """
    return {
        "k_values": [1, 3, 5, 10, 15, 20],  # Number of retrieved passages
        "expansion_factors": [1, 2, 3, 5],   # Candidate expansion (c*k strategy)
        "chunk_sizes": [150, 250, 350],      # Chunk size in tokens
        "overlap_ratios": [0.1, 0.15, 0.2]   # Chunk overlap
    }


if __name__ == "__main__":
    print("RAG Evaluation Module")
    print("Create test questions with create_test_questions()")
    
    questions = create_test_questions()
    print(f"\nCreated {len(questions)} test questions:")
    for q in questions[:5]:
        print(f"  - [{q.category}] {q.question}")

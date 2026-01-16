# -*- coding: utf-8-sig -*-
"""
Score a manually labeled annotation pack and compute retrieval metrics.

This turns your manual inspection into:
- Recall@k (required by Assignment 3)
- Precision@k, F1@k, MRR, nDCG@k, AP, MAP (extra analysis)

Usage (do this yourself when ready):
  cd rag_system
  source venv/bin/activate
  python scripts/score_annotations.py \
    --in evaluation_results/annotation_pack.json \
    --out evaluation_results/retrieval_metrics.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import TOP_K  # noqa: E402
from src.evaluation import RAGEvaluator  # noqa: E402


def _extract_relevant_ids(item: Dict[str, Any]) -> List[str]:
    relevant = []
    for c in item.get("candidates", []):
        if c.get("is_relevant") is True:
            relevant.append(c["chunk_id"])
    return relevant


def _extract_ranked_ids(item: Dict[str, Any]) -> List[str]:
    return [c["chunk_id"] for c in item.get("candidates", [])]


def score_pack(pack: Dict[str, Any], k: int) -> Dict[str, Any]:
    evaluator = RAGEvaluator()

    per_question: List[Dict[str, Any]] = []

    for item in pack.get("items", []):
        question = item["question"]
        ranked_ids = _extract_ranked_ids(item)
        relevant_ids = _extract_relevant_ids(item)

        # If nothing is labeled relevant, treat as "no answerable evidence found"
        # (this is useful for boundary questions)
        metrics = {
            "recall_at_k": evaluator.recall_at_k(ranked_ids, relevant_ids, k=k) if relevant_ids else None,
            "precision_at_k": evaluator.precision_at_k(ranked_ids, relevant_ids, k=k) if relevant_ids else None,
            "f1_at_k": evaluator.f1_at_k(ranked_ids, relevant_ids, k=k) if relevant_ids else None,
            "mrr": evaluator.mrr(ranked_ids, relevant_ids) if relevant_ids else None,
            "ndcg_at_k": evaluator.ndcg_at_k(ranked_ids, relevant_ids, k=k) if relevant_ids else None,
            "average_precision": evaluator.average_precision(ranked_ids, relevant_ids) if relevant_ids else None,
            "n_relevant_labeled": len(relevant_ids),
        }

        per_question.append(
            {
                "question": question,
                "category": item.get("category", "general"),
                "k": k,
                "metrics": metrics,
                "relevant_chunks": relevant_ids,
                "top_k": ranked_ids[:k],
            }
        )

    # Aggregate over questions that have at least 1 relevant label
    def _vals(name: str) -> List[float]:
        vals = []
        for q in per_question:
            v = q["metrics"].get(name)
            if v is not None:
                vals.append(float(v))
        return vals

    summary = {}
    for m in ["recall_at_k", "precision_at_k", "f1_at_k", "mrr", "ndcg_at_k", "average_precision"]:
        vals = _vals(m)
        if vals:
            summary[f"mean_{m}"] = float(np.mean(vals))
            summary[f"std_{m}"] = float(np.std(vals))
            summary[f"n_questions_scored_{m}"] = len(vals)

    ap_vals = _vals("average_precision")
    if ap_vals:
        summary["MAP"] = float(np.mean(ap_vals))

    return {
        "k": k,
        "summary": summary,
        "per_question": per_question,
        "notes": "Metrics are aggregated only over questions where at least one candidate was labeled relevant.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a labeled annotation pack (JSON) and compute Recall@k.")
    parser.add_argument("--in", dest="in_path", type=str, required=True, help="Input annotation_pack.json")
    parser.add_argument("--out", dest="out_path", type=str, required=True, help="Output metrics JSON")
    parser.add_argument("-k", type=int, default=TOP_K, help=f"Recall@k cutoff (default: {TOP_K})")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8-sig") as f:
        pack = json.load(f)

    scored = score_pack(pack, k=args.k)

    with open(out_path, "w", encoding="utf-8-sig") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote retrieval metrics to: {out_path}")
    if scored["summary"].get("mean_recall_at_k") is not None:
        print(f"Mean Recall@{args.k}: {scored['summary']['mean_recall_at_k']:.4f}")


if __name__ == "__main__":
    main()


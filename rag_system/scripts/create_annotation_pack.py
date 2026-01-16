# -*- coding: utf-8-sig -*-
"""
Create an annotation pack for manual relevance labeling (Retrieval Evaluation).

Why this exists (aligns with Assignment 3 evaluation requirements):
- You must report Recall@k.
- You must manually inspect relevance of top-k passages.

Ground truth in small RAG projects is typically created by:
1) Sampling evaluation questions
2) Retrieving top-N candidates (N >= k)
3) Manually labeling each candidate as relevant / not relevant
4) Computing Recall@k, Precision@k, etc. from the labels

This script produces a single JSON file you can annotate easily (no API calls).

Usage (do this yourself when ready):
  cd rag_system
  source venv/bin/activate
  python scripts/create_annotation_pack.py --top-n 20 --out evaluation_results/annotation_pack.json
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import sys

# Ensure we can import project modules when running as a script
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import DATA_DIR, EMBEDDINGS_DIR, TOP_K  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.evaluation import create_test_questions  # noqa: E402


def build_annotation_pack(top_n: int) -> Dict[str, Any]:
    chunks_path = DATA_DIR / "chunks.json"
    index_path = EMBEDDINGS_DIR / "faiss_index"
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"

    retriever = Retriever()
    retriever.load_components(
        chunks_path=chunks_path,
        index_path=index_path,
        embeddings_path=embeddings_path,
    )

    questions = create_test_questions()

    items: List[Dict[str, Any]] = []
    for q in questions:
        query = q.question
        results = retriever.retrieve(query, k=top_n)

        candidates = []
        for r in results:
            candidates.append(
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk_id,
                    "source_title": r.source_title,
                    "source_section": r.source_section,
                    "similarity_score": r.similarity_score,
                    "text": r.text,
                    # Fill this manually: true/false
                    "is_relevant": None,
                    # Optional short note explaining your decision
                    "note": "",
                }
            )

        items.append(
            {
                "question": query,
                "category": q.category,
                "k": TOP_K,
                "top_n": top_n,
                "candidates": candidates,
                # Optional: store a free-form rationale in the report
                "question_note": "",
            }
        )

    return {
        "schema_version": "1.0",
        "description": "Manual relevance labels for Recall@k evaluation (Assignment 3).",
        "k_default": TOP_K,
        "top_n_default": top_n,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a manual relevance annotation pack (JSON).")
    parser.add_argument("--top-n", type=int, default=20, help="How many candidates to retrieve per question.")
    parser.add_argument(
        "--out",
        type=str,
        default=str((PROJECT_DIR / "evaluation_results" / "annotation_pack.json")),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pack = build_annotation_pack(top_n=args.top_n)
    with open(out_path, "w", encoding="utf-8-sig") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote annotation pack to: {out_path}")
    print("Next: open the JSON, fill `is_relevant` for each candidate, then run score_annotations.py")


if __name__ == "__main__":
    main()


# -*- coding: utf-8-sig -*-
"""
Create a pooled manual-annotation pack for comparing multiple retrieval systems
on the SAME question set (agentic vs baseline).

Why pooling matters:
- If you label separate candidate lists per system, the "relevant set" changes.
- Pooling labels the UNION of candidates once, then evaluates each system against
  a fixed relevance set (standard IR pooling approach for small evaluations).

This script performs NO OpenAI calls.

Usage (do this yourself when ready):
  cd rag_system
  source venv/bin/activate
  python scripts/create_pooled_ablation_pack.py \
    --top-n 20 \
    --out evaluation_results/pooled_ablation_pack.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import DATA_DIR, EMBEDDINGS_DIR, TOP_K  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.evaluation import create_test_questions  # noqa: E402
from src.langgraph_agentic_rag import AgenticConfig, agentic_retrieve_only  # noqa: E402


def _system_id(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _pack_schema(top_n: int) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "description": "Pooled relevance labels for ablation: baseline vs agentic (LangGraph).",
        "k_default": TOP_K,
        "top_n_per_system": top_n,
        "systems": [
            {"id": "baseline", "description": "Single-query dense retrieval (Retriever.retrieve)"},
            {"id": "agentic", "description": "LangGraph-style agentic retrieval (tag+rewrite+RRF+MMR)"},
        ],
        "items": [],
    }


def create_pack(top_n: int) -> Dict[str, Any]:
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
    pack = _pack_schema(top_n=top_n)

    agentic_cfg = AgenticConfig(top_k=TOP_K, top_n_per_query=top_n)

    for q in questions:
        query = q.question

        # System A: baseline retrieval
        baseline_results = retriever.retrieve(query, k=top_n)
        baseline_ranked = [r.chunk_id for r in baseline_results]
        baseline_scores = [float(r.similarity_score) for r in baseline_results]

        # System B: agentic retrieval (no generation)
        # NOTE: agentic_retrieve_only returns top-k (final selected context),
        # which is what we actually evaluate for Recall@k in the report.
        # Baseline contributes additional pooled candidates for labeling.
        agentic_results, agentic_trace = agentic_retrieve_only(query, retriever=retriever, cfg=agentic_cfg)
        agentic_ranked = [r.chunk_id for r in agentic_results]
        agentic_scores = [float(r.similarity_score) for r in agentic_results]

        # Union of candidates to label
        union_ids = []
        seen = set()
        for cid in baseline_ranked + agentic_ranked:
            if cid not in seen:
                union_ids.append(cid)
                seen.add(cid)

        candidates = []
        for cid in union_ids:
            ch = retriever.chunk_lookup.get(cid)
            if not ch:
                continue
            candidates.append(
                {
                    "chunk_id": cid,
                    "source_title": ch.source_title,
                    "source_section": ch.source_section,
                    "source_file": ch.source_file,
                    "text": ch.text,
                    # Manual labels
                    "is_relevant": None,
                    "note": "",
                    # Per-system rank/score (if present)
                    "systems": {
                        "baseline": {
                            "rank": (baseline_ranked.index(cid) + 1) if cid in baseline_ranked else None,
                            "score": baseline_scores[baseline_ranked.index(cid)] if cid in baseline_ranked else None,
                        },
                        "agentic": {
                            "rank": (agentic_ranked.index(cid) + 1) if cid in agentic_ranked else None,
                            "score": agentic_scores[agentic_ranked.index(cid)] if cid in agentic_ranked else None,
                        },
                    },
                }
            )

        pack["items"].append(
            {
                "question": query,
                "category": q.category,
                "k": TOP_K,
                "systems_ranked_ids": {
                    "baseline": baseline_ranked,
                    "agentic": agentic_ranked,
                },
                "systems_meta": {
                    "agentic_trace": agentic_trace,  # helpful for report appendix/debug
                },
                "candidates": candidates,
                "question_note": "",
            }
        )

    return pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Create pooled ablation annotation pack (baseline vs agentic).")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N candidates per system (baseline).")
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_DIR / "evaluation_results" / "pooled_ablation_pack.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pack = create_pack(top_n=args.top_n)
    out_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print(f"✅ Wrote pooled ablation pack: {out_path}")
    print("Next: label `is_relevant` for candidates, then run score_pooled_ablation_pack.py")


if __name__ == "__main__":
    main()


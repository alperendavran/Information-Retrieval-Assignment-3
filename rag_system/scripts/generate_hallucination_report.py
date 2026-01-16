# -*- coding: utf-8-sig -*-
"""
Generate answers (baseline vs RAG vs agentic RAG) and surface hallucination cases.

This script CAN call OpenAI (GPT-4o) if OPENAI_API_KEY is set.
It is designed to support the assignment requirement:
- Compare answers with retrieval to answers without retrieval (baseline).
- Provide ≥3 hallucination cases with explanations.

Usage (you run this when ready):
  cd rag_system
  source venv/bin/activate
  export OPENAI_API_KEY='...'
  python scripts/generate_hallucination_report.py \\
    --pack evaluation_results/pooled_ablation_pack.json \\
    --out evaluation_results/hallucination_report.json \\
    --k 5 \\
    --limit 12

Notes:
- We reuse your pooled pack, so the questions are fixed.
- We run:
  1) baseline: GPT-4o without retrieval
  2) normal RAG: Retriever top-k context
  3) agentic RAG: agentic_retrieve_only top-k context
- If you want, you can enable reflection/judge via config (optional).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv  # noqa: E402

# Load `.env` so users can run this script without manually exporting vars.
load_dotenv(dotenv_path=PROJECT_DIR / ".env")

from config import DATA_DIR, EMBEDDINGS_DIR, TOP_K  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.generation import AnswerGenerator  # noqa: E402
from src.evaluation import RAGEvaluator  # noqa: E402
from src.langgraph_agentic_rag import AgenticConfig, agentic_retrieve_only  # noqa: E402


def _format_context(selected: List[Dict[str, Any]]) -> str:
    parts = []
    for item in selected:
        title = item.get("source_title", "")
        section = item.get("source_section")
        src = f"[Source: {title}" + (f" - {section}]" if section else "]")
        parts.append(f"{src}\n{item.get('text','')}")
    return "\n\n---\n\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate hallucination report (requires OpenAI).")
    p.add_argument("--pack", required=True, help="pooled_ablation_pack.json")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("-k", type=int, default=TOP_K, help="top-k for retrieval context")
    p.add_argument("--limit", type=int, default=12, help="number of questions to run (cost control)")
    p.add_argument("--judge", action="store_true", help="also run LLM-as-judge faithfulness/relevance (more cost)")
    args = p.parse_args()

    pack = json.loads(Path(args.pack).read_text(encoding="utf-8-sig"))
    items = pack.get("items", [])[: max(int(args.limit), 1)]

    # Load retriever + generator
    retriever = Retriever()
    retriever.load_components(
        chunks_path=DATA_DIR / "chunks.json",
        index_path=EMBEDDINGS_DIR / "faiss_index",
        embeddings_path=EMBEDDINGS_DIR / "embeddings.npy",
    )

    generator = AnswerGenerator(retriever=retriever)  # uses OPENAI_API_KEY env
    evaluator = RAGEvaluator(retriever=retriever, generator=generator) if args.judge else None

    agentic_cfg = AgenticConfig(top_k=args.k, top_n_per_query=max(20, args.k))

    outputs: List[Dict[str, Any]] = []
    for it in items:
        q = it["question"]

        # 1) baseline (no retrieval)
        baseline = generator.generate_without_retrieval(q)

        # 2) normal RAG
        context_normal, retrieved_normal = retriever.retrieve_with_context(q, k=args.k)
        with_rag = generator.generate(q, context=context_normal, retrieved_results=retrieved_normal)

        # 3) agentic RAG (retrieval only)
        agentic_sel, agentic_trace = agentic_retrieve_only(q, retriever=retriever, cfg=agentic_cfg)
        agentic_ctx = _format_context([r.to_dict() for r in agentic_sel])
        agentic = generator.generate(q, context=agentic_ctx, retrieved_results=agentic_sel)

        row: Dict[str, Any] = {
            "question": q,
            "baseline_no_retrieval": baseline,
            "normal_rag": with_rag,
            "agentic_rag": agentic,
            "agentic_trace": agentic_trace,
        }

        if evaluator and evaluator.client:
            row["judge"] = {
                "normal_faithfulness": evaluator.evaluate_faithfulness(with_rag["answer"], context_normal),
                "agentic_faithfulness": evaluator.evaluate_faithfulness(agentic["answer"], agentic_ctx),
                "baseline_vs_normal": evaluator.compare_answers(q, with_rag["answer"], baseline["answer"]),
            }

        outputs.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"k": args.k, "items": outputs}, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"✅ Wrote hallucination report JSON: {out_path}")
    print("Next: manually pick ≥3 hallucination examples (or filter by low faithfulness scores if judge enabled).")


if __name__ == "__main__":
    main()


# -*- coding: utf-8-sig -*-
"""
Create a timestamped markdown report documenting:
- dataset/chunk stats (data-driven)
- artifacts created (chunks, embeddings, FAISS index, pooled pack)
- OFFLINE cost estimate for running generation with a chosen model (e.g., gpt-4o-mini)

This script does NOT call OpenAI.

Usage:
  cd rag_system
  source venv/bin/activate
  python scripts/create_run_report.py --model gpt-4o-mini --out evaluation_results/run_report.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tiktoken

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import (  # noqa: E402
    DATA_DIR,
    EMBEDDINGS_DIR,
    EVALUATION_DIR,
    TOP_K,
    SYSTEM_PROMPT,
    MAX_TOKENS,
    MODEL_PRICING_USD_PER_1M,
)
from src.retrieval import Retriever  # noqa: E402
from src.langgraph_agentic_rag import AgenticConfig, agentic_retrieve_only  # noqa: E402
from src.cost_tracking import estimate_cost_usd  # noqa: E402
from src.evaluation import create_test_questions  # noqa: E402


def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def format_rag_prompt(query: str, context: str) -> str:
    user_content = f"""Context information from the University of Antwerp CS Masters program:

{context}

---

Based on the above context, please answer the following question:

Question: {query}

Answer:"""
    # Approx token count: sum of system + user content
    return SYSTEM_PROMPT + "\n\n" + user_content


def format_baseline_prompt(query: str) -> str:
    system = (
        "You are a helpful assistant. Answer questions about the University of Antwerp Computer Science Masters program "
        "to the best of your knowledge."
    )
    user = f"Question: {query}\n\nAnswer:"
    return system + "\n\n" + user


def estimate_call_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
    cost = estimate_cost_usd(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pricing_table=MODEL_PRICING_USD_PER_1M,
    )
    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_estimate_usd": float(cost) if cost is not None else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Create a run report + offline cost estimate.")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name for cost estimate.")
    p.add_argument("--answer-tokens", type=int, default=350, help="Assumed average answer completion tokens.")
    p.add_argument("--judge-tokens", type=int, default=200, help="Assumed average judge completion tokens (JSON).")
    p.add_argument("--limit", type=int, default=12, help="How many questions to estimate for (cost control).")
    p.add_argument(
        "--out",
        type=str,
        default=str(EVALUATION_DIR / "run_report.md"),
        help="Output markdown path.",
    )
    args = p.parse_args()

    model = args.model
    answer_tokens = int(args.answer_tokens)
    judge_tokens = int(args.judge_tokens)
    limit = int(args.limit)

    ts = datetime.now().isoformat()

    # Artifact stats
    chunks_path = DATA_DIR / "chunks.json"
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    index_faiss = EMBEDDINGS_DIR / "faiss_index.faiss"
    pooled_pack = EVALUATION_DIR / "pooled_ablation_pack.json"

    def file_meta(p: Path) -> Dict[str, Any]:
        if not p.exists():
            return {"exists": False}
        st = p.stat()
        return {
            "exists": True,
            "size_bytes": int(st.st_size),
            "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
        }

    chunks = json.loads(chunks_path.read_text(encoding="utf-8-sig")) if chunks_path.exists() else []
    tok = np.array([c.get("token_count", 0) for c in chunks], dtype=np.int64) if chunks else np.array([], dtype=np.int64)

    emb_shape = None
    if embeddings_path.exists():
        emb = np.load(embeddings_path)
        emb_shape = tuple(emb.shape)

    # Load retriever for context-building (no API)
    retriever = Retriever()
    retriever.load_components(
        chunks_path=chunks_path,
        index_path=EMBEDDINGS_DIR / "faiss_index",
        embeddings_path=embeddings_path,
    )

    questions = create_test_questions()[: max(limit, 1)]
    agentic_cfg = AgenticConfig(top_k=TOP_K, top_n_per_query=max(20, TOP_K))

    # Offline prompt token estimates per question
    per_q = []
    for q in questions:
        query = q.question

        # Baseline prompt
        base_prompt = format_baseline_prompt(query)
        base_prompt_tokens = count_tokens(base_prompt)

        # Normal RAG context
        normal_context, normal_results = retriever.retrieve_with_context(query, k=TOP_K)
        normal_prompt = format_rag_prompt(query, normal_context)
        normal_prompt_tokens = count_tokens(normal_prompt)

        # Agentic context
        agentic_sel, _trace = agentic_retrieve_only(query, retriever=retriever, cfg=agentic_cfg)
        parts = []
        for r in agentic_sel:
            src = f"[Source: {r.source_title}" + (f" - {r.source_section}]" if r.source_section else "]")
            parts.append(f"{src}\n{r.text}")
        agentic_context = "\n\n---\n\n".join(parts)
        agentic_prompt = format_rag_prompt(query, agentic_context)
        agentic_prompt_tokens = count_tokens(agentic_prompt)

        per_q.append(
            {
                "question": query,
                "category": q.category,
                "baseline_prompt_tokens": int(base_prompt_tokens),
                "normal_rag_prompt_tokens": int(normal_prompt_tokens),
                "agentic_rag_prompt_tokens": int(agentic_prompt_tokens),
            }
        )

    # Aggregate prompt token stats
    def avg(key: str) -> float:
        return float(np.mean([x[key] for x in per_q])) if per_q else 0.0

    base_avg = int(round(avg("baseline_prompt_tokens")))
    normal_avg = int(round(avg("normal_rag_prompt_tokens")))
    agentic_avg = int(round(avg("agentic_rag_prompt_tokens")))

    # Cost estimates for a "hallucination report" run:
    # per question: baseline + normal RAG + agentic RAG = 3 calls
    per_question_costs = {
        "baseline_call": estimate_call_cost(model, base_avg, answer_tokens),
        "normal_rag_call": estimate_call_cost(model, normal_avg, answer_tokens),
        "agentic_rag_call": estimate_call_cost(model, agentic_avg, answer_tokens),
    }

    # Total for N questions
    n = len(per_q)
    total_est = 0.0
    for kname in ["baseline_call", "normal_rag_call", "agentic_rag_call"]:
        ce = per_question_costs[kname]["cost_estimate_usd"]
        if ce is not None:
            total_est += float(ce) * n

    # Worst-case upper bound (max_tokens)
    worst_total = 0.0
    for kname, avg_prompt in [("baseline", base_avg), ("normal_rag", normal_avg), ("agentic_rag", agentic_avg)]:
        ce = estimate_cost_usd(model, avg_prompt, MAX_TOKENS, MODEL_PRICING_USD_PER_1M)
        if ce is not None:
            worst_total += float(ce) * n

    # Markdown report
    lines: List[str] = []
    lines.append("## Run Report (Data-driven) + Offline Cost Estimate")
    lines.append("")
    lines.append(f"- **Generated at**: {ts}")
    lines.append(f"- **Assumed model for estimate**: `{model}`")
    lines.append(f"- **Assumed pricing (USD per 1M tokens)**: {MODEL_PRICING_USD_PER_1M.get(model, 'UNKNOWN')} (update in `config.py` if needed)")
    lines.append("")
    lines.append("### Artifacts")
    lines.append("")
    chunks_meta = file_meta(chunks_path)
    lines.append(
        f"- **Chunks**: `{chunks_path}` exists={chunks_meta.get('exists')} "
        f"count={len(chunks)} mtime={chunks_meta.get('modified_at')} size={chunks_meta.get('size_bytes')}"
    )
    if tok.size:
        lines.append(f"  - token_count p50={int(np.percentile(tok,50))}, p90={int(np.percentile(tok,90))}, p99={int(np.percentile(tok,99))}, max={int(np.max(tok))}")
    emb_meta = file_meta(embeddings_path)
    lines.append(
        f"- **Embeddings**: `{embeddings_path}` exists={emb_meta.get('exists')} "
        f"shape={emb_shape} mtime={emb_meta.get('modified_at')} size={emb_meta.get('size_bytes')}"
    )
    index_meta = file_meta(index_faiss)
    lines.append(
        f"- **FAISS index**: `{index_faiss}` exists={index_meta.get('exists')} "
        f"mtime={index_meta.get('modified_at')} size={index_meta.get('size_bytes')}"
    )
    pack_meta = file_meta(pooled_pack)
    lines.append(
        f"- **Pooled pack**: `{pooled_pack}` exists={pack_meta.get('exists')} "
        f"mtime={pack_meta.get('modified_at')} size={pack_meta.get('size_bytes')}"
    )
    lines.append("")

    lines.append("### Offline cost estimate (no API calls made)")
    lines.append("")
    lines.append(f"Estimated for a run over **{n} questions** with assumed average completion of **{answer_tokens} tokens** per answer.")
    lines.append("")
    lines.append("| Scenario | Avg prompt tokens | Assumed completion tokens | Est. cost per call (USD) | Est. total (USD) |")
    lines.append("|---|---:|---:|---:|---:|")

    def row(label: str, avg_prompt_tokens: int, call_key: str) -> None:
        ce = per_question_costs[call_key]["cost_estimate_usd"]
        per_call = float(ce) if ce is not None else None
        total = (per_call * n) if per_call is not None else None
        lines.append(
            f"| {label} | {avg_prompt_tokens} | {answer_tokens} | "
            f"{(f'{per_call:.6f}' if per_call is not None else 'n/a')} | "
            f"{(f'{total:.6f}' if total is not None else 'n/a')} |"
        )

    row("Baseline (no retrieval)", base_avg, "baseline_call")
    row("Normal RAG (Retriever top-k)", normal_avg, "normal_rag_call")
    row("Agentic RAG (tag+rewrite+RRF+MMR)", agentic_avg, "agentic_rag_call")

    lines.append("")
    lines.append(f"- **Estimated total for baseline+normal+agentic (3 calls per question)**: **${total_est:.6f}**")
    lines.append(f"- **Worst-case upper bound** (completion={MAX_TOKENS} tokens): **${worst_total:.6f}**")
    lines.append("")
    lines.append("### How to run with GPT-4o-mini (actual)")
    lines.append("")
    lines.append("Set environment variables (do NOT commit your key):")
    lines.append("")
    lines.append("```bash")
    lines.append("export OPENAI_MODEL='gpt-4o-mini'")
    lines.append("export OPENAI_API_KEY='...your key...'")
    lines.append("```")
    lines.append("")
    lines.append("Then generate real logs and summarize them:")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/generate_hallucination_report.py --pack evaluation_results/pooled_ablation_pack.json --out evaluation_results/hallucination_report.json --k 5 --limit 12 --judge")
    lines.append("python scripts/summarize_openai_costs.py")
    lines.append("```")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"✅ Wrote run report: {out_path}")


if __name__ == "__main__":
    main()


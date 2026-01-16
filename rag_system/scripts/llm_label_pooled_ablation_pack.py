# -*- coding: utf-8-sig -*-
"""
LLM-based relevance labeling for pooled ablation pack (using function calling).

Why this exists:
- Manual labeling is tedious and time-consuming
- Heuristic auto-labeling (auto_label_pooled_ablation_pack.py) is limited
- LLM can understand context and judge relevance more accurately

This script uses OpenAI function calling to:
1. Present each candidate to the LLM with the question
2. LLM returns a structured judgment (is_relevant: bool, confidence: float, reasoning: str)
3. We log costs and allow resume/budget controls

Usage:
  cd rag_system
  source venv/bin/activate
  python scripts/llm_label_pooled_ablation_pack.py \
    --in evaluation_results/pooled_ablation_pack.json \
    --out evaluation_results/pooled_ablation_pack_labeled_llm.json \
    --model gpt-4o-mini \
    --budget-usd 5.0 \
    --resume
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv

load_dotenv(dotenv_path=PROJECT_DIR / ".env")

from openai import OpenAI

from config import MODEL_PRICING_USD_PER_1M, OPENAI_API_KEY
from src.cost_tracking import estimate_cost_usd, log_event


def _relevance_judge_schema() -> Dict[str, Any]:
    """OpenAI function schema for relevance judgment."""
    return {
        "type": "function",
        "function": {
            "name": "judge_relevance",
            "description": "Judge whether a candidate chunk is relevant to answering the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "is_relevant": {
                        "type": "boolean",
                        "description": "True if the chunk contains information needed to answer the question, False otherwise.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score 0.0-1.0 (1.0 = very confident, 0.5 = uncertain).",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this chunk is/isn't relevant.",
                    },
                },
                "required": ["is_relevant", "confidence", "reasoning"],
                "additionalProperties": False,
            },
        },
    }


def judge_candidate_relevance(
    question: str,
    candidate: Dict[str, Any],
    client: OpenAI,
    model: str,
) -> Dict[str, Any]:
    """
    Use LLM to judge if a candidate chunk is relevant to the question.
    Returns: {"is_relevant": bool, "confidence": float, "reasoning": str, "cost_estimate_usd": float}
    """
    chunk_text = candidate.get("text", "")
    source_title = candidate.get("source_title", "")
    source_section = candidate.get("source_section", "")

    prompt = f"""You are evaluating whether a retrieved chunk is relevant to answering a user question.

QUESTION: {question}

CANDIDATE CHUNK:
Source: {source_title}
Section: {source_section or "(no section)"}
Text: {chunk_text[:2000]}  # Truncate to avoid token limits

INSTRUCTIONS:
- A chunk is RELEVANT if it contains information that directly helps answer the question.
- A chunk is NOT RELEVANT if it:
  * Is about a different topic/course
  * Contains only tangential information
  * Is from the wrong section (e.g., question asks for "prerequisites" but chunk is from "study material")
- Be strict: partial relevance or weak connections should be marked as NOT RELEVANT unless the information is clearly useful.

Judge the relevance and provide your reasoning."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=[_relevance_judge_schema()],
            tool_choice={"type": "function", "function": {"name": "judge_relevance"}},
            temperature=0.0,
            max_tokens=200,
        )

        prompt_tokens = getattr(response.usage, "prompt_tokens", None)
        completion_tokens = getattr(response.usage, "completion_tokens", None)
        total_tokens = getattr(response.usage, "total_tokens", None)
        cost_est = None
        if prompt_tokens is not None and completion_tokens is not None:
            cost_est = estimate_cost_usd(
                model=model,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                pricing_table=MODEL_PRICING_USD_PER_1M,
            )

        log_event(
            {
                "operation": "llm_label_relevance",
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_estimate_usd": cost_est,
            }
        )

        tool_calls = getattr(response.choices[0].message, "tool_calls", None) or []
        if not tool_calls:
            return {"is_relevant": False, "confidence": 0.0, "reasoning": "No tool call returned", "cost_estimate_usd": cost_est}

        fn = tool_calls[0].function
        args_str = getattr(fn, "arguments", "") or "{}"
        args = json.loads(args_str)

        return {
            "is_relevant": bool(args.get("is_relevant", False)),
            "confidence": float(args.get("confidence", 0.0)),
            "reasoning": str(args.get("reasoning", "")),
            "cost_estimate_usd": cost_est,
        }

    except Exception as e:
        return {
            "is_relevant": False,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}",
            "cost_estimate_usd": None,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-based relevance labeling for pooled ablation pack.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input pooled_ablation_pack.json")
    ap.add_argument("--out", dest="out_path", required=True, help="Output labeled pack path")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    ap.add_argument("--budget-usd", type=float, default=10.0, help="Maximum USD to spend (default: 10.0)")
    ap.add_argument("--resume", action="store_true", help="Resume labeling (skip already-labeled candidates)")
    ap.add_argument("--only-null", action="store_true", help="Only label candidates where is_relevant is null/None")
    ap.add_argument("--limit-items", type=int, default=0, help="Limit number of questions to process (0 = all)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = OPENAI_API_KEY or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Set it in .env or export it.")

    client = OpenAI(api_key=api_key)
    model = args.model

    # Load pack
    pack = json.loads(in_path.read_text(encoding="utf-8-sig"))

    # Resume: load existing output if it exists
    if args.resume and out_path.exists():
        pack = json.loads(out_path.read_text(encoding="utf-8-sig"))
        print(f"✅ Resuming from existing file: {out_path}")

    items = list(pack.get("items", []))
    if args.limit_items > 0:
        items = items[: args.limit_items]

    total_cost = 0.0
    total_candidates = 0
    labeled_count = 0
    skipped_count = 0

    for item_idx, item in enumerate(items, 1):
        question = str(item.get("question", ""))
        candidates = list(item.get("candidates", []))

        print(f"\n[{item_idx}/{len(items)}] Question: {question[:60]}...")
        print(f"  Candidates: {len(candidates)}")

        for cand_idx, candidate in enumerate(candidates, 1):
            # Skip if already labeled (resume mode)
            current_label = candidate.get("is_relevant")
            if args.resume or args.only_null:
                if current_label is not None:
                    skipped_count += 1
                    continue

            # Budget check
            if total_cost >= args.budget_usd:
                print(f"\n⚠️  Budget limit reached: ${total_cost:.2f} >= ${args.budget_usd:.2f}")
                print("   Stopping. Use --resume to continue later.")
                break

            total_candidates += 1
            print(f"  [{cand_idx}/{len(candidates)}] Judging candidate: {candidate.get('source_title', '')[:40]}...")

            result = judge_candidate_relevance(question, candidate, client, model)
            cost = result.get("cost_estimate_usd") or 0.0
            total_cost += cost

            candidate["is_relevant"] = result["is_relevant"]
            candidate["confidence"] = result["confidence"]
            if not candidate.get("note"):
                candidate["note"] = f"llm: {result['reasoning'][:100]}"
            candidate["llm_labeling"] = {
                "model": model,
                "confidence": result["confidence"],
                "reasoning": result["reasoning"],
                "cost_estimate_usd": cost,
            }

            labeled_count += 1
            print(f"    → Relevant: {result['is_relevant']} (confidence: {result['confidence']:.2f}, cost: ${cost:.4f})")

        # Save progress after each question (resume-friendly)
        out_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("✅ LLM Labeling Complete")
    print("=" * 60)
    print(f"  Total candidates processed: {total_candidates}")
    print(f"  Labeled: {labeled_count}")
    print(f"  Skipped (already labeled): {skipped_count}")
    print(f"  Total cost (estimated): ${total_cost:.4f}")
    print(f"  Output: {out_path}")
    print("\nNext: Run score_pooled_ablation_pack.py to generate metrics.")


if __name__ == "__main__":
    main()

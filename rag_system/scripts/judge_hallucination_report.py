# -*- coding: utf-8-sig -*-
"""
Re-run ONLY the judge steps for an existing hallucination_report.json.

Why:
- If generation already ran, we can avoid re-generating answers (cheaper).
- Fixes judge parsing robustness in src/evaluation.py and re-computes scores.

This script DOES call OpenAI (uses OPENAI_API_KEY from .env/.environment).

Usage:
  cd rag_system
  source venv/bin/activate
  python scripts/judge_hallucination_report.py \\
    --in evaluation_results/hallucination_report.json \\
    --out evaluation_results/hallucination_report_judged.json \\
    --limit 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=PROJECT_DIR / ".env")

from src.evaluation import RAGEvaluator  # noqa: E402


def _format_context_from_passages(passages: List[Dict[str, Any]]) -> str:
    parts = []
    for p in passages:
        title = p.get("source_title", "")
        section = p.get("source_section")
        src = f"[Source: {title}" + (f" - {section}]" if section else "]")
        parts.append(f"{src}\n{p.get('text','')}")
    return "\n\n---\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge an existing hallucination_report.json (OpenAI calls).")
    ap.add_argument("--in", dest="in_path", required=True, help="Input hallucination_report.json")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSON path (with refreshed judge fields)")
    ap.add_argument("--limit", type=int, default=20, help="How many items to judge (cost control).")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = json.loads(in_path.read_text(encoding="utf-8-sig"))
    items = list(report.get("items", []))[: max(int(args.limit), 1)]

    evaluator = RAGEvaluator(retriever=None, generator=None)
    if not evaluator.client:
        raise RuntimeError("OpenAI client not initialized. Check OPENAI_API_KEY in .env.")

    out_items: List[Dict[str, Any]] = []
    for it in items:
        q = it.get("question", "")
        baseline = (it.get("baseline_no_retrieval") or {}).get("answer", "")
        normal = (it.get("normal_rag") or {}).get("answer", "")
        agentic = (it.get("agentic_rag") or {}).get("answer", "")

        normal_ctx = _format_context_from_passages((it.get("normal_rag") or {}).get("retrieved_passages", []))
        agentic_ctx = _format_context_from_passages((it.get("agentic_rag") or {}).get("retrieved_passages", []))

        judge = {
            "normal_faithfulness": evaluator.evaluate_faithfulness(normal, normal_ctx),
            "agentic_faithfulness": evaluator.evaluate_faithfulness(agentic, agentic_ctx),
            "baseline_vs_normal": evaluator.compare_answers(q, normal, baseline),
        }

        it2 = dict(it)
        it2["judge"] = judge
        out_items.append(it2)

    out = dict(report)
    out["items"] = out_items
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"✅ Wrote judged report: {out_path}")


if __name__ == "__main__":
    main()


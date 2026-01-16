# -*- coding: utf-8-sig -*-
"""
Score a pooled ablation pack and generate:
- Retrieval metrics per system (Recall@k required)
- A markdown ablation table for the report

This script does NOT call OpenAI.

Usage:
  cd rag_system
  source venv/bin/activate
  python scripts/score_pooled_ablation_pack.py \
    --in evaluation_results/pooled_ablation_pack.json \
    --out evaluation_results/pooled_ablation_metrics.json \
    --md evaluation_results/ablation_table.md
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import TOP_K  # noqa: E402
from src.evaluation import RAGEvaluator  # noqa: E402


def _relevant_ids(item: Dict[str, Any]) -> List[str]:
    rel = []
    for c in item.get("candidates", []):
        if c.get("is_relevant") is True:
            rel.append(c["chunk_id"])
    return rel


def _ranked_ids(item: Dict[str, Any], system_id: str) -> List[str]:
    return list(item.get("systems_ranked_ids", {}).get(system_id, []))


def score(pack: Dict[str, Any], k: int) -> Dict[str, Any]:
    evaluator = RAGEvaluator()

    system_ids = [s["id"] for s in pack.get("systems", [])]
    per_question: List[Dict[str, Any]] = []

    for item in pack.get("items", []):
        q = item["question"]
        relevant = _relevant_ids(item)

        row = {"question": q, "category": item.get("category", "general"), "n_relevant": len(relevant), "systems": {}}

        for sid in system_ids:
            ranked = _ranked_ids(item, sid)
            metrics = {
                "recall_at_k": evaluator.recall_at_k(ranked, relevant, k=k) if relevant else None,
                "precision_at_k": evaluator.precision_at_k(ranked, relevant, k=k) if relevant else None,
                "f1_at_k": evaluator.f1_at_k(ranked, relevant, k=k) if relevant else None,
                "mrr": evaluator.mrr(ranked, relevant) if relevant else None,
                "ndcg_at_k": evaluator.ndcg_at_k(ranked, relevant, k=k) if relevant else None,
                "average_precision": evaluator.average_precision(ranked, relevant) if relevant else None,
            }
            row["systems"][sid] = {"top_k": ranked[:k], "metrics": metrics}

        per_question.append(row)

    def _agg(metric_name: str, sid: str) -> Dict[str, Any]:
        vals = []
        for r in per_question:
            v = r["systems"][sid]["metrics"].get(metric_name)
            if v is not None:
                vals.append(float(v))
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    summary = {}
    for sid in system_ids:
        summary[sid] = {
            "recall_at_k": _agg("recall_at_k", sid),
            "precision_at_k": _agg("precision_at_k", sid),
            "f1_at_k": _agg("f1_at_k", sid),
            "mrr": _agg("mrr", sid),
            "ndcg_at_k": _agg("ndcg_at_k", sid),
            "MAP": _agg("average_precision", sid),
        }

    return {"k": k, "summary": summary, "per_question": per_question}


def render_markdown_table(scored: Dict[str, Any], pack: Dict[str, Any]) -> str:
    k = scored["k"]
    systems = pack.get("systems", [])

    # Table header
    lines = []
    lines.append(f"## Ablation: Retrieval Quality (pooled labels, Recall@{k})")
    lines.append("")
    lines.append("| System | Recall@k (mean±std) | MRR (mean±std) | nDCG@k (mean±std) | MAP (mean±std) | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for s in systems:
        sid = s["id"]
        name = s.get("description", sid)
        summ = scored["summary"].get(sid, {})

        def fmt(m: str) -> str:
            d = summ.get(m, {})
            if d.get("mean") is None:
                return "n/a"
            return f"{d['mean']:.3f}±{d['std']:.3f}"

        n = summ.get("recall_at_k", {}).get("n", 0)
        lines.append(
            f"| `{sid}` | {fmt('recall_at_k')} | {fmt('mrr')} | {fmt('ndcg_at_k')} | {fmt('MAP')} | {n} |"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("- Metrics are aggregated only over questions where at least one candidate was labeled relevant.")
    lines.append("- This table compares **systems on the same relevance labels** (pooled candidates).")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score pooled ablation pack (baseline vs agentic).")
    parser.add_argument("--in", dest="in_path", type=str, required=True, help="Input pooled_ablation_pack.json")
    parser.add_argument("--out", dest="out_path", type=str, required=True, help="Output metrics JSON")
    parser.add_argument("--md", dest="md_path", type=str, required=True, help="Output markdown table")
    parser.add_argument("-k", type=int, default=TOP_K, help=f"Recall@k cutoff (default: {TOP_K})")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    md_path = Path(args.md_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    pack = json.loads(in_path.read_text(encoding="utf-8-sig"))
    scored = score(pack, k=args.k)

    out_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    md_path.write_text(render_markdown_table(scored, pack), encoding="utf-8-sig")

    print(f"✅ Wrote metrics JSON: {out_path}")
    print(f"✅ Wrote markdown table: {md_path}")


if __name__ == "__main__":
    main()


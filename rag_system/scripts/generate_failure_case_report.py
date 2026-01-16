# -*- coding: utf-8-sig -*-
"""
Generate a human-readable failure-case report from pooled ablation results.

Outputs:
- markdown with at least N worst retrieval cases per system
- includes top-k passages + your manual relevance labels for easy copy into report

No OpenAI calls.

Usage:
  cd rag_system
  source venv/bin/activate
  python scripts/generate_failure_case_report.py \\
    --pack evaluation_results/pooled_ablation_pack.json \\
    --scored evaluation_results/pooled_ablation_metrics.json \\
    --out evaluation_results/failure_cases.md \\
    --k 5 \\
    --top 3
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _relevant_ids(item: Dict[str, Any]) -> List[str]:
    rel = []
    for c in item.get("candidates", []):
        if c.get("is_relevant") is True:
            rel.append(c["chunk_id"])
    return rel


def _candidate_lookup(item: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {c["chunk_id"]: c for c in item.get("candidates", [])}


def main() -> None:
    p = argparse.ArgumentParser(description="Generate failure-case report (markdown).")
    p.add_argument("--pack", required=True, help="pooled_ablation_pack.json")
    p.add_argument("--scored", required=True, help="pooled_ablation_metrics.json")
    p.add_argument("--out", required=True, help="output markdown path")
    p.add_argument("--k", type=int, default=5, help="top-k cutoff")
    p.add_argument("--top", type=int, default=3, help="how many worst cases per system")
    args = p.parse_args()

    pack = json.loads(Path(args.pack).read_text(encoding="utf-8-sig"))
    scored = json.loads(Path(args.scored).read_text(encoding="utf-8-sig"))

    systems = [s["id"] for s in pack.get("systems", [])]

    # Build a quick index by question string
    pack_items = {it["question"]: it for it in pack.get("items", [])}

    lines: List[str] = []
    lines.append("# Failure cases (Retrieval)")
    lines.append("")
    lines.append("This file is auto-generated from pooled manual relevance labels.")
    lines.append("")

    for sid in systems:
        # Collect (question, recall) pairs where recall is defined
        cases = []
        for qrow in scored.get("per_question", []):
            q = qrow["question"]
            m = qrow["systems"][sid]["metrics"].get("recall_at_k")
            if m is None:
                continue
            cases.append((q, float(m)))

        cases.sort(key=lambda x: x[1])  # worst first
        worst = cases[: max(int(args.top), 1)]

        lines.append(f"## System: `{sid}` — worst {len(worst)} cases by Recall@{args.k}")
        lines.append("")

        for q, recall in worst:
            item = pack_items.get(q, {})
            relevant = set(_relevant_ids(item))
            cand = _candidate_lookup(item)
            ranked = item.get("systems_ranked_ids", {}).get(sid, [])[: args.k]

            lines.append(f"### Q: {q}")
            lines.append(f"- **Recall@{args.k}**: {recall:.3f}")
            lines.append(f"- **#relevant labeled in pool**: {len(relevant)}")
            lines.append("")
            lines.append("#### Top-k retrieved passages (with your relevance labels)")
            lines.append("")

            for rank, cid in enumerate(ranked, 1):
                c = cand.get(cid)
                if not c:
                    continue
                is_rel = c.get("is_relevant")
                label = "relevant" if is_rel is True else ("not relevant" if is_rel is False else "UNLABELED")
                preview = c.get("text", "")[:500].replace("\n", " ")
                lines.append(f"- **[{rank}]** `{cid}` — **{label}** — {c.get('source_title','')} / {c.get('source_section')}")
                lines.append(f"  - preview: {preview}...")
            lines.append("")

            if relevant:
                missing = [rid for rid in relevant if rid not in set(ranked)]
                if missing:
                    lines.append("#### Relevant chunks missed in top-k")
                    for mid in missing[:10]:
                        c = cand.get(mid, {})
                        lines.append(f"- `{mid}` — {c.get('source_title','')} / {c.get('source_section')}")
                    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"✅ Wrote failure case report: {out_path}")


if __name__ == "__main__":
    main()


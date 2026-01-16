# -*- coding: utf-8-sig -*-
"""
Summarize a judged hallucination report into a markdown file:
- pick worst faithfulness cases (normal + agentic)
- pick baseline hallucination candidates from judge hallucinations_in_B

No OpenAI calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_score(obj: Dict[str, Any]) -> float:
    try:
        s = obj.get("score")
        if s is None:
            return 1e9
        return float(s)
    except Exception:
        return 1e9


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize judged hallucination report (markdown). No OpenAI calls.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input hallucination_report_judged.json")
    ap.add_argument("--out", dest="out_path", required=True, help="Output markdown path")
    ap.add_argument("--top", type=int, default=3, help="How many cases per section")
    args = ap.parse_args()

    report = json.loads(Path(args.in_path).read_text(encoding="utf-8-sig"))
    items = list(report.get("items", []))

    normal_cases: List[Tuple[float, Dict[str, Any]]] = []
    agentic_cases: List[Tuple[float, Dict[str, Any]]] = []
    baseline_halluc: List[Tuple[int, Dict[str, Any], List[str]]] = []

    for it in items:
        judge = it.get("judge") or {}
        nf = judge.get("normal_faithfulness") or {}
        af = judge.get("agentic_faithfulness") or {}
        cmp = judge.get("baseline_vs_normal") or {}

        normal_cases.append((_safe_score(nf), it))
        agentic_cases.append((_safe_score(af), it))

        hb = cmp.get("hallucinations_in_B") or []
        if isinstance(hb, list) and hb:
            hb_str = [str(x) for x in hb if str(x).strip()]
            if hb_str:
                baseline_halluc.append((len(hb_str), it, hb_str))

    normal_cases.sort(key=lambda x: x[0])
    agentic_cases.sort(key=lambda x: x[0])
    baseline_halluc.sort(key=lambda x: x[0], reverse=True)

    top = max(int(args.top), 1)

    lines: List[str] = []
    lines.append("## Hallucination / Answer Quality Summary (LLM-as-judge)")
    lines.append("")
    lines.append(f"- Input: `{args.in_path}`")
    lines.append("")

    def add_faithfulness_block(title: str, key: str, cases: List[Tuple[float, Dict[str, Any]]]) -> None:
        lines.append(f"### Worst {top}: {title}")
        lines.append("")
        for score, it in cases[:top]:
            q = it.get("question", "")
            obj = ((it.get("judge") or {}).get(key) or {})
            exp = obj.get("explanation", "")
            score_txt = str(int(score)) if score < 1e8 else "n/a"
            lines.append(f"- **score={score_txt}** — {q}")
            if exp:
                lines.append(f"  - explanation: {exp}")
        lines.append("")

    add_faithfulness_block("Normal RAG faithfulness", "normal_faithfulness", normal_cases)
    add_faithfulness_block("Agentic RAG faithfulness", "agentic_faithfulness", agentic_cases)

    lines.append(f"### Baseline hallucinations (from judge baseline-vs-normal) — top {top}")
    lines.append("")
    for n, it, hb in baseline_halluc[:top]:
        q = it.get("question", "")
        lines.append(f"- **{n} hallucinated claims** — {q}")
        for h in hb[:10]:
            lines.append(f"  - {h}")
    lines.append("")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"✅ Wrote hallucination summary: {out_path}")


if __name__ == "__main__":
    main()


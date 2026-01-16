# -*- coding: utf-8-sig -*-
"""
Summarize OpenAI cost logs and generate a timestamped markdown report.

Reads:
  evaluation_results/openai_cost_log.jsonl

Outputs:
  evaluation_results/cost_summary_<YYYYMMDD_HHMMSS>.md

No OpenAI calls.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import EVALUATION_DIR  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize OpenAI cost log and write a markdown report.")
    p.add_argument(
        "--log",
        type=str,
        default=str(EVALUATION_DIR / "openai_cost_log.jsonl"),
        help="Path to openai_cost_log.jsonl",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output markdown path (default: evaluation_results/cost_summary_<ts>.md)",
    )
    p.add_argument("--tail", type=int, default=50, help="How many last events to include.")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Cost log not found: {log_path}")

    # Read JSONL
    events: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            # tolerate partial lines
            continue

    total_cost = 0.0
    cost_known = 0.0
    total_prompt = 0
    total_completion = 0

    by_model: Dict[str, Dict[str, float]] = {}
    by_op: Dict[str, Dict[str, float]] = {}

    for e in events:
        model = str(e.get("model", "unknown"))
        op = str(e.get("operation", "unknown"))
        pt = e.get("prompt_tokens") or 0
        ct = e.get("completion_tokens") or 0
        total_prompt += int(pt)
        total_completion += int(ct)

        ce = e.get("cost_estimate_usd")
        if isinstance(ce, (int, float)):
            total_cost += float(ce)
            cost_known += float(ce)

        by_model.setdefault(model, {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "n": 0})
        by_model[model]["prompt_tokens"] += int(pt)
        by_model[model]["completion_tokens"] += int(ct)
        by_model[model]["n"] += 1
        if isinstance(ce, (int, float)):
            by_model[model]["cost"] += float(ce)

        by_op.setdefault(op, {"cost": 0.0, "n": 0})
        by_op[op]["n"] += 1
        if isinstance(ce, (int, float)):
            by_op[op]["cost"] += float(ce)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else (EVALUATION_DIR / f"cost_summary_{ts}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("## OpenAI Cost Summary (Estimated)")
    lines.append("")
    lines.append(f"- **Generated at**: {datetime.now().isoformat()}")
    lines.append(f"- **Log file**: `{log_path}`")
    lines.append("")
    lines.append("### Totals")
    lines.append("")
    lines.append(f"- **Estimated total cost (USD)**: {total_cost:.6f}")
    lines.append(f"- **Total prompt tokens**: {total_prompt}")
    lines.append(f"- **Total completion tokens**: {total_completion}")
    lines.append(f"- **Total events**: {len(events)}")
    lines.append("")
    lines.append("### By model")
    lines.append("")
    lines.append("| Model | Events | Prompt tokens | Completion tokens | Est. cost (USD) |")
    lines.append("|---|---:|---:|---:|---:|")
    for model, d in sorted(by_model.items(), key=lambda x: x[1]["cost"], reverse=True):
        lines.append(f"| `{model}` | {int(d['n'])} | {int(d['prompt_tokens'])} | {int(d['completion_tokens'])} | {float(d['cost']):.6f} |")
    lines.append("")
    lines.append("### By operation")
    lines.append("")
    lines.append("| Operation | Events | Est. cost (USD) |")
    lines.append("|---|---:|---:|")
    for op, d in sorted(by_op.items(), key=lambda x: x[1]["cost"], reverse=True):
        lines.append(f"| `{op}` | {int(d['n'])} | {float(d['cost']):.6f} |")
    lines.append("")
    lines.append(f"### Last {min(int(args.tail), len(events))} events")
    lines.append("")
    tail = events[-min(int(args.tail), len(events)) :]
    lines.append("| ts_utc | operation | model | prompt | completion | cost_usd |")
    lines.append("|---|---|---|---:|---:|---:|")
    for e in tail:
        lines.append(
            f"| {e.get('ts_utc','')} | `{e.get('operation','')}` | `{e.get('model','')}` | "
            f"{e.get('prompt_tokens','')} | {e.get('completion_tokens','')} | {e.get('cost_estimate_usd','')} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"✅ Wrote cost summary: {out_path}")


if __name__ == "__main__":
    main()


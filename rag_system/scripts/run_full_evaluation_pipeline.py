# -*- coding: utf-8-sig -*-
"""
End-to-end evaluation pipeline: runs all evaluation steps automatically.

This script automates the entire evaluation workflow:
1. Create pooled ablation pack
2. (Optional) LLM-label relevance
3. Score and generate ablation table
4. Generate failure case report
5. Generate hallucination report (optional, requires OpenAI)
6. Summarize costs
7. Assemble final report markdown

Usage:
  cd rag_system
  source venv/bin/activate
  python scripts/run_full_evaluation_pipeline.py \
    --top-n 20 \
    --use-llm-labeling \
    --llm-model gpt-4o-mini \
    --budget-usd 10.0 \
    --generate-hallucination-report \
    --out-dir evaluation_results/full_pipeline_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv

load_dotenv(dotenv_path=PROJECT_DIR / ".env")


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=PROJECT_DIR)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def assemble_final_report(
    out_dir: Path,
    ablation_table_path: Path,
    failure_cases_path: Path,
    hallucination_summary_path: Optional[Path],
    cost_summary_path: Optional[Path],
) -> Path:
    """Assemble all reports into a single markdown file."""
    report_path = out_dir / "FULL_EVALUATION_REPORT.md"

    lines: List[str] = []
    lines.append("# Full Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    # Ablation table
    if ablation_table_path.exists():
        lines.append("## 1. Ablation Study: Baseline vs Agentic RAG")
        lines.append("")
        lines.append(ablation_table_path.read_text(encoding="utf-8-sig"))
        lines.append("")

    # Failure cases
    if failure_cases_path.exists():
        lines.append("## 2. Retrieval Failure Cases")
        lines.append("")
        lines.append(failure_cases_path.read_text(encoding="utf-8-sig"))
        lines.append("")

    # Hallucination summary
    if hallucination_summary_path and hallucination_summary_path.exists():
        lines.append("## 3. Hallucination Analysis")
        lines.append("")
        lines.append(hallucination_summary_path.read_text(encoding="utf-8-sig"))
        lines.append("")

    # Cost summary
    if cost_summary_path and cost_summary_path.exists():
        lines.append("## 4. Cost Summary")
        lines.append("")
        lines.append(cost_summary_path.read_text(encoding="utf-8-sig"))
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full end-to-end evaluation pipeline.")
    ap.add_argument("--top-n", type=int, default=20, help="Top-N candidates per system (default: 20)")
    ap.add_argument("--use-llm-labeling", action="store_true", help="Use LLM-based relevance labeling (requires OpenAI)")
    ap.add_argument("--use-heuristic-labeling", action="store_true", help="Use heuristic auto-labeling (no API calls)")
    ap.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="Model for LLM labeling (default: gpt-4o-mini)")
    ap.add_argument("--budget-usd", type=float, default=10.0, help="Budget for LLM labeling (default: 10.0)")
    ap.add_argument("--generate-hallucination-report", action="store_true", help="Generate hallucination report (requires OpenAI)")
    ap.add_argument("--hallucination-limit", type=int, default=12, help="Limit questions for hallucination report (default: 12)")
    ap.add_argument("--out-dir", type=str, default="", help="Output directory (default: evaluation_results/full_pipeline_TIMESTAMP)")
    args = ap.parse_args()

    # Setup output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_DIR / "evaluation_results" / f"full_pipeline_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("🚀 Full Evaluation Pipeline")
    print("=" * 60)
    print(f"Output directory: {out_dir}")

    # Step 1: Create pooled ablation pack
    pack_path = out_dir / "pooled_ablation_pack.json"
    if not run_command(
        [
            sys.executable,
            "scripts/create_pooled_ablation_pack.py",
            "--top-n",
            str(args.top_n),
            "--out",
            str(pack_path),
        ],
        "Step 1: Creating pooled ablation pack",
    ):
        print("❌ Failed to create ablation pack. Exiting.")
        return

    # Step 2: Label relevance
    labeled_pack_path = out_dir / "pooled_ablation_pack_labeled.json"
    if args.use_llm_labeling:
        if not run_command(
            [
                sys.executable,
                "scripts/llm_label_pooled_ablation_pack.py",
                "--in",
                str(pack_path),
                "--out",
                str(labeled_pack_path),
                "--model",
                args.llm_model,
                "--budget-usd",
                str(args.budget_usd),
            ],
            "Step 2: LLM-based relevance labeling",
        ):
            print("⚠️  LLM labeling failed. Falling back to unlabeled pack.")
            labeled_pack_path = pack_path
    elif args.use_heuristic_labeling:
        if not run_command(
            [
                sys.executable,
                "scripts/auto_label_pooled_ablation_pack.py",
                "--in",
                str(pack_path),
                "--out",
                str(labeled_pack_path),
                "--summary",
                str(out_dir / "auto_label_summary.md"),
            ],
            "Step 2: Heuristic auto-labeling",
        ):
            print("⚠️  Heuristic labeling failed. Falling back to unlabeled pack.")
            labeled_pack_path = pack_path
    else:
        print("\n⚠️  No labeling method selected. Using unlabeled pack.")
        print("   (You can manually label or run with --use-llm-labeling or --use-heuristic-labeling)")
        labeled_pack_path = pack_path

    # Step 3: Score and generate ablation table
    metrics_path = out_dir / "pooled_ablation_metrics.json"
    ablation_table_path = out_dir / "ablation_table.md"
    if not run_command(
        [
            sys.executable,
            "scripts/score_pooled_ablation_pack.py",
            "--in",
            str(labeled_pack_path),
            "--out",
            str(metrics_path),
            "--md",
            str(ablation_table_path),
            "-k",
            "5",
        ],
        "Step 3: Scoring and generating ablation table",
    ):
        print("❌ Failed to score ablation pack. Exiting.")
        return

    # Step 4: Generate failure case report
    failure_cases_path = out_dir / "failure_cases.md"
    if not run_command(
        [
            sys.executable,
            "scripts/generate_failure_case_report.py",
            "--pack",
            str(labeled_pack_path),
            "--scored",
            str(metrics_path),
            "--out",
            str(failure_cases_path),
            "--k",
            "5",
            "--top",
            "3",
        ],
        "Step 4: Generating failure case report",
    ):
        print("⚠️  Failed to generate failure case report.")

    # Step 5: Generate hallucination report (optional)
    hallucination_report_path = None
    hallucination_summary_path = None
    if args.generate_hallucination_report:
        hallucination_report_path = out_dir / "hallucination_report.json"
        if run_command(
            [
                sys.executable,
                "scripts/generate_hallucination_report.py",
                "--pack",
                str(labeled_pack_path),
                "--out",
                str(hallucination_report_path),
                "--k",
                "5",
                "--limit",
                str(args.hallucination_limit),
                "--judge",
            ],
            "Step 5: Generating hallucination report",
        ):
            # Judge hallucination report
            judged_path = out_dir / "hallucination_report_judged.json"
            if run_command(
                [
                    sys.executable,
                    "scripts/judge_hallucination_report.py",
                    "--in",
                    str(hallucination_report_path),
                    "--out",
                    str(judged_path),
                    "--limit",
                    str(args.hallucination_limit),
                ],
                "Step 5b: Judging hallucination report",
            ):
                # Summarize
                hallucination_summary_path = out_dir / "hallucination_summary.md"
                run_command(
                    [
                        sys.executable,
                        "scripts/summarize_hallucination_report.py",
                        "--in",
                        str(judged_path),
                        "--out",
                        str(hallucination_summary_path),
                    ],
                    "Step 5c: Summarizing hallucination report",
                )

    # Step 6: Summarize costs
    cost_summary_path = None
    cost_log_path = PROJECT_DIR / "evaluation_results" / "openai_cost_log.jsonl"
    if cost_log_path.exists():
        cost_summary_path = out_dir / "cost_summary.md"
        run_command(
            [
                sys.executable,
                "scripts/summarize_openai_costs.py",
                "--log",
                str(cost_log_path),
                "--out",
                str(cost_summary_path),
            ],
            "Step 6: Summarizing costs",
        )

    # Step 7: Assemble final report
    final_report_path = assemble_final_report(
        out_dir=out_dir,
        ablation_table_path=ablation_table_path,
        failure_cases_path=failure_cases_path,
        hallucination_summary_path=hallucination_summary_path,
        cost_summary_path=cost_summary_path,
    )

    print("\n" + "=" * 60)
    print("✅ Full Evaluation Pipeline Complete")
    print("=" * 60)
    print(f"📁 Output directory: {out_dir}")
    print(f"📄 Final report: {final_report_path}")
    print("\nGenerated files:")
    print(f"  - Ablation table: {ablation_table_path}")
    print(f"  - Failure cases: {failure_cases_path}")
    if hallucination_summary_path:
        print(f"  - Hallucination summary: {hallucination_summary_path}")
    if cost_summary_path:
        print(f"  - Cost summary: {cost_summary_path}")


if __name__ == "__main__":
    main()

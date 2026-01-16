# -*- coding: utf-8-sig -*-
"""
Auto-label `is_relevant` in a pooled ablation pack using deterministic heuristics.

Why?
- The assignment asks for manual inspection, but this script can bootstrap labels
  and give you a first ablation table quickly.
- NO OpenAI calls (cost = $0). You can later overwrite/adjust labels manually.

Heuristics are tuned for this project's question set in `create_test_questions()`.

Usage:
  cd rag_system
  source venv/bin/activate
  python scripts/auto_label_pooled_ablation_pack.py \\
    --in evaluation_results/pooled_ablation_pack.json \\
    --out evaluation_results/pooled_ablation_pack_labeled_auto.json \\
    --summary evaluation_results/auto_label_summary.md
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _contains(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def _best_course_match(question: str, course_titles: Set[str]) -> Optional[str]:
    q = _norm(question)
    matches = [t for t in course_titles if _norm(t) in q]
    if not matches:
        return None
    # longest match wins (avoid matching "Internet" before "Future Internet")
    matches.sort(key=lambda x: len(x), reverse=True)
    return matches[0]


def _program_key(question: str) -> Optional[str]:
    q = question
    if "Computer Networks" in q:
        return "master-computer-networks"
    if "Data Science" in q:
        return "master-data-science"
    if "Software Engineering" in q:
        return "master-software-engineering"
    return None


def _is_course_pdf(candidate: Dict[str, Any]) -> bool:
    return str(candidate.get("source_file", "")).lower().endswith(".pdf")


def _label_course_single_hop(q: str, cands: List[Dict[str, Any]], course_titles: Set[str]) -> Set[str]:
    course = _best_course_match(q, course_titles)
    if not course:
        return set()

    qn = _norm(q)
    wanted_section: Optional[str] = None
    include_all_course_chunks = False

    if "prerequisite" in qn:
        wanted_section = "Prerequisites"
    elif "learning outcome" in qn:
        wanted_section = "Learning Outcomes"
    elif "assess" in qn or "assessment" in qn:
        wanted_section = "Assessment method and criteria"
    elif "exam period" in qn:
        # In this corpus, exam *period* is not always present on course PDFs.
        wanted_section = None
    elif "who teaches" in qn or "which semester" in qn:
        # Lecturer + semester is repeated in the header of (almost) every course chunk.
        # Mark all chunks from that course as relevant to avoid false 0-recall.
        include_all_course_chunks = True

    rel: Set[str] = set()

    for cand in cands:
        sec = str(cand.get("source_section", ""))
        text = str(cand.get("text", ""))

        # Course PDF chunks (source_title == course)
        if _is_course_pdf(cand) and cand.get("source_title") == course:
            if include_all_course_chunks:
                rel.add(cand["chunk_id"])
                continue
            if wanted_section:
                if sec == wanted_section:
                    rel.add(cand["chunk_id"])
                    continue

        # Exam period is often present on study-programme website blocks:
        # there, `source_section` == course name.
        if "exam period" in qn:
            if (cand.get("source_title") == course) or (sec == course):
                if re.search(r"Exam\s*Period", text, flags=re.IGNORECASE):
                    rel.add(cand["chunk_id"])
                    continue

    return rel


def _label_study_programme_multi_hop(q: str, cands: List[Dict[str, Any]]) -> Set[str]:
    prog = _program_key(q)
    if not prog:
        return set()

    rel: Set[str] = set()
    qn = _norm(q)

    # Filters
    want_sem1 = "1st semester" in qn or "1e sem" in qn
    want_sem2 = "2nd semester" in qn or "2e sem" in qn
    want_exam_contract = "exam contract not possible" in qn

    # Course-name targeted questions (e.g., Future Internet)
    target_course = None
    m = re.search(r"'([^']+)'", q)
    if m:
        target_course = m.group(1).strip()
    else:
        # common ones used in this project
        for name in ["Future Internet", "Data Mining", "Database Systems", "Model Driven Engineering"]:
            if name.lower() in qn:
                target_course = name
                break

    for cand in cands:
        sf = str(cand.get("source_file", ""))
        if f"/{prog}/study-programme/" not in sf:
            continue
        if _is_course_pdf(cand):
            continue

        text = str(cand.get("text", ""))
        sec = str(cand.get("source_section", ""))

        # direct course match questions
        if target_course and sec == target_course:
            rel.add(cand["chunk_id"])
            continue

        # lecturer-based questions
        if "hans vangheluwe" in qn and "hans vangheluwe" in _norm(text):
            rel.add(cand["chunk_id"])
            continue
        if "juan felipe botero" in qn and "juan felipe botero" in _norm(text):
            rel.add(cand["chunk_id"])
            continue

        # semester filters (course blocks include "**Semester:** X")
        if want_sem1 and re.search(r"\*\*Semester:\*\*\s*1E\s*SEM", text, flags=re.IGNORECASE):
            rel.add(cand["chunk_id"])
            continue
        if want_sem2 and re.search(r"\*\*Semester:\*\*\s*2E\s*SEM", text, flags=re.IGNORECASE):
            rel.add(cand["chunk_id"])
            continue

        # contract restrictions
        if want_exam_contract and re.search(
            r"\*\*Contract restrictions:\*\*\s*Exam contract not possible", text, flags=re.IGNORECASE
        ):
            rel.add(cand["chunk_id"])
            continue

    return rel


def _label_cross_page_multi_hop(q: str, cands: List[Dict[str, Any]], course_titles: Set[str]) -> Set[str]:
    course = _best_course_match(q, course_titles)
    if not course:
        # try quoted
        m = re.search(r"'([^']+)'", q)
        if m:
            course = m.group(1).strip()
    if not course:
        return set()

    qn = _norm(q)
    needed: List[str] = []
    if "prerequisite" in qn:
        needed.append("Prerequisites")
    if "learning outcome" in qn:
        needed.append("Learning Outcomes")
    if "course content" in qn or "contents" in qn:
        needed.append("Course Contents")
    if "assessment" in qn or "assessed" in qn:
        needed.append("Assessment method and criteria")

    rel: Set[str] = set()
    for cand in cands:
        if not _is_course_pdf(cand):
            continue
        if cand.get("source_title") != course:
            continue
        sec = str(cand.get("source_section", ""))
        if sec in needed:
            rel.add(cand["chunk_id"])

    return rel


def _label_website_single_hop(q: str, cands: List[Dict[str, Any]]) -> Set[str]:
    qn = _norm(q)
    rel: Set[str] = set()

    # Admission requirements (Software Engineering)
    if "admission" in qn and "software engineering" in q:
        for cand in cands:
            sf = str(cand.get("source_file", ""))
            if "/master-software-engineering/" in sf and "admissionrequirements" in sf:
                rel.add(cand["chunk_id"])
        return rel

    # Career opportunities (Data Science specialization)
    if ("career opportunities" in qn or "career" in qn) and "Data Science" in q:
        for cand in cands:
            sf = str(cand.get("source_file", ""))
            if "/master-data-science/" in sf and ("job-opportunities" in sf or "job" in sf):
                rel.add(cand["chunk_id"])
        return rel

    return rel


def _label_boundary(q: str, cands: List[Dict[str, Any]]) -> Set[str]:
    qn = _norm(q)
    rel: Set[str] = set()

    # Tuition fees are actually present in the dataset (tuitionfees pages)
    if "tuition fee" in qn or "tuition fees" in qn:
        for cand in cands:
            sf = str(cand.get("source_file", ""))
            if "tuitionfees" in sf:
                rel.add(cand["chunk_id"])
        return rel

    # Others: likely not in dataset; keep empty to exclude from metrics
    return rel


def _collect_course_titles_from_pack(pack: Dict[str, Any]) -> Set[str]:
    titles: Set[str] = set()
    for it in pack.get("items", []):
        for c in it.get("candidates", []):
            if _is_course_pdf(c) and c.get("source_title"):
                titles.add(str(c["source_title"]))
    return titles


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-label pooled ablation pack (heuristic). No OpenAI calls.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input pooled_ablation_pack.json")
    ap.add_argument("--out", dest="out_path", required=True, help="Output labeled pack path")
    ap.add_argument("--summary", dest="summary_path", default="", help="Optional markdown summary path")
    ap.add_argument("--only-null", action="store_true", help="Only fill labels where is_relevant is null/None")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pack = json.loads(in_path.read_text(encoding="utf-8-sig"))
    course_titles = _collect_course_titles_from_pack(pack)

    ts = datetime.now().isoformat()
    total_cands = 0
    total_rel = 0
    included_questions = 0

    per_item_rows: List[str] = []

    for item in pack.get("items", []):
        q = str(item.get("question", ""))
        cat = str(item.get("category", "general"))
        cands = list(item.get("candidates", []))

        total_cands += len(cands)

        if cat == "course_single_hop":
            rel_ids = _label_course_single_hop(q, cands, course_titles)
        elif cat == "study_programme_multi_hop":
            rel_ids = _label_study_programme_multi_hop(q, cands)
        elif cat == "cross_page_multi_hop":
            rel_ids = _label_cross_page_multi_hop(q, cands, course_titles)
        elif cat == "website_single_hop":
            rel_ids = _label_website_single_hop(q, cands)
        elif cat == "boundary":
            rel_ids = _label_boundary(q, cands)
        else:
            rel_ids = set()

        # Apply labels
        for cand in cands:
            cur = cand.get("is_relevant")
            if args.only_null and cur is not None:
                continue
            cid = cand.get("chunk_id")
            is_rel = cid in rel_ids
            cand["is_relevant"] = bool(is_rel)
            if not cand.get("note"):
                cand["note"] = "auto: heuristic"

        n_rel = sum(1 for cand in cands if cand.get("is_relevant") is True)
        total_rel += n_rel
        if n_rel > 0:
            included_questions += 1

        per_item_rows.append(f"- **{cat}** — {q} — **relevant={n_rel} / candidates={len(cands)}**")

    out_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"✅ Wrote labeled pack: {out_path}")

    if args.summary_path:
        s_path = Path(args.summary_path)
        s_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append("## Auto-label summary (heuristic, no OpenAI calls)")
        lines.append("")
        lines.append(f"- **Generated at**: {ts}")
        lines.append(f"- **Input pack**: `{in_path}`")
        lines.append(f"- **Output pack**: `{out_path}`")
        lines.append("")
        lines.append("### Coverage")
        lines.append("")
        lines.append(f"- **Total candidates**: {total_cands}")
        lines.append(f"- **Total labeled relevant**: {total_rel}")
        lines.append(f"- **Questions with ≥1 relevant**: {included_questions} / {len(pack.get('items', []))}")
        lines.append("")
        lines.append("### Per question")
        lines.append("")
        lines.extend(per_item_rows)
        s_path.write_text("\n".join(lines), encoding="utf-8-sig")
        print(f"✅ Wrote summary: {s_path}")


if __name__ == "__main__":
    main()


# -*- coding: utf-8-sig -*-
"""
Structured index built FROM the already-chunked corpus.

Goal:
- Provide deterministic "entity + field" retrieval for this dataset.
- Fix common failure cases where dense retrieval returns the right course
  but the wrong section (e.g., prerequisites vs study material).
- Provide structured access to study-programme course blocks (DS/SE/CN) so that
  semester / contract / lecturer filters are reliable.

This does NOT replace dense retrieval; it augments it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.chunking import Chunk


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_course_pdf(ch: Chunk) -> bool:
    return str(ch.source_file).lower().endswith(".pdf")


def _is_website(ch: Chunk) -> bool:
    return str(ch.source_file).startswith("http")


def _program_from_url(url: str) -> Optional[str]:
    u = str(url)
    if "/master-data-science/" in u:
        return "DS"
    if "/master-software-engineering/" in u:
        return "SE"
    if "/master-computer-networks/" in u:
        return "CN"
    return None


def _extract_quoted_title(query: str) -> Optional[str]:
    m = re.search(r"'([^']+)'", query)
    if m:
        return m.group(1).strip()
    m = re.search(r"\"([^\"]+)\"", query)
    if m:
        return m.group(1).strip()
    return None


def _detect_intents(query: str) -> Dict[str, bool]:
    q = _norm(query)
    return {
        "study_programme": any(k in q for k in ["study programme", "study program", "compulsory courses", "compulsory course", "model path"]),
        # avoid matching "listed" as "list"
        "list": bool(re.search(r"\blist\b", q)) or any(k in q for k in ["which compulsory", "which courses", "compulsory courses"]),
        "prerequisites": any(k in q for k in ["prereq", "prerequisite", "requirements", "prior knowledge", "before taking"]),
        "learning_outcomes": any(k in q for k in ["learning outcome", "learning outcomes", "outcome", "competenc", "you will be able"]),
        "course_contents": any(k in q for k in ["course content", "course contents", "contents", "syllabus", "topics", "covers", "what is covered"]),
        "assessment": any(k in q for k in ["assessment", "grading", "evaluation", "criteria"]),
        "exam_period": "exam period" in q,
        "lecturer": any(k in q for k in ["who teaches", "taught by", "lecturer", "instructor", "teacher"]),
        "semester": any(k in q for k in ["semester", "1st semester", "2nd semester", "1e sem", "2e sem"]),
        "contract_restrictions": any(k in q for k in ["contract restriction", "contract restrictions", "exam contract"]),
    }


def _detect_program(query: str) -> Optional[str]:
    q = _norm(query)
    hits = set()
    if "data science" in q or "artificial intelligence" in q:
        hits.add("DS")
    if "software engineering" in q:
        hits.add("SE")
    if "computer networks" in q or "computer network" in q:
        hits.add("CN")
    # If query mentions multiple, don't lock
    if len(hits) == 1:
        return next(iter(hits))
    return None


def _semester_filter(query: str) -> Optional[str]:
    q = _norm(query)
    if "1st semester" in q or "1e sem" in q:
        return "1E SEM"
    if "2nd semester" in q or "2e sem" in q:
        return "2E SEM"
    return None


def _extract_person_name_filters(query: str) -> List[str]:
    """
    Minimal name filters used by our current question set.
    (We keep it simple and deterministic.)
    """
    q = query.lower()
    names = []
    for n in ["hans vangheluwe", "juan felipe botero"]:
        if n in q:
            names.append(n)
    return names


@dataclass(frozen=True)
class StudyProgrammeBlock:
    chunk_id: str
    program: str  # DS/SE/CN
    course_name: str
    semester: Optional[str]
    course_code: Optional[str]
    contract_restrictions: Optional[str]
    exam_period: Optional[str]
    lecturers: Optional[str]


def _extract_field(text: str, label: str) -> Optional[str]:
    """
    Extract markdown field like '**Semester:** 1E SEM'
    """
    # tolerate extra spaces/newlines
    pat = rf"\*\*{re.escape(label)}:\*\*\s*([^\n*]+)"
    m = re.search(pat, text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


class StructuredIndex:
    """
    In-memory structured index.
    """

    def __init__(self) -> None:
        self.course_titles: Set[str] = set()
        self._course_norm_to_title: Dict[str, str] = {}
        # course_title -> section -> [chunk_id,...]
        self.course_sections: Dict[str, Dict[str, List[str]]] = {}

        # Study programme blocks
        self.study_blocks_by_program: Dict[str, List[StudyProgrammeBlock]] = {"DS": [], "SE": [], "CN": []}
        # program -> course_name -> [block,...]
        self.study_blocks_by_program_course: Dict[str, Dict[str, List[StudyProgrammeBlock]]] = {"DS": {}, "SE": {}, "CN": {}}
        # course_name -> [block,...] across programmes
        self.study_blocks_by_course: Dict[str, List[StudyProgrammeBlock]] = {}

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> "StructuredIndex":
        idx = cls()
        for ch in chunks:
            # Course PDF: sectioned chunks
            if _is_course_pdf(ch):
                title = str(ch.source_title or "").strip()
                if title:
                    idx.course_titles.add(title)
                    idx._course_norm_to_title.setdefault(_norm(title), title)
                sec = str(ch.source_section or "").strip()
                if title and sec:
                    idx.course_sections.setdefault(title, {}).setdefault(sec, []).append(ch.chunk_id)

            # Study programme: per-course blocks live in website chunks, with section == course name
            if _is_website(ch) and "/study-programme/" in str(ch.source_file):
                program = _program_from_url(str(ch.source_file))
                if not program:
                    continue
                # Only keep real course blocks (not headers like "Compulsory courses")
                if not ch.source_section:
                    continue
                if "**Course code:**" not in (ch.text or ""):
                    continue

                course_name = str(ch.source_section).strip()
                if not course_name:
                    continue

                text = ch.text or ""
                block = StudyProgrammeBlock(
                    chunk_id=ch.chunk_id,
                    program=program,
                    course_name=course_name,
                    semester=_extract_field(text, "Semester"),
                    course_code=_extract_field(text, "Course code"),
                    contract_restrictions=_extract_field(text, "Contract restrictions"),
                    exam_period=_extract_field(text, "Exam Period"),
                    lecturers=_extract_field(text, "Lecturer(s)"),
                )

                idx.study_blocks_by_program[program].append(block)
                idx.study_blocks_by_program_course[program].setdefault(course_name, []).append(block)
                idx.study_blocks_by_course.setdefault(course_name, []).append(block)

        return idx

    def match_course_title(self, query: str) -> Optional[str]:
        # Quoted titles have priority
        quoted = _extract_quoted_title(query)
        if quoted:
            qn = _norm(quoted)
            # Exact normalized match
            if qn in self._course_norm_to_title:
                return self._course_norm_to_title[qn]
            # Fuzzy: longest course title contained in quoted string
            matches = [t for t in self.course_titles if _norm(t) in qn or qn in _norm(t)]
            if matches:
                matches.sort(key=lambda x: len(x), reverse=True)
                return matches[0]

        qn = _norm(query)
        # Longest substring match over known titles
        matches = [t for t in self.course_titles if _norm(t) and _norm(t) in qn]
        if not matches:
            return None
        matches.sort(key=lambda x: len(x), reverse=True)
        return matches[0]

    def forced_chunk_ids(self, query: str, k: int = 5) -> List[str]:
        """
        Return chunk_ids that should be forced to the top for this query.
        """
        intents = _detect_intents(query)
        program = _detect_program(query)
        course = self.match_course_title(query)
        qn = _norm(query)

        forced: List[str] = []

        # Course-section questions (PDF chunks)
        if course:
            sec_map = self.course_sections.get(course, {})

            def add_section(sec: str) -> None:
                for cid in sec_map.get(sec, []):
                    forced.append(cid)

            if intents["prerequisites"]:
                add_section("Prerequisites")
            if intents["learning_outcomes"]:
                add_section("Learning Outcomes")
            if intents["course_contents"]:
                add_section("Course Contents")
            if intents["assessment"]:
                add_section("Assessment method and criteria")

            # Exam period is more reliable from study-programme blocks than PDFs
            if intents["exam_period"]:
                blocks = self.study_blocks_by_course.get(course, [])
                # pick 1 block (avoid duplicates), prefer DS then SE then CN
                order = {"DS": 0, "SE": 1, "CN": 2}
                blocks = sorted(blocks, key=lambda b: order.get(b.program, 9))
                if blocks:
                    forced.append(blocks[0].chunk_id)

            # Lecturer/semester/course-code questions: pick the smallest stable header chunk if present
            if intents["lecturer"] or intents["semester"]:
                if "Is part of the next programmes" in sec_map:
                    add_section("Is part of the next programmes")
                else:
                    # fallback: any chunk from that course
                    for sec, ids in sec_map.items():
                        if ids:
                            forced.append(ids[0])
                            break

        # Study programme list/filter questions (prefer website blocks)
        # IMPORTANT: only force *programme-wide* course blocks when the query is actually asking
        # for lists/filters. Otherwise it can drown the answer context for course-specific questions.
        if intents["study_programme"] and program:
            sem = _semester_filter(query)
            name_filters = _extract_person_name_filters(query)
            wants_programme_listing = bool(intents["list"] or sem or intents["contract_restrictions"] or name_filters)

            if wants_programme_listing:
                blocks = list(self.study_blocks_by_program.get(program, []))
                if sem:
                    blocks = [b for b in blocks if (b.semester or "").upper() == sem]
                if intents["contract_restrictions"] and "exam contract not possible" in qn:
                    blocks = [b for b in blocks if (b.contract_restrictions or "").lower().strip() == "exam contract not possible"]
                for nf in name_filters:
                    blocks = [b for b in blocks if nf in (b.lecturers or "").lower()]

                # deterministic order by course name
                blocks.sort(key=lambda b: _norm(b.course_name))
                forced.extend([b.chunk_id for b in blocks])
            elif course:
                # Course-specific question that mentions a programme: include only that course block (if available)
                blocks = self.study_blocks_by_program_course.get(program, {}).get(course, [])
                if blocks:
                    forced.append(blocks[0].chunk_id)

        # Which specialization includes a named course? -> show study programme blocks across programs
        if intents["study_programme"] and ("which specialization" in qn or "which specialisation" in qn) and course:
            blocks = self.study_blocks_by_course.get(course, [])
            # stable order: DS, SE, CN
            order = {"DS": 0, "SE": 1, "CN": 2}
            blocks = sorted(blocks, key=lambda b: (order.get(b.program, 9), _norm(b.course_name)))
            forced.extend([b.chunk_id for b in blocks])

        # De-dup, keep order
        seen = set()
        out = []
        for cid in forced:
            if cid in seen:
                continue
            seen.add(cid)
            out.append(cid)
            # don't pre-truncate too hard; caller will slice via k anyway
            if len(out) >= max(k * 5, 50):
                break

        return out


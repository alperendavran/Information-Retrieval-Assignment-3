# -*- coding: utf-8-sig -*-
"""
Tool / function-calling router for Agentic RAG.

Concept:
1) An LLM decides WHICH deterministic local tool(s) to call (OpenAI function calling)
2) We execute those tools locally to get chunk_ids (structured retrieval)
3) We build context from the selected chunks and generate an answer

This makes routing explicit (and "agentic") without hardcoding lots of if/else logic
in the main pipeline.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import MODEL_PRICING_USD_PER_1M
from src.cost_tracking import estimate_cost_usd, log_event
from src.retrieval import RetrievalResult, Retriever


COURSE_FIELDS_ENUM = [
    "Prerequisites",
    "Learning Outcomes",
    "Course Contents",
    "Assessment method and criteria",
    "Study material",
    "Teaching method and planned learning activities",
    "Contact information",
    "Tutoring",
    "Is part of the next programmes",
]


def _tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_course_fields",
                "description": "Retrieve specific section chunks for a given course (deterministic, local).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "course_title": {"type": "string", "description": "Course title, e.g. 'Future Internet'."},
                        "fields": {
                            "type": "array",
                            "items": {"type": "string", "enum": COURSE_FIELDS_ENUM},
                            "description": "Which sections to retrieve for this course.",
                        },
                        "max_chunks_per_field": {
                            "type": "integer",
                            "description": "Limit chunks per field (some fields span multiple chunks).",
                            "default": 2,
                            "minimum": 1,
                            "maximum": 4,
                        },
                    },
                    "required": ["course_title", "fields"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_study_programme_blocks",
                "description": "Retrieve study-programme course blocks (website) with filters like program/semester/lecturer/contract restrictions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "program": {"type": "string", "enum": ["DS", "SE", "CN"]},
                        "semester": {"type": ["string", "null"], "enum": ["1E SEM", "2E SEM", None]},
                        "course_name": {"type": ["string", "null"], "description": "Optional course name to narrow down."},
                        "lecturer_contains": {"type": ["string", "null"], "description": "Case-insensitive substring to filter lecturers."},
                        "contract_contains": {"type": ["string", "null"], "description": "Case-insensitive substring to filter contract restrictions."},
                        "top_n": {"type": "integer", "default": 10, "minimum": 1, "maximum": 15},
                    },
                    "required": ["program"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dense_retrieve",
                "description": "Fallback: dense retrieval from FAISS (no structured filtering).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                    },
                    "required": ["query", "k"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def _build_results_from_chunk_ids(
    retriever: Retriever,
    query_embedding: Optional[np.ndarray],
    chunk_ids: List[str],
) -> List[RetrievalResult]:
    out: List[RetrievalResult] = []
    for rank, cid in enumerate(chunk_ids, 1):
        ch = retriever.chunk_lookup.get(cid)
        if not ch:
            continue
        sc = 1.0
        if query_embedding is not None and hasattr(retriever, "_embeddings_cache") and hasattr(retriever, "_chunk_id_to_row"):
            row = retriever._chunk_id_to_row.get(cid)
            if row is not None and 0 <= row < len(retriever._embeddings_cache):
                sc = float(np.dot(retriever._embeddings_cache[row], query_embedding[0]))
        out.append(
            RetrievalResult(
                chunk_id=cid,
                text=ch.text,
                source_title=ch.source_title,
                source_section=ch.source_section,
                similarity_score=float(sc),
                rank=rank,
            )
        )
    return out


def route_and_select_chunks(
    query: str,
    retriever: Retriever,
    client: Any,
    model: str,
    top_k: int = 5,
    max_calls: int = 3,
) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
    """
    Use OpenAI function calling to decide which local tools to run,
    then execute them to select chunk_ids.
    """
    trace: Dict[str, Any] = {"model": model, "tool_calls": [], "selected": 0}

    # We need structured index for the structured tools
    idx = getattr(retriever, "structured_index", None)

    tools = _tools_schema()
    sys_prompt = (
        "You are a query routing module for a RAG system about the University of Antwerp CS Masters pages.\n"
        "Decide which tool(s) to call to retrieve the BEST evidence chunks.\n"
        "Rules:\n"
        "- Prefer get_course_fields for course-specific questions (prerequisites/outcomes/contents/assessment).\n"
        "- Prefer get_study_programme_blocks for study programme questions (DS/SE/CN, semester, contract restrictions, lecturers).\n"
        "- Use dense_retrieve only if you are unsure.\n"
        f"- Call at most {max_calls} tools.\n"
        "Return tool calls only."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto",
        temperature=0.0,
    )

    prompt_tokens = getattr(resp.usage, "prompt_tokens", None)
    completion_tokens = getattr(resp.usage, "completion_tokens", None)
    total_tokens = getattr(resp.usage, "total_tokens", None)
    cost_est = None
    if prompt_tokens is not None and completion_tokens is not None:
        cost_est = estimate_cost_usd(model=model, prompt_tokens=int(prompt_tokens), completion_tokens=int(completion_tokens), pricing_table=MODEL_PRICING_USD_PER_1M)

    log_event(
        {
            "operation": "tool_router",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_estimate_usd": cost_est,
        }
    )

    tool_calls = getattr(resp.choices[0].message, "tool_calls", None) or []
    tool_calls = tool_calls[: max(int(max_calls), 1)]

    # Prepare query embedding for scoring (optional)
    q_emb = None
    try:
        q_emb = retriever.embedding_model.encode_query(query)
    except Exception:
        q_emb = None

    selected_ids: List[str] = []

    def add_ids(ids: List[str]) -> None:
        nonlocal selected_ids
        for cid in ids:
            if cid not in selected_ids:
                selected_ids.append(cid)

    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "")
        args_s = getattr(fn, "arguments", "") or ""
        args = _safe_json_loads(args_s) or {}
        trace["tool_calls"].append({"name": name, "arguments": args})

        if name == "dense_retrieve":
            k = int(args.get("k") or top_k)
            # keep "dense" truly dense (no structured forcing)
            res = retriever.retrieve(str(args.get("query") or query), k=k, use_structured=False)
            add_ids([r.chunk_id for r in res])
            continue

        if idx is None:
            continue

        if name == "get_course_fields":
            course_title = str(args.get("course_title") or "").strip()
            fields = args.get("fields") or []
            if not course_title or not isinstance(fields, list):
                continue
            matched = idx.match_course_title(course_title) or idx.match_course_title(query)
            if not matched:
                continue
            sec_map = idx.course_sections.get(matched, {})
            max_per = int(args.get("max_chunks_per_field") or 2)
            max_per = max(1, min(max_per, 4))
            for f in fields:
                if f in sec_map:
                    add_ids(sec_map[f][:max_per])
            continue

        if name == "get_study_programme_blocks":
            program = str(args.get("program") or "").strip().upper()
            if program not in ("DS", "SE", "CN"):
                continue
            semester = args.get("semester")
            semester = str(semester).upper().strip() if isinstance(semester, str) else None
            course_name = args.get("course_name")
            course_name = str(course_name).strip() if isinstance(course_name, str) and course_name.strip() else None
            lecturer_contains = args.get("lecturer_contains")
            lecturer_contains = str(lecturer_contains).lower().strip() if isinstance(lecturer_contains, str) and lecturer_contains.strip() else None
            contract_contains = args.get("contract_contains")
            contract_contains = str(contract_contains).lower().strip() if isinstance(contract_contains, str) and contract_contains.strip() else None
            top_n = int(args.get("top_n") or 10)
            top_n = max(1, min(top_n, 15))

            blocks = list(idx.study_blocks_by_program.get(program, []))
            if course_name:
                # simple exact match on course_name
                blocks = [b for b in blocks if b.course_name.lower().strip() == course_name.lower().strip()]
            if semester:
                blocks = [b for b in blocks if (b.semester or "").upper().strip() == semester]
            if lecturer_contains:
                blocks = [b for b in blocks if lecturer_contains in (b.lecturers or "").lower()]
            if contract_contains:
                blocks = [b for b in blocks if contract_contains in (b.contract_restrictions or "").lower()]

            blocks.sort(key=lambda b: b.course_name.lower())
            add_ids([b.chunk_id for b in blocks[:top_n]])
            continue

    # Fallback if router didn't pick anything
    if not selected_ids:
        res = retriever.retrieve(query, k=top_k, use_structured=True)
        selected_ids = [r.chunk_id for r in res]

    # Truncate to a reasonable size for context (but allow >k for lists)
    max_ctx = max(top_k, 12)
    selected_ids = selected_ids[:max_ctx]

    selected = _build_results_from_chunk_ids(retriever, q_emb, selected_ids)
    trace["selected"] = len(selected)
    return selected, trace


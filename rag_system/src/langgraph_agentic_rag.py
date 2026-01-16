# -*- coding: utf-8-sig -*-
"""
Agentic RAG workflow powered by LangGraph.

Goal (assignment-aligned):
- Keep the core RAG components (chunk → embed → index → retrieve → generate)
- Add an "advanced" orchestration layer that makes retrieval more robust by:
  - Query understanding (rule-based tagger; optional LLM tagger)
  - Multi-query expansion + fusion (RRF)
  - Post-retrieval deduplication + MMR diversification
  - Optional reflection / LLM-as-judge hooks

References (for report + design motivation):
- LangGraph docs (StateGraph, START/END): https://docs.langchain.com/oss/python/langgraph/overview
- Advanced RAG techniques (query expansion, dedup, reranking): 15_Advanced_RAG_Techniques.pdf
- Agentic RAG patterns (planning/reflection/tool use): Agentic_RAG survey (2025)
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Tuple

import numpy as np

_LANGGRAPH_AVAILABLE = True
try:
    from langgraph.graph import StateGraph, START, END
except Exception:  # pragma: no cover
    # Keep module importable even if langgraph isn't installed.
    # We only require LangGraph when building/executing the graph.
    _LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore
    START = None  # type: ignore
    END = None  # type: ignore

from config import (
    TOP_K,
    USE_RERANKING,
    EXPANSION_FACTOR,
    SECTION_AWARE_BOOST,
    DIVERSIFY_SOURCES,
    MAX_CHUNKS_PER_SOURCE_TITLE,
    AGENTIC_USE_LLM_TAGGER,
    AGENTIC_TAGGER_MODEL,
    AGENTIC_USE_HYDE,
    AGENTIC_HYDE_MODEL,
    AGENTIC_USE_REFLECTION,
    AGENTIC_REFLECTION_MODEL,
    AGENTIC_USE_TOOL_ROUTER,
    AGENTIC_TOOL_ROUTER_MODEL,
    AGENTIC_TOOL_ROUTER_MAX_CALLS,
)
from src.retrieval import Retriever, RetrievalResult
from src.generation import AnswerGenerator
from src.evaluation import RAGEvaluator


class AgenticRAGState(TypedDict, total=False):
    # Inputs
    query: str

    # Understanding / planning
    tags: Dict[str, Any]
    rewritten_queries: List[str]
    scope: str  # "all" | "course" | "website"

    # Retrieval
    top_n_per_query: int
    fused_candidates: List[Dict[str, Any]]  # list of {chunk_id, rrf_score, best_sim}
    selected: List[Dict[str, Any]]  # list of RetrievalResult.to_dict()
    context: str

    # Generation
    answer: str
    sources: List[Dict[str, Any]]

    # Optional baseline + judge
    baseline_answer: Optional[str]
    judge: Optional[Dict[str, Any]]

    # Debug / tracing
    trace: List[Dict[str, Any]]
    tool_routed: bool


# ---------------------------------------------------------------------------
# Query understanding (rule-based, dataset-aware)
# ---------------------------------------------------------------------------


def _normalize_query(q: str) -> str:
    return " ".join(q.lower().split())


def _extract_course_codes(query: str) -> List[str]:
    # Dataset shows course codes like 2500WETINT
    q_up = query.upper()
    return re.findall(r"\b\d{4}[A-Z]{3,}\b", q_up)


def rule_based_tag_query(query: str) -> Dict[str, Any]:
    """
    Very lightweight tagger based on dataset structure (cs-data).

    Outputs:
    - intents: list[str]
    - preferred_sections: list[str]
    - program: optional 'DS'|'SE'|'CN'
    - scope: 'course'|'website'|'all'
    - course_codes: list[str]
    """
    q = _normalize_query(query)
    codes = _extract_course_codes(query)

    intents: List[str] = []
    preferred_sections: List[str] = []

    def add_intent(name: str, section: Optional[str] = None) -> None:
        if name not in intents:
            intents.append(name)
        if section and section not in preferred_sections:
            preferred_sections.append(section)

    # Section-intent mapping driven by course-pages.json section names
    if any(k in q for k in ["prereq", "prerequisite", "requirements", "before taking", "prior knowledge"]):
        add_intent("prerequisites", "Prerequisites")
    if any(k in q for k in ["learning outcome", "outcome", "competenc", "you will be able"]):
        add_intent("learning_outcomes", "Learning Outcomes")
    if any(k in q for k in ["course content", "contents", "syllabus", "topics", "covers", "what is covered"]):
        add_intent("course_contents", "Course Contents")
    if any(k in q for k in ["exam", "assessment", "grading", "evaluation", "criteria"]):
        add_intent("assessment", "Assessment method and criteria")
    if any(k in q for k in ["study material", "textbook", "book", "slides", "reading", "literature"]):
        add_intent("study_material", "Study material")
    if any(k in q for k in ["lecturer", "teacher", "taught by", "who teaches", "instructor"]):
        add_intent("lecturers", "Contact information")
    if any(k in q for k in ["credit", "ects"]):
        add_intent("credits")
    if any(k in q for k in ["semester"]):
        add_intent("semester")

    # Website-scraped intents
    if any(k in q for k in ["admission", "apply", "application", "enrol", "enroll", "international students"]):
        add_intent("admission")
    if any(k in q for k in ["tuition", "fees", "cost"]):
        add_intent("tuition")

    # Program detection:
    # If the query mentions multiple programmes (e.g., "Data Science, Software Engineering, or Computer Networks"),
    # do NOT lock to one programme; keep it None.
    prog_hits: List[str] = []
    if "data science" in q or "artificial intelligence" in q:
        prog_hits.append("DS")
    if "software engineering" in q:
        prog_hits.append("SE")
    if "computer networks" in q or "computer network" in q:
        prog_hits.append("CN")
    program: Optional[str] = prog_hits[0] if len(set(prog_hits)) == 1 else None

    # Scope routing (dataset-driven):
    # - admission/tuition -> website pages
    # - explicit study-programme queries -> website pages (study-programme blocks live on website)
    #   BUT if the query explicitly references a specific course AND asks for course-page fields,
    #   keep scope="all" so we can retrieve both website + course PDFs (cross-page multi-hop).
    # - "exam period" is typically present on study-programme pages -> include website (scope="all")
    scope = "all"
    is_study_programme_query = any(
        k in q for k in ["study programme", "study-programme", "compulsory courses", "compulsory course", "model path over"]
    )
    course_page_intent = any(
        i in intents for i in ["prerequisites", "learning_outcomes", "course_contents", "assessment", "study_material"]
    )
    # Explicit course reference heuristic (keeps agent focused on course pages only when needed)
    has_explicit_course_ref = bool(codes) or bool(re.search(r"course\s*['\"]", query, flags=re.IGNORECASE))

    if any(i in intents for i in ["admission", "tuition"]):
        scope = "website"
    elif is_study_programme_query:
        scope = "all" if (course_page_intent and has_explicit_course_ref) else "website"
    elif "exam period" in q:
        scope = "all"
    elif any(i in intents for i in ["prerequisites", "learning_outcomes", "course_contents", "assessment", "study_material", "credits", "semester", "lecturers"]):
        scope = "course"

    return {
        "intents": intents,
        "preferred_sections": preferred_sections,
        "program": program,
        "scope": scope,
        "course_codes": codes,
    }


def build_query_variants(query: str, tags: Dict[str, Any], max_variants: int = 5) -> List[str]:
    """
    Query expansion (Advanced RAG): create multiple semantically-focused queries.

    This improves recall/precision on structured corpora where intent maps to known sections.
    """
    q = query.strip()
    variants: List[str] = [q]

    # Section-focused variants
    for sec in tags.get("preferred_sections", []):
        variants.append(f"{sec}: {q}")

    # Program-focused variants
    program = tags.get("program")
    if program:
        variants.append(f"{program} specialization: {q}")

    # Course code focused
    for code in tags.get("course_codes", []):
        variants.append(f"Course code {code}: {q}")

    # Dedup, keep order
    seen = set()
    out = []
    for v in variants:
        v2 = " ".join(v.split())
        if v2 and v2 not in seen:
            out.append(v2)
            seen.add(v2)
        if len(out) >= max_variants:
            break
    return out


# ---------------------------------------------------------------------------
# Optional LLM helpers (tagging / HyDE / reflection)
# ---------------------------------------------------------------------------


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON extraction.
    LangGraph itself is not responsible for structured output; we keep this robust.
    """
    text = text.strip()
    if not text:
        return None
    # Fast path: full JSON
    try:
        import json

        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find the first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        import json

        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def llm_tag_query(query: str, client: Any, model: str) -> Optional[Dict[str, Any]]:
    """
    Use an LLM to output structured query tags (intent + routing).
    This is optional and OFF by default in config.
    """
    prompt = f"""You are a query understanding module for a RAG system about the University of Antwerp CS Masters program.

Return ONLY valid JSON with this schema:
{{
  "intents": ["prerequisites"|"learning_outcomes"|"course_contents"|"assessment"|"study_material"|"lecturers"|"credits"|"semester"|"admission"|"tuition"],
  "preferred_sections": ["Prerequisites"|"Learning Outcomes"|"Course Contents"|"Assessment method and criteria"|"Study material"|"Teaching method and planned learning activities"|"Contact information"|"Tutoring"|"Is part of the next programmes"],
  "program": "DS"|"SE"|"CN"|null,
  "scope": "course"|"website"|"all",
  "course_codes": ["<course code like 2500WETINT>", ...]
}}

Rules:
- Keep intents/sections strictly to the allowed values above.
- If unsure, choose scope=\"all\".
- Extract course codes if present in the query.

Query: {query}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        txt = resp.choices[0].message.content or ""
        obj = _extract_first_json_object(txt)
        return obj
    except Exception:
        return None


def llm_hyde(query: str, client: Any, model: str) -> Optional[str]:
    """
    HyDE: generate a hypothetical passage that would answer the question.
    We then use that passage as an additional retrieval query variant.
    """
    prompt = f"""Write a short hypothetical passage (2-5 sentences) that would likely appear in the University of Antwerp CS Masters pages and would help answer the question.

Important:
- Do NOT mention that this is hypothetical.
- Do NOT cite sources.
- Do NOT include disclaimers.
- Output ONLY the passage text.

Question: {query}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None


def llm_reflect(query: str, context: str, draft_answer: str, client: Any, model: str) -> Optional[str]:
    """
    Reflection: rewrite the draft answer to be strictly grounded in context.
    """
    prompt = f"""You are a strict verifier for a RAG system.

QUESTION:
{query}

CONTEXT:
{context}

DRAFT ANSWER:
{draft_answer}

Task:
- Produce the FINAL answer using ONLY the CONTEXT.
- If the context is insufficient, answer exactly:
  \"I don't have enough information in the provided context to answer this question.\"
- Keep the answer concise and factual.
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fusion + reranking utilities
# ---------------------------------------------------------------------------


def rrf_fuse(result_lists: List[List[RetrievalResult]], k0: int = 60) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF):
      score(d) = Σ 1 / (k0 + rank_i(d))
    """
    scores: Dict[str, float] = {}
    for results in result_lists:
        for r in results:
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k0 + r.rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def _is_website_source(source_file: str) -> bool:
    return str(source_file).startswith("http")


def _normalize_text_for_hash(text: str) -> str:
    # aggressive normalization for dedup on boilerplate
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def dedup_by_text(results: List[RetrievalResult]) -> List[RetrievalResult]:
    seen = set()
    out = []
    for r in results:
        h = hashlib.sha1(_normalize_text_for_hash(r.text).encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


def mmr_select(
    query_emb: np.ndarray,
    candidate_ids: List[str],
    candidate_scores: List[float],
    get_embedding_by_chunk_id,
    k: int,
    lambda_mult: float = 0.7,
) -> List[str]:
    """
    Maximal Marginal Relevance (MMR) selection.

    Selects a diverse subset of candidates:
      argmax_d  λ * sim(q,d) - (1-λ) * max_{s in selected} sim(d,s)
    """
    if not candidate_ids:
        return []

    # Preload embeddings
    emb_map: Dict[str, np.ndarray] = {}
    for cid in candidate_ids:
        e = get_embedding_by_chunk_id(cid)
        if e is not None:
            emb_map[cid] = e

    # Keep only candidates that have embeddings
    filtered = [(cid, sc) for cid, sc in zip(candidate_ids, candidate_scores) if cid in emb_map]
    if not filtered:
        return candidate_ids[:k]

    cand_ids = [cid for cid, _ in filtered]
    rel = np.array([sc for _, sc in filtered], dtype=np.float32)

    selected: List[str] = []
    remaining = set(cand_ids)

    # Initialize with best relevance
    first = cand_ids[int(np.argmax(rel))]
    selected.append(first)
    remaining.remove(first)

    # Greedy selection
    while remaining and len(selected) < k:
        best_cid = None
        best_val = -1e9
        for cid in list(remaining):
            # relevance term
            sim_q = float(rel[cand_ids.index(cid)])
            # diversity term
            sim_to_sel = 0.0
            for sid in selected:
                sim_to_sel = max(sim_to_sel, float(np.dot(emb_map[cid], emb_map[sid])))
            val = lambda_mult * sim_q - (1.0 - lambda_mult) * sim_to_sel
            if val > best_val:
                best_val = val
                best_cid = cid
        if best_cid is None:
            break
        selected.append(best_cid)
        remaining.remove(best_cid)

    return selected


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


@dataclass
class AgenticConfig:
    top_k: int = TOP_K
    top_n_per_query: int = 20
    max_query_variants: int = 5
    rrf_k0: int = 60
    use_mmr: bool = True
    mmr_lambda: float = 0.7
    use_dedup: bool = True


def agentic_retrieve_only(
    query: str,
    retriever: Retriever,
    cfg: Optional[AgenticConfig] = None,
) -> Tuple[List[RetrievalResult], List[Dict[str, Any]]]:
    """
    Run the agentic retrieval logic WITHOUT generation (no API calls).

    Returns:
      - selected RetrievalResult list (ranked 1..k)
      - trace events (for debugging / report appendix)
    """
    cfg = cfg or AgenticConfig()
    trace: List[Dict[str, Any]] = []

    def t(event: str, payload: Dict[str, Any]) -> None:
        trace.append({"event": event, "payload": payload, "ts": time.time()})

    tags = rule_based_tag_query(query)
    scope = tags.get("scope", "all")
    t("tag_query", {"tags": tags})

    # For list-like / filter queries (especially website study-programme pages),
    # MMR diversity can hurt Recall@k because we actually WANT many similar blocks.
    qn = _normalize_query(query)
    intents = set(tags.get("intents", []) or [])
    use_mmr = bool(cfg.use_mmr)
    if re.search(r"\blist\b", qn) or any(k in qn for k in ["compulsory courses", "study programme", "study-programme"]):
        use_mmr = False
    if scope == "website" and (("semester" in intents) or ("tuition" in intents) or ("admission" in intents)):
        use_mmr = False
    if scope == "website" and any(k in qn for k in ["taught by", "who teaches"]):
        use_mmr = False

    rewrites = build_query_variants(query, tags, max_variants=cfg.max_query_variants)
    t("rewrite", {"rewritten_queries": rewrites})

    # Retrieval per rewrite
    result_lists: List[List[RetrievalResult]] = []
    for q in rewrites:
        result_lists.append(
            retriever.retrieve(
                q,
                k=cfg.top_n_per_query,
                use_reranking=USE_RERANKING,
                expansion_factor=EXPANSION_FACTOR,
            )
        )

    fused = rrf_fuse(result_lists, k0=cfg.rrf_k0)

    # Keep best similarity as a secondary signal
    best_sim: Dict[str, float] = {}
    for results in result_lists:
        for r in results:
            best_sim[r.chunk_id] = max(best_sim.get(r.chunk_id, -1e9), float(r.similarity_score))

    # Scope filter based on source_file
    filtered: List[Tuple[str, float]] = []
    for cid, rrf_score in fused:
        ch = retriever.chunk_lookup.get(cid)
        if ch is None:
            continue
        if scope == "website" and not _is_website_source(ch.source_file):
            continue
        if scope == "course" and _is_website_source(ch.source_file):
            continue
        filtered.append((cid, rrf_score))

    fused_candidates = [{"chunk_id": cid, "rrf_score": float(rrf_score), "best_sim": float(best_sim.get(cid, 0.0))} for cid, rrf_score in filtered]
    t("retrieve_fuse", {"n_variants": len(rewrites), "scope": scope, "n_fused": len(fused_candidates)})

    if not fused_candidates:
        return [], trace

    # Candidate pool sorted by (best_sim, rrf_score)
    fused_sorted = sorted(fused_candidates, key=lambda x: (x.get("best_sim", 0.0), x.get("rrf_score", 0.0)), reverse=True)
    candidate_ids = [c["chunk_id"] for c in fused_sorted[: max(cfg.top_k * 10, 50)]]

    # Re-score with original query embedding when possible
    q_emb = None
    candidate_scores: List[float] = []
    if hasattr(retriever, "_embeddings_cache") and hasattr(retriever, "_chunk_id_to_row"):
        try:
            q_emb = retriever.embedding_model.encode_query(query)
        except Exception:
            q_emb = None

    for cid in candidate_ids:
        # IMPORTANT: keep `best_sim` (which already includes dataset-aware boosts + reranking)
        # as a signal; avoid overwriting it with a raw dot-product score.
        sc = float(best_sim.get(cid, 0.0))
        if q_emb is not None:
            row = retriever._chunk_id_to_row.get(cid)
            if row is not None and 0 <= row < len(retriever._embeddings_cache):
                dot_sc = float(np.dot(retriever._embeddings_cache[row], q_emb[0]))
                sc = max(sc, dot_sc)
        candidate_scores.append(float(sc))

    # Build candidate RetrievalResult objects
    candidates: List[RetrievalResult] = []
    for i, (cid, sc) in enumerate(zip(candidate_ids, candidate_scores), 1):
        ch = retriever.chunk_lookup.get(cid)
        if not ch:
            continue
        candidates.append(
            RetrievalResult(
                chunk_id=cid,
                text=ch.text,
                source_title=ch.source_title,
                source_section=ch.source_section,
                similarity_score=float(sc),
                rank=i,
            )
        )

    if cfg.use_dedup:
        candidates = dedup_by_text(candidates)

    # Optional source diversification (same knobs as Retriever)
    if DIVERSIFY_SOURCES:
        per_source: Dict[str, int] = {}
        diversified: List[RetrievalResult] = []
        for r in candidates:
            per_source.setdefault(r.source_title, 0)
            if per_source[r.source_title] >= max(int(MAX_CHUNKS_PER_SOURCE_TITLE), 1):
                continue
            diversified.append(r)
            per_source[r.source_title] += 1
        candidates = diversified

    # Prefer coverage of intent-relevant sections (if present) before diversity.
    preferred_secs = list(tags.get("preferred_sections", []) or [])
    forced_ids: List[str] = []
    if preferred_secs:
        for sec in preferred_secs:
            for r in candidates:
                if r.source_section == sec and r.chunk_id not in forced_ids:
                    forced_ids.append(r.chunk_id)
                    break

    selected_ids = [r.chunk_id for r in candidates[: cfg.top_k]]

    def mmr_select_seeded(
        query_emb: np.ndarray,
        candidate_ids: List[str],
        candidate_scores: List[float],
        get_embedding_by_chunk_id,
        k: int,
        lambda_mult: float = 0.7,
        seed_selected: Optional[List[str]] = None,
    ) -> List[str]:
        if not candidate_ids:
            return []

        # Preload embeddings
        emb_map: Dict[str, np.ndarray] = {}
        for cid in candidate_ids:
            e = get_embedding_by_chunk_id(cid)
            if e is not None:
                emb_map[cid] = e

        # Keep only candidates that have embeddings
        filtered = [(cid, sc) for cid, sc in zip(candidate_ids, candidate_scores) if cid in emb_map]
        if not filtered:
            return candidate_ids[:k]

        cand_ids = [cid for cid, _ in filtered]
        rel_map = {cid: float(sc) for cid, sc in filtered}

        selected: List[str] = []
        remaining = set(cand_ids)

        # Seed (forced) selections first
        for cid in (seed_selected or []):
            if cid in remaining:
                selected.append(cid)
                remaining.remove(cid)
            if len(selected) >= k:
                return selected[:k]

        # Initialize with best relevance if nothing seeded
        if not selected:
            first = max(cand_ids, key=lambda cid: rel_map.get(cid, -1e9))
            selected.append(first)
            remaining.remove(first)

        # Greedy selection
        while remaining and len(selected) < k:
            best_cid = None
            best_val = -1e9
            for cid in list(remaining):
                sim_q = float(rel_map.get(cid, 0.0))
                sim_to_sel = 0.0
                for sid in selected:
                    sim_to_sel = max(sim_to_sel, float(np.dot(emb_map[cid], emb_map[sid])))
                val = lambda_mult * sim_q - (1.0 - lambda_mult) * sim_to_sel
                if val > best_val:
                    best_val = val
                    best_cid = cid
            if best_cid is None:
                break
            selected.append(best_cid)
            remaining.remove(best_cid)

        return selected

    if use_mmr and q_emb is not None and hasattr(retriever, "_embeddings_cache"):
        def get_emb(cid: str) -> Optional[np.ndarray]:
            row = retriever._chunk_id_to_row.get(cid)
            if row is None:
                return None
            if row < 0 or row >= len(retriever._embeddings_cache):
                return None
            return retriever._embeddings_cache[row]

        selected_ids = mmr_select_seeded(
            query_emb=q_emb,
            candidate_ids=[r.chunk_id for r in candidates],
            candidate_scores=[float(r.similarity_score) for r in candidates],
            get_embedding_by_chunk_id=get_emb,
            k=cfg.top_k,
            lambda_mult=cfg.mmr_lambda,
            seed_selected=forced_ids,
        )
    elif forced_ids:
        # If MMR is disabled/unavailable, still honor forced section coverage.
        filled = []
        used = set()
        for cid in forced_ids:
            if cid in {r.chunk_id for r in candidates} and cid not in used:
                filled.append(cid)
                used.add(cid)
        for r in candidates:
            if r.chunk_id in used:
                continue
            filled.append(r.chunk_id)
            used.add(r.chunk_id)
            if len(filled) >= cfg.top_k:
                break
        selected_ids = filled[: cfg.top_k]

    id_to_r = {r.chunk_id: r for r in candidates}
    selected = [id_to_r[cid] for cid in selected_ids if cid in id_to_r]
    for i, r in enumerate(selected, 1):
        r.rank = i

    t("select", {"selected": len(selected), "mmr": use_mmr, "dedup": cfg.use_dedup})
    return selected, trace


def build_agentic_rag_graph(
    retriever: Retriever,
    generator: AnswerGenerator,
    evaluator: Optional[RAGEvaluator] = None,
    cfg: Optional[AgenticConfig] = None,
):
    """
    Returns a compiled LangGraph runnable.
    """
    if not _LANGGRAPH_AVAILABLE:  # pragma: no cover
        raise ImportError("LangGraph is not installed. Install with: pip install -U langgraph")

    cfg = cfg or AgenticConfig()

    def add_trace(state: AgenticRAGState, event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        trace = list(state.get("trace", []))
        trace.append({"event": event, "payload": payload, "ts": time.time()})
        return {"trace": trace}

    def tag_node(state: AgenticRAGState) -> Dict[str, Any]:
        query = state["query"]
        tags = rule_based_tag_query(query)

        # Optional: LLM-based tagger (more advanced intent reading)
        if AGENTIC_USE_LLM_TAGGER and getattr(generator, "client", None) is not None:
            llm_tags = llm_tag_query(query, generator.client, model=AGENTIC_TAGGER_MODEL)
            if isinstance(llm_tags, dict):
                # Merge: prefer LLM outputs when present; fallback to rule-based.
                merged = {**tags, **{k: v for k, v in llm_tags.items() if v is not None}}
                tags = merged
        out = {"tags": tags, "scope": tags.get("scope", "all")}
        out.update(add_trace(state, "tag_query", {"tags": tags}))
        return out

    def tool_route_node(state: AgenticRAGState) -> Dict[str, Any]:
        """
        Optional function-calling tool router:
        - LLM chooses which deterministic local tool(s) to call
        - We execute them to get selected chunks
        - If it succeeds, we can skip rewrite/retrieve/select
        """
        if not AGENTIC_USE_TOOL_ROUTER:
            return add_trace(state, "tool_route_skipped", {"enabled": False})
        if getattr(generator, "client", None) is None:
            return add_trace(state, "tool_route_skipped", {"enabled": True, "reason": "no client"})

        from src.tool_router import route_and_select_chunks  # local import to keep dependencies optional

        selected, info = route_and_select_chunks(
            query=state["query"],
            retriever=retriever,
            client=generator.client,
            model=AGENTIC_TOOL_ROUTER_MODEL,
            top_k=cfg.top_k,
            max_calls=int(AGENTIC_TOOL_ROUTER_MAX_CALLS),
        )

        if not selected:
            out = {"tool_routed": False}
            out.update(add_trace(state, "tool_route", {"ok": False, **info}))
            return out

        out = {"selected": [r.to_dict() for r in selected], "tool_routed": True}
        out.update(add_trace(state, "tool_route", {"ok": True, **info}))
        return out

    def rewrite_node(state: AgenticRAGState) -> Dict[str, Any]:
        query = state["query"]
        tags = state.get("tags", {})
        rewrites = build_query_variants(query, tags, max_variants=cfg.max_query_variants)
        out = {"rewritten_queries": rewrites, "top_n_per_query": cfg.top_n_per_query}
        out.update(add_trace(state, "rewrite", {"rewritten_queries": rewrites}))
        return out

    def hyde_node(state: AgenticRAGState) -> Dict[str, Any]:
        if not AGENTIC_USE_HYDE:
            return add_trace(state, "hyde_skipped", {"enabled": False})
        if getattr(generator, "client", None) is None:
            return add_trace(state, "hyde_skipped", {"enabled": True, "reason": "no client"})

        hyde_text = llm_hyde(state["query"], generator.client, model=AGENTIC_HYDE_MODEL)
        if not hyde_text:
            return add_trace(state, "hyde_failed", {})

        rewrites = list(state.get("rewritten_queries", []))
        rewrites.append(hyde_text)
        # keep within max_variants + 1 (HyDE)
        rewrites = rewrites[: cfg.max_query_variants + 1]

        out = {"rewritten_queries": rewrites}
        out.update(add_trace(state, "hyde", {"added": True}))
        return out

    def retrieve_node(state: AgenticRAGState) -> Dict[str, Any]:
        rewrites = state.get("rewritten_queries", [state["query"]])
        top_n = int(state.get("top_n_per_query", cfg.top_n_per_query))

        # Run retrieval per variant (uses our dataset-aware boosts already)
        result_lists: List[List[RetrievalResult]] = []
        for q in rewrites:
            result_lists.append(
                retriever.retrieve(
                    q,
                    k=top_n,
                    use_reranking=USE_RERANKING,
                    expansion_factor=EXPANSION_FACTOR,
                )
            )

        # Fuse with RRF
        fused = rrf_fuse(result_lists, k0=cfg.rrf_k0)

        # Keep best similarity score as a secondary signal
        best_sim: Dict[str, float] = {}
        for results in result_lists:
            for r in results:
                best_sim[r.chunk_id] = max(best_sim.get(r.chunk_id, -1e9), float(r.similarity_score))

        # Scope filtering (website vs course) based on source_file
        scope = state.get("scope", "all")
        filtered: List[Tuple[str, float]] = []
        for cid, rrf_score in fused:
            ch = retriever.chunk_lookup.get(cid)
            if ch is None:
                continue
            if scope == "website" and not _is_website_source(ch.source_file):
                continue
            if scope == "course" and _is_website_source(ch.source_file):
                continue
            filtered.append((cid, rrf_score))

        fused_candidates = [
            {"chunk_id": cid, "rrf_score": float(rrf_score), "best_sim": float(best_sim.get(cid, 0.0))}
            for cid, rrf_score in filtered[: max(cfg.top_k * 10, 50)]
        ]

        out = {"fused_candidates": fused_candidates}
        out.update(
            add_trace(
                state,
                "retrieve_fuse",
                {
                    "n_variants": len(rewrites),
                    "scope": scope,
                    "n_fused": len(fused_candidates),
                    "top_n_per_query": top_n,
                },
            )
        )
        return out

    def select_node(state: AgenticRAGState) -> Dict[str, Any]:
        fused = state.get("fused_candidates", [])
        if not fused:
            return add_trace(state, "select", {"selected": 0})

        # For list-like / filter queries (especially website study-programme pages),
        # avoid MMR diversity as it can hurt Recall@k (we want many similar blocks).
        qn = _normalize_query(state.get("query", ""))
        tags = state.get("tags") or {}
        scope = state.get("scope", "all")
        intents = set(tags.get("intents", []) or [])
        use_mmr = bool(cfg.use_mmr)
        if re.search(r"\blist\b", qn) or any(k in qn for k in ["compulsory courses", "study programme", "study-programme"]):
            use_mmr = False
        if scope == "website" and (("semester" in intents) or ("tuition" in intents) or ("admission" in intents)):
            use_mmr = False
        if scope == "website" and any(k in qn for k in ["taught by", "who teaches"]):
            use_mmr = False

        # Build candidate RetrievalResult list (keep best_sim for ordering)
        # Start from best_sim (dense similarity) to get a reasonable candidate pool.
        fused_sorted = sorted(fused, key=lambda x: (x.get("best_sim", 0.0), x.get("rrf_score", 0.0)), reverse=True)
        candidate_ids = [c["chunk_id"] for c in fused_sorted]
        candidate_scores = [float(c.get("best_sim", 0.0)) for c in fused_sorted]

        # If we have embeddings cache, re-score candidates with the ORIGINAL query embedding.
        # This makes MMR and selection consistent (vs. using prefixed rewrite queries).
        q_emb = None
        if hasattr(retriever, "_embeddings_cache") and hasattr(retriever, "_chunk_id_to_row"):
            try:
                q_emb = retriever.embedding_model.encode_query(state["query"])
            except Exception:
                q_emb = None

        if q_emb is not None:
            rescored = []
            for cid, fallback_sc in zip(candidate_ids, candidate_scores):
                row = retriever._chunk_id_to_row.get(cid)
                if row is None or row < 0 or row >= len(retriever._embeddings_cache):
                    rescored.append(float(fallback_sc))
                else:
                    # Keep `best_sim` as a signal; do not overwrite with raw dot-product.
                    rescored.append(max(float(fallback_sc), float(np.dot(retriever._embeddings_cache[row], q_emb[0]))))
            candidate_scores = rescored

        # Build RetrievalResult objects using retriever chunk lookup
        candidates: List[RetrievalResult] = []
        for rank, cid in enumerate(candidate_ids, 1):
            ch = retriever.chunk_lookup.get(cid)
            if not ch:
                continue
            candidates.append(
                RetrievalResult(
                    chunk_id=cid,
                    text=ch.text,
                    source_title=ch.source_title,
                    source_section=ch.source_section,
                    similarity_score=candidate_scores[rank - 1],
                    rank=rank,
                )
            )

        if cfg.use_dedup:
            candidates = dedup_by_text(candidates)

        # Optional: reuse retriever's built-in diversification knobs (source caps)
        # (These are applied in Retriever.retrieve already; kept here for safety if using fusion.)
        if DIVERSIFY_SOURCES:
            per_source: Dict[str, int] = {}
            diversified: List[RetrievalResult] = []
            for r in candidates:
                per_source.setdefault(r.source_title, 0)
                if per_source[r.source_title] >= max(int(MAX_CHUNKS_PER_SOURCE_TITLE), 1):
                    continue
                diversified.append(r)
                per_source[r.source_title] += 1
            candidates = diversified

        # MMR selection (diversity in context)
        selected_ids = [r.chunk_id for r in candidates[: cfg.top_k]]
        if use_mmr and hasattr(retriever, "_embeddings_cache"):
            q_emb = q_emb or retriever.embedding_model.encode_query(state["query"])

            def get_emb(cid: str) -> Optional[np.ndarray]:
                row = retriever._chunk_id_to_row.get(cid)
                if row is None:
                    return None
                if row < 0 or row >= len(retriever._embeddings_cache):
                    return None
                return retriever._embeddings_cache[row]

            selected_ids = mmr_select(
                query_emb=q_emb,
                candidate_ids=[r.chunk_id for r in candidates],
                candidate_scores=[float(r.similarity_score) for r in candidates],
                get_embedding_by_chunk_id=get_emb,
                k=cfg.top_k,
                lambda_mult=cfg.mmr_lambda,
            )

        # Build final selected list preserving order in selected_ids
        id_to_r = {r.chunk_id: r for r in candidates}
        selected = [id_to_r[cid] for cid in selected_ids if cid in id_to_r]

        # Re-rank positions 1..k
        selected_out = []
        for i, r in enumerate(selected, 1):
            r.rank = i
            selected_out.append(r.to_dict())

        out = {"selected": selected_out}
        out.update(add_trace(state, "select", {"selected": len(selected_out), "mmr": use_mmr, "dedup": cfg.use_dedup}))
        return out

    def context_node(state: AgenticRAGState) -> Dict[str, Any]:
        selected = state.get("selected", [])
        parts: List[str] = []
        for item in selected:
            title = item.get("source_title", "")
            section = item.get("source_section")
            source_info = f"[Source: {title}" + (f" - {section}]" if section else "]")
            parts.append(f"{source_info}\n{item.get('text','')}")
        context = "\n\n---\n\n".join(parts)
        out = {"context": context}
        out.update(add_trace(state, "build_context", {"chars": len(context), "k": len(selected)}))
        return out

    def generate_node(state: AgenticRAGState) -> Dict[str, Any]:
        selected_dicts = state.get("selected", [])

        # Convert back into RetrievalResult for sources (optional)
        selected_rr: List[RetrievalResult] = []
        for item in selected_dicts:
            selected_rr.append(
                RetrievalResult(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    source_title=item["source_title"],
                    source_section=item.get("source_section"),
                    similarity_score=float(item.get("similarity_score", 0.0)),
                    rank=int(item.get("rank", 0)),
                )
            )

        result = generator.generate(
            query=state["query"],
            context=state.get("context", ""),
            include_sources=True,
            retrieved_results=selected_rr,
        )

        out = {"answer": result.get("answer", ""), "sources": result.get("sources", [])}
        out.update(add_trace(state, "generate", {"model": result.get("model"), "num_passages": result.get("num_passages")}))
        return out

    def reflect_node(state: AgenticRAGState) -> Dict[str, Any]:
        # Optional reflection step to reduce hallucinations (Agentic RAG pattern).
        if not AGENTIC_USE_REFLECTION:
            return add_trace(state, "reflect_skipped", {"enabled": False})
        if getattr(generator, "client", None) is None:
            return add_trace(state, "reflect_skipped", {"enabled": True, "reason": "no client"})

        improved = llm_reflect(
            query=state["query"],
            context=state.get("context", ""),
            draft_answer=state.get("answer", ""),
            client=generator.client,
            model=AGENTIC_REFLECTION_MODEL,
        )
        if not improved:
            return add_trace(state, "reflect_failed", {})

        out = {"answer": improved}
        out.update(add_trace(state, "reflect", {"model": AGENTIC_REFLECTION_MODEL}))
        return out

    def judge_node(state: AgenticRAGState) -> Dict[str, Any]:
        if evaluator is None or evaluator.client is None:
            return add_trace(state, "judge_skipped", {"reason": "no evaluator client"})

        answer = state.get("answer", "")
        context = state.get("context", "")
        faith = evaluator.evaluate_faithfulness(answer, context)
        rel = evaluator.evaluate_answer_relevance(state["query"], answer)
        out = {"judge": {"faithfulness": faith, "answer_relevance": rel}}
        out.update(add_trace(state, "judge", {"faithfulness": faith.get("score"), "relevance": rel.get("score")}))
        return out

    graph = StateGraph(AgenticRAGState)
    graph.add_node("tag_query", tag_node)
    graph.add_node("tool_route", tool_route_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("hyde", hyde_node)
    graph.add_node("retrieve_fuse", retrieve_node)
    graph.add_node("select", select_node)
    graph.add_node("build_context", context_node)
    graph.add_node("generate", generate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("judge", judge_node)

    graph.add_edge(START, "tag_query")
    graph.add_edge("tag_query", "tool_route")

    def _after_tool_route(state: AgenticRAGState) -> str:
        return "skip" if state.get("tool_routed") else "continue"

    graph.add_conditional_edges(
        "tool_route",
        _after_tool_route,
        {
            "skip": "build_context",
            "continue": "rewrite",
        },
    )
    graph.add_edge("rewrite", "hyde")
    graph.add_edge("hyde", "retrieve_fuse")
    graph.add_edge("retrieve_fuse", "select")
    graph.add_edge("select", "build_context")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", "reflect")
    graph.add_edge("reflect", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


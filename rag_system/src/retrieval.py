# -*- coding: utf-8-sig -*-
"""
Retrieval Module for RAG System.
Handles query encoding and passage retrieval.

University of Antwerp - Information Retrieval Assignment 3
References:
- Assignment: "Encode the query into an embedding, retrieve top-k most similar passages"
- Karpukhin et al. (2020): Dense Passage Retrieval
- Feedback from IR Assignment 2: "When looking for top-k results, select the top c*k 
  and compute exact distances for the c*k resulting vectors. Then make a top-k out 
  of these c*k candidates." (Candidate Expansion + Refinement strategy)
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    TOP_K,
    DATA_DIR,
    EMBEDDINGS_DIR,
    USE_RERANKING,
    EXPANSION_FACTOR,
    SECTION_AWARE_BOOST,
    SECTION_BOOST_WEIGHT,
    DIVERSIFY_SOURCES,
    MAX_CHUNKS_PER_SOURCE_TITLE,
    LEXICAL_CODE_BOOST,
    LEXICAL_CODE_BOOST_WEIGHT,
)
from src.embedding import EmbeddingModel
from src.indexing import FAISSIndex
from src.chunking import Chunk
from src.structured_index import StructuredIndex


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    chunk_id: str
    text: str
    source_title: str
    source_section: Optional[str]
    similarity_score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_title": self.source_title,
            "source_section": self.source_section,
            "similarity_score": self.similarity_score,
            "rank": self.rank
        }


class Retriever:
    """
    Retrieves relevant passages for a given query.
    
    Pipeline:
    1. Encode query using the same embedding model as documents
    2. Search FAISS index for top-k similar passages
    3. Return ranked results with similarity scores
    """
    
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        index: Optional[FAISSIndex] = None,
        chunks: Optional[List[Chunk]] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: Pre-loaded embedding model (optional)
            index: Pre-loaded FAISS index (optional)
            chunks: Pre-loaded chunks (optional)
        """
        self.embedding_model = embedding_model
        self.index = index
        self.chunks = chunks
        self.chunk_lookup: Dict[str, Chunk] = {}
        self._chunk_id_to_row: Dict[str, int] = {}
        self.structured_index: Optional[StructuredIndex] = None
        
        if chunks:
            self._build_chunk_lookup()
            self._build_structured_index()
    
    def _build_chunk_lookup(self) -> None:
        """Build a lookup dictionary from chunk_id to chunk."""
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in self.chunks}
        self._chunk_id_to_row = {chunk.chunk_id: i for i, chunk in enumerate(self.chunks)}

    def _build_structured_index(self) -> None:
        """
        Build a lightweight structured index from chunks.
        This enables deterministic "entity+field" retrieval in this dataset.
        """
        try:
            if not self.chunks:
                self.structured_index = None
                return
            self.structured_index = StructuredIndex.from_chunks(self.chunks)
        except Exception:
            # Keep retrieval usable even if structured index build fails
            self.structured_index = None

    def _apply_forced_chunks(
        self,
        query: str,
        query_embedding: np.ndarray,
        ranked_pairs: List[tuple[str, float]],
        forced_ids: List[str],
    ) -> List[tuple[str, float]]:
        if not forced_ids:
            return ranked_pairs

        existing = {cid for cid, _ in ranked_pairs}
        # Remove forced ids from the normal list
        tail = [(cid, sc) for cid, sc in ranked_pairs if cid not in set(forced_ids)]

        base_top = ranked_pairs[0][1] if ranked_pairs else 0.0
        forced_pairs: List[tuple[str, float]] = []

        for cid in forced_ids:
            ch = self.chunk_lookup.get(cid)
            if not ch:
                continue
            # Compute a reasonable similarity score if possible
            sc = float(base_top)
            if hasattr(self, "_embeddings_cache") and getattr(self, "_chunk_id_to_row", None):
                row = self._chunk_id_to_row.get(cid)
                if row is not None and 0 <= row < len(self._embeddings_cache):
                    sc = float(np.dot(self._embeddings_cache[row], query_embedding[0]))

            # Apply the same dataset-aware bonuses
            q_lc = self._normalize_query(query)
            sc += self._section_bonus(q_lc, ch.source_section)
            sc += self._lexical_bonus(q_lc, ch.text)

            # Force to the top (cosine sim is typically in [-1,1])
            sc = sc + 1.0
            forced_pairs.append((cid, float(sc)))

        # Keep order of forced_ids but ensure they're sorted (highest first) for sanity
        forced_pairs.sort(key=lambda x: x[1], reverse=True)
        return forced_pairs + tail

    def _apply_course_scoping(
        self,
        query: str,
        ranked_pairs: List[tuple[str, float]],
        forced_ids: List[str],
        course_title: Optional[str],
    ) -> List[tuple[str, float]]:
        """
        If a query is clearly about a specific course, avoid contaminating the top-k
        with other courses' sections (common dense-retrieval failure in this dataset).
        """
        if not course_title:
            return ranked_pairs

        ql = self._normalize_query(query)
        is_course_specific = any(
            k in ql
            for k in [
                "prereq",
                "prerequisite",
                "learning outcome",
                "course content",
                "contents",
                "assessment",
                "exam period",
                "who teaches",
                "taught by",
                "semester",
                "course code",
                "credits",
            ]
        )
        if not is_course_specific:
            return ranked_pairs

        allowed = set(forced_ids)
        for ch in self.chunks or []:
            if str(ch.source_file).lower().endswith(".pdf") and ch.source_title == course_title:
                allowed.add(ch.chunk_id)

        filtered = [(cid, sc) for cid, sc in ranked_pairs if cid in allowed]

        # If we filtered too aggressively, fall back-fill with original ranking
        if len(filtered) < 5:
            seen = {cid for cid, _ in filtered}
            for cid, sc in ranked_pairs:
                if cid in seen:
                    continue
                filtered.append((cid, sc))
                seen.add(cid)
                if len(filtered) >= len(ranked_pairs):
                    break

        return filtered

    def _normalize_query(self, query: str) -> str:
        return " ".join(query.lower().split())

    def _section_bonus(self, query_lc: str, section: Optional[str]) -> float:
        """
        Dataset-aware section boosting.

        The course dataset has stable section headers like:
        - Prerequisites
        - Learning Outcomes
        - Course Contents
        - Assessment method and criteria
        - Study material

        Many questions directly mention these intents; boosting the matching
        section helps retrieval accuracy without changing the embedder.
        """
        if not section:
            return 0.0
        s = section.lower()

        rules = [
            ("prerequisites", ["prereq", "prerequisite", "requirements", "prior knowledge", "before taking"]),
            ("learning outcomes", ["learning outcome", "outcome", "competenc", "after completing", "you will be able"]),
            ("course contents", ["content", "syllabus", "topics", "covers", "what is covered"]),
            ("assessment method and criteria", ["exam", "assessment", "grading", "evaluation", "criteria"]),
            ("study material", ["study material", "textbook", "book", "slides", "literature", "reading"]),
            ("teaching method", ["teaching method", "lectures", "lab", "tutorial", "planned learning"]),
            ("contact information", ["contact", "email", "reach", "office"]),
            ("tutoring", ["tutor", "tutoring", "guidance", "support"]),
            ("is part of the next programmes", ["track", "specialisation", "programme", "program", "part of"]),
        ]

        for section_key, kw in rules:
            if section_key in s and any(k in query_lc for k in kw):
                return float(SECTION_BOOST_WEIGHT)

        return 0.0

    def _lexical_bonus(self, query_lc: str, chunk_text: str) -> float:
        """
        Dataset-aware lexical boost for exact identifiers.

        Why (based on cs-data):
        - Course codes are structured tokens like '2500WETINT'.
        - Dense embeddings sometimes underweight these exact IDs.
        - A small exact-match bonus makes 'course code' queries much more stable.
        """
        if not LEXICAL_CODE_BOOST:
            return 0.0

        import re

        # Extract likely course codes from the query (uppercased pattern)
        q_up = query_lc.upper()
        codes = re.findall(r"\b\d{4}[A-Z]{3,}\b", q_up)
        if not codes:
            return 0.0

        t_up = chunk_text.upper()
        for code in codes:
            if code in t_up:
                return float(LEXICAL_CODE_BOOST_WEIGHT)
        return 0.0

    def _diversify_by_source(self, ranked: List[tuple[str, float]], k: int) -> List[tuple[str, float]]:
        """
        Optional: limit how many chunks come from the same source_title.
        Useful for comparative / multi-hop questions.
        """
        if not DIVERSIFY_SOURCES:
            return ranked[:k]

        max_per_source = max(int(MAX_CHUNKS_PER_SOURCE_TITLE), 1)
        out: List[tuple[str, float]] = []
        per_source = {}

        for cid, score in ranked:
            ch = self.chunk_lookup.get(cid)
            title = ch.source_title if ch else ""
            per_source.setdefault(title, 0)
            if per_source[title] >= max_per_source:
                continue
            out.append((cid, score))
            per_source[title] += 1
            if len(out) >= k:
                break

        return out
    
    def load_components(
        self,
        chunks_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        embeddings_path: Optional[Path] = None
    ) -> None:
        """
        Load retriever components from files.
        
        Args:
            chunks_path: Path to chunks.json
            index_path: Path to FAISS index
            embeddings_path: Path to embeddings (if index needs to be rebuilt)
        """
        # Load embedding model if not provided
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel()
        
        # Load chunks
        if chunks_path and chunks_path.exists():
            print(f"📂 Loading chunks from {chunks_path}")
            with open(chunks_path, 'r', encoding='utf-8-sig') as f:
                chunks_data = json.load(f)
            
            self.chunks = [
                Chunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    source_file=c["source_file"],
                    source_title=c["source_title"],
                    source_section=c.get("source_section"),
                    token_count=c.get("token_count", 0)
                )
                for c in chunks_data
            ]
            self._build_chunk_lookup()
            self._build_structured_index()
            print(f"✅ Loaded {len(self.chunks)} chunks")
        
        # Load embeddings cache for reranking (IR HW2 feedback improvement)
        if embeddings_path and embeddings_path.exists():
            self._embeddings_cache = np.load(embeddings_path)
            print(f"✅ Loaded embeddings cache for reranking: {self._embeddings_cache.shape}")
        
        # Load FAISS index
        if index_path and Path(str(index_path) + '.faiss').exists():
            self.index = FAISSIndex()
            self.index.load(index_path)
        elif embeddings_path and embeddings_path.exists():
            # Rebuild index from embeddings
            print("🔄 Rebuilding index from embeddings...")
            embeddings = self._embeddings_cache if hasattr(self, '_embeddings_cache') else np.load(embeddings_path)
            chunk_ids = [c.chunk_id for c in self.chunks]
            
            self.index = FAISSIndex(dimension=embeddings.shape[1])
            self.index.build_index(embeddings, chunk_ids)
    
    def retrieve(
        self, 
        query: str, 
        k: int = TOP_K,
        return_scores: bool = True,
        use_reranking: Optional[bool] = None,
        expansion_factor: Optional[int] = None,
        use_structured: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k passages for a query with optional candidate expansion.
        
        IMPROVEMENT FROM IR ASSIGNMENT 2 FEEDBACK:
        "When looking for top-k results, select the top c*k and compute exact 
        distances for the c*k resulting vectors. Then make a top-k out of 
        these c*k candidates."
        
        Args:
            query: User query string
            k: Number of passages to retrieve
            return_scores: Whether to include similarity scores
            use_reranking: If True, use candidate expansion + exact reranking
            expansion_factor: Retrieve this many times k candidates, then refine
            
        Returns:
            List of RetrievalResult objects
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded")
        if self.index is None:
            raise ValueError("Index not loaded")
        if not self.chunks:
            raise ValueError("Chunks not loaded")
        
        # Encode query
        query_embedding = self.embedding_model.encode_query(query)

        # Structured forcing (entity+field / study-programme filters)
        forced_ids: List[str] = []
        matched_course_title: Optional[str] = None
        if use_structured and self.structured_index is not None:
            forced_ids = self.structured_index.forced_chunk_ids(query, k=k)
            matched_course_title = self.structured_index.match_course_title(query)
        
        use_reranking_effective = USE_RERANKING if use_reranking is None else use_reranking
        expansion_factor_effective = EXPANSION_FACTOR if expansion_factor is None else expansion_factor

        ranked_pairs: List[tuple[str, float]] = []

        if use_reranking_effective:
            # CANDIDATE EXPANSION STRATEGY (from IR HW2 feedback)
            # Step 1: Retrieve expanded candidate set (c*k)
            expanded_k = min(k * max(int(expansion_factor_effective), 1), self.index.size)
            chunk_ids, approx_scores = self.index.search(query_embedding, k=expanded_k)
            
            # Step 2: Compute exact cosine similarities for candidates
            candidate_embeddings: List[np.ndarray] = []
            valid_chunk_ids: List[str] = []

            if hasattr(self, "_embeddings_cache") and self._chunk_id_to_row:
                for chunk_id in chunk_ids:
                    row = self._chunk_id_to_row.get(chunk_id)
                    if row is None:
                        continue
                    if row < 0 or row >= len(self._embeddings_cache):
                        continue
                    candidate_embeddings.append(self._embeddings_cache[row])
                    valid_chunk_ids.append(chunk_id)
            
            if candidate_embeddings:
                candidate_embeddings = np.array(candidate_embeddings)
                # Exact cosine similarity (dot product for normalized vectors)
                exact_scores = self.embedding_model.compute_similarity(
                    query_embedding, candidate_embeddings
                )

                # Step 3: Re-rank by exact scores (keep the FULL candidate pool)
                # NOTE: We intentionally do NOT slice to top-k yet. We first apply dataset-aware
                # boosting (section/code) on the candidate pool; otherwise boosting cannot
                # promote a relevant chunk that is slightly below top-k by raw similarity.
                ranked_pairs = list(zip(valid_chunk_ids, exact_scores.tolist()))
            else:
                # Fallback to approximate scores
                ranked_pairs = list(zip(chunk_ids, approx_scores))
        else:
            # Standard search without reranking
            chunk_ids, scores = self.index.search(query_embedding, k=k)
            ranked_pairs = list(zip(chunk_ids, scores))
        
        # Dataset-aware post-processing: section boosting + optional diversification
        if SECTION_AWARE_BOOST:
            query_lc = self._normalize_query(query)
            boosted: List[tuple[str, float]] = []
            for cid, sc in ranked_pairs:
                ch = self.chunk_lookup.get(cid)
                bonus = 0.0
                if ch:
                    bonus += self._section_bonus(query_lc, ch.source_section)
                    bonus += self._lexical_bonus(query_lc, ch.text)
                boosted.append((cid, float(sc) + bonus))
            boosted.sort(key=lambda x: x[1], reverse=True)
            ranked_pairs = boosted
        else:
            ranked_pairs = sorted(ranked_pairs, key=lambda x: x[1], reverse=True)

        # Force structured chunks to the top (if available)
        if use_structured:
            ranked_pairs = self._apply_forced_chunks(
                query=query,
                query_embedding=query_embedding,
                ranked_pairs=ranked_pairs,
                forced_ids=forced_ids,
            )

            ranked_pairs = self._apply_course_scoping(
                query=query,
                ranked_pairs=ranked_pairs,
                forced_ids=forced_ids,
                course_title=matched_course_title,
            )

        ranked_pairs = self._diversify_by_source(ranked_pairs, k=k)

        chunk_ids = [cid for cid, _ in ranked_pairs]
        scores = [sc for _, sc in ranked_pairs]

        # Build results
        results = []
        for rank, (chunk_id, score) in enumerate(zip(chunk_ids, scores), 1):
            chunk = self.chunk_lookup.get(chunk_id)
            if chunk:
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    source_title=chunk.source_title,
                    source_section=chunk.source_section,
                    similarity_score=score if return_scores else 0.0,
                    rank=rank
                ))
        
        return results
    
    def retrieve_with_context(
        self, 
        query: str, 
        k: int = TOP_K
    ) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve passages and format them as context for the LLM.
        
        Args:
            query: User query string
            k: Number of passages to retrieve
            
        Returns:
            Tuple of (formatted_context, results)
        """
        results = self.retrieve(query, k=k)
        
        # Format context for LLM
        context_parts = []
        for result in results:
            source_info = f"[Source: {result.source_title}"
            if result.source_section:
                source_info += f" - {result.source_section}"
            source_info += "]"
            
            context_parts.append(f"{source_info}\n{result.text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return context, results
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        k: int = TOP_K
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve passages for multiple queries.
        
        Args:
            queries: List of query strings
            k: Number of passages per query
            
        Returns:
            List of result lists, one per query
        """
        # Encode all queries at once
        query_embeddings = self.embedding_model.encode(queries, show_progress=False)
        
        # Batch search
        batch_results = self.index.batch_search(query_embeddings, k=k)
        
        # Build results
        all_results = []
        for chunk_ids, scores in batch_results:
            query_results = []
            for rank, (chunk_id, score) in enumerate(zip(chunk_ids, scores), 1):
                chunk = self.chunk_lookup.get(chunk_id)
                if chunk:
                    query_results.append(RetrievalResult(
                        chunk_id=chunk_id,
                        text=chunk.text,
                        source_title=chunk.source_title,
                        source_section=chunk.source_section,
                        similarity_score=score,
                        rank=rank
                    ))
            all_results.append(query_results)
        
        return all_results
    
    def get_passage_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get a specific chunk by ID."""
        return self.chunk_lookup.get(chunk_id)


if __name__ == "__main__":
    # Test the retriever
    print("Testing Retriever...")
    
    # Initialize retriever
    retriever = Retriever()
    
    # Try to load existing components
    chunks_path = DATA_DIR / "chunks.json"
    index_path = EMBEDDINGS_DIR / "faiss_index"
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    
    if chunks_path.exists():
        retriever.load_components(
            chunks_path=chunks_path,
            index_path=index_path,
            embeddings_path=embeddings_path
        )
        
        # Test retrieval
        test_queries = [
            "What are the prerequisites for the IoT course?",
            "Who teaches the master thesis?",
            "What programming languages are used in the courses?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            results = retriever.retrieve(query, k=3)
            for result in results:
                print(f"  [{result.rank}] {result.source_title} ({result.similarity_score:.4f})")
                print(f"      {result.text[:100]}...")
    else:
        print("❌ Chunks not found. Run chunking first.")

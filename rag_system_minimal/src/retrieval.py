# -*- coding: utf-8-sig -*-
"""
Retrieval Module (Assignment 4.3 - 20%)

This module implements the retrieval component:
1. Encode the query into an embedding
2. Retrieve the top-k most similar passages
3. Rank results using similarity scores

References:
- Assignment 3: "Encode the query into an embedding. Retrieve the top-k most similar passages. Rank results using similarity scores."
- Karpukhin et al. (2020): Dense Passage Retrieval for Open-Domain Question Answering
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from src.embedding import EmbeddingModel
from src.indexing import FAISSIndex
from src.chunking import Chunk


@dataclass
class RetrievalResult:
    """Represents a retrieved passage with metadata."""
    chunk_id: str
    text: str
    source_title: str
    source_section: str
    similarity_score: float
    rank: int


class Retriever:
    """
    Retrieves top-k most similar passages for a given query.
    
    Steps:
    1. Encode query using the same embedding model
    2. Search FAISS index for top-k results
    3. Return ranked list of passages with similarity scores
    """
    
    def __init__(self, embedding_model: EmbeddingModel, index: FAISSIndex):
        """
        Initialize retriever.
        
        Args:
            embedding_model: Model for encoding queries
            index: FAISS index for similarity search
        """
        self.embedding_model = embedding_model
        self.index = index
        self.chunk_lookup: dict = {}  # chunk_id -> Chunk object
    
    def load_chunks(self, chunks: List[Chunk]) -> None:
        """Load chunks into lookup dictionary."""
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve top-k most similar passages for a query.
        
        Args:
            query: Natural language query
            k: Number of passages to retrieve (default: 5)
            
        Returns:
            List of RetrievalResult objects, ranked by similarity (highest first)
        """
        # Step 1: Encode query
        query_embedding = self.embedding_model.encode_query(query)
        
        # Step 2: Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Step 3: Build results with ranking
        results: List[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(distances, indices), 1):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue
            
            chunk_id = self.index.chunk_ids[idx]
            chunk = self.chunk_lookup.get(chunk_id)
            
            if chunk:
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    source_title=chunk.source_title,
                    source_section=chunk.source_section,
                    similarity_score=float(score),
                    rank=rank
                )
                results.append(result)
        
        return results

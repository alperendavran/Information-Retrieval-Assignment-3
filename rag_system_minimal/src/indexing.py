# -*- coding: utf-8-sig -*-
"""
Indexing Module (Assignment 4.2 - 20%)

This module builds a similarity index using FAISS (Facebook AI Similarity Search).
FAISS provides efficient similarity search for dense vectors.

Index Type: IndexFlatIP (Inner Product)
- For L2-normalized vectors, inner product = cosine similarity
- Exact search (no approximation)
- Suitable for small to medium datasets (< 1M vectors)

References:
- Assignment 3: "Build a similarity index (e.g., cosine similarity, FAISS, or sklearn)"
- Johnson et al. (2019): Billion-scale similarity search with GPUs
"""

import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np
import faiss


class FAISSIndex:
    """
    FAISS-based similarity index for dense vectors.
    
    Uses IndexFlatIP (Inner Product) which computes exact cosine similarity
    for L2-normalized vectors.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.chunk_ids: List[str] = []
    
    def build_index(self, embeddings: np.ndarray, chunk_ids: List[str]) -> None:
        """
        Build the FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of shape (n_vectors, dimension)
                       Should be L2-normalized for cosine similarity
            chunk_ids: List of chunk IDs corresponding to each embedding
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} "
                f"does not match index dimension {self.dimension}"
            )
        
        # Create IndexFlatIP (Inner Product) for cosine similarity
        # (Inner product = cosine similarity when vectors are L2-normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add vectors to index
        self.index.add(embeddings.astype('float32'))
        self.chunk_ids = chunk_ids
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int) -> tuple:
        """
        Search for top-k most similar vectors.
        
        Args:
            query_embedding: Query vector of shape (1, dimension)
            k: Number of top results to return
            
        Returns:
            Tuple of (distances, indices)
            - distances: cosine similarity scores (higher is better)
            - indices: indices into chunk_ids list
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Ensure query is float32 and has correct shape
        query = query_embedding.astype('float32')
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query, min(k, self.index.ntotal))
        
        return distances[0], indices[0]
    
    def save(self, index_path: Path) -> None:
        """Save index and chunk IDs to disk."""
        if self.index is None:
            raise ValueError("Index not built. Cannot save.")
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path) + '.faiss')
        
        # Save chunk IDs
        with open(index_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.chunk_ids, f)
        
        print(f"Saved index to {index_path}")
    
    def load(self, index_path: Path) -> None:
        """Load index and chunk IDs from disk."""
        # Load FAISS index
        self.index = faiss.read_index(str(index_path) + '.faiss')
        
        # Load chunk IDs
        with open(index_path.with_suffix('.pkl'), 'rb') as f:
            self.chunk_ids = pickle.load(f)
        
        self.dimension = self.index.d
        print(f"Loaded index with {self.index.ntotal} vectors")

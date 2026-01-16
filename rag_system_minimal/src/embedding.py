# -*- coding: utf-8-sig -*-
"""
Embedding Module (Assignment 4.2 - 20%)

This module uses a pretrained sentence embedding model to compute embeddings
for all passages. The model runs locally (no API calls).

Model Choice: sentence-transformers/all-MiniLM-L6-v2
- Dimension: 384
- Fast inference, good quality for semantic similarity
- Runs locally (no external API required)
- Suitable for cosine similarity search

References:
- Assignment 3: "Use a pretrained sentence embedding model"
- Reimers & Gurevych (2019): Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wraps a pretrained sentence embedding model for encoding documents and queries.
    
    The model is loaded once and reused for all encoding operations.
    Embeddings are L2-normalized for cosine similarity computation.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
                        (default: all-MiniLM-L6-v2, 384 dimensions)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of documents into embeddings.
        
        Args:
            texts: List of document texts to encode
            
        Returns:
            numpy array of shape (n_documents, embedding_dim)
            Embeddings are L2-normalized for cosine similarity.
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
            show_progress_bar=True
        )
        return np.array(embeddings)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query into an embedding.
        
        Args:
            query: Query text to encode
            
        Returns:
            numpy array of shape (1, embedding_dim)
            Embedding is L2-normalized for cosine similarity.
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        return np.array(embedding)
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        """Save embeddings to a .npy file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)
        print(f"Saved embeddings to {output_path}")

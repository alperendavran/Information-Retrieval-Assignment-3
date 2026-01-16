# -*- coding: utf-8-sig -*-
"""
Embedding Module for RAG System.
Computes embeddings for passages using sentence transformers.

University of Antwerp - Information Retrieval Assignment 3
References:
- Lewis et al. (2020): Dense Passage Retrieval
- Karpukhin et al. (2020): DPR for open-domain QA
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, EMBEDDINGS_DIR


class EmbeddingModel:
    """
    Wrapper for sentence transformer embedding models.
    
    Supported models (all run locally):
    - sentence-transformers/all-MiniLM-L6-v2 (384d, fast, good baseline)
    - sentence-transformers/all-mpnet-base-v2 (768d, more accurate)
    - BAAI/bge-small-en-v1.5 (384d, SOTA performance)
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"🔄 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension from a test embedding
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.dimension = len(test_embedding)
            print(f"✅ Model loaded. Embedding dimension: {self.dimension}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            
        Returns:
            Numpy array of embeddings (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize  # L2 normalize for cosine similarity
        )
        
        return embeddings
    
    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query.
        
        Args:
            query: Query text
            normalize: Whether to L2-normalize
            
        Returns:
            Query embedding (1, dimension)
        """
        return self.encode([query], show_progress=False, normalize=normalize)
    
    def encode_documents(
        self, 
        documents: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode multiple documents.
        
        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize
            
        Returns:
            Document embeddings (n_docs, dimension)
        """
        print(f"🔄 Encoding {len(documents)} documents...")
        embeddings = self.encode(
            documents, 
            batch_size=batch_size, 
            show_progress=True,
            normalize=normalize
        )
        print(f"✅ Encoded {len(embeddings)} documents")
        return embeddings
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        output_path: Path
    ) -> None:
        """Save embeddings to numpy file."""
        np.save(output_path, embeddings)
        print(f"💾 Saved embeddings to {output_path}")
    
    def load_embeddings(self, input_path: Path) -> np.ndarray:
        """Load embeddings from numpy file."""
        embeddings = np.load(input_path)
        print(f"📂 Loaded embeddings with shape {embeddings.shape}")
        return embeddings
    
    def compute_similarity(
        self, 
        query_embedding: np.ndarray, 
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        If embeddings are L2-normalized, dot product = cosine similarity.
        
        Args:
            query_embedding: Query embedding (1, d) or (d,)
            document_embeddings: Document embeddings (n, d)
            
        Returns:
            Similarity scores (n,)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Dot product of normalized vectors = cosine similarity
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        return similarities


if __name__ == "__main__":
    # Test the embedding model
    model = EmbeddingModel()
    
    # Test encoding
    test_texts = [
        "What are the prerequisites for the IoT course?",
        "This course covers Internet of Things concepts.",
        "Machine learning is about training models on data."
    ]
    
    embeddings = model.encode(test_texts, show_progress=False)
    print(f"\nEmbedding shape: {embeddings.shape}")
    
    # Test similarity
    query = "What do I need to know before taking IoT?"
    query_emb = model.encode_query(query)
    
    similarities = model.compute_similarity(query_emb, embeddings)
    print(f"\nSimilarities to '{query}':")
    for text, sim in zip(test_texts, similarities):
        print(f"  {sim:.4f}: {text[:50]}...")

# -*- coding: utf-8-sig -*-
"""
Unit Tests for RAG System.
University of Antwerp - Information Retrieval Assignment 3

Run with: pytest tests/test_rag.py -v
"""

import pytest
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import DocumentChunker, Chunk
from src.embedding import EmbeddingModel
from src.indexing import FAISSIndex
from src.retrieval import Retriever, RetrievalResult
from src.evaluation import RAGEvaluator, EvaluationQuestion


class TestDocumentChunker:
    """Tests for DocumentChunker class."""
    
    def test_initialization(self):
        """Test chunker initialization with default parameters."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 250
        assert chunker.overlap_ratio == 0.15
        assert chunker.min_chunk_size == 50
    
    def test_count_tokens(self):
        """Test token counting."""
        chunker = DocumentChunker()
        
        # Simple test
        text = "Hello world"
        tokens = chunker.count_tokens(text)
        assert tokens > 0
        assert tokens < 10
        
        # Longer text
        long_text = "This is a longer text " * 50
        long_tokens = chunker.count_tokens(long_text)
        assert long_tokens > tokens
    
    def test_split_into_sentences(self):
        """Test sentence splitting."""
        chunker = DocumentChunker()
        
        text = "First sentence. Second sentence! Third sentence?"
        sentences = chunker.split_into_sentences(text)
        
        assert len(sentences) == 3
        assert "First" in sentences[0]
        assert "Second" in sentences[1]
        assert "Third" in sentences[2]
    
    def test_chunk_by_tokens(self):
        """Test chunking by token count."""
        chunker = DocumentChunker(chunk_size=50, min_chunk_size=10)
        
        text = "This is sentence one. This is sentence two. This is sentence three. " * 5
        chunks = chunker.chunk_by_tokens(
            text=text,
            source_file="test.json",
            source_title="Test Document"
        )
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.token_count >= chunker.min_chunk_size
            assert isinstance(chunk, Chunk)
    
    def test_chunk_dataclass(self):
        """Test Chunk dataclass."""
        chunk = Chunk(
            chunk_id="test_1",
            text="Test content",
            source_file="test.json",
            source_title="Test",
            source_section="Section 1",
            token_count=5
        )
        
        d = chunk.to_dict()
        assert d["chunk_id"] == "test_1"
        assert d["text"] == "Test content"
        assert d["source_section"] == "Section 1"


class TestEmbeddingModel:
    """Tests for EmbeddingModel class."""
    
    @pytest.fixture
    def model(self):
        """Create embedding model instance."""
        return EmbeddingModel()
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.model is not None
        assert model.dimension == 384  # MiniLM default
    
    def test_encode_single(self, model):
        """Test encoding a single text."""
        text = "This is a test sentence."
        embedding = model.encode(text, show_progress=False)
        
        assert embedding.shape == (1, 384)
        assert np.isfinite(embedding).all()
    
    def test_encode_multiple(self, model):
        """Test encoding multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = model.encode(texts, show_progress=False)
        
        assert embeddings.shape == (3, 384)
        assert np.isfinite(embeddings).all()
    
    def test_encode_query(self, model):
        """Test query encoding."""
        query = "What are the prerequisites?"
        embedding = model.encode_query(query)
        
        assert embedding.shape == (1, 384)
    
    def test_normalization(self, model):
        """Test that embeddings are L2 normalized."""
        text = "Test sentence"
        embedding = model.encode(text, normalize=True, show_progress=False)
        
        # L2 norm should be 1.0
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)
    
    def test_similarity(self, model):
        """Test similarity computation."""
        texts = [
            "What is machine learning?",
            "Machine learning is a type of AI.",
            "The weather is nice today."
        ]
        embeddings = model.encode(texts, show_progress=False)
        
        query = model.encode_query("Tell me about machine learning")
        similarities = model.compute_similarity(query, embeddings)
        
        # First two should be more similar than the third
        assert similarities[0] > similarities[2]
        assert similarities[1] > similarities[2]


class TestFAISSIndex:
    """Tests for FAISSIndex class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample embeddings and chunk IDs."""
        n_docs = 50
        dim = 384
        embeddings = np.random.randn(n_docs, dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        chunk_ids = [f"chunk_{i}" for i in range(n_docs)]
        return embeddings, chunk_ids
    
    def test_initialization(self):
        """Test index initialization."""
        index = FAISSIndex(dimension=384, index_type="flat")
        assert index.dimension == 384
        assert index.index_type == "flat"
        assert index.index is None
    
    def test_build_index(self, sample_data):
        """Test building the index."""
        embeddings, chunk_ids = sample_data
        
        index = FAISSIndex(dimension=384)
        index.build_index(embeddings, chunk_ids)
        
        assert index.index is not None
        assert index.size == 50
        assert len(index.chunk_ids) == 50
    
    def test_search(self, sample_data):
        """Test search functionality."""
        embeddings, chunk_ids = sample_data
        
        index = FAISSIndex(dimension=384)
        index.build_index(embeddings, chunk_ids)
        
        # Search with first document embedding
        query = embeddings[0:1]
        result_ids, scores = index.search(query, k=5)
        
        assert len(result_ids) == 5
        assert len(scores) == 5
        # First result should be the query itself
        assert result_ids[0] == "chunk_0"
        assert scores[0] > 0.99  # Should be very close to 1.0
    
    def test_batch_search(self, sample_data):
        """Test batch search."""
        embeddings, chunk_ids = sample_data
        
        index = FAISSIndex(dimension=384)
        index.build_index(embeddings, chunk_ids)
        
        queries = embeddings[:3]
        results = index.batch_search(queries, k=3)
        
        assert len(results) == 3
        for result_ids, scores in results:
            assert len(result_ids) == 3
            assert len(scores) == 3
    
    def test_save_load(self, sample_data, tmp_path):
        """Test saving and loading index."""
        embeddings, chunk_ids = sample_data
        
        index = FAISSIndex(dimension=384)
        index.build_index(embeddings, chunk_ids)
        
        # Save
        save_path = tmp_path / "test_index"
        index.save(save_path)
        
        # Load
        new_index = FAISSIndex()
        new_index.load(save_path)
        
        assert new_index.size == 50
        assert len(new_index.chunk_ids) == 50
        
        # Verify search works on loaded index
        query = embeddings[0:1]
        result_ids, _ = new_index.search(query, k=1)
        assert result_ids[0] == "chunk_0"


class TestRetriever:
    """Tests for Retriever class."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks."""
        return [
            Chunk(
                chunk_id=f"chunk_{i}",
                text=f"This is test content for chunk {i}. It contains some information.",
                source_file="test.json",
                source_title=f"Test Document {i}",
                source_section="Section",
                token_count=20
            )
            for i in range(10)
        ]
    
    def test_initialization(self, sample_chunks):
        """Test retriever initialization."""
        retriever = Retriever(chunks=sample_chunks)
        
        assert len(retriever.chunks) == 10
        assert len(retriever.chunk_lookup) == 10
        assert "chunk_0" in retriever.chunk_lookup
    
    def test_get_passage_by_id(self, sample_chunks):
        """Test getting passage by ID."""
        retriever = Retriever(chunks=sample_chunks)
        
        chunk = retriever.get_passage_by_id("chunk_5")
        assert chunk is not None
        assert chunk.chunk_id == "chunk_5"
        
        missing = retriever.get_passage_by_id("nonexistent")
        assert missing is None


class TestEvaluator:
    """Tests for RAGEvaluator class."""
    
    def test_recall_at_k(self):
        """Test Recall@k calculation."""
        evaluator = RAGEvaluator()
        
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["b", "d", "f"]
        
        recall = evaluator.recall_at_k(retrieved, relevant, k=5)
        # 2 out of 3 relevant documents retrieved
        assert recall == pytest.approx(2/3)
        
        recall_at_2 = evaluator.recall_at_k(retrieved, relevant, k=2)
        # Only 1 relevant in top-2
        assert recall_at_2 == pytest.approx(1/3)
    
    def test_mrr(self):
        """Test Mean Reciprocal Rank."""
        evaluator = RAGEvaluator()
        
        # First relevant at position 1
        mrr1 = evaluator.mrr(["a", "b", "c"], ["a"])
        assert mrr1 == 1.0
        
        # First relevant at position 2
        mrr2 = evaluator.mrr(["a", "b", "c"], ["b"])
        assert mrr2 == 0.5
        
        # First relevant at position 3
        mrr3 = evaluator.mrr(["a", "b", "c"], ["c"])
        assert mrr3 == pytest.approx(1/3)
        
        # No relevant found
        mrr0 = evaluator.mrr(["a", "b", "c"], ["x", "y"])
        assert mrr0 == 0.0
    
    def test_precision_at_k(self):
        """Test Precision@k calculation."""
        evaluator = RAGEvaluator()
        
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["b", "d"]
        
        precision = evaluator.precision_at_k(retrieved, relevant, k=5)
        assert precision == pytest.approx(2/5)
        
        precision_at_2 = evaluator.precision_at_k(retrieved, relevant, k=2)
        assert precision_at_2 == pytest.approx(1/2)
    
    def test_f1_at_k(self):
        """Test F1@k calculation (from IR HW2)."""
        evaluator = RAGEvaluator()
        
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["b", "d"]
        
        f1 = evaluator.f1_at_k(retrieved, relevant, k=5)
        # P@5 = 2/5, R@5 = 2/2 = 1.0
        # F1 = 2 * (2/5 * 1) / (2/5 + 1) = 2 * 0.4 / 1.4 = 0.8/1.4
        p = 2/5
        r = 2/2
        expected_f1 = 2 * p * r / (p + r)
        assert f1 == pytest.approx(expected_f1)
    
    def test_average_precision(self):
        """Test Average Precision calculation (from IR HW2)."""
        evaluator = RAGEvaluator()
        
        # Perfect ranking: all relevant at top
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b"]
        ap = evaluator.average_precision(retrieved, relevant)
        # P@1 = 1, P@2 = 1, AP = (1 + 1)/2 = 1.0
        assert ap == 1.0
        
        # Imperfect ranking
        retrieved2 = ["x", "a", "y", "b", "z"]
        ap2 = evaluator.average_precision(retrieved2, relevant)
        # P@2 = 1/2, P@4 = 2/4, AP = (0.5 + 0.5)/2 = 0.5
        assert ap2 == pytest.approx(0.5)
    
    def test_ndcg_at_k(self):
        """Test nDCG@k calculation (from IR HW2)."""
        evaluator = RAGEvaluator()
        
        # Perfect ranking
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b"]
        ndcg = evaluator.ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg == 1.0  # All relevant at top
        
        # Imperfect ranking
        retrieved2 = ["c", "a", "b"]
        ndcg2 = evaluator.ndcg_at_k(retrieved2, relevant, k=3)
        assert ndcg2 < 1.0  # Relevant not at top
    
    def test_precision_recall_curve(self):
        """Test PR curve computation (from IR HW2)."""
        evaluator = RAGEvaluator()
        
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = ["a", "b", "c"]
        
        recalls, precisions = evaluator.compute_precision_recall_curve(
            retrieved, relevant, max_k=5
        )
        
        assert len(recalls) == 5
        assert len(precisions) == 5
        # At k=1: recall=1/3, precision=1/1
        assert recalls[0] == pytest.approx(1/3)
        assert precisions[0] == 1.0
    
    def test_evaluation_question(self):
        """Test EvaluationQuestion dataclass."""
        question = EvaluationQuestion(
            question="Test question?",
            expected_answer="Test answer",
            relevant_chunks=["chunk_1", "chunk_2"],
            category="test"
        )
        
        d = question.to_dict()
        assert d["question"] == "Test question?"
        assert d["category"] == "test"
        assert len(d["relevant_chunks"]) == 2
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics computation (from IR HW2)."""
        evaluator = RAGEvaluator()
        
        metrics = evaluator.compute_efficiency_metrics(
            n_candidates=100,
            n_total=1000,
            query_time=0.01,
            baseline_time=0.1
        )
        
        assert metrics["candidate_ratio"] == 0.1
        assert metrics["speedup"] == 10.0
        assert metrics["query_time_ms"] == 10.0


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_chunking_to_embedding_pipeline(self):
        """Test chunking and embedding together."""
        # Create simple test data
        chunker = DocumentChunker(chunk_size=50, min_chunk_size=10)
        text = "This is test content. " * 10
        
        chunks = chunker.chunk_by_tokens(
            text=text,
            source_file="test.json",
            source_title="Test"
        )
        
        if chunks:
            # Embed chunks
            model = EmbeddingModel()
            texts = [c.text for c in chunks]
            embeddings = model.encode(texts, show_progress=False)
            
            assert embeddings.shape[0] == len(chunks)
            assert embeddings.shape[1] == 384
    
    def test_embedding_to_retrieval_pipeline(self):
        """Test embedding and retrieval together."""
        # Create and embed sample documents
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "The weather forecast predicts rain tomorrow.",
            "Natural language processing handles text data.",
            "Computer vision analyzes images and videos."
        ]
        
        model = EmbeddingModel()
        embeddings = model.encode(docs, show_progress=False)
        
        # Build index
        chunk_ids = [f"doc_{i}" for i in range(len(docs))]
        index = FAISSIndex(dimension=384)
        index.build_index(embeddings, chunk_ids)
        
        # Search
        query = "Tell me about AI and machine learning"
        query_emb = model.encode_query(query)
        result_ids, scores = index.search(query_emb, k=2)
        
        # First two should be ML/AI related
        assert "doc_0" in result_ids or "doc_1" in result_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

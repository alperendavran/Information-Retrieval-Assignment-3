# -*- coding: utf-8-sig -*-
"""
FAISS Indexing Module for RAG System.
Builds and manages vector indices for fast similarity search.

University of Antwerp - Information Retrieval Assignment 3
References:
- Lecture slides: "ANN index; e.g. FAISS"
- Lewis et al. (2020): Using FAISS with HNSW for fast retrieval
- IR HW2 Feedback: FAISS-LSH uses single-table scheme, different from classical LSH
- IR HW2: IVF with proper nprobe tuning for better recall
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import EMBEDDING_DIMENSION, TOP_K


class FAISSIndex:
    """
    FAISS-based vector index for efficient similarity search.
    
    Supports multiple index types (enhanced based on IR HW2 experience):
    
    | Type     | Search  | Memory   | Best For                |
    |----------|---------|----------|-------------------------|
    | flat     | Exact   | High     | Small datasets (<10K)   |
    | ivf      | Approx  | Medium   | Medium datasets (10K-1M)|
    | ivf_pq   | Approx  | Low      | Large datasets (>1M)    |
    | hnsw     | Approx  | Medium   | High recall requirement |
    
    For the assignment dataset (~500 chunks), flat index is sufficient.
    
    INCREMENTAL UPDATES (from IR HW1 lessons):
    - add_vectors(): Add new documents without full rebuild
    - remove_vectors(): Remove documents by ID
    - update_vectors(): Update existing documents
    - Uses IndexIDMap for efficient ID management
    """
    
    def __init__(
        self, 
        dimension: int = EMBEDDING_DIMENSION,
        index_type: str = "flat",
        nprobe: int = 10,  # Number of clusters to search (for IVF)
        use_id_map: bool = True  # Enable ID-based operations (from HW1)
    ):
        """
        Initialize the FAISS index.
        
        Args:
            dimension: Embedding dimension
            index_type: "flat", "ivf", "ivf_pq", or "hnsw"
            nprobe: Number of clusters to probe during search (IVF only)
                    Higher = better recall, slower search
            use_id_map: If True, wrap index with IndexIDMap for updates
                       (Lesson from HW1: enables incremental updates)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nprobe = nprobe
        self.use_id_map = use_id_map
        self.index = None
        self.chunk_ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}  # chunk_id -> internal FAISS id
        self.idx_to_id: Dict[int, str] = {}  # internal FAISS id -> chunk_id
        self.next_id: int = 0  # Counter for internal IDs
        self.is_trained = False
        self.index_params: Dict[str, Any] = {}
        
        # Auxiliary index for buffered updates (HW1 lesson: batch updates)
        self._aux_embeddings: List[np.ndarray] = []
        self._aux_ids: List[str] = []
        self._aux_buffer_size: int = 100  # Flush after this many additions
        
    def build_index(
        self, 
        embeddings: np.ndarray,
        chunk_ids: List[str],
        nlist: int = None,  # Number of clusters for IVF (auto-calculated if None)
        m_pq: int = 8,      # Number of subquantizers for PQ
        nbits: int = 8,     # Bits per subquantizer (PQ)
        M_hnsw: int = 32,   # Number of connections per node (HNSW)
        efConstruction: int = 200  # Construction time parameter (HNSW)
    ) -> None:
        """
        Build the FAISS index from embeddings.
        
        Args:
            embeddings: Document embeddings (n, d)
            chunk_ids: List of chunk IDs corresponding to embeddings
            nlist: Number of clusters for IVF (auto: sqrt(n))
            m_pq: Number of subquantizers for Product Quantization
            nbits: Bits per subquantizer code
            M_hnsw: HNSW connections per node (higher = better recall)
            efConstruction: HNSW construction parameter
        """
        n_vectors, dim = embeddings.shape
        assert dim == self.dimension, f"Expected dimension {self.dimension}, got {dim}"
        assert len(chunk_ids) == n_vectors, "Mismatch between embeddings and chunk_ids"
        
        self.chunk_ids = chunk_ids
        embeddings = embeddings.astype(np.float32)
        
        # Auto-calculate nlist based on dataset size (rule of thumb: sqrt(n))
        if nlist is None:
            nlist = max(int(np.sqrt(n_vectors)), 1)
        
        # Store parameters for reference
        self.index_params = {
            'n_vectors': n_vectors,
            'nlist': nlist,
            'm_pq': m_pq,
            'M_hnsw': M_hnsw
        }
        
        if self.index_type == "flat":
            # Exact search - best for small datasets
            # Uses Inner Product (dot product = cosine for normalized vectors)
            base_index = faiss.IndexFlatIP(self.dimension)
            
            if self.use_id_map:
                # Wrap with IndexIDMap for incremental updates (HW1 lesson)
                self.index = faiss.IndexIDMap(base_index)
                internal_ids = np.arange(n_vectors, dtype=np.int64)
                self.index.add_with_ids(embeddings, internal_ids)
                
                # Build ID mappings
                for i, chunk_id in enumerate(chunk_ids):
                    self.id_to_idx[chunk_id] = i
                    self.idx_to_id[i] = chunk_id
                self.next_id = n_vectors
            else:
                self.index = base_index
                self.index.add(embeddings)
            
            self.is_trained = True
            
        elif self.index_type == "ivf":
            # IVF (Inverted File) - cluster-based approximate search
            # Good balance of speed and accuracy
            quantizer = faiss.IndexFlatIP(self.dimension)
            actual_nlist = min(nlist, n_vectors // 10 + 1)
            
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.dimension, 
                actual_nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = self.nprobe  # Set search-time parameter
            self.is_trained = True
            self.index_params['actual_nlist'] = actual_nlist
            
        elif self.index_type == "ivf_pq":
            # IVF with Product Quantization - memory efficient
            # From IR HW2: PQ compresses vectors into codes
            quantizer = faiss.IndexFlatIP(self.dimension)
            actual_nlist = min(nlist, n_vectors // 10 + 1)
            
            # Ensure dimension is divisible by m_pq
            actual_m = min(m_pq, self.dimension)
            while self.dimension % actual_m != 0 and actual_m > 1:
                actual_m -= 1
            
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                actual_nlist,
                actual_m,  # Number of subquantizers
                nbits      # Bits per code
            )
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = self.nprobe
            self.is_trained = True
            self.index_params['actual_m_pq'] = actual_m
            
        elif self.index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World)
            # Best recall for approximate search, graph-based
            self.index = faiss.IndexHNSWFlat(self.dimension, M_hnsw)
            self.index.hnsw.efConstruction = efConstruction
            self.index.hnsw.efSearch = max(self.nprobe * 4, 64)  # Search-time param
            self.index.add(embeddings)
            self.is_trained = True
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}. "
                           f"Supported: flat, ivf, ivf_pq, hnsw")
        
        print(f"✅ Built {self.index_type} index with {self.index.ntotal} vectors")
        if self.index_type in ["ivf", "ivf_pq"]:
            print(f"   nlist={self.index_params.get('actual_nlist', nlist)}, nprobe={self.nprobe}")
        elif self.index_type == "hnsw":
            print(f"   M={M_hnsw}, efConstruction={efConstruction}")
    
    def set_nprobe(self, nprobe: int) -> None:
        """
        Update nprobe parameter for IVF indices.
        Higher nprobe = better recall but slower.
        
        IR HW2 Feedback: "More combinations of parameters could have been tried"
        """
        self.nprobe = nprobe
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
            print(f"📊 Updated nprobe to {nprobe}")
    
    def set_ef_search(self, ef: int) -> None:
        """Update efSearch parameter for HNSW index."""
        if self.index_type == "hnsw":
            self.index.hnsw.efSearch = ef
            print(f"📊 Updated efSearch to {ef}")
    
    # =========================================================================
    # INCREMENTAL UPDATE METHODS (Lessons from IR HW1)
    # =========================================================================
    # HW1 Feedback: "Logarithmic merge would have been better"
    # HW1 Feedback: "Documents stored in auxiliary index, merged when overflow"
    # 
    # For vector indices, we implement:
    # - add_vectors(): Add new vectors without full rebuild
    # - remove_vectors(): Remove by ID (requires IndexIDMap)
    # - update_vectors(): Remove + Add
    # - flush_auxiliary(): Batch merge auxiliary index
    # =========================================================================
    
    def add_vectors(
        self, 
        embeddings: np.ndarray, 
        chunk_ids: List[str],
        immediate: bool = False
    ) -> None:
        """
        Add new vectors incrementally without full rebuild.
        
        HW1 Lesson: Use auxiliary index + periodic merge for efficiency.
        
        Args:
            embeddings: New document embeddings (n, d)
            chunk_ids: New chunk IDs
            immediate: If True, add to main index immediately
                      If False, buffer in auxiliary index (more efficient)
        """
        if not self.use_id_map:
            raise ValueError("Incremental updates require use_id_map=True")
        
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        embeddings = embeddings.astype(np.float32)
        n_new = len(chunk_ids)
        
        if immediate:
            # Add directly to main index
            internal_ids = np.arange(self.next_id, self.next_id + n_new, dtype=np.int64)
            self.index.add_with_ids(embeddings, internal_ids)
            
            for i, chunk_id in enumerate(chunk_ids):
                internal_id = self.next_id + i
                self.id_to_idx[chunk_id] = internal_id
                self.idx_to_id[internal_id] = chunk_id
                self.chunk_ids.append(chunk_id)
            
            self.next_id += n_new
            print(f"✅ Added {n_new} vectors immediately. Total: {self.size}")
        else:
            # Buffer in auxiliary index (HW1 pattern)
            for i, chunk_id in enumerate(chunk_ids):
                self._aux_embeddings.append(embeddings[i])
                self._aux_ids.append(chunk_id)
            
            print(f"📦 Buffered {n_new} vectors. Buffer size: {len(self._aux_ids)}")
            
            # Auto-flush if buffer is full
            if len(self._aux_ids) >= self._aux_buffer_size:
                self.flush_auxiliary()
    
    def flush_auxiliary(self) -> None:
        """
        Merge auxiliary index into main index.
        
        HW1 Lesson: Batch merge is more efficient than individual adds.
        """
        if not self._aux_embeddings:
            return
        
        n_flush = len(self._aux_ids)
        embeddings = np.array(self._aux_embeddings, dtype=np.float32)
        internal_ids = np.arange(self.next_id, self.next_id + n_flush, dtype=np.int64)
        
        self.index.add_with_ids(embeddings, internal_ids)
        
        for i, chunk_id in enumerate(self._aux_ids):
            internal_id = self.next_id + i
            self.id_to_idx[chunk_id] = internal_id
            self.idx_to_id[internal_id] = chunk_id
            self.chunk_ids.append(chunk_id)
        
        self.next_id += n_flush
        
        # Clear auxiliary buffer
        self._aux_embeddings = []
        self._aux_ids = []
        
        print(f"🔄 Flushed {n_flush} vectors. Total: {self.size}")
    
    def remove_vectors(self, chunk_ids_to_remove: List[str]) -> int:
        """
        Remove vectors by chunk ID.
        
        Note: FAISS IndexIDMap supports remove_ids for some index types.
        For flat index, we need to rebuild (mark as deleted + periodic rebuild).
        
        Args:
            chunk_ids_to_remove: List of chunk IDs to remove
            
        Returns:
            Number of vectors removed
        """
        if not self.use_id_map:
            raise ValueError("Remove requires use_id_map=True")
        
        # Get internal IDs to remove
        internal_ids_to_remove = []
        for chunk_id in chunk_ids_to_remove:
            if chunk_id in self.id_to_idx:
                internal_ids_to_remove.append(self.id_to_idx[chunk_id])
        
        if not internal_ids_to_remove:
            return 0
        
        # Try to remove from FAISS (works for some index types)
        try:
            ids_array = np.array(internal_ids_to_remove, dtype=np.int64)
            n_removed = self.index.remove_ids(ids_array)
            
            # Update mappings
            for chunk_id in chunk_ids_to_remove:
                if chunk_id in self.id_to_idx:
                    internal_id = self.id_to_idx[chunk_id]
                    del self.id_to_idx[chunk_id]
                    if internal_id in self.idx_to_id:
                        del self.idx_to_id[internal_id]
                    if chunk_id in self.chunk_ids:
                        self.chunk_ids.remove(chunk_id)
            
            print(f"🗑️ Removed {n_removed} vectors. Total: {self.size}")
            return n_removed
            
        except RuntimeError as e:
            # Some index types don't support remove
            print(f"⚠️ Remove not supported for {self.index_type}: {e}")
            print("   Consider rebuild with filtered data.")
            return 0
    
    def update_vectors(
        self, 
        embeddings: np.ndarray, 
        chunk_ids: List[str]
    ) -> None:
        """
        Update existing vectors (remove old + add new).
        
        HW1 Lesson: Updates should trigger norm/weight recomputation.
        For embeddings, this means re-embedding the updated documents.
        
        Args:
            embeddings: Updated embeddings
            chunk_ids: Chunk IDs to update
        """
        # Remove old versions
        self.remove_vectors(chunk_ids)
        
        # Add updated versions
        self.add_vectors(embeddings, chunk_ids, immediate=True)
        
        print(f"🔄 Updated {len(chunk_ids)} vectors")
    
    def needs_rebuild(self, fragmentation_threshold: float = 0.3) -> bool:
        """
        Check if index needs rebuild due to fragmentation.
        
        HW1 Lesson: After many updates, index may become fragmented.
        Periodic rebuild improves performance.
        
        Args:
            fragmentation_threshold: Rebuild if deleted/total > threshold
            
        Returns:
            True if rebuild recommended
        """
        if not self.chunk_ids:
            return False
        
        # Estimate fragmentation (internal IDs vs active IDs)
        active_count = len(self.id_to_idx)
        total_allocated = self.next_id
        
        if total_allocated == 0:
            return False
        
        fragmentation = 1 - (active_count / total_allocated)
        
        if fragmentation > fragmentation_threshold:
            print(f"⚠️ Index fragmentation: {fragmentation:.1%} > {fragmentation_threshold:.1%}")
            return True
        
        return False
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = TOP_K
    ) -> Tuple[List[str], List[float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embedding: Query embedding (1, d) or (d,)
            k: Number of results to return
            
        Returns:
            Tuple of (chunk_ids, similarity_scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Map indices to chunk IDs
        result_ids = []
        result_scores = []
        
        for idx, score in zip(indices[0], scores[0]):
            idx = int(idx)
            if idx < 0:  # -1 means no result
                continue

            # If we have a stable ID mapping (IndexIDMap mode), use it.
            if self.idx_to_id:
                chunk_id = self.idx_to_id.get(idx)
                if chunk_id is None:
                    continue
            else:
                # Positional mapping (classic FAISS indices)
                if idx >= len(self.chunk_ids):
                    continue
                chunk_id = self.chunk_ids[idx]

            result_ids.append(chunk_id)
            result_scores.append(float(score))
        
        return result_ids, result_scores
    
    def batch_search(
        self, 
        query_embeddings: np.ndarray, 
        k: int = TOP_K
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Search for multiple queries at once.
        
        Args:
            query_embeddings: Query embeddings (n_queries, d)
            k: Number of results per query
            
        Returns:
            List of (chunk_ids, scores) tuples for each query
        """
        query_embeddings = query_embeddings.astype(np.float32)
        scores, indices = self.index.search(query_embeddings, k)
        
        results = []
        for query_indices, query_scores in zip(indices, scores):
            result_ids = []
            result_scores = []
            for idx, score in zip(query_indices, query_scores):
                idx = int(idx)
                if idx < 0:
                    continue

                if self.idx_to_id:
                    chunk_id = self.idx_to_id.get(idx)
                    if chunk_id is None:
                        continue
                else:
                    if idx >= len(self.chunk_ids):
                        continue
                    chunk_id = self.chunk_ids[idx]

                result_ids.append(chunk_id)
                result_scores.append(float(score))
            results.append((result_ids, result_scores))
        
        return results
    
    def save(self, path: Path) -> None:
        """Save the index and metadata."""
        path = Path(path)
        
        # Flush any pending auxiliary data before save
        if self._aux_embeddings:
            self.flush_auxiliary()
        
        # Save FAISS index
        faiss.write_index(self.index, str(path.with_suffix('.faiss')))
        
        # Save metadata (extended for incremental updates - HW1 lesson)
        metadata = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'chunk_ids': self.chunk_ids,
            'is_trained': self.is_trained,
            'use_id_map': self.use_id_map,
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id,
            'next_id': self.next_id,
            'index_params': self.index_params
        }
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"💾 Saved index to {path}")
    
    def load(self, path: Path) -> None:
        """Load the index and metadata."""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path.with_suffix('.faiss')))
        
        # Load metadata
        with open(path.with_suffix('.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.dimension = metadata['dimension']
        self.index_type = metadata['index_type']
        self.chunk_ids = metadata['chunk_ids']
        self.is_trained = metadata['is_trained']
        
        # Load incremental update state if available (backwards compatible)
        self.use_id_map = metadata.get('use_id_map', False)
        self.id_to_idx = metadata.get('id_to_idx', {})
        self.idx_to_id = metadata.get('idx_to_id', {})
        self.next_id = metadata.get('next_id', len(self.chunk_ids))
        self.index_params = metadata.get('index_params', {})
        
        print(f"📂 Loaded index with {self.index.ntotal} vectors")
    
    @property
    def size(self) -> int:
        """Get the number of vectors in the index."""
        return self.index.ntotal if self.index else 0
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the index for analysis."""
        return {
            "type": self.index_type,
            "dimension": self.dimension,
            "n_vectors": self.size,
            "is_trained": self.is_trained,
            "params": self.index_params
        }


def compare_index_types(
    embeddings: np.ndarray,
    chunk_ids: List[str],
    query_embeddings: np.ndarray,
    k: int = 10,
    ground_truth: Optional[List[List[str]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different index types on the same data.
    
    Based on IR HW2: Comprehensive comparison of indexing techniques.
    
    Args:
        embeddings: Document embeddings (n, d)
        chunk_ids: List of chunk IDs
        query_embeddings: Query embeddings for testing
        k: Number of results to retrieve
        ground_truth: Optional ground truth for recall calculation
        
    Returns:
        Dictionary with results for each index type
    """
    import time
    
    index_types = ["flat", "ivf", "hnsw"]
    results = {}
    
    # Use flat as ground truth if not provided
    if ground_truth is None:
        print("Computing ground truth with flat index...")
        flat_index = FAISSIndex(dimension=embeddings.shape[1], index_type="flat")
        flat_index.build_index(embeddings, chunk_ids)
        ground_truth = []
        for q in query_embeddings:
            ids, _ = flat_index.search(q, k=k)
            ground_truth.append(ids)
    
    for idx_type in index_types:
        print(f"\n🔧 Testing {idx_type} index...")
        
        # Build index
        start_build = time.time()
        index = FAISSIndex(dimension=embeddings.shape[1], index_type=idx_type)
        index.build_index(embeddings, chunk_ids)
        build_time = time.time() - start_build
        
        # Search queries
        query_times = []
        recalls = []
        
        for i, q in enumerate(query_embeddings):
            start_query = time.time()
            result_ids, _ = index.search(q, k=k)
            query_times.append(time.time() - start_query)
            
            # Calculate recall vs ground truth
            gt_set = set(ground_truth[i])
            result_set = set(result_ids)
            recall = len(gt_set & result_set) / len(gt_set) if gt_set else 0
            recalls.append(recall)
        
        results[idx_type] = {
            "build_time_ms": build_time * 1000,
            "avg_query_time_ms": np.mean(query_times) * 1000,
            "avg_recall": np.mean(recalls),
            "index_info": index.get_index_info()
        }
        
        print(f"   Build: {build_time*1000:.2f}ms, Query: {np.mean(query_times)*1000:.4f}ms, Recall: {np.mean(recalls):.4f}")
    
    return results


if __name__ == "__main__":
    # Test the FAISS index with multiple index types
    print("="*60)
    print("FAISS Index Comparison (based on IR HW2 methodology)")
    print("="*60)
    
    # Create random test embeddings
    n_docs = 500
    dim = 384
    n_queries = 20
    
    print(f"\nGenerating {n_docs} documents and {n_queries} queries...")
    embeddings = np.random.randn(n_docs, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    chunk_ids = [f"chunk_{i}" for i in range(n_docs)]
    
    query_embeddings = np.random.randn(n_queries, dim).astype(np.float32)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    
    # Compare index types
    print("\n" + "="*60)
    print("Comparing Index Types")
    print("="*60)
    
    results = compare_index_types(embeddings, chunk_ids, query_embeddings, k=10)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Index':<10} {'Build(ms)':<12} {'Query(ms)':<12} {'Recall':<10}")
    print("-"*44)
    for idx_type, data in results.items():
        print(f"{idx_type:<10} {data['build_time_ms']:<12.2f} {data['avg_query_time_ms']:<12.4f} {data['avg_recall']:<10.4f}")
    
    # Test save/load
    print("\n" + "="*60)
    print("Testing Save/Load")
    print("="*60)
    
    from config import EMBEDDINGS_DIR
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    
    index = FAISSIndex(dimension=dim, index_type="flat")
    index.build_index(embeddings, chunk_ids)
    index.save(EMBEDDINGS_DIR / "test_index")
    
    new_index = FAISSIndex(dimension=dim)
    new_index.load(EMBEDDINGS_DIR / "test_index")
    print(f"✅ Reloaded index size: {new_index.size}")

# -*- coding: utf-8-sig -*-
"""
Main Application for Minimal RAG System (Assignment 3).

This implements the required components:
- 4.1 Document Chunking (10%)
- 4.2 Embedding & Indexing (20%)
- 4.3 Retrieval Module (20%)
- 4.4 Answer Generation Module (30%)
- 4.5 System Evaluation (20%)

Usage:
    python main.py --setup          # First-time setup (chunk, embed, index)
    python main.py --query "..."    # Single query
    python main.py --interactive    # Interactive mode
    python main.py --evaluate       # Run evaluation
"""

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from config import (
    DATA_DIR, EMBEDDINGS_DIR, COURSE_PAGES_PATH, WEBSITE_SCRAPED_PATH,
    TOP_K, OPENAI_API_KEY
)
from src.chunking import DocumentChunker
from src.embedding import EmbeddingModel
from src.indexing import FAISSIndex
from src.retrieval import Retriever
from src.generation import AnswerGenerator


def setup_directories():
    """Create necessary directories."""
    DATA_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)


def run_setup(force: bool = False):
    """
    Run initial setup: chunking, embedding, and indexing.
    
    This implements:
    - 4.1 Document Chunking: Split dataset into passages (100-300 words)
    - 4.2 Embedding & Indexing: Compute embeddings, build similarity index
    """
    print("\n" + "="*60)
    print("🚀 RAG System Setup")
    print("="*60)
    
    setup_directories()
    
    chunks_path = DATA_DIR / "chunks.json"
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    index_path = EMBEDDINGS_DIR / "faiss_index"
    
    # Step 1: Chunking (4.1)
    if not chunks_path.exists() or force:
        print("\n📄 Step 1: Document Chunking (Assignment 4.1)")
        print("-" * 40)
        
        chunker = DocumentChunker(chunk_size=250, overlap=0.15)
        chunks = chunker.load_and_chunk_all(
            course_pages_path=COURSE_PAGES_PATH,
            website_scraped_path=WEBSITE_SCRAPED_PATH if WEBSITE_SCRAPED_PATH.exists() else None
        )
        chunker.save_chunks(chunks, chunks_path)
        print(f"✅ Created {len(chunks)} chunks")
    else:
        print("\n✅ Chunks already exist. Loading...")
        with open(chunks_path, 'r', encoding='utf-8-sig') as f:
            chunks_data = json.load(f)
        print(f"   Loaded {len(chunks_data)} chunks")
    
    # Step 2: Embedding (4.2)
    if not embeddings_path.exists() or force:
        print("\n🧠 Step 2: Computing Embeddings (Assignment 4.2)")
        print("-" * 40)
        
        with open(chunks_path, 'r', encoding='utf-8-sig') as f:
            chunks_data = json.load(f)
        
        texts = [c["text"] for c in chunks_data]
        
        embedding_model = EmbeddingModel()
        embeddings = embedding_model.encode_documents(texts)
        embedding_model.save_embeddings(embeddings, embeddings_path)
        print(f"✅ Computed embeddings: {embeddings.shape}")
    else:
        print("\n✅ Embeddings already exist.")
    
    # Step 3: Indexing (4.2)
    if not Path(str(index_path) + '.faiss').exists() or force:
        print("\n📊 Step 3: Building FAISS Index (Assignment 4.2)")
        print("-" * 40)
        
        with open(chunks_path, 'r', encoding='utf-8-sig') as f:
            chunks_data = json.load(f)
        
        import numpy as np
        embeddings = np.load(embeddings_path)
        
        chunk_ids = [c["chunk_id"] for c in chunks_data]
        
        index = FAISSIndex(dimension=embeddings.shape[1])
        index.build_index(embeddings, chunk_ids)
        index.save(index_path)
        print(f"✅ Built index with {index.index.ntotal} vectors")
    else:
        print("\n✅ FAISS index already exists.")
    
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print(f"   📁 Chunks: {chunks_path}")
    print(f"   📁 Embeddings: {embeddings_path}")
    print(f"   📁 Index: {index_path}")
    print("\nYou can now run queries with: python main.py --interactive")


def load_rag_system():
    """Load the complete RAG system."""
    print("\n🔄 Loading RAG System...")
    
    chunks_path = DATA_DIR / "chunks.json"
    index_path = EMBEDDINGS_DIR / "faiss_index"
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    
    if not chunks_path.exists():
        raise FileNotFoundError("Chunks not found. Run setup first: python main.py --setup")
    
    # Load chunks
    with open(chunks_path, 'r', encoding='utf-8-sig') as f:
        chunks_data = json.load(f)
    
    from src.chunking import Chunk
    chunks = [
        Chunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            source_title=c["source_title"],
            source_section=c["source_section"],
            source_file=c["source_file"]
        )
        for c in chunks_data
    ]
    
    # Load embedding model
    embedding_model = EmbeddingModel()
    
    # Load index
    index = FAISSIndex(dimension=384)  # all-MiniLM-L6-v2 dimension
    index.load(index_path)
    
    # Create retriever
    retriever = Retriever(embedding_model, index)
    retriever.load_chunks(chunks)
    
    # Create generator
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set. Generation will fail.")
    
    generator = AnswerGenerator(api_key=api_key)
    
    print("✅ RAG System loaded successfully!")
    return retriever, generator


def run_query(query: str, k: int = TOP_K):
    """Run a single query (Assignment 4.3 + 4.4)."""
    retriever, generator = load_rag_system()
    
    print(f"\n🔍 Query: {query}")
    print("-" * 60)
    
    # Retrieve (4.3)
    print("\n📚 Retrieving top-k passages...")
    retrieved = retriever.retrieve(query, k=k)
    
    print(f"\nRetrieved {len(retrieved)} passages:")
    for r in retrieved:
        print(f"  [{r.rank}] {r.source_title} (score: {r.similarity_score:.4f})")
    
    # Generate (4.4)
    print("\n🤖 Generating answer with GPT-4o...")
    result = generator.generate(query, retrieved, include_sources=True)
    
    print("\n📝 Answer:")
    print(result["answer"])
    
    if "sources" in result:
        print("\n📖 Sources:")
        for src in result["sources"]:
            print(f"  - {src['title']} (score: {src['relevance_score']:.4f})")


def run_interactive():
    """Run interactive mode."""
    retriever, generator = load_rag_system()
    
    print("\n" + "="*60)
    print("🎓 RAG System - University of Antwerp CS Masters")
    print("="*60)
    print("Type 'quit' or 'exit' to end the session.")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("🔍 Your question: ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            # Retrieve
            retrieved = retriever.retrieve(query, k=TOP_K)
            
            # Generate
            result = generator.generate(query, retrieved, include_sources=True)
            
            print("\n📝 Answer:")
            print("-" * 40)
            print(result["answer"])
            
            if "sources" in result:
                print("\n📖 Sources:")
                for src in result["sources"]:
                    print(f"  - {src['title']} (score: {src['relevance_score']:.4f})")
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_evaluation():
    """
    Run evaluation (Assignment 4.5).
    
    This demonstrates:
    - Retrieval quality: Recall@k calculation
    - Answer quality: Comparison with/without retrieval
    - Error analysis: Finding retrieval failures and hallucinations
    """
    print("\n" + "="*60)
    print("📊 Running RAG Evaluation (Assignment 4.5)")
    print("="*60)
    
    retriever, generator = load_rag_system()
    
    from src.evaluation import RAGEvaluator, EvaluationQuestion
    
    # Sample test questions (in a real evaluation, these would have ground truth)
    test_questions = [
        "What are the prerequisites for the Internet of Things course?",
        "Who teaches the Master thesis?",
        "How many credits is the Advanced Networking Lab worth?",
    ]
    
    evaluator = RAGEvaluator()
    
    # For demonstration: retrieve for each question
    retrieval_results = {}
    rag_answers = {}
    baseline_answers = {}
    
    print("\n1. Retrieval Quality (Recall@k)")
    print("-" * 40)
    print("Note: For proper evaluation, you need ground truth relevant chunk IDs.")
    print("This is a demonstration of the evaluation framework.\n")
    
    for q in test_questions:
        retrieved = retriever.retrieve(q, k=TOP_K)
        retrieval_results[q] = [r.chunk_id for r in retrieved]
        
        # Generate RAG answer
        rag_result = generator.generate(q, retrieved, include_sources=False)
        rag_answers[q] = rag_result["answer"]
        
        # Generate baseline answer
        baseline_result = generator.generate_without_retrieval(q)
        baseline_answers[q] = baseline_result["answer"]
    
    print(f"Retrieved passages for {len(test_questions)} questions.")
    
    # Compare RAG vs baseline
    print("\n2. Answer Quality (RAG vs Baseline)")
    print("-" * 40)
    comparison = evaluator.compare_rag_vs_baseline(test_questions, rag_answers, baseline_answers)
    print(f"Compared {comparison['n_questions']} questions.")
    print("\nExample comparison:")
    if comparison['comparisons']:
        ex = comparison['comparisons'][0]
        print(f"Q: {ex['question']}")
        print(f"RAG: {ex['rag_answer'][:100]}...")
        print(f"Baseline: {ex['baseline_answer'][:100]}...")
    
    # Error analysis
    print("\n3. Error Analysis")
    print("-" * 40)
    print("Note: For proper error analysis, you need ground truth relevant chunks.")
    print("This demonstrates the framework for finding failures.\n")
    
    # Save results
    results = {
        "retrieval_results": retrieval_results,
        "rag_answers": rag_answers,
        "baseline_answers": baseline_answers,
        "comparison": comparison
    }
    
    output_path = Path(__file__).parent / "evaluation_results.json"
    evaluator.save_evaluation_results(results, output_path)
    
    print("\n✅ Evaluation complete!")
    print(f"   Results saved to: {output_path}")
    print("\nFor proper evaluation:")
    print("  1. Create test questions with ground truth relevant chunk IDs")
    print("  2. Manually inspect relevance of top-k passages")
    print("  3. Compare RAG vs baseline answers")
    print("  4. Identify ≥3 retrieval failures and ≥3 hallucinations")


def main():
    parser = argparse.ArgumentParser(
        description="Minimal RAG System for Assignment 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                    # Initial setup
  python main.py --query "What is IoT?"     # Single query
  python main.py --interactive              # Interactive mode
  python main.py --evaluate                 # Run evaluation
        """
    )
    
    parser.add_argument('--setup', action='store_true',
                       help='Run initial setup (chunking, embedding, indexing)')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild during setup')
    parser.add_argument('--query', type=str,
                       help='Run a single query')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive Q&A mode')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation (Assignment 4.5)')
    parser.add_argument('-k', type=int, default=TOP_K,
                       help=f'Number of passages to retrieve (default: {TOP_K})')
    
    args = parser.parse_args()
    
    try:
        if args.setup:
            run_setup(force=args.force)
        elif args.query:
            run_query(args.query, k=args.k)
        elif args.interactive:
            run_interactive()
        elif args.evaluate:
            run_evaluation()
        else:
            parser.print_help()
            print("\n💡 Quick start:")
            print("   1. python main.py --setup       # Setup the system")
            print("   2. python main.py --interactive # Start asking questions!")
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Run 'python main.py --setup' first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

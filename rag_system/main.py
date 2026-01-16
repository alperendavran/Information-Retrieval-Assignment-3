# -*- coding: utf-8-sig -*-
"""
Main Application for RAG System.
University of Antwerp - Information Retrieval Assignment 3

Usage:
    python main.py --setup          # First-time setup (chunk, embed, index)
    python main.py --query "..."    # Single query
    python main.py --interactive    # Interactive mode
    python main.py --evaluate       # Run evaluation
    python main.py --demo           # Demo with sample queries
"""

import argparse
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
_BASE_DIR = Path(__file__).resolve().parent
# Explicit path makes behavior deterministic (and avoids find_dotenv quirks).
load_dotenv(dotenv_path=_BASE_DIR / ".env")

from config import (
    DATA_DIR, 
    EMBEDDINGS_DIR, 
    EVALUATION_DIR,
    COURSE_PAGES_PATH,
    WEBSITE_SCRAPED_PATH,
    TOP_K,
    OPENAI_API_KEY
)
from src.chunking import DocumentChunker
from src.embedding import EmbeddingModel
from src.indexing import FAISSIndex
from src.retrieval import Retriever
from src.generation import AnswerGenerator, RAGPipeline
from src.evaluation import RAGEvaluator, create_test_questions
from src.langgraph_agentic_rag import build_agentic_rag_graph, AgenticConfig


def setup_directories() -> None:
    """Create necessary directories."""
    DATA_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    EVALUATION_DIR.mkdir(exist_ok=True)


def run_setup(force: bool = False) -> None:
    """
    Run initial setup: chunking, embedding, and indexing.
    
    Args:
        force: If True, rebuild even if files exist
    """
    print("\n" + "="*60)
    print("🚀 RAG System Setup")
    print("="*60)
    
    setup_directories()
    
    chunks_path = DATA_DIR / "chunks.json"
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    index_path = EMBEDDINGS_DIR / "faiss_index"
    
    # Step 1: Chunking
    if not chunks_path.exists() or force:
        print("\n📄 Step 1: Document Chunking")
        print("-" * 40)
        
        chunker = DocumentChunker()
        chunks = chunker.load_and_chunk_all(
            course_pages_path=COURSE_PAGES_PATH,
            website_scraped_path=WEBSITE_SCRAPED_PATH if WEBSITE_SCRAPED_PATH.exists() else None
        )
        chunker.save_chunks(chunks_path)
    else:
        print("\n✅ Chunks already exist. Loading...")
        with open(chunks_path, 'r', encoding='utf-8-sig') as f:
            chunks_data = json.load(f)
        print(f"   Loaded {len(chunks_data)} chunks")
    
    # Step 2: Embedding
    if not embeddings_path.exists() or force:
        print("\n🧠 Step 2: Computing Embeddings")
        print("-" * 40)
        
        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8-sig') as f:
            chunks_data = json.load(f)
        
        texts = [c["text"] for c in chunks_data]
        
        embedding_model = EmbeddingModel()
        embeddings = embedding_model.encode_documents(texts)
        embedding_model.save_embeddings(embeddings, embeddings_path)
    else:
        print("\n✅ Embeddings already exist.")
    
    # Step 3: Indexing
    if not Path(str(index_path) + '.faiss').exists() or force:
        print("\n📊 Step 3: Building FAISS Index")
        print("-" * 40)
        
        # Load chunks and embeddings
        with open(chunks_path, 'r', encoding='utf-8-sig') as f:
            chunks_data = json.load(f)
        
        import numpy as np
        embeddings = np.load(embeddings_path)
        
        chunk_ids = [c["chunk_id"] for c in chunks_data]
        
        index = FAISSIndex(dimension=embeddings.shape[1])
        index.build_index(embeddings, chunk_ids)
        index.save(index_path)
    else:
        print("\n✅ FAISS index already exists.")
    
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print(f"   📁 Chunks: {chunks_path}")
    print(f"   📁 Embeddings: {embeddings_path}")
    print(f"   📁 Index: {index_path}")
    print("\nYou can now run queries with: python main.py --interactive")


def load_rag_system() -> RAGPipeline:
    """Load the complete RAG system."""
    print("\n🔄 Loading RAG System...")
    
    chunks_path = DATA_DIR / "chunks.json"
    index_path = EMBEDDINGS_DIR / "faiss_index"
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    
    # Check if setup has been run
    if not chunks_path.exists():
        raise FileNotFoundError(
            "Chunks not found. Run setup first: python main.py --setup"
        )
    
    # Load retriever
    retriever = Retriever()
    retriever.load_components(
        chunks_path=chunks_path,
        index_path=index_path,
        embeddings_path=embeddings_path
    )
    
    # Initialize generator
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set. Generation will fail.")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
    
    generator = AnswerGenerator(retriever=retriever, api_key=api_key)
    
    # Create pipeline
    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    
    print("✅ RAG System loaded successfully!")
    return pipeline


def run_query(query: str, k: int = TOP_K) -> None:
    """Run a single query."""
    pipeline = load_rag_system()
    
    print(f"\n🔍 Query: {query}")
    print("-" * 60)
    
    result = pipeline.answer(query, k=k)
    
    print("\n📝 Answer:")
    print(result["answer"])
    
    if "sources" in result:
        print("\n📖 Sources:")
        for src in result["sources"]:
            print(f"  - {src['title']} (score: {src['relevance_score']:.4f})")


def load_agentic_graph(k: int = TOP_K, enable_judge: bool = False):
    """
    Load the agentic LangGraph-based RAG pipeline.
    This does NOT run any OpenAI calls until you invoke the graph.
    """
    pipeline = load_rag_system()

    evaluator = None
    if enable_judge:
        evaluator = RAGEvaluator(retriever=pipeline.retriever, generator=pipeline.generator)

    cfg = AgenticConfig(top_k=k)
    return build_agentic_rag_graph(
        retriever=pipeline.retriever,
        generator=pipeline.generator,
        evaluator=evaluator,
        cfg=cfg,
    )


def run_query_agentic(query: str, k: int = TOP_K, show_trace: bool = False) -> None:
    """Run a single query via LangGraph agentic pipeline."""
    graph = load_agentic_graph(k=k, enable_judge=False)

    print(f"\n🤖🔍 Agentic Query: {query}")
    print("-" * 60)

    out = graph.invoke({"query": query})

    print("\n📝 Answer:")
    print(out.get("answer", ""))

    sources = out.get("sources") or []
    if sources:
        print("\n📖 Sources:")
        for src in sources:
            print(f"  - {src['title']} (score: {src['relevance_score']:.4f})")

    if show_trace:
        trace = out.get("trace") or []
        print("\n🧭 Trace (debug):")
        for step in trace:
            print(f"  - {step.get('event')} @ {step.get('ts')}")


def run_interactive() -> None:
    """Run interactive mode."""
    pipeline = load_rag_system()
    pipeline.interactive_mode()


def run_interactive_agentic() -> None:
    """Run interactive mode using the LangGraph agentic pipeline."""
    graph = load_agentic_graph(k=TOP_K, enable_judge=False)

    print("\n" + "="*60)
    print("🤖🎓 Agentic RAG (LangGraph) — University of Antwerp CS Masters")
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

            out = graph.invoke({"query": query})
            print("\n📝 Answer:")
            print("-" * 40)
            print(out.get("answer", ""))

            sources = out.get("sources") or []
            if sources:
                print("\n📖 Sources:")
                for src in sources:
                    print(f"  - {src['title']} (score: {src['relevance_score']:.4f})")

            print("\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_demo() -> None:
    """Run demo with sample queries."""
    pipeline = load_rag_system()
    
    demo_queries = [
        "What are the prerequisites for the Internet of Things course?",
        "Who teaches the Master thesis?",
        "How many credits is the Advanced Networking Lab worth?",
        "What programming languages are required for the CS Masters?",
        "What topics are covered in the IoT course?"
    ]
    
    print("\n" + "="*60)
    print("🎓 RAG System Demo")
    print("="*60)
    
    for query in demo_queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print("-" * 60)
        
        result = pipeline.answer(query, k=3)
        
        print("\n📝 Answer:")
        print(result["answer"])
        
        if "sources" in result:
            print("\n📖 Sources:")
            for src in result["sources"][:2]:
                print(f"  - {src['title']}")
        
        print()


def run_demo_agentic() -> None:
    """Run demo queries through the LangGraph agentic pipeline."""
    graph = load_agentic_graph(k=3, enable_judge=False)

    demo_queries = [
        "What are the prerequisites for the Internet of Things course?",
        "Who teaches the Master thesis?",
        "How many credits is the Advanced Networking Lab worth?",
        "What programming languages are required for the CS Masters?",
        "What topics are covered in the IoT course?",
    ]

    print("\n" + "="*60)
    print("🤖🎓 Agentic RAG Demo (LangGraph)")
    print("="*60)

    for query in demo_queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print("-" * 60)

        out = graph.invoke({"query": query})

        print("\n📝 Answer:")
        print(out.get("answer", ""))

        sources = out.get("sources") or []
        if sources:
            print("\n📖 Sources:")
            for src in sources[:2]:
                print(f"  - {src['title']}")

        print()


def run_evaluation(save: bool = True) -> None:
    """Run full evaluation."""
    pipeline = load_rag_system()
    
    print("\n" + "="*60)
    print("📊 Running RAG Evaluation")
    print("="*60)
    
    # Create evaluator
    evaluator = RAGEvaluator(
        retriever=pipeline.retriever,
        generator=pipeline.generator
    )
    
    # Get test questions
    questions = create_test_questions()
    
    # Run evaluation
    results = evaluator.evaluate_batch(questions, k=TOP_K, save_results=save)
    
    # Print summary
    print("\n" + "="*60)
    print("📈 Evaluation Summary")
    print("="*60)
    
    summary = results["summary"]
    
    if "mean_recall_at_k" in summary:
        print(f"  📊 Mean Recall@{TOP_K}: {summary['mean_recall_at_k']:.4f}")
    if "mean_mrr" in summary:
        print(f"  📊 Mean MRR: {summary['mean_mrr']:.4f}")
    if "MAP" in summary:
        print(f"  📊 MAP: {summary['MAP']:.4f}")
    
    if "avg_faithfulness" in summary:
        print(f"  📊 Average Faithfulness: {summary['avg_faithfulness']:.1f}/100")
    
    if "avg_relevance" in summary:
        print(f"  📊 Average Relevance: {summary['avg_relevance']:.1f}/100")
    
    if "rag_vs_baseline" in summary:
        rvb = summary["rag_vs_baseline"]
        print(f"\n  🏆 RAG vs Baseline:")
        print(f"     RAG wins: {rvb['rag_wins']}")
        print(f"     Baseline wins: {rvb['baseline_wins']}")
        print(f"     Ties: {rvb['ties']}")
        print(f"     RAG win rate: {rvb['rag_win_rate']*100:.1f}%")
    
    # Error analysis
    print("\n📋 Error Analysis")
    print("-" * 40)
    
    failures = evaluator.find_retrieval_failures(results["individual_results"])
    if failures:
        print(f"\n⚠️  Retrieval Failures ({len(failures)}):")
        for f in failures[:3]:
            print(f"  - {f['question'][:50]}... (recall: {f['recall']:.2f})")
    
    hallucinations = evaluator.find_hallucination_cases(results["individual_results"])
    if hallucinations:
        print(f"\n⚠️  Potential Hallucinations ({len(hallucinations)}):")
        for h in hallucinations[:3]:
            print(f"  - {h['question'][:50]}... (faithfulness: {h['faithfulness_score']:.0f})")


def run_comparison(query: str) -> None:
    """Compare RAG vs non-RAG for a specific query."""
    pipeline = load_rag_system()
    
    print(f"\n🔍 Comparing: {query}")
    print("="*60)
    
    result = pipeline.generator.compare_with_without_retrieval(query)
    
    print("\n📚 WITH RETRIEVAL:")
    print("-" * 40)
    print(result["with_retrieval"]["answer"])
    
    print("\n🚫 WITHOUT RETRIEVAL (baseline):")
    print("-" * 40)
    print(result["without_retrieval"]["answer"])
    
    if "sources" in result["with_retrieval"]:
        print("\n📖 Sources used:")
        for src in result["with_retrieval"]["sources"]:
            print(f"  - {src['title']} (score: {src['relevance_score']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="RAG System for University of Antwerp CS Masters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                    # Initial setup
  python main.py --query "What is IoT?"     # Single query
  python main.py --interactive              # Interactive mode
  python main.py --evaluate                 # Run evaluation
  python main.py --demo                     # Demo queries
  python main.py --compare "query"          # Compare RAG vs no-RAG
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
                       help='Run full evaluation')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with sample queries')
    parser.add_argument('--agentic', action='store_true',
                       help='Use LangGraph agentic pipeline for query/demo/interactive')
    parser.add_argument('--trace', action='store_true',
                       help='(Agentic) Print a simple debug trace')
    parser.add_argument('--compare', type=str,
                       help='Compare RAG vs non-RAG for a query')
    parser.add_argument('-k', type=int, default=TOP_K,
                       help=f'Number of passages to retrieve (default: {TOP_K})')
    
    args = parser.parse_args()
    
    try:
        if args.setup:
            run_setup(force=args.force)
        elif args.query:
            if args.agentic:
                run_query_agentic(args.query, k=args.k, show_trace=args.trace)
            else:
                run_query(args.query, k=args.k)
        elif args.interactive:
            if args.agentic:
                run_interactive_agentic()
            else:
                run_interactive()
        elif args.evaluate:
            run_evaluation()
        elif args.demo:
            if args.agentic:
                run_demo_agentic()
            else:
                run_demo()
        elif args.compare:
            run_comparison(args.compare)
        else:
            parser.print_help()
            print("\n💡 Quick start:")
            print("   1. python main.py --setup       # Setup the system")
            print("   2. python main.py --demo        # Try demo queries")
            print("   3. python main.py --interactive # Start asking questions!")
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Run 'python main.py --setup' first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

# -*- coding: utf-8-sig -*-
"""
Configuration settings for the RAG System.
University of Antwerp - Information Retrieval Assignment 3
"""

import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATIONS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EVALUATION_DIR = BASE_DIR / "evaluation_results"

# Source data paths
CS_DATA_DIR = BASE_DIR.parent / "cs-data"
COURSE_PAGES_PATH = CS_DATA_DIR / "course-pages.json"
WEBSITE_SCRAPED_PATH = CS_DATA_DIR / "website-scraped.json"

# ============================================================================
# CHUNKING CONFIGURATIONS
# ============================================================================
CHUNK_SIZE = 250  # tokens (recommended: 200-300)
CHUNK_OVERLAP = 0.15  # 15% overlap
MIN_CHUNK_SIZE = 50  # minimum tokens per chunk

# ============================================================================
# EMBEDDING CONFIGURATIONS
# ============================================================================
# Model options (all run locally):
# - "sentence-transformers/all-MiniLM-L6-v2" (384d, fast)
# - "sentence-transformers/all-mpnet-base-v2" (768d, more accurate)
# - "BAAI/bge-small-en-v1.5" (384d, SOTA)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ============================================================================
# RETRIEVAL CONFIGURATIONS
# ============================================================================
TOP_K = 5  # Number of passages to retrieve (1 too small, 20 too big)

# Candidate Expansion Strategy (from IR HW2 feedback)
# "When looking for top-k results, select the top c*k and compute exact 
#  distances for the c*k resulting vectors. Then make a top-k out of 
#  these c*k candidates."
USE_RERANKING = True         # Enable candidate expansion + reranking
EXPANSION_FACTOR = 3         # Retrieve 3*k candidates, then refine to k

# Dataset-aware post-retrieval tweaks (driven by cs-data structure)
# - Course pages have consistent section names (Prerequisites, Learning Outcomes, Assessment, etc.)
# - Many user queries explicitly mention these intents ("prerequisites", "exam", "credits", "lecturer")
SECTION_AWARE_BOOST = True
SECTION_BOOST_WEIGHT = 0.05  # small additive bonus applied during reranking

# Strong lexical boost for exact identifiers in the dataset (course codes like 2500WETINT)
LEXICAL_CODE_BOOST = True
LEXICAL_CODE_BOOST_WEIGHT = 0.20

# ============================================================================
# LANGGRAPH / AGENTIC RAG (Optional, Advanced)
# ============================================================================
# These settings control the LangGraph-based workflow in `src/langgraph_agentic_rag.py`.

# Use an LLM to produce structured query tags (intent + routing).
# Default OFF to avoid extra API calls; you can enable for a more "agentic" demo.
AGENTIC_USE_LLM_TAGGER = False
AGENTIC_TAGGER_MODEL = os.getenv("AGENTIC_TAGGER_MODEL", "gpt-4o-mini")

# HyDE (Hypothetical Document Embeddings): generate a hypothetical answer/passage
# and use it as an additional retrieval query variant.
AGENTIC_USE_HYDE = False
AGENTIC_HYDE_MODEL = os.getenv("AGENTIC_HYDE_MODEL", "gpt-4o-mini")

# Reflection: post-generation self-check to reduce hallucinations.
AGENTIC_USE_REFLECTION = False
# For assignment compliance, default reflection model follows OPENAI_MODEL (if set),
# otherwise defaults to GPT-4o.
AGENTIC_REFLECTION_MODEL = os.getenv("AGENTIC_REFLECTION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))

# Optional: diversify retrieved context for multi-hop questions (kept OFF by default)
DIVERSIFY_SOURCES = False
MAX_CHUNKS_PER_SOURCE_TITLE = 2

# ============================================================================
# AGENTIC TOOL ROUTER (Function Calling) — Optional
# ============================================================================
# Advanced: Use OpenAI tool/function-calling to route queries to deterministic local tools
# backed by our structured index (course sections, study programme filters, etc.).
# This reduces "manual heuristics" in the agent and makes routing decisions explicit/inspectable.
AGENTIC_USE_TOOL_ROUTER = os.getenv("AGENTIC_USE_TOOL_ROUTER", "0").strip().lower() in ("1", "true", "yes", "y")
AGENTIC_TOOL_ROUTER_MODEL = os.getenv("AGENTIC_TOOL_ROUTER_MODEL", "gpt-4o-mini")
AGENTIC_TOOL_ROUTER_MAX_CALLS = int(os.getenv("AGENTIC_TOOL_ROUTER_MAX_CALLS", "3"))

# ============================================================================
# GENERATION CONFIGURATIONS
# ============================================================================
# Model options (from cheapest to most capable):
# - "gpt-4o-mini"     : $0.15/1M input, $0.60/1M output (RECOMMENDED - best value)
# - "gpt-4.1-nano"    : $0.10/1M input, $1.40/1M output (fastest)
# - "gpt-4.1-mini"    : $0.40/1M input, $1.60/1M output (balanced)
# - "gpt-4o"          : $2.50/1M input, $10.0/1M output (standard)
# - "gpt-4.1"         : $2.00/1M input, $8.00/1M output (strongest reasoning)
# IMPORTANT (assignment requirement): Use GPT-4o for answer generation.
# For development/cost reasons you can override via env var, e.g.:
#   export OPENAI_MODEL="gpt-4o-mini"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_TOKENS = 1024
TEMPERATURE = 0.1  # Low temperature for factual responses

# System prompt for the RAG system
SYSTEM_PROMPT = """You are a helpful assistant for the University of Antwerp Computer Science Masters program. 
Your role is to answer questions about courses, programs, requirements, and other academic information.

IMPORTANT INSTRUCTIONS:
1. Answer questions based ONLY on the provided context below.
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
3. Be specific and cite course names or details when relevant.
4. Keep your answers concise but complete.
5. Do not make up information that is not in the context."""

# ============================================================================
# EVALUATION CONFIGURATIONS
# ============================================================================
EVALUATION_METRICS = ["recall_at_k", "mrr", "faithfulness", "answer_relevance"]

# OpenAI API Key (set via environment variable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================================================
# COST ESTIMATION (Optional)
# ============================================================================
# Prices are ESTIMATES. Update if OpenAI pricing changes.
# Source note: user requested GPT-4o-mini for development cost control.
MODEL_PRICING_USD_PER_1M = {
    # model: { "input": USD per 1M input tokens, "output": USD per 1M output tokens }
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

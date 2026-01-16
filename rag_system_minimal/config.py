# -*- coding: utf-8-sig -*-
"""
Configuration settings for Minimal RAG System (Assignment 3 Only).
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# Source data paths (relative to parent directory)
CS_DATA_DIR = BASE_DIR.parent / "cs-data"
COURSE_PAGES_PATH = CS_DATA_DIR / "course-pages.json"
WEBSITE_SCRAPED_PATH = CS_DATA_DIR / "website-scraped.json"

# Chunking settings
CHUNK_SIZE = 250  # Target words per chunk (100-300 range)
CHUNK_OVERLAP = 0.15  # 15% overlap between chunks

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions

# Retrieval settings
TOP_K = 5  # Number of passages to retrieve (1 too small, 20 too big)

# Generation settings
OPENAI_MODEL = "gpt-4o"  # Assignment requirement: use GPT-4o
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

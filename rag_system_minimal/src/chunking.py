# -*- coding: utf-8-sig -*-
"""
Document Chunking Module (Assignment 4.1 - 10%)

This module splits documents into passages of 100-300 words as required by the assignment.
It handles preprocessing steps like cleaning markup and preserving semantic coherence.

References:
- Manning et al. (2008): Introduction to Information Retrieval, Chapter 2 (tokenization)
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class Chunk:
    """Represents a single chunk of text."""
    chunk_id: str
    text: str
    source_title: str
    source_section: str
    source_file: str


class DocumentChunker:
    """
    Splits documents into passages of 100-300 words.
    
    Design decisions:
    - Chunk size: 250 words (middle of 100-300 range)
    - Overlap: 15% to preserve context across boundaries
    - Preprocessing: Remove extra whitespace, preserve sentence boundaries
    - For course PDFs: Preserve section headers (Prerequisites, Learning Outcomes, etc.)
    - For website pages: Clean markdown but preserve structure
    """
    
    def __init__(self, chunk_size: int = 250, overlap: float = 0.15):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target number of words per chunk (default: 250)
            overlap: Overlap ratio between consecutive chunks (default: 0.15 = 15%)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = 50  # Minimum words to keep a chunk
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocessing steps:
        1. Remove excessive whitespace (normalize to single spaces)
        2. Remove markdown artifacts (for website-scraped pages)
        3. Preserve sentence boundaries
        4. Keep course metadata (course codes, credits, etc.)
        
        Note: The provided dataset (cs-data) is already mostly clean,
        so this focuses on normalization rather than heavy cleaning.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove markdown artifacts (if any remain)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Remove markdown links, keep text
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # Remove bold markers, keep text
        
        return text
    
    def _count_words(self, text: str) -> int:
        """Count words in text (simple whitespace-based)."""
        return len(text.split())
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple period-based, can be improved)."""
        # Split on sentence endings, but preserve abbreviations
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(
        self, 
        text: str, 
        source_title: str = "", 
        source_section: str = "",
        source_file: str = ""
    ) -> List[Chunk]:
        """
        Split text into chunks of approximately chunk_size words.
        
        Strategy:
        - Split into sentences first
        - Group sentences until reaching chunk_size words
        - Apply overlap between consecutive chunks
        - Preserve sentence boundaries (don't split mid-sentence)
        
        Args:
            text: Input text to chunk
            source_title: Title of the source document
            source_section: Section name (e.g., "Prerequisites", "Learning Outcomes")
            source_file: Source file path
            
        Returns:
            List of Chunk objects
        """
        text = self._preprocess_text(text)
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks: List[Chunk] = []
        overlap_words = int(self.chunk_size * self.overlap)
        
        current_chunk_sentences: List[str] = []
        current_word_count = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence_words = self._count_words(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_word_count + sentence_words > self.chunk_size and current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                if self._count_words(chunk_text) >= self.min_chunk_size:
                    chunk = Chunk(
                        chunk_id=f"{source_file}_{chunk_idx}",
                        text=chunk_text,
                        source_title=source_title,
                        source_section=source_section,
                        source_file=source_file
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                
                # Start new chunk with overlap
                # Keep last N words from previous chunk
                if overlap_words > 0:
                    prev_text = ' '.join(current_chunk_sentences)
                    prev_words = prev_text.split()
                    overlap_text = ' '.join(prev_words[-overlap_words:])
                    current_chunk_sentences = [overlap_text] if overlap_text else []
                    current_word_count = self._count_words(overlap_text)
                else:
                    current_chunk_sentences = []
                    current_word_count = 0
            
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_words
        
        # Add final chunk if it has enough words
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            if self._count_words(chunk_text) >= self.min_chunk_size:
                chunk = Chunk(
                    chunk_id=f"{source_file}_{chunk_idx}",
                    text=chunk_text,
                    source_title=source_title,
                    source_section=source_section,
                    source_file=source_file
                )
                chunks.append(chunk)
        
        return chunks
    
    def load_and_chunk_all(
        self,
        course_pages_path: Path,
        website_scraped_path: Path = None
    ) -> List[Chunk]:
        """
        Load course pages and website data, then chunk all documents.
        
        Args:
            course_pages_path: Path to course-pages.json
            website_scraped_path: Optional path to website-scraped.json
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks: List[Chunk] = []
        
        # Load course pages
        if course_pages_path.exists():
            with open(course_pages_path, 'r', encoding='utf-8-sig') as f:
                course_data = json.load(f)
            
            # course-pages.json structure: list of course objects
            for course in course_data:
                title = course.get('course_title', '')
                sections = course.get('course_description_sections', {})
                
                for section_name, section_text in sections.items():
                    if section_text:
                        section_chunks = self.chunk_text(
                            text=section_text,
                            source_title=title,
                            source_section=section_name,
                            source_file=str(course_pages_path)
                        )
                        all_chunks.extend(section_chunks)
        
        # Load website scraped pages
        if website_scraped_path and website_scraped_path.exists():
            with open(website_scraped_path, 'r', encoding='utf-8-sig') as f:
                website_data = json.load(f)
            
            # website-scraped.json structure: {"pages": [...]}
            pages = website_data.get('pages', [])
            for page in pages:
                page_title = page.get('title', '')
                # Use markdown content if available, otherwise html_clean
                page_content = page.get('markdown', '') or page.get('html_clean', '')
                
                if page_content:
                    page_chunks = self.chunk_text(
                        text=page_content,
                        source_title=page_title,
                        source_section='',
                        source_file=str(website_scraped_path)
                    )
                    all_chunks.extend(page_chunks)
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Chunk], output_path: Path) -> None:
        """Save chunks to JSON file."""
        chunks_data = [
            {
                'chunk_id': c.chunk_id,
                'text': c.text,
                'source_title': c.source_title,
                'source_section': c.source_section,
                'source_file': c.source_file
            }
            for c in chunks
        ]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

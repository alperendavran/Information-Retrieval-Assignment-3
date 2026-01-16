# -*- coding: utf-8-sig -*-
"""
Document Chunking Module for RAG System.
Splits documents into passages of 100-300 words with overlap.

University of Antwerp - Information Retrieval Assignment 3
References:
- Lecture slides: "Chunk size: 200–300 tokens, Overlap: 10–20%"
- 15 Advanced RAG Techniques whitepaper
"""

import json
import tiktoken
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    chunk_id: str
    text: str
    source_file: str
    source_title: str
    source_section: Optional[str] = None
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "source_title": self.source_title,
            "source_section": self.source_section,
            "token_count": self.token_count
        }


class DocumentChunker:
    """
    Splits documents into semantically coherent chunks.
    
    Chunking Strategy:
    1. For course pages: chunk by section (prerequisites, learning outcomes, etc.)
    2. For website pages: chunk by paragraph with overlap
    3. Maintain semantic coherence - don't split mid-sentence
    """
    
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE,
        overlap_ratio: float = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE
    ):
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.min_chunk_size = min_chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunks: List[Chunk] = []
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentence-like units while preserving structure.

        IMPORTANT (dataset-driven):
        - The provided dataset contains lots of newline-separated bullet lists
          (e.g., study programme pages; course sections).
        - A pure punctuation-based splitter collapses these into a single
          mega-"sentence", producing chunks of thousands of tokens.

        Strategy:
        - Treat headings and list items as hard boundaries.
        - Treat blank lines as boundaries.
        - Within normal lines/paragraphs, fall back to punctuation splitting.
        """
        import re

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        lines = [ln.strip() for ln in text.split("\n")]

        units: List[str] = []
        paragraph_buf: List[str] = []

        def flush_paragraph() -> None:
            if not paragraph_buf:
                return
            paragraph = " ".join(paragraph_buf).strip()
            paragraph_buf.clear()
            if not paragraph:
                return
            # Split paragraph into sentences
            for s in re.split(r"(?<=[.!?])\s+", paragraph):
                s = s.strip()
                if s:
                    units.append(s)

        heading_re = re.compile(r"^\s*#{1,6}\s+(.*\S)\s*$")
        bullet_re = re.compile(r"^\s*(?:[-*•]|\d+[\).])\s+(.*\S)\s*$")

        for ln in lines:
            if not ln:
                flush_paragraph()
                continue

            m_h = heading_re.match(ln)
            if m_h:
                flush_paragraph()
                units.append(m_h.group(1).strip())
                continue

            m_b = bullet_re.match(ln)
            if m_b:
                flush_paragraph()
                units.append(m_b.group(1).strip())
                continue

            # Normal line: keep accumulating into a paragraph buffer
            paragraph_buf.append(ln)

        flush_paragraph()
        return units

    def _clean_html_preserve_newlines(self, text: str) -> str:
        """Remove HTML tags while keeping newline structure intact."""
        import re
        # Keep newlines; replace tags with spaces
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace per-line (do NOT collapse across newlines)
        cleaned_lines = []
        for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            cleaned_lines.append(" ".join(ln.split()).strip())
        # Collapse multiple empty lines
        out_lines = []
        empty_run = 0
        for ln in cleaned_lines:
            if ln == "":
                empty_run += 1
                if empty_run <= 1:
                    out_lines.append("")
            else:
                empty_run = 0
                out_lines.append(ln)
        return "\n".join(out_lines).strip()

    def _split_markdown_blocks(self, markdown: str) -> List[tuple[Optional[str], str]]:
        """
        Split markdown into blocks by headings.

        Returns a list of (heading, block_text) where heading may be None.
        This prevents mixing multiple course entries into the same chunk on
        'Study programme' pages.
        """
        import re

        markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
        lines = markdown.split("\n")
        heading_re = re.compile(r"^\s*#{1,6}\s+(.*\S)\s*$")

        blocks: List[tuple[Optional[str], List[str]]] = []
        current_heading: Optional[str] = None
        current_lines: List[str] = []

        def push_block() -> None:
            nonlocal current_heading, current_lines
            if not current_lines:
                return
            text = "\n".join(current_lines).strip()
            if text:
                blocks.append((current_heading, current_lines))
            current_lines = []

        for ln in lines:
            m = heading_re.match(ln)
            if m:
                # Start a new block at every heading
                push_block()
                current_heading = m.group(1).strip()
                current_lines = [ln]  # keep original heading line (for structure)
            else:
                current_lines.append(ln)

        push_block()

        if not blocks:
            return [(None, markdown)]

        # Convert to (heading, text) with cleaned whitespace per-line
        out: List[tuple[Optional[str], str]] = []
        for heading, blines in blocks:
            text = "\n".join(blines)
            out.append((heading, text))
        return out
    
    def chunk_by_tokens(self, text: str, source_file: str, source_title: str, 
                        section_name: Optional[str] = None) -> List[Chunk]:
        """
        Chunk text by token count with overlap.
        Respects sentence boundaries for semantic coherence.
        """
        chunks = []
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return chunks
        
        current_chunk = []
        current_tokens = 0
        overlap_tokens = int(self.chunk_size * self.overlap_ratio)
        chunk_counter = len(self.chunks)
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds chunk size, add it as its own chunk
            if sentence_tokens > self.chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if self.count_tokens(chunk_text) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            chunk_id=f"chunk_{chunk_counter}",
                            text=chunk_text,
                            source_file=source_file,
                            source_title=source_title,
                            source_section=section_name,
                            token_count=self.count_tokens(chunk_text)
                        ))
                        chunk_counter += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Add the long sentence as its own chunk
                chunks.append(Chunk(
                    chunk_id=f"chunk_{chunk_counter}",
                    text=sentence,
                    source_file=source_file,
                    source_title=source_title,
                    source_section=section_name,
                    token_count=sentence_tokens
                ))
                chunk_counter += 1
                continue
            
            # Check if adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                if self.count_tokens(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        chunk_id=f"chunk_{chunk_counter}",
                        text=chunk_text,
                        source_file=source_file,
                        source_title=source_title,
                        source_section=section_name,
                        token_count=self.count_tokens(chunk_text)
                    ))
                    chunk_counter += 1
                
                # Start new chunk with overlap
                # Find sentences to include for overlap
                overlap_sentences = []
                overlap_count = 0
                for s in reversed(current_chunk):
                    s_tokens = self.count_tokens(s)
                    if overlap_count + s_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, s)
                        overlap_count += s_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if self.count_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    chunk_id=f"chunk_{chunk_counter}",
                    text=chunk_text,
                    source_file=source_file,
                    source_title=source_title,
                    source_section=section_name,
                    token_count=self.count_tokens(chunk_text)
                ))
        
        return chunks
    
    def process_course_page(self, course_data: Dict[str, Any]) -> List[Chunk]:
        """
        Process a single course page from course-pages.json.
        Chunks each section separately to maintain semantic coherence.
        """
        chunks = []
        file_name = course_data.get("file_name", "unknown")
        course_title = course_data.get("course_title", "Unknown Course")
        
        # Get course details as a header
        details = course_data.get("course_details", {})
        header = f"Course: {course_title}\n"
        if details.get("course_code"):
            header += f"Code: {details['course_code']}\n"
        if details.get("credits"):
            header += f"Credits: {details['credits']}\n"
        if details.get("semester"):
            header += f"Semester: {details['semester']}\n"
        if details.get("lecturers"):
            lecturers = details['lecturers']
            if isinstance(lecturers, list):
                header += f"Lecturers: {', '.join(lecturers)}\n"
        
        # Process each section
        sections = course_data.get("course_description_sections", {})
        
        for section_name, section_content in sections.items():
            if not section_content or not section_content.strip():
                continue
            
            # Combine header with section for context
            full_text = f"{header}\n{section_name}:\n{section_content}"
            
            section_chunks = self.chunk_by_tokens(
                text=full_text,
                source_file=file_name,
                source_title=course_title,
                section_name=section_name
            )
            chunks.extend(section_chunks)
        
        # If no sections, use the full course description
        if not chunks:
            full_desc = course_data.get("course_description", "")
            if full_desc:
                chunks = self.chunk_by_tokens(
                    text=f"{header}\n{full_desc}",
                    source_file=file_name,
                    source_title=course_title,
                    section_name="Full Description"
                )
        
        return chunks
    
    def process_website_page(self, page_data: Dict[str, Any]) -> List[Chunk]:
        """Process a single page from website-scraped.json."""
        chunks = []
        
        # Handle the nested structure from website-scraped.json
        url = page_data.get("url", "unknown")
        title = page_data.get("title", "Website Page")
        
        # Use markdown content (cleaner) or fall back to html_clean
        content = page_data.get("markdown", "")
        if not content:
            content = page_data.get("html_clean", page_data.get("html_content", ""))
        
        if not content:
            return chunks
        
        # Clean up HTML tags if present, but preserve markdown/newline structure.
        content = self._clean_html_preserve_newlines(content)
        
        # Extract metadata for context
        metadata = page_data.get("metadata", {})
        breadcrumbs = metadata.get("breadcrumbs", "")
        program = metadata.get("program", "")
        page_type = metadata.get("page_type", "")
        
        # Add title as context
        header = f"Page: {title}\n"
        if breadcrumbs:
            header += f"Navigation: {breadcrumbs}\n"
        if program:
            header += f"Program: {program}\n"
        if page_type:
            header += f"Page type: {page_type}\n"
        header += f"Source: {url}\n\n"

        # Dataset-driven: split by headings to avoid giant mixed chunks in study programme pages.
        blocks = self._split_markdown_blocks(content)
        for heading, block_text in blocks:
            full_text = header + block_text
            block_chunks = self.chunk_by_tokens(
                text=full_text,
                source_file=url,
                source_title=title,
                section_name=heading,
            )
            chunks.extend(block_chunks)
        
        return chunks
    
    def load_and_chunk_all(
        self, 
        course_pages_path: Path,
        website_scraped_path: Optional[Path] = None
    ) -> List[Chunk]:
        """
        Load and chunk all documents from the dataset.
        
        Args:
            course_pages_path: Path to course-pages.json
            website_scraped_path: Path to website-scraped.json (optional)
            
        Returns:
            List of all chunks
        """
        self.chunks = []
        
        # Process course pages
        print(f"Loading course pages from {course_pages_path}...")
        with open(course_pages_path, 'r', encoding='utf-8-sig') as f:
            course_pages = json.load(f)
        
        print(f"Processing {len(course_pages)} course pages...")
        for course in course_pages:
            course_chunks = self.process_course_page(course)
            self.chunks.extend(course_chunks)
        
        print(f"Created {len(self.chunks)} chunks from course pages")
        
        # Process website pages if provided
        if website_scraped_path and website_scraped_path.exists():
            print(f"\nLoading website pages from {website_scraped_path}...")
            with open(website_scraped_path, 'r', encoding='utf-8-sig') as f:
                website_data = json.load(f)
            
            # Handle the nested structure: {"pages": [...]}
            if isinstance(website_data, dict) and "pages" in website_data:
                website_pages = website_data["pages"]
            elif isinstance(website_data, list):
                website_pages = website_data
            else:
                website_pages = []
            
            initial_count = len(self.chunks)
            print(f"Processing {len(website_pages)} website pages...")
            for page in website_pages:
                if isinstance(page, dict):  # Skip if not a dict
                    page_chunks = self.process_website_page(page)
                    self.chunks.extend(page_chunks)
            
            print(f"Created {len(self.chunks) - initial_count} chunks from website pages")
        
        # Update chunk IDs to be unique
        for i, chunk in enumerate(self.chunks):
            chunk.chunk_id = f"chunk_{i}"
        
        print(f"\n✅ Total chunks created: {len(self.chunks)}")
        
        # Statistics
        total_tokens = sum(c.token_count for c in self.chunks)
        avg_tokens = total_tokens / len(self.chunks) if self.chunks else 0
        print(f"📊 Average tokens per chunk: {avg_tokens:.1f}")
        print(f"📊 Total tokens: {total_tokens}")
        
        return self.chunks
    
    def save_chunks(self, output_path: Path) -> None:
        """Save chunks to JSON file."""
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(self.chunks)} chunks to {output_path}")
    
    def load_chunks(self, input_path: Path) -> List[Chunk]:
        """Load chunks from JSON file."""
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            chunks_data = json.load(f)
        
        self.chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                source_file=c["source_file"],
                source_title=c["source_title"],
                source_section=c.get("source_section"),
                token_count=c.get("token_count", 0)
            )
            for c in chunks_data
        ]
        print(f"📂 Loaded {len(self.chunks)} chunks from {input_path}")
        return self.chunks


if __name__ == "__main__":
    # Test the chunker
    from config import COURSE_PAGES_PATH, WEBSITE_SCRAPED_PATH, DATA_DIR
    
    chunker = DocumentChunker()
    chunks = chunker.load_and_chunk_all(
        course_pages_path=COURSE_PAGES_PATH,
        website_scraped_path=WEBSITE_SCRAPED_PATH
    )
    
    # Save chunks
    DATA_DIR.mkdir(exist_ok=True)
    chunker.save_chunks(DATA_DIR / "chunks.json")
    
    # Show sample chunks
    print("\n📝 Sample chunks:")
    for chunk in chunks[:3]:
        print(f"\n--- {chunk.chunk_id} ({chunk.token_count} tokens) ---")
        print(f"Source: {chunk.source_title}")
        print(f"Section: {chunk.source_section}")
        print(f"Text: {chunk.text[:200]}...")

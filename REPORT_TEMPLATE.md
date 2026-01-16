# Information Retrieval — Optional Assignment 3 (RAG)

**Course**: Information Retrieval (2025–2026)  
**Instructor**: Prof. Toon Calders  
**Deadline**: 30/01/2026  

## Group

- **Member 1**: Name — Student number
- **Member 2**: Name — Student number
- **Member 3**: Name — Student number
- **Member 4**: Name — Student number

## Repository

- **GitHub link**: `<paste link here>` (public or shared with `tcalders`)

Repository should include all code and data needed to run:
- Raw dataset (`cs-data/`)
- Code (`rag_system/`)
- Instructions to reproduce derived artifacts (chunks/embeddings/index)

## 1. System Overview

We implement a small-scale **Retrieval-Augmented Generation (RAG)** system for answering questions about the University of Antwerp CS Masters program.

Pipeline:
1. Chunk documents into passages
2. Embed passages with a **local** pretrained sentence embedding model
3. Build a FAISS similarity index (cosine similarity)
4. Retrieve top-\(k\) passages for a query
5. Generate an answer using **GPT‑4o** conditioned on retrieved passages

(Optional, advanced): We also provide an **agentic orchestration layer** using **LangGraph** to structure the workflow and improve retrieval robustness via:
- query tagging (intent + routing)
- multi-query expansion + fusion (RRF)
- post-retrieval deduplication + MMR diversification
- optional HyDE / reflection hooks (discussed explicitly in the report)

## 2. Dataset

We use the provided dataset:
- `cs-data/course-pages.json`
- `cs-data/website-scraped.json`

Report:
- number of pages/documents
- number of chunks/passages after chunking (must be ≥ 150–200)

## 3. Document Chunking (10%)

Implementation: `rag_system/src/chunking.py`

### 3.1 Chunking strategy

- **Chunk size**: \<e.g. ~250 tokens ≈ 150–200 words\>
- **Overlap**: \<e.g. 15%\>
- **Boundary rule**: split on sentence boundaries (avoid mid-sentence cuts)

### 3.2 Preprocessing

- remove/clean markup (for scraped pages)
- whitespace normalization
- keep course metadata as header context (course code, credits, semester, lecturers)

Include a short table:

| Setting | Value | Motivation |
|--------|-------|------------|
| chunk_size |  |  |
| overlap |  |  |

## 4. Embedding & Indexing (20%)

Implementation: `rag_system/src/embedding.py`, `rag_system/src/indexing.py`

### 4.1 Embedding model choice

- Model: `sentence-transformers/all-MiniLM-L6-v2` (local)
- Why: fast, strong baseline, 384 dimensions, cosine similarity friendly

(Optional experiments: compare with `all-mpnet-base-v2` or `bge-small-en-v1.5`)

### 4.2 Indexing

- Similarity: cosine similarity (implemented as inner product on L2-normalized vectors)
- Index: FAISS `IndexFlatIP` (exact) for this dataset size

Scalability note:
- For larger corpora, approximate indices (e.g., HNSW / IVF) are supported in code, but not required for this dataset.

## 5. Retrieval Module (20%)

Implementation: `rag_system/src/retrieval.py`

Steps:
1. embed query
2. retrieve top-\(k\) passages
3. rank by similarity score

Optional improvement (if used):
- candidate expansion + reranking (retrieve \(c \cdot k\), then refine to \(k\))

## 6. Answer Generation Module (30%)

Implementation: `rag_system/src/generation.py`

Requirement compliance:
- **Generator model**: GPT‑4o (mandatory)

Prompting:
- system prompt instructs “answer only from provided context”
- retrieved passages are inserted with source headers
- output optionally includes sources (for inspection)

Baseline:
- compare with “no retrieval” (LLM only)

## 7. System Evaluation (20%)

Implementation: `rag_system/src/evaluation.py` + manual annotation scripts in `rag_system/scripts/`

### 7.1 Retrieval quality

**Required**: report Recall@k and manually inspect top-\(k\) passages.

We use a simple manual labeling workflow:
1. Generate candidate packs:
   - `rag_system/scripts/create_annotation_pack.py` (no API calls)
2. Manually label `is_relevant` for each candidate
3. Compute metrics:
   - `rag_system/scripts/score_annotations.py`

Report:
- mean Recall@k (and optionally Precision@k, MRR, nDCG@k)
- include 2–3 example questions with top-\(k\) passages and your relevance labels

#### Ablation (baseline vs agentic)

To compare **normal vs agentic (LangGraph)** retrieval fairly, we use *pooled labeling*:

1. Create pooled pack: `rag_system/scripts/create_pooled_ablation_pack.py`
2. Label `is_relevant` once for the union of candidates
3. Score + generate ablation table: `rag_system/scripts/score_pooled_ablation_pack.py`
4. Auto-generate 3 worst retrieval failures per system: `rag_system/scripts/generate_failure_case_report.py`

### 7.2 Answer quality

Required:
- compare answers with retrieval vs without retrieval (baseline)
- evaluate correctness, completeness, hallucination rate

Suggested method:
- manual assessment for a subset of questions
- optional: LLM-as-a-judge scoring for faithfulness/relevance (discuss validity limits)

### 7.3 Error analysis

Required:
- ≥ 3 cases where retrieval failed
- ≥ 3 cases where GPT‑4o produced incorrect/hallucinated answer

For each case:
- question
- retrieved passages (top-\(k\))
- what went wrong (retrieval mismatch? chunking issue? prompt/context overload?)
- proposed fix (e.g., chunk size, metadata-aware rerank, deduplication, different embedder)

## 8. Notes for the Instructor (Optional)

- Mention anything special you want considered (limitations, compute constraints, design decisions).

## 9. Plagiarism / Use of External Material

- We used GenAI assistance for brainstorming and code review.  
- All external sources (papers, slides, websites) are cited in References.  
- Any copied/adapted code is clearly marked in source files with a reference comment.

## References

Include:
- assignment pdf
- lecture slides
- inspected papers (RAG, GraphRAG, Agentic RAG, etc.)
- any libraries/docs you relied on (FAISS, sentence-transformers, OpenAI API docs)


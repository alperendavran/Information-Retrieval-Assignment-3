# Information Retrieval — Assignment 3 Report  
## Retrieval-Augmented Generation System

**Course**: Information Retrieval (2025–2026)  
**Instructor**: Prof. Toon Calders  
**Deadline**: 30/01/2026

---

*This report describes the RAG system we developed for Assignment 3.*

---

## Group

- **Member 1**: [Name] — [Student number]
- **Member 2**: [Name] — [Student number]
- *(Add more if applicable)*

## Repository

**GitHub**: [Paste your repository link here]  
(Public or shared with tcalders)

The repository includes all code and data needed to run the system: raw dataset in `cs-data/`, code in `rag_system/`, and instructions in the README.

---

## 1. System Overview

We built a RAG system for the UAntwerp CS Masters programme. Given a question, it retrieves relevant passages from course descriptions and programme pages, then generates an answer with GPT-4o using that context.

The pipeline works as follows:

1. **Chunk** documents into passages (around 250 tokens each, 15% overlap)
2. **Embed** passages with a local sentence transformer (`all-MiniLM-L6-v2`)
3. **Index** embeddings in FAISS (cosine similarity via L2-normalised inner product)
4. **Retrieve** the top-*k* passages for a query (we use *k* = 5)
5. **Generate** an answer with GPT-4o conditioned on the retrieved passages

We also implemented an optional agentic workflow with LangGraph (query tagging, multi-query expansion, RRF fusion, MMR diversification) to explore whether it improves retrieval and answers. Details and results are discussed in the evaluation section.

---

## 2. Dataset

We use the provided dataset:

- `cs-data/course-pages.json` — course pages
- `cs-data/website-scraped.json` — scraped programme pages

After chunking we obtain around 500 passages, which exceeds the required 150–200.

---

## 3. Document Chunking (4.1)

**Implementation**: `rag_system/src/chunking.py`

### 3.1 Strategy

| Setting    | Value | Motivation |
|-----------|-------|------------|
| Chunk size | 250 tokens | ~150–200 words, within the suggested range. Keeps chunks coherent and informative. |
| Overlap    | 15%  | Preserves context across boundaries and avoids cutting mid-topic. |
| Boundary   | Sentence / section | We split on sentence and section boundaries to avoid cutting in the middle of a thought. |

The dataset has many bullet lists and headings. We treat headings and list items as hard boundaries so chunks stay readable and semantically coherent.

### 3.2 Preprocessing

- Whitespace normalisation (trim, collapse multiple spaces)
- No lowercasing (course codes and names stay as-is)
- Markup removal was largely done in the dataset; we handle remaining structure (headings, bullets) in the splitter
- Course metadata (code, credits, semester, lecturers) is kept in chunk headers for better retrieval

---

## 4. Embedding & Indexing (4.2)

**Implementation**: `rag_system/src/embedding.py`, `rag_system/src/indexing.py`

### 4.1 Embedding model

We use **sentence-transformers/all-MiniLM-L6-v2**:

- 384 dimensions
- Runs locally (no API calls)
- Reasonable quality for semantic similarity
- Fast enough for our corpus size

We also tried `all-mpnet-base-v2` and `bge-small-en-v1.5` during development; MiniLM gave a good balance of speed and quality.

### 4.2 Indexing

- **Similarity**: Cosine similarity (inner product on L2-normalised vectors)
- **Index**: FAISS `IndexFlatIP` for exact search; sufficient for our dataset size
- **Scalability**: For larger corpora we could switch to IVF or HNSW; the code supports it but we did not use it here.

---

## 5. Retrieval Module (4.3)

**Implementation**: `rag_system/src/retrieval.py`

1. Encode the query with the same embedding model
2. Search the FAISS index for the top-*k* passages
3. Rank results by similarity score

We use *k* = 5 (as suggested: 1 is too few, 20 too many for our collection).

Following feedback from the previous assignment, we added candidate expansion: we first retrieve 3×*k* candidates, then rerank them by exact distances and select the final top-*k*. We also apply small boosts for section matches (e.g. “Prerequisites”) and exact course code matches when they appear in the query.

---

## 6. Answer Generation Module (4.4)

**Implementation**: `rag_system/src/generation.py`

- **Model**: GPT-4o (as required)
- **Prompt**: The system prompt tells the model to answer only from the provided context and to say when information is missing
- **Temperature**: 0.1 to keep answers factual and stable
- **Sources**: Retrieved passages are included in the output for inspection (`--compare` shows RAG vs. no-retrieval)

---

## 7. System Evaluation (4.5)

**Implementation**: `rag_system/src/evaluation.py` and scripts in `rag_system/scripts/`

### 7.1 Retrieval quality

We report **Recall@5** and **MRR** on a manually labelled set (pooled across baseline and agentic runs). Example results:

| System   | Recall@5 (mean±std) | MRR    | n     |
|----------|---------------------|--------|-------|
| Baseline | 0.80 ± 0.28         | 1.00   | 19    |
| Agentic  | 0.73 ± 0.30         | 1.00   | 19    |

We manually inspected top-*k* passages for a subset of questions. The labelling workflow uses `create_pooled_ablation_pack.py` and `score_pooled_ablation_pack.py`.

### 7.2 Answer quality

We compared answers **with** vs. **without** retrieval (baseline = GPT-4o only). RAG generally gives more accurate and grounded answers when the relevant context is retrieved.

We also used an LLM-as-judge to score faithfulness and relevance. Note: this is an automatic approximation; manual review remains important for validity.

### 7.3 Error analysis

#### Retrieval failures (≥3 cases)

From `generate_failure_case_report.py`:

1. **Question**: “In the Data Science study programme (2025–2026), which compulsory courses are in the 2nd semester?”  
   **Problem**: Many relevant chunks are spread across the study programme page; top-5 does not cover all of them (Recall@5 = 0.25).  
   **Possible fix**: Increase *k* for list-style questions, or use programme-specific retrieval.

2. **Question**: “Which compulsory course in the Computer Networks programme is taught by Juan Felipe Botero, and what is its exam period?”  
   **Problem**: The query mixes lecturer name and exam info. Dense retrieval favours general “Future Internet” chunks; lecturer-specific chunks rank lower.  
   **Possible fix**: Metadata filters (e.g. by lecturer) or multi-vector retrieval for structured fields.

3. **Question**: “In the Computer Networks study programme, list the compulsory courses in the 1st semester (names + course codes).”  
   **Problem**: Partially successful but some courses missed (Recall@5 = 0.31). Chunks are per-course, so a single programme overview chunk would help.  
   **Possible fix**: Add programme-level summary chunks or use a structured index for programme layout.

#### Hallucinations / incorrect answers (≥3 cases)

From `generate_hallucination_report.py` and LLM-judge:

1. **Question**: “Who teaches Database Systems and in which semester is it offered?”  
   **Problem**: Model invented “typically taught by faculty members” and “instructor may vary from year to year” when the context did not specify the lecturer.  
   **Cause**: Generic fallback when retrieval does not return lecturer info.

2. **Question**: “What is the thesis submission deadline?”  
   **Problem**: Model said “not enough information” despite context containing Master’s Thesis details.  
   **Cause**: Retrieval may have missed the right chunk, or the prompt led to an over-cautious answer.

3. **Question**: “Which compulsory course in the Software Engineering programme is taught by Hans Vangheluwe?”  
   **Problem**: Model gave an incorrect course name (“Model-Driven Software Engineering”).  
   **Cause**: Wrong chunk ranked high or confusion between similar course names.

---

## 8. Notes for the Instructor

*(Optional — add any points you want the instructor to consider: limitations, compute constraints, design choices, etc.)*

---

## 9. Plagiarism / Use of External Material

We used GenAI tools for brainstorming and code review. All external sources (papers, slides, websites) are cited below. Any copied or adapted code is marked in the source with a reference comment.

---

## References

- Assignment PDF: IR___Assignment_3___25_26.pdf
- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.
- Karpukhin, V., et al. (2020). Dense passage retrieval for open-domain question answering. EMNLP.
- Es, S., et al. (2024). RAGAS: Automated evaluation of retrieval augmented generation. EACL.
- FAISS documentation: https://github.com/facebookresearch/faiss
- Sentence-Transformers: https://www.sbert.net/
- OpenAI API documentation

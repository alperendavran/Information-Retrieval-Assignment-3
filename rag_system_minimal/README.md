# Minimal RAG System - Assignment 3

**Information Retrieval - Assignment 3**  
Retrieval Augmented Generation System

This is a **minimal implementation** that focuses solely on the assignment requirements. It implements all required components with clear explanations and no extra features.

## Assignment Components

This system implements:

- **4.1 Document Chunking (10%)**: Split dataset into passages (100-300 words), with preprocessing explanation
- **4.2 Embedding & Indexing (20%)**: Pretrained sentence embedding model, compute embeddings, build FAISS similarity index
- **4.3 Retrieval Module (20%)**: Encode query, retrieve top-k passages, rank by similarity
- **4.4 Answer Generation Module (30%)**: GPT-4o, accept query, retrieve top-k, insert into prompt, generate answer
- **4.5 System Evaluation (20%)**: Recall@k, manual inspection, compare with/without retrieval, error analysis

## Quick Start

### 1. Setup Environment

```bash
cd rag_system_minimal

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API Key

```bash
# Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 3. Run Setup

```bash
python main.py --setup
```

This will:
- Chunk documents into passages (100-300 words)
- Compute embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Build FAISS similarity index

### 4. Start Asking Questions

```bash
# Interactive mode
python main.py --interactive

# Single query
python main.py --query "What are the prerequisites for IoT?"

# Run evaluation
python main.py --evaluate
```

## System Architecture

```
User Query
    ↓
Query Encoding (Embedding Model)
    ↓
FAISS Index Search (Top-k)
    ↓
Retrieved Passages
    ↓
GPT-4o Generation (with context)
    ↓
Answer + Sources
```

## Configuration

Edit `config.py` to customize:

- `CHUNK_SIZE`: Target words per chunk (default: 250, range: 100-300)
- `CHUNK_OVERLAP`: Overlap ratio (default: 0.15 = 15%)
- `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `TOP_K`: Number of passages to retrieve (default: 5)
- `OPENAI_MODEL`: LLM for generation (default: gpt-4o, as required)

## Project Structure

```
rag_system_minimal/
├── config.py              # Configuration settings
├── main.py                # Main application entry point
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── src/                   # Source modules
│   ├── chunking.py       # Document chunking (4.1)
│   ├── embedding.py      # Embedding model (4.2)
│   ├── indexing.py       # FAISS indexing (4.2)
│   ├── retrieval.py      # Retrieval module (4.3)
│   ├── generation.py     # Answer generation (4.4)
│   └── evaluation.py     # Evaluation metrics (4.5)
│
├── data/                  # Processed data
│   └── chunks.json       # Chunked documents
│
└── embeddings/           # Saved embeddings and index
    ├── embeddings.npy    # Document embeddings
    ├── faiss_index.faiss # FAISS index
    └── faiss_index.pkl   # Index metadata
```

## Evaluation

The evaluation module (`src/evaluation.py`) provides:

1. **Retrieval Quality**:
   - `recall_at_k()`: Calculate Recall@k metric
   - Manual inspection framework

2. **Answer Quality**:
   - `compare_rag_vs_baseline()`: Compare RAG vs. no-RAG answers
   - Framework for correctness, completeness, hallucination evaluation

3. **Error Analysis**:
   - `find_retrieval_failures()`: Identify cases where retrieval failed
   - `find_hallucination_cases()`: Identify potential hallucinations

Run evaluation:
```bash
python main.py --evaluate
```

**Note**: For proper evaluation, you need:
- Test questions with ground truth relevant chunk IDs
- Manual inspection of top-k passages
- Manual review of answers for correctness/hallucination

## Design Decisions

### Chunking (4.1)
- **Chunk size**: 250 words (middle of 100-300 range)
- **Overlap**: 15% to preserve context across boundaries
- **Preprocessing**: Normalize whitespace, preserve sentence boundaries
- **Strategy**: Split by sentences, group until reaching chunk size

### Embedding (4.2)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - 384 dimensions
  - Fast inference
  - Good quality for semantic similarity
  - Runs locally (no API calls)
- **Normalization**: L2-normalize embeddings for cosine similarity

### Indexing (4.2)
- **Index type**: FAISS `IndexFlatIP` (Inner Product)
  - For L2-normalized vectors, inner product = cosine similarity
  - Exact search (no approximation)
  - Suitable for small to medium datasets

### Retrieval (4.3)
- **Top-k**: 5 passages (reasonable for this dataset)
- **Ranking**: By cosine similarity score (higher is better)

### Generation (4.4)
- **Model**: GPT-4o (assignment requirement)
- **Prompt**: System prompt instructs model to answer only from context
- **Temperature**: 0.1 (low for factual responses)

## References

1. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.
2. Karpukhin, V., et al. (2020). Dense passage retrieval for open-domain question answering. EMNLP.
3. Manning, C. D., et al. (2008). Introduction to Information Retrieval. Cambridge University Press.
4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.

## License

Academic use only - Assignment submission

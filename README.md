**Information Retrieval - Assignment 3**  
Retrieval Augmented Generation System

## 📋 Overview

This project implements a complete RAG (Retrieval Augmented Generation) system for answering questions about the University of Antwerp Computer Science Masters program. The system retrieves relevant passages from course descriptions and generates answers using GPT-4o (default) — you can override the model (e.g. `gpt-4o-mini`) via the `OPENAI_MODEL` environment variable for cheaper development runs.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query                                                     │
│      ↓                                                          │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Query Encoder │────│   FAISS Index   │                    │
│  │   (MiniLM-L6)   │    │   (Flat/IVF)    │                    │
│  └─────────────────┘    └────────┬────────┘                    │
│                                  ↓                              │
│                         ┌─────────────────┐                    │
│                         │  Top-k Passages │                    │
│                         └────────┬────────┘                    │
│                                  ↓                              │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              GPT-4o Generator                        │       │
│  │   System Prompt + Context + Query → Answer           │       │
│  └─────────────────────────────────────────────────────┘       │
│                                  ↓                              │
│                         Generated Answer                        │
│                         + Source Citations                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd rag_system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API Key

```bash
# Create .env file
echo "OPENAI_API_KEY=key" > .env

# Or export directly
export OPENAI_API_KEY= 

# Optional: cheaper dev model (assignment default remains GPT-4o)
export OPENAI_MODEL='gpt-4o-mini'
```

### 3. Run Setup (Chunking + Embedding + Indexing)

```bash
python main.py --setup
```

### 4. Start Asking Questions

```bash
# Interactive mode
python main.py --interactive

# Single query
python main.py --query "What are the prerequisites for IoT?"

# Demo with sample queries
python main.py --demo

# Compare RAG vs baseline
python main.py --compare "Who teaches the Master thesis?"
```

## 🤖 Agentic Mode (LangGraph) — Advanced Workflow (Optional)

This project also includes an **agentic RAG workflow** built with **LangGraph**:
- query understanding (rule-based tags)
- multi-query expansion + fusion (RRF)
- post-retrieval dedup + MMR diversification

Run it with the `--agentic` flag:

```bash
# Agentic interactive mode
python main.py --interactive --agentic

# Agentic single query
python main.py --query "What are the prerequisites for IoT?" --agentic

# Agentic demo
python main.py --demo --agentic
```

### Optional advanced knobs (config)

In `config.py` you can enable additional agentic behaviors:
- `AGENTIC_USE_LLM_TAGGER`: LLM-based intent + routing tags (more “agentic”)
- `AGENTIC_USE_HYDE`: HyDE query expansion (adds a hypothetical passage as retrieval query)
- `AGENTIC_USE_REFLECTION`: post-generation self-check (reduces hallucinations)

These are **OFF by default** to avoid extra API calls.

## 📁 Project Structure

```
rag_system/
├── config.py              # Configuration settings
├── main.py                # Main application entry point
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── src/                   # Source modules
│   ├── __init__.py
│   ├── chunking.py       # Document chunking (10%)
│   ├── embedding.py      # Embedding model (20%)
│   ├── indexing.py       # FAISS indexing (20%)
│   ├── retrieval.py      # Retrieval module (20%)
│   ├── generation.py     # Answer generation (30%)
│   └── evaluation.py     # Evaluation metrics (20%)
│
├── data/                  # Processed data
│   └── chunks.json       # Chunked documents
│
├── embeddings/           # Saved embeddings and index
│   ├── embeddings.npy    # Document embeddings
│   ├── faiss_index.faiss # FAISS index
│   └── faiss_index.pkl   # Index metadata
│
├── evaluation_results/   # Evaluation outputs
│   └── evaluation_*.json # Evaluation results
│
└── tests/                # Unit tests
    └── test_rag.py
```

## 🔧 Configuration

Edit `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 250 | Tokens per chunk |
| `CHUNK_OVERLAP` | 0.15 | Overlap ratio (15%) |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `TOP_K` | 5 | Passages to retrieve |
| `OPENAI_MODEL` | gpt-4o | LLM for generation |
| `TEMPERATURE` | 0.1 | Generation temperature |

## 📊 Evaluation

Run the full evaluation suite:

```bash
python main.py --evaluate
```

This evaluates:
- **Retrieval Quality**: Recall@k, MRR
- **Answer Quality**: Faithfulness, Relevance (LLM-as-judge)
- **RAG vs Baseline**: Comparison with/without retrieval
- **Error Analysis**: Retrieval failures, hallucinations

## 🧪 Ablation: baseline vs agentic (pooled labels)

### Quick Start: Full Automated Pipeline (Recommended)

**One command to run everything:**

```bash
python scripts/run_full_evaluation_pipeline.py \
  --top-n 20 \
  --use-llm-labeling \
  --llm-model gpt-4o-mini \
  --budget-usd 10.0 \
  --generate-hallucination-report \
  --out-dir evaluation_results/full_pipeline_$(date +%Y%m%d_%H%M%S)
```

This automatically:
1. Creates pooled ablation pack
2. Labels relevance using LLM (function calling)
3. Scores and generates ablation table
4. Generates failure case report
5. Generates hallucination report (optional)
6. Summarizes costs
7. Assembles final report markdown

### Manual Workflow (Step-by-Step)

If you prefer manual control:

1) Create a pooled annotation pack (NO API calls):

```bash
python scripts/create_pooled_ablation_pack.py --top-n 20 --out evaluation_results/pooled_ablation_pack.json
```

2) Label relevance (choose one method):

**Option A: LLM-based labeling (recommended, requires OpenAI):**
```bash
python scripts/llm_label_pooled_ablation_pack.py \
  --in evaluation_results/pooled_ablation_pack.json \
  --out evaluation_results/pooled_ablation_pack_labeled_llm.json \
  --model gpt-4o-mini \
  --budget-usd 10.0 \
  --resume
```

**Option B: Heuristic auto-labeling (no API calls, less accurate):**
```bash
python scripts/auto_label_pooled_ablation_pack.py \
  --in evaluation_results/pooled_ablation_pack.json \
  --out evaluation_results/pooled_ablation_pack_labeled_auto.json \
  --summary evaluation_results/auto_label_summary.md
```

**Option C: Manual labeling (open JSON and fill `is_relevant` for each candidate)**

3) Score and generate a markdown ablation table:

```bash
python scripts/score_pooled_ablation_pack.py \
  --in evaluation_results/pooled_ablation_pack_labeled_llm.json \
  --out evaluation_results/pooled_ablation_metrics.json \
  --md evaluation_results/ablation_table.md
```

4) Generate "worst retrieval failures" markdown (auto):

```bash
python scripts/generate_failure_case_report.py \
  --pack evaluation_results/pooled_ablation_pack_labeled_llm.json \
  --scored evaluation_results/pooled_ablation_metrics.json \
  --out evaluation_results/failure_cases.md \
  --k 5 --top 3
```
5) (Optional, requires OpenAI) Generate answer outputs for hallucination analysis:

```bash
python scripts/generate_hallucination_report.py \
  --pack evaluation_results/pooled_ablation_pack_labeled_llm.json \
  --out evaluation_results/hallucination_report.json \
  --k 5 --limit 12 --judge
```

## 💰 Cost tracking (estimated) + timestamped logs

When OpenAI calls are enabled, the system logs token usage and an **estimated USD cost** (based on `MODEL_PRICING_USD_PER_1M` in `config.py`) to:
- `evaluation_results/openai_cost_log.jsonl`

Summarize the log into a markdown table:

```bash
python scripts/summarize_openai_costs.py
```

If you want an **offline** cost estimate (no API calls), generate a run report:

```bash
python scripts/create_run_report.py --model gpt-4o-mini --limit 20 --out evaluation_results/run_report_gpt4o-mini_20q.md
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📖 Components

### 1. Document Chunking (`src/chunking.py`)
- Splits documents into 200-300 token passages
- Maintains semantic coherence (sentence boundaries)
- 15% overlap for context preservation

### 2. Embedding (`src/embedding.py`)
- Uses `all-MiniLM-L6-v2` (384-dimensional)
- L2-normalized for cosine similarity
- Runs locally (no API calls)

### 3. Indexing (`src/indexing.py`)
- FAISS flat index for exact search
- Inner product for normalized vectors = cosine similarity
- Supports save/load for persistence

### 4. Retrieval (`src/retrieval.py`)
- Encodes query with same model
- Retrieves top-k most similar passages
- Returns passages with similarity scores

### 5. Generation (`src/generation.py`)
- GPT-4o with custom system prompt
- Grounded in retrieved context
- Includes source citations

### 6. Evaluation (`src/evaluation.py`)
- Recall@k, MRR, Precision@k
- LLM-as-judge for faithfulness/relevance
- Error analysis utilities

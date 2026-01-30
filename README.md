# Information Retrieval — Assignment 3: RAG System

**University of Antwerp — Computer Science Masters**  
**Course**: Information Retrieval (2025–2026)  
**Instructor**: Prof. Toon Calders  
**Deadline**: 30/01/2026

---

This repository contains our submission for the third assignment. Below we describe how to run the system and where to find each component. The full report (design choices, evaluation, error analysis) is in `REPORT.md`. The system answers questions about the UAntwerp CS Masters programme using retrieved course descriptions and GPT-4o.

---

## How to Run

### 1. Setup

```bash
cd rag_system

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. API Key

Create a `.env` file in the `rag_system` folder:

```
OPENAI_API_KEY=your-api-key-here
```

(For development you can use `OPENAI_MODEL=gpt-4o-mini` to reduce costs; the assignment default is GPT-4o.)

### 3. Initial Setup (chunking, embeddings, index)

```bash
python main.py --setup
```

This creates chunks from the dataset, computes embeddings, and builds the FAISS index. Run it once before queries.

### 4. Queries

```bash
# Interactive Q&A
python main.py --interactive

# Single query
python main.py --query "What are the prerequisites for IoT?"

# RAG vs no-retrieval comparison
python main.py --compare "Who teaches the Master thesis?"

# Demo with sample questions
python main.py --demo
```

### 5. Optional: Agentic pipeline (LangGraph)

```bash
python main.py --interactive --agentic
```

---

## Dataset

We use the provided CS Masters dataset:

- `cs-data/course-pages.json` — course descriptions
- `cs-data/website-scraped.json` — scraped website content

After chunking we obtain around 500 passages (well above the required 150–200).

---

## Project Structure

```
Information-Retrieval-Assignment-3/
├── README.md
├── REPORT.md                 # Assignment report
├── IR___Assignment_3___25_26.pdf
│
├── cs-data/
│   ├── course-pages.json
│   └── website-scraped.json
│
└── rag_system/               # Main implementation
    ├── config.py             # Chunk size, top-k, model, etc.
    ├── main.py               # Entry point
    ├── requirements.txt
    ├── src/
    │   ├── chunking.py       # 4.1 Document chunking
    │   ├── embedding.py      # 4.2 Embeddings
    │   ├── indexing.py       # 4.2 FAISS index
    │   ├── retrieval.py      # 4.3 Retrieval
    │   ├── generation.py     # 4.4 GPT-4o answer generation
    │   ├── evaluation.py     # 4.5 Evaluation metrics
    │   └── ...               # Optional: agentic workflow, structured index
    ├── scripts/              # Evaluation and labelling scripts
    ├── evaluation_results/   # Outputs from evaluation runs
    └── tests/
```

---

## Evaluation

To run the full evaluation pipeline:

```bash
cd rag_system
python scripts/run_full_evaluation_pipeline.py \
  --top-n 20 \
  --use-llm-labeling \
  --llm-model gpt-4o-mini \
  --budget-usd 10.0 \
  --out-dir evaluation_results/full_pipeline_$(date +%Y%m%d_%H%M%S)
```

For manual evaluation steps, see the scripts in `rag_system/scripts/`:
- `create_pooled_ablation_pack.py` — create annotation pack
- `score_pooled_ablation_pack.py` — compute Recall@k, MRR, etc.
- `generate_failure_case_report.py` — retrieval failure analysis
- `generate_hallucination_report.py` — hallucination analysis

---

## Configuration

Main settings in `rag_system/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| CHUNK_SIZE | 250 | Tokens per chunk |
| CHUNK_OVERLAP | 0.15 | Overlap between chunks |
| TOP_K | 5 | Passages retrieved per query |
| OPENAI_MODEL | gpt-4o | Model for answer generation |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | Local sentence transformer |

---

## Report

The assignment report is in `REPORT.md`. It includes the system description, design choices, evaluation results, and error analysis.

---

## References

1. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.
2. Karpukhin, V., et al. (2020). Dense passage retrieval for open-domain question answering. EMNLP.
3. RAGAS: Automated evaluation of retrieval augmented generation. EACL 2024.

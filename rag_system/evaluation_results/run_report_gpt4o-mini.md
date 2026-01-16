## Run Report (Data-driven) + Offline Cost Estimate

- **Generated at**: 2026-01-12T18:11:49.264684
- **Assumed model for estimate**: `gpt-4o-mini`
- **Assumed pricing (USD per 1M tokens)**: {'input': 0.15, 'output': 0.6} (update in `config.py` if needed)

### Artifacts

- **Chunks**: `/Users/alperendavran/Desktop/ir hw3/rag_system/data/chunks.json` exists=True count=588
  - token_count p50=159, p90=239, p99=257, max=370
- **Embeddings**: `/Users/alperendavran/Desktop/ir hw3/rag_system/embeddings/embeddings.npy` exists=True shape=(588, 384)
- **FAISS index**: `/Users/alperendavran/Desktop/ir hw3/rag_system/embeddings/faiss_index.faiss` exists=True
- **Pooled pack**: `/Users/alperendavran/Desktop/ir hw3/rag_system/evaluation_results/pooled_ablation_pack.json` exists=True

### Offline cost estimate (no API calls made)

Estimated for a run over **12 questions** with assumed average completion of **350 tokens** per answer.

| Scenario | Avg prompt tokens | Assumed completion tokens | Est. cost per call (USD) | Est. total (USD) |
|---|---:|---:|---:|---:|
| Baseline (no retrieval) | 48 | 350 | 0.000217 | 0.002606 |
| Normal RAG (Retriever top-k) | 964 | 350 | 0.000355 | 0.004255 |
| Agentic RAG (tag+rewrite+RRF+MMR) | 662 | 350 | 0.000309 | 0.003712 |

- **Estimated total for baseline+normal+agentic (3 calls per question)**: **$0.010573**
- **Worst-case upper bound** (completion=1024 tokens): **$0.025132**

### How to run with GPT-4o-mini (actual)

Set environment variables (do NOT commit your key):

```bash
export OPENAI_MODEL='gpt-4o-mini'
export OPENAI_API_KEY='...your key...'
```

Then generate real logs and summarize them:

```bash
python scripts/generate_hallucination_report.py --pack evaluation_results/pooled_ablation_pack.json --out evaluation_results/hallucination_report.json --k 5 --limit 12 --judge
python scripts/summarize_openai_costs.py
```
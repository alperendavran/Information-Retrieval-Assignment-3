## Ablation: Retrieval Quality (pooled labels, Recall@5)

| System | Recall@k (mean±std) | MRR (mean±std) | nDCG@k (mean±std) | MAP (mean±std) | n |
|---|---:|---:|---:|---:|---:|
| `baseline` | 0.579±0.364 | 0.728±0.345 | 0.606±0.361 | 0.630±0.314 | 15 |
| `agentic` | 0.631±0.364 | 0.772±0.286 | 0.625±0.312 | 0.763±0.273 | 15 |

Notes:
- Metrics are aggregated only over questions where at least one candidate was labeled relevant.
- This table compares **systems on the same relevance labels** (pooled candidates).
## Ablation: Retrieval Quality (pooled labels, Recall@5)

| System | Recall@k (mean±std) | MRR (mean±std) | nDCG@k (mean±std) | MAP (mean±std) | n |
|---|---:|---:|---:|---:|---:|
| `baseline` | 0.822±0.252 | 0.971±0.118 | 0.964±0.100 | 0.958±0.126 | 17 |
| `agentic` | 0.822±0.252 | 1.000±0.000 | 0.986±0.055 | 1.000±0.000 | 17 |

Notes:
- Metrics are aggregated only over questions where at least one candidate was labeled relevant.
- This table compares **systems on the same relevance labels** (pooled candidates).
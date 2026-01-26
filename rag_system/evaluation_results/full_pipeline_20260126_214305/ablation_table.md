## Ablation: Retrieval Quality (pooled labels, Recall@5)

| System | Recall@k (mean±std) | MRR (mean±std) | nDCG@k (mean±std) | MAP (mean±std) | n |
|---|---:|---:|---:|---:|---:|
| `baseline` | 0.796±0.284 | 1.000±0.000 | 0.958±0.137 | 0.950±0.132 | 19 |
| `agentic` | 0.730±0.303 | 1.000±0.000 | 0.902±0.189 | 0.974±0.078 | 19 |

Notes:
- Metrics are aggregated only over questions where at least one candidate was labeled relevant.
- This table compares **systems on the same relevance labels** (pooled candidates).
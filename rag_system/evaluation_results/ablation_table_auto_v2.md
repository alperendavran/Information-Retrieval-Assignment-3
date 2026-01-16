## Ablation: Retrieval Quality (pooled labels, Recall@5)

| System | Recall@k (mean±std) | MRR (mean±std) | nDCG@k (mean±std) | MAP (mean±std) | n |
|---|---:|---:|---:|---:|---:|
| `baseline` | 0.567±0.397 | 0.641±0.376 | 0.562±0.397 | 0.594±0.357 | 14 |
| `agentic` | 0.489±0.415 | 0.685±0.432 | 0.521±0.407 | 0.667±0.424 | 14 |

Notes:
- Metrics are aggregated only over questions where at least one candidate was labeled relevant.
- This table compares **systems on the same relevance labels** (pooled candidates).
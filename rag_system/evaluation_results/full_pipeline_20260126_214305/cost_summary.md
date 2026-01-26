## OpenAI Cost Summary (Estimated)

- **Generated at**: 2026-01-26T21:52:27.711467
- **Log file**: `/Users/shakhzod/Desktop/Information-Retrieval-Assignment-3/rag_system/evaluation_results/openai_cost_log.jsonl`

### Totals

- **Estimated total cost (USD)**: 0.064062
- **Total prompt tokens**: 289991
- **Total completion tokens**: 34272
- **Total events**: 486

### By model

| Model | Events | Prompt tokens | Completion tokens | Est. cost (USD) |
|---|---:|---:|---:|---:|
| `gpt-4o-mini` | 486 | 289991 | 34272 | 0.064062 |

### By operation

| Operation | Events | Est. cost (USD) |
|---|---:|---:|
| `llm_label_relevance` | 322 | 0.035297 |
| `llm_judge_faithfulness` | 78 | 0.015455 |
| `answer_generation` | 65 | 0.010682 |
| `llm_judge_compare_answers` | 20 | 0.002531 |
| `tool_router` | 1 | 0.000096 |

### Last 50 events

| ts_utc | operation | model | prompt | completion | cost_usd |
|---|---|---|---:|---:|---:|
| 2026-01-26T20:51:00.402963+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 500 | 69 | 0.00011639999999999998 |
| 2026-01-26T20:51:01.838022+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 499 | 64 | 0.00011324999999999999 |
| 2026-01-26T20:51:03.579627+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 496 | 69 | 0.0001158 |
| 2026-01-26T20:51:05.216398+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 499 | 69 | 0.00011624999999999998 |
| 2026-01-26T20:51:06.855048+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 384 | 62 | 9.48e-05 |
| 2026-01-26T20:51:08.083777+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 429 | 51 | 9.495e-05 |
| 2026-01-26T20:51:09.619682+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 496 | 69 | 0.0001158 |
| 2026-01-26T20:51:11.053107+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 501 | 69 | 0.00011654999999999999 |
| 2026-01-26T20:51:12.691867+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 488 | 68 | 0.00011399999999999998 |
| 2026-01-26T20:51:14.534149+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 496 | 68 | 0.00011520000000000001 |
| 2026-01-26T20:51:15.918822+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 542 | 56 | 0.00011489999999999997 |
| 2026-01-26T20:51:17.812701+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 545 | 56 | 0.00011534999999999998 |
| 2026-01-26T20:51:19.348494+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 544 | 55 | 0.0001146 |
| 2026-01-26T20:51:20.781879+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 527 | 55 | 0.00011205 |
| 2026-01-26T20:51:22.013057+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 530 | 55 | 0.0001125 |
| 2026-01-26T20:51:24.263297+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 529 | 56 | 0.00011294999999999998 |
| 2026-01-26T20:51:25.900450+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 542 | 55 | 0.00011429999999999999 |
| 2026-01-26T20:51:27.334626+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 547 | 56 | 0.00011564999999999999 |
| 2026-01-26T20:51:28.770255+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 545 | 59 | 0.00011715 |
| 2026-01-26T20:51:30.069136+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 546 | 55 | 0.0001149 |
| 2026-01-26T20:51:32.045464+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 516 | 58 | 0.00011219999999999999 |
| 2026-01-26T20:51:33.274452+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 548 | 52 | 0.00011339999999999999 |
| 2026-01-26T20:51:35.117717+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 520 | 57 | 0.00011219999999999999 |
| 2026-01-26T20:51:36.652276+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 541 | 57 | 0.00011535000000000001 |
| 2026-01-26T20:51:38.397990+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 522 | 64 | 0.00011669999999999999 |
| 2026-01-26T20:51:40.651441+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 440 | 56 | 9.960000000000001e-05 |
| 2026-01-26T20:51:42.046315+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 442 | 49 | 9.57e-05 |
| 2026-01-26T20:51:43.309068+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 496 | 56 | 0.000108 |
| 2026-01-26T20:51:44.844438+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 440 | 51 | 9.66e-05 |
| 2026-01-26T20:51:46.380938+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 443 | 51 | 9.704999999999999e-05 |
| 2026-01-26T20:51:51.707328+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 511 | 74 | 0.00012104999999999999 |
| 2026-01-26T20:51:54.366122+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 519 | 77 | 0.00012405 |
| 2026-01-26T20:51:55.905305+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 511 | 56 | 0.00011025 |
| 2026-01-26T20:51:57.849851+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 464 | 65 | 0.00010859999999999998 |
| 2026-01-26T20:51:59.898835+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 437 | 66 | 0.00010515 |
| 2026-01-26T20:52:01.563206+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 459 | 61 | 0.00010544999999999999 |
| 2026-01-26T20:52:02.971409+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 392 | 62 | 9.6e-05 |
| 2026-01-26T20:52:05.119994+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 392 | 62 | 9.6e-05 |
| 2026-01-26T20:52:06.860030+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 395 | 62 | 9.645e-05 |
| 2026-01-26T20:52:08.747557+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 479 | 61 | 0.00010845 |
| 2026-01-26T20:52:10.137566+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 383 | 57 | 9.164999999999999e-05 |
| 2026-01-26T20:52:11.571187+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 444 | 66 | 0.00010619999999999999 |
| 2026-01-26T20:52:13.415846+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 453 | 54 | 0.00010035 |
| 2026-01-26T20:52:15.054582+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 472 | 61 | 0.0001074 |
| 2026-01-26T20:52:16.691294+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 488 | 62 | 0.0001104 |
| 2026-01-26T20:52:18.521301+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 470 | 52 | 0.00010169999999999999 |
| 2026-01-26T20:52:20.274144+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 456 | 61 | 0.00010499999999999999 |
| 2026-01-26T20:52:22.015328+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 463 | 61 | 0.00010604999999999999 |
| 2026-01-26T20:52:23.392009+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 482 | 65 | 0.0001113 |
| 2026-01-26T20:52:26.416524+00:00 | `llm_label_relevance` | `gpt-4o-mini` | 352 | 67 | 9.3e-05 |
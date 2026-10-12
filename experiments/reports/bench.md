# Throughput benchmark

15-epoch time_shaping trainings run by run_batch.py with P concurrent sessions; 32768 samples/epoch in all cases.
Per-session numbers are averages. est run = startup + 300 x s/epoch + eval (eval measured under load).

| P | envs/session | s/epoch | startup s | eval s | est min/run | runs/hour | bench wall s |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 0.20 | 0.6 | 0.5 | 1.0 | 59 | 5 |
| 2 | 4 | 0.32 | 0.5 | 0.5 | 1.6 | 74 | 7 |
| 3 | 3 | 0.45 | 0.5 | 0.6 | 2.2 | 80 | 9 |
| 5 | 2 | 0.65 | 0.6 | 0.7 | 3.3 | 92 | 13 |

Best: P=5, workers (num_envs)=2 -> 92 runs/hour.
Standalone evaluate(): 0.5 s.

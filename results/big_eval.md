n = 500 problems (all conditions complete)

| Condition | Accuracy | Msg size | Solver input | Reader s | 3B prefill s | 3B decode s | Answer tokens | Total s |
|---|---|---|---|---|---|---|---|---|
| Text message | 60.2% | 59.4 | 113.9 | 2.725 | 0.173 | 10.48 | 139 | 13.38 |
| Latent, stride 1 | 70.4% | 41.0 | 95.5 | 0.059 | 0.128 | 9.95 | 132 | 10.13 |
| Latent, stride 2 | 58.6% | 20.8 | 75.3 | 0.059 | 0.117 | 9.79 | 130 | 9.96 |
| Raw context (ref) | 77.6% | 41.3 | 95.8 | 0.000 | 0.138 | 10.71 | 142 | 10.84 |

McNemar (exact, two-sided) vs text message:

| Condition | Text wrong, it right | Text right, it wrong | p |
|---|---|---|---|
| Latent, stride 1 | 94 | 43 | 1.6e-05 |
| Latent, stride 2 | 75 | 83 | 0.58 |
| Raw context (ref) | 111 | 24 | 1.4e-14 |

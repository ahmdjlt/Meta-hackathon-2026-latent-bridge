| Condition | Accuracy | Tokens sent | Bytes sent | Receiver input len | Sender s | Receiver s | Total s |
|---|---|---|---|---|---|---|---|
| B: no message | 2.0% | 0.0 | 0 | 43.4 | 0.000 | 8.273 | 8.273 |
| A: text message | 59.0% | 59.9 | 243 | 114.3 | 2.724 | 10.973 | 13.697 |
| Ours: latent (token bridge, stride 1, L8) | 68.0% | 40.0 | 245576 | 94.3 | 0.058 | 9.828 | 9.886 |
| Ref: raw context text | 73.0% | 40.3 | 163 | 94.7 | 0.000 | 10.889 | 10.889 |

# canet_b4: CANet, EfficientNet-B4 encoder (ours)

512px windows at 1× scale. Test: 10 photographs, every pixel scored once at native resolution. Threshold chosen per run on validation; metrics pooled over all test pixels. The paper patches column is F1 on the NWRD paper's test patches with the authors' averaging at 0.5.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | Val IoU | Threshold | Best epoch | Paper patches F1 | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.4189 | 0.5905 | 0.5620 | 0.6220 | 0.3925 | 0.4422 | 0.90 | 24 | 0.5524 | 2.0 |
| 43 | 0.4318 | 0.6031 | 0.6061 | 0.6002 | 0.4149 | 0.4299 | 0.94 | 28 | 0.5671 | 1.9 |
| 44 | 0.4199 | 0.5914 | 0.5656 | 0.6197 | 0.4009 | 0.4357 | 0.92 | 27 | 0.5489 | 1.9 |
| **mean ± sd** | 0.4235 ± 0.0072 | 0.5950 ± 0.0070 | 0.5779 ± 0.0245 | 0.6140 ± 0.0120 | 0.4027 ± 0.0113 | 0.4359 ± 0.0061 | 0.92 | – | 0.5562 ± 0.0097 | 1.9 |

Side by side with the other models: comparison.md of the scale's results folder (python -m nwrd_public.compare results).

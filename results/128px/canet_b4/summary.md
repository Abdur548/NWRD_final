# canet_b4: CANet, EfficientNet-B4 encoder (ours)

128px windows at 0.5× scale. Test: 10 photographs, every pixel scored once at native resolution. Threshold chosen per run on validation; metrics pooled over all test pixels. The paper patches column is F1 on the NWRD paper's test patches with the authors' averaging at 0.5.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | Val IoU | Threshold | Best epoch | Paper patches F1 | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.4181 | 0.5897 | 0.5573 | 0.6260 | 0.3774 | 0.4054 | 0.90 | 24 | 0.5254 | 1.7 |
| 43 | 0.3933 | 0.5645 | 0.5301 | 0.6037 | 0.3554 | 0.4059 | 0.90 | 12 | 0.5065 | 1.7 |
| 44 | 0.4106 | 0.5822 | 0.5470 | 0.6223 | 0.3691 | 0.3904 | 0.94 | 26 | 0.5194 | 1.7 |
| **mean ± sd** | 0.4073 ± 0.0127 | 0.5788 ± 0.0129 | 0.5448 ± 0.0137 | 0.6173 ± 0.0119 | 0.3673 ± 0.0111 | 0.4006 ± 0.0088 | 0.91 | – | 0.5171 ± 0.0096 | 1.7 |

Side by side with the other models: comparison.md of the scale's results folder (python -m nwrd_public.compare results).

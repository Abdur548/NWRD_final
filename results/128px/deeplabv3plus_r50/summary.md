# deeplabv3plus_r50: DeepLabV3+, ResNet-50 encoder (baseline)

128px windows at 0.5× scale. Test: 10 photographs, every pixel scored once at native resolution. Threshold chosen per run on validation; metrics pooled over all test pixels. The paper patches column is F1 on the NWRD paper's test patches with the authors' averaging at 0.5.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | Val IoU | Threshold | Best epoch | Paper patches F1 | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.4252 | 0.5967 | 0.5471 | 0.6562 | 0.3888 | 0.4232 | 0.90 | 30 | 0.5427 | 0.7 |
| 43 | 0.3951 | 0.5665 | 0.5089 | 0.6387 | 0.3694 | 0.4174 | 0.88 | 28 | 0.5229 | 0.6 |
| 44 | 0.4090 | 0.5806 | 0.5205 | 0.6563 | 0.3772 | 0.4084 | 0.90 | 29 | 0.5384 | 0.6 |
| **mean ± sd** | 0.4098 ± 0.0150 | 0.5812 ± 0.0151 | 0.5255 ± 0.0196 | 0.6504 ± 0.0101 | 0.3784 ± 0.0098 | 0.4163 ± 0.0075 | 0.89 | – | 0.5347 ± 0.0104 | 0.6 |

Side by side with the other models: comparison.md of the scale's results folder (python -m nwrd_public.compare results).

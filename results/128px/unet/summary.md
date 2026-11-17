# unet: UNet, 4 encoder / 4 decoder blocks, from scratch (NWRD paper baseline)

128px windows at 0.5× scale. Test: 10 photographs, every pixel scored once at native resolution. Threshold chosen per run on validation; metrics pooled over all test pixels. The paper patches column is F1 on the NWRD paper's test patches with the authors' averaging at 0.5.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | Val IoU | Threshold | Best epoch | Paper patches F1 | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.3787 | 0.5494 | 0.4581 | 0.6861 | 0.3589 | 0.3459 | 0.72 | 19 | 0.5189 | 1.0 |
| 43 | 0.4433 | 0.6143 | 0.5601 | 0.6801 | 0.3722 | 0.3875 | 0.96 | 30 | 0.5339 | 1.0 |
| 44 | 0.3815 | 0.5522 | 0.4559 | 0.7001 | 0.2966 | 0.3634 | 0.95 | 10 | 0.4544 | 1.0 |
| **mean ± sd** | 0.4011 ± 0.0365 | 0.5720 ± 0.0367 | 0.4914 ± 0.0595 | 0.6888 ± 0.0103 | 0.3426 ± 0.0403 | 0.3656 ± 0.0209 | 0.88 | – | 0.5024 ± 0.0423 | 1.0 |

Side by side with the other models: comparison.md of the scale's results folder (python -m nwrd_public.compare results).

# unet: UNet, 4 encoder / 4 decoder blocks, from scratch (NWRD paper baseline)

512px windows at 1× scale. Test: 10 photographs, every pixel scored once at native resolution. Threshold chosen per run on validation; metrics pooled over all test pixels. The paper patches column is F1 on the NWRD paper's test patches with the authors' averaging at 0.5.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | Val IoU | Threshold | Best epoch | Paper patches F1 | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.4113 | 0.5829 | 0.5516 | 0.6179 | 0.2813 | 0.3382 | 0.98 | 24 | 0.4243 | 4.2 |
| 43 | 0.3879 | 0.5590 | 0.4586 | 0.7158 | 0.3103 | 0.3611 | 0.95 | 28 | 0.4658 | 4.1 |
| 44 | 0.3957 | 0.5671 | 0.5161 | 0.6293 | 0.2784 | 0.3370 | 0.95 | 22 | 0.4356 | 4.1 |
| **mean ± sd** | 0.3983 ± 0.0119 | 0.5697 ± 0.0121 | 0.5087 ± 0.0469 | 0.6543 ± 0.0535 | 0.2900 ± 0.0176 | 0.3454 ± 0.0136 | 0.96 | – | 0.4419 ± 0.0214 | 4.1 |

Side by side with the other models: comparison.md of the scale's results folder (python -m nwrd_public.compare results).

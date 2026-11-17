# deeplabv3plus_r50: DeepLabV3+, ResNet-50 encoder (baseline)

512px windows at 1× scale. Test: 10 photographs, every pixel scored once at native resolution. Threshold chosen per run on validation; metrics pooled over all test pixels. The paper patches column is F1 on the NWRD paper's test patches with the authors' averaging at 0.5.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | Val IoU | Threshold | Best epoch | Paper patches F1 | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.4477 | 0.6185 | 0.5672 | 0.6801 | 0.4418 | 0.4539 | 0.63 | 29 | 0.5835 | 1.2 |
| 43 | 0.4636 | 0.6335 | 0.6508 | 0.6171 | 0.4440 | 0.4360 | 0.96 | 27 | 0.5930 | 1.2 |
| 44 | 0.4602 | 0.6304 | 0.6054 | 0.6575 | 0.4412 | 0.4458 | 0.88 | 23 | 0.5991 | 1.2 |
| **mean ± sd** | 0.4572 ± 0.0084 | 0.6275 ± 0.0079 | 0.6078 ± 0.0419 | 0.6516 ± 0.0319 | 0.4423 ± 0.0015 | 0.4452 ± 0.0090 | 0.82 | – | 0.5919 ± 0.0079 | 1.2 |

Side by side with the other models: comparison.md of the scale's results folder (python -m nwrd_public.compare results).

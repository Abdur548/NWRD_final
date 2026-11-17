# CANet-B4 against the NWRD paper's baselines on the public NWRD release: 512px

512px windows of photographs at 1× scale. Test: the release's 10 test photographs, every pixel scored once at native resolution (0.63% rust). Threshold chosen per run on the 8 validation photographs (IoU-maximising, grid 0.01–0.99), then frozen for test. Metrics pooled over all test pixels; mean ± sample sd over seeds.

## Accuracy (test photographs, validation-selected threshold)

| Model | Seeds | IoU | Dice | Precision | Recall | Specificity | Accuracy | IoU @0.5 | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| **canet_b4** (ours) | 3 | 0.4235 ± 0.0072 | 0.5950 ± 0.0070 | 0.5779 ± 0.0245 | 0.6140 ± 0.0120 | 0.9972 ± 0.0003 | 0.9947 ± 0.0003 | 0.4027 ± 0.0113 | 0.92 |
| deeplabv3plus_r50 | 3 | 0.4572 ± 0.0084 | 0.6275 ± 0.0079 | 0.6078 ± 0.0419 | 0.6516 ± 0.0319 | 0.9973 ± 0.0006 | 0.9951 ± 0.0004 | 0.4423 ± 0.0015 | 0.82 |
| unet | 3 | 0.3983 ± 0.0119 | 0.5697 ± 0.0121 | 0.5087 ± 0.0469 | 0.6543 ± 0.0535 | 0.9959 ± 0.0011 | 0.9938 ± 0.0008 | 0.2900 ± 0.0176 | 0.96 |

## Each run

| Model | Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | Val IoU | Threshold | Best epoch | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| canet_b4 | 42 | 0.4189 | 0.5905 | 0.5620 | 0.6220 | 0.3925 | 0.4422 | 0.90 | 24 | 2.0 |
| canet_b4 | 43 | 0.4318 | 0.6031 | 0.6061 | 0.6002 | 0.4149 | 0.4299 | 0.94 | 28 | 1.9 |
| canet_b4 | 44 | 0.4199 | 0.5914 | 0.5656 | 0.6197 | 0.4009 | 0.4357 | 0.92 | 27 | 1.9 |
| deeplabv3plus_r50 | 42 | 0.4477 | 0.6185 | 0.5672 | 0.6801 | 0.4418 | 0.4539 | 0.63 | 29 | 1.2 |
| deeplabv3plus_r50 | 43 | 0.4636 | 0.6335 | 0.6508 | 0.6171 | 0.4440 | 0.4360 | 0.96 | 27 | 1.2 |
| deeplabv3plus_r50 | 44 | 0.4602 | 0.6304 | 0.6054 | 0.6575 | 0.4412 | 0.4458 | 0.88 | 23 | 1.2 |
| unet | 42 | 0.4113 | 0.5829 | 0.5516 | 0.6179 | 0.2813 | 0.3382 | 0.98 | 24 | 4.2 |
| unet | 43 | 0.3879 | 0.5590 | 0.4586 | 0.7158 | 0.3103 | 0.3611 | 0.95 | 28 | 4.1 |
| unet | 44 | 0.3957 | 0.5671 | 0.5161 | 0.6293 | 0.2784 | 0.3370 | 0.95 | 22 | 4.1 |

## Each test photograph (IoU at each run's threshold, mean over seeds)

| Photograph | Camera | Rust | canet_b4 | deeplabv3plus_r50 | unet |
|---|---|---|---|---|---|
| 105 | PAR-LX1M | 0.57% | 0.3606 | 0.3857 | 0.2603 |
| 123 | PAR-LX1M | 0.97% | 0.4513 | 0.4708 | 0.4329 |
| 25 | NIKON D3200 | 0.32% | 0.4100 | 0.3980 | 0.2942 |
| 29 | NIKON D3200 | 0.35% | 0.4948 | 0.5257 | 0.4661 |
| 62 | NIKON D3200 | 1.04% | 0.3959 | 0.4363 | 0.3760 |
| 63 | NIKON D3200 | 1.24% | 0.4369 | 0.4875 | 0.4493 |
| 67 | PAR-LX1M | 0.76% | 0.5373 | 0.6136 | 0.5772 |
| 77 | PAR-LX1M | 0.08% | 0.2358 | 0.2226 | 0.1719 |
| 80 | PAR-LX1M | 0.25% | 0.2534 | 0.3051 | 0.3613 |
| 89 | PAR-LX1M | 0.47% | 0.3592 | 0.3482 | 0.2995 |

## By camera (pooled test IoU, mean over seeds)

| Camera | Photographs | canet_b4 | deeplabv3plus_r50 | unet |
|---|---|---|---|---|
| NIKON D3200 | 4 | 0.4263 | 0.4632 | 0.4016 |
| PAR-LX1M | 6 | 0.4191 | 0.4488 | 0.3925 |

## Is the difference real? (CANet-B4 minus baseline, test photographs)

| Baseline | Paired seeds | ΔIoU | 95% CI, resampling photographs | ΔDice | ΔDice 95% CI | Seed-level t-test p | CANet-B4 higher (seeds) | Verdict |
|---|---|---|---|---|---|---|---|---|
| deeplabv3plus_r50 | 3 | -0.0337 | [-0.0476, -0.0127] (p 0.0020) | -0.0325 | [-0.0442, -0.0127] (p 0.0020) | 0.0103 | 0 of 3 | CANet-B4 is worse than deeplabv3plus_r50 |
| unet | 3 | +0.0252 | [-0.0006, +0.0612] (p 0.0616) | +0.0254 | [-0.0006, +0.0651] (p 0.0608) | 0.1378 | 3 of 3 | CANet-B4 is comparable to unet |

Verdict: the decision rule fixed before any result existed (docs/protocol.md §5). CANet-B4 *improves on* a baseline if its mean test IoU is higher, the 95% CI of ΔIoU that resamples test photographs excludes 0, and it is higher on at least 2 of 3 seeds; it *is worse* if that CI lies below 0; otherwise the two are *comparable*. The interval captures test-set sampling, not training-run variance; the seed-level t-test does, with little power at 3 seeds.

## On the NWRD paper's own test patches

The 2,737 128px patches of the half-size test photographs that the paper's test.py scores (394 with rust), in its order. Our predictions are those scored above, resized by area to half size; masks halved by area.

| Result | Precision | Recall | F1 / Dice | IoU |
|---|---|---|---|---|
| Paper, Table 3, UNet + APF, all test patches | 0.506 | 0.624 | 0.557 | – |
| Paper, Table 4, UNet + APF, background-only patches removed | 0.593 | 0.552 | 0.564 | 0.438 |
| canet_b4: all patches, the authors' averaging, threshold 0.5 | 0.476 | 0.678 | 0.556 | – |
| canet_b4: rust patches (≈ Table 4), the authors' averaging, 0.5 | 0.592 | 0.716 | 0.648 | – |
| canet_b4: all patches, pooled, validation threshold | 0.598 | 0.610 | 0.603 | 0.432 |
| canet_b4: rust patches, pooled, validation threshold | 0.685 | 0.610 | 0.645 | 0.476 |
| deeplabv3plus_r50: all patches, the authors' averaging, threshold 0.5 | 0.519 | 0.700 | 0.592 | – |
| deeplabv3plus_r50: rust patches (≈ Table 4), the authors' averaging, 0.5 | 0.631 | 0.732 | 0.677 | – |
| deeplabv3plus_r50: all patches, pooled, validation threshold | 0.630 | 0.645 | 0.636 | 0.466 |
| deeplabv3plus_r50: rust patches, pooled, validation threshold | 0.706 | 0.645 | 0.673 | 0.507 |
| unet: all patches, the authors' averaging, threshold 0.5 | 0.319 | 0.827 | 0.442 | – |
| unet: rust patches (≈ Table 4), the authors' averaging, 0.5 | 0.475 | 0.834 | 0.605 | – |
| unet: all patches, pooled, validation threshold | 0.530 | 0.646 | 0.579 | 0.407 |
| unet: rust patches, pooled, validation threshold | 0.656 | 0.646 | 0.648 | 0.480 |

The paper's rows are as published, on the same photographs. Table 3 is its UNet trained with APF (focal loss, 500 epochs, 128px patches at stride 32) and scored by its test.py on all the patches. Table 4 is a separately trained UNet (stride 128, background-only patches removed from training as well as test, paper §5), and its IoU was not computed by test.py. The paper's training set held twins of test photographs 80 and 89 (docs/dataset.md); no model here trained on a twin of a validation or test photograph. The `unet` rows are the paper's architecture re-trained under this protocol, so CANet-B4 against `unet` is the like-for-like comparison.

## Speed (Tesla T4, measured by NWRD_final; results/nwrd_final/)

Speed depends on the architecture and the window, not on the photographs, so it is not measured again. Seconds per photograph = windows per photograph / fp16 throughput at batch 8.

| Model | Params (M) | GFLOPs per window | Windows/s | s per photograph: Nikon D3200, 6016×4000 (96 windows) | s per photograph: Huawei PAR-LX1M, 4608×3456 or 3456×4608 (63 windows) |
|---|---|---|---|---|---|
| canet_b4 | 29.4 | 36.9 | 73 | 1.3 | 0.9 |
| deeplabv3plus_r50 | 26.7 | 73.1 | 95 | 1.0 | 0.7 |
| unet | 31.0 | 385.3 | 28 | 3.4 | 2.3 |

## Models

- `canet_b4` — CANet, EfficientNet-B4 encoder (ours)
- `deeplabv3plus_r50` — DeepLabV3+, ResNet-50 encoder (baseline)
- `unet` — UNet, 4 encoder / 4 decoder blocks, from scratch (NWRD paper baseline)

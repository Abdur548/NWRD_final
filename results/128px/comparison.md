# CANet-B4 against the NWRD paper's baselines on the public NWRD release: 128px

128px windows of photographs at 0.5× scale. Test: the release's 10 test photographs, every pixel scored once at native resolution (0.63% rust). Threshold chosen per run on the 8 validation photographs (IoU-maximising, grid 0.01–0.99), then frozen for test. Metrics pooled over all test pixels; mean ± sample sd over seeds.

## Accuracy (test photographs, validation-selected threshold)

| Model | Seeds | IoU | Dice | Precision | Recall | Specificity | Accuracy | IoU @0.5 | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| **canet_b4** (ours) | 3 | 0.4073 ± 0.0127 | 0.5788 ± 0.0129 | 0.5448 ± 0.0137 | 0.6173 ± 0.0119 | 0.9967 ± 0.0001 | 0.9944 ± 0.0002 | 0.3673 ± 0.0111 | 0.91 |
| deeplabv3plus_r50 | 3 | 0.4098 ± 0.0150 | 0.5812 ± 0.0151 | 0.5255 ± 0.0196 | 0.6504 ± 0.0101 | 0.9963 ± 0.0002 | 0.9941 ± 0.0003 | 0.3784 ± 0.0098 | 0.89 |
| unet | 3 | 0.4011 ± 0.0365 | 0.5720 ± 0.0367 | 0.4914 ± 0.0595 | 0.6888 ± 0.0103 | 0.9954 ± 0.0011 | 0.9935 ± 0.0010 | 0.3426 ± 0.0403 | 0.88 |

## Each run

| Model | Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | Val IoU | Threshold | Best epoch | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| canet_b4 | 42 | 0.4181 | 0.5897 | 0.5573 | 0.6260 | 0.3774 | 0.4054 | 0.90 | 24 | 1.7 |
| canet_b4 | 43 | 0.3933 | 0.5645 | 0.5301 | 0.6037 | 0.3554 | 0.4059 | 0.90 | 12 | 1.7 |
| canet_b4 | 44 | 0.4106 | 0.5822 | 0.5470 | 0.6223 | 0.3691 | 0.3904 | 0.94 | 26 | 1.7 |
| deeplabv3plus_r50 | 42 | 0.4252 | 0.5967 | 0.5471 | 0.6562 | 0.3888 | 0.4232 | 0.90 | 30 | 0.7 |
| deeplabv3plus_r50 | 43 | 0.3951 | 0.5665 | 0.5089 | 0.6387 | 0.3694 | 0.4174 | 0.88 | 28 | 0.6 |
| deeplabv3plus_r50 | 44 | 0.4090 | 0.5806 | 0.5205 | 0.6563 | 0.3772 | 0.4084 | 0.90 | 29 | 0.6 |
| unet | 42 | 0.3787 | 0.5494 | 0.4581 | 0.6861 | 0.3589 | 0.3459 | 0.72 | 19 | 1.0 |
| unet | 43 | 0.4433 | 0.6143 | 0.5601 | 0.6801 | 0.3722 | 0.3875 | 0.96 | 30 | 1.0 |
| unet | 44 | 0.3815 | 0.5522 | 0.4559 | 0.7001 | 0.2966 | 0.3634 | 0.95 | 10 | 1.0 |

## Each test photograph (IoU at each run's threshold, mean over seeds)

| Photograph | Camera | Rust | canet_b4 | deeplabv3plus_r50 | unet |
|---|---|---|---|---|---|
| 105 | PAR-LX1M | 0.57% | 0.3309 | 0.3754 | 0.2956 |
| 123 | PAR-LX1M | 0.97% | 0.3780 | 0.4596 | 0.4159 |
| 25 | NIKON D3200 | 0.32% | 0.3369 | 0.3270 | 0.3299 |
| 29 | NIKON D3200 | 0.35% | 0.4911 | 0.4997 | 0.4682 |
| 62 | NIKON D3200 | 1.04% | 0.3833 | 0.3964 | 0.3661 |
| 63 | NIKON D3200 | 1.24% | 0.4490 | 0.4286 | 0.4361 |
| 67 | PAR-LX1M | 0.76% | 0.5582 | 0.5680 | 0.5933 |
| 77 | PAR-LX1M | 0.08% | 0.2099 | 0.1284 | 0.2175 |
| 80 | PAR-LX1M | 0.25% | 0.3236 | 0.3268 | 0.4111 |
| 89 | PAR-LX1M | 0.47% | 0.3147 | 0.3264 | 0.3049 |

## By camera (pooled test IoU, mean over seeds)

| Camera | Photographs | canet_b4 | deeplabv3plus_r50 | unet |
|---|---|---|---|---|
| NIKON D3200 | 4 | 0.4176 | 0.4132 | 0.3985 |
| PAR-LX1M | 6 | 0.3916 | 0.4045 | 0.4068 |

## Is the difference real? (CANet-B4 minus baseline, test photographs)

| Baseline | Paired seeds | ΔIoU | 95% CI, resampling photographs | ΔDice | ΔDice 95% CI | Seed-level t-test p | CANet-B4 higher (seeds) | Verdict |
|---|---|---|---|---|---|---|---|---|
| deeplabv3plus_r50 | 3 | -0.0024 | [-0.0298, +0.0236] (p 0.8280) | -0.0024 | [-0.0300, +0.0247] (p 0.8300) | 0.4366 | 1 of 3 | CANet-B4 is comparable to deeplabv3plus_r50 |
| unet | 3 | +0.0062 | [-0.0154, +0.0177] (p 0.4924) | +0.0068 | [-0.0151, +0.0189] (p 0.4612) | 0.8468 | 2 of 3 | CANet-B4 is comparable to unet |

Verdict: the decision rule fixed before any result existed (docs/protocol.md §5). CANet-B4 *improves on* a baseline if its mean test IoU is higher, the 95% CI of ΔIoU that resamples test photographs excludes 0, and it is higher on at least 2 of 3 seeds; it *is worse* if that CI lies below 0; otherwise the two are *comparable*. The interval captures test-set sampling, not training-run variance; the seed-level t-test does, with little power at 3 seeds.

## On the NWRD paper's own test patches

The 2,737 128px patches of the half-size test photographs that the paper's test.py scores (394 with rust), in its order. Our predictions are those scored above, taken at half size; masks halved by area.

| Result | Precision | Recall | F1 / Dice | IoU |
|---|---|---|---|---|
| Paper, Table 3, UNet + APF, all test patches | 0.506 | 0.624 | 0.557 | – |
| Paper, Table 4, UNet + APF, background-only patches removed | 0.593 | 0.552 | 0.564 | 0.438 |
| canet_b4: all patches, the authors' averaging, threshold 0.5 | 0.408 | 0.735 | 0.517 | – |
| canet_b4: rust patches (≈ Table 4), the authors' averaging, 0.5 | 0.559 | 0.743 | 0.638 | – |
| canet_b4: all patches, pooled, validation threshold | 0.552 | 0.612 | 0.580 | 0.409 |
| canet_b4: rust patches, pooled, validation threshold | 0.679 | 0.612 | 0.644 | 0.475 |
| deeplabv3plus_r50: all patches, the authors' averaging, threshold 0.5 | 0.433 | 0.735 | 0.535 | – |
| deeplabv3plus_r50: rust patches (≈ Table 4), the authors' averaging, 0.5 | 0.578 | 0.752 | 0.653 | – |
| deeplabv3plus_r50: all patches, pooled, validation threshold | 0.541 | 0.652 | 0.591 | 0.420 |
| deeplabv3plus_r50: rust patches, pooled, validation threshold | 0.664 | 0.652 | 0.658 | 0.490 |
| unet: all patches, the authors' averaging, threshold 0.5 | 0.393 | 0.769 | 0.502 | – |
| unet: rust patches (≈ Table 4), the authors' averaging, 0.5 | 0.518 | 0.782 | 0.621 | – |
| unet: all patches, pooled, validation threshold | 0.499 | 0.688 | 0.577 | 0.406 |
| unet: rust patches, pooled, validation threshold | 0.616 | 0.688 | 0.649 | 0.481 |

The paper's rows are as published, on the same photographs. Table 3 is its UNet trained with APF (focal loss, 500 epochs, 128px patches at stride 32) and scored by its test.py on all the patches. Table 4 is a separately trained UNet (stride 128, background-only patches removed from training as well as test, paper §5), and its IoU was not computed by test.py. The paper's training set held twins of test photographs 80 and 89 (docs/dataset.md); no model here trained on a twin of a validation or test photograph. The `unet` rows are the paper's architecture re-trained under this protocol, so CANet-B4 against `unet` is the like-for-like comparison.

## Speed (Tesla T4, measured by NWRD_final; results/nwrd_final/)

Speed depends on the architecture and the window, not on the photographs, so it is not measured again. Seconds per photograph = windows per photograph / fp16 throughput at batch 8.

| Model | Params (M) | GFLOPs per window | Windows/s | s per photograph: Nikon D3200, 6016×4000 (384 windows) | s per photograph: Huawei PAR-LX1M, 4608×3456 or 3456×4608 (252 windows) |
|---|---|---|---|---|---|
| canet_b4 | 29.4 | 2.3 | 309 | 1.2 | 0.8 |
| deeplabv3plus_r50 | 26.7 | 4.6 | 987 | 0.4 | 0.3 |
| unet | 31.0 | 24.1 | 474 | 0.8 | 0.5 |

## Models

- `canet_b4` — CANet, EfficientNet-B4 encoder (ours)
- `deeplabv3plus_r50` — DeepLabV3+, ResNet-50 encoder (baseline)
- `unet` — UNet, 4 encoder / 4 decoder blocks, from scratch (NWRD paper baseline)

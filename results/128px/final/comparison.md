# CANet-B4 vs baselines: comparison

Test split: 3840 patches, 722 with rust, 8.1% rust pixels. Threshold chosen per run on val (IoU-maximising, grid 0.05–0.95), then frozen for test. Metrics pooled over all test pixels; mean ± sample std over seeds.

## Accuracy (test split, val-selected threshold)

| Model | Seeds | IoU | Dice | Precision | Recall | Specificity | Accuracy | IoU @0.5 | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| **canet_b4** (ours) | 3 | 0.7041 ± 0.0045 | 0.8264 ± 0.0031 | 0.8167 ± 0.0191 | 0.8367 ± 0.0134 | 0.9835 ± 0.0023 | 0.9716 ± 0.0011 | 0.6838 ± 0.0078 | 0.72 |
| deeplabv3plus_r50 | 3 | 0.7320 ± 0.0050 | 0.8452 ± 0.0034 | 0.8445 ± 0.0142 | 0.8462 ± 0.0095 | 0.9863 ± 0.0016 | 0.9750 ± 0.0008 | 0.7177 ± 0.0074 | 0.71 |
| unet | 3 | 0.7495 ± 0.0094 | 0.8568 ± 0.0061 | 0.8505 ± 0.0124 | 0.8633 ± 0.0061 | 0.9867 ± 0.0013 | 0.9767 ± 0.0011 | 0.7311 ± 0.0104 | 0.65 |

## Against the NWRD paper (UNet + APF)

| Result | Precision | Recall | F1 / Dice | IoU |
|---|---|---|---|---|
| Paper, Table 3, full test set | 0.506 | 0.624 | 0.557 | – |
| Paper, Table 4, background-only patches removed | 0.593 | 0.552 | 0.564 | 0.438 |
| canet_b4, re-run: all test patches | 0.817 | 0.837 | 0.826 | 0.704 |
| canet_b4, re-run: rust-positive test patches (≈ Table 4) | 0.918 | 0.837 | 0.875 | 0.778 |
| deeplabv3plus_r50, re-run: all test patches | 0.844 | 0.846 | 0.845 | 0.732 |
| deeplabv3plus_r50, re-run: rust-positive test patches (≈ Table 4) | 0.921 | 0.846 | 0.882 | 0.789 |
| unet, re-run: all test patches | 0.850 | 0.863 | 0.857 | 0.749 |
| unet, re-run: rust-positive test patches (≈ Table 4) | 0.914 | 0.863 | 0.888 | 0.798 |

Every model here sees 128px tiles re-cut at 0.5× scale, the paper's own patch size and field of view (nwrd/tiles.py). The paper's test photographs, training (APF, 500 epochs, focal loss) and patch sampling still differ, so its rows are reference points, not a paired comparison. The `unet` row is the paper's architecture re-trained under this protocol, so CANet vs `unet` is the like-for-like comparison.

## Is the difference real? (CANet-B4 minus baseline, test split)

| Baseline | Paired seeds | ΔIoU | 95% CI, resampling photographs | 95% CI, resampling patches | ΔDice | ΔDice 95% CI, photographs | seed-level t-test p | CANet higher (seeds) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| deeplabv3plus_r50 | 3 | -0.0279 | [-0.0544, -0.0026] (p 0.0328) | [-0.0426, -0.0138] (p 0.0000) | -0.0189 | [-0.0354, -0.0018] (p 0.0332) | 0.0130 | 0 of 3 | CANet-B4 is worse than deeplabv3plus_r50 |
| unet | 3 | -0.0454 | [-0.0809, -0.0056] (p 0.0276) | [-0.0608, -0.0299] (p 0.0000) | -0.0304 | [-0.0556, -0.0037] (p 0.0276) | 0.0297 | 0 of 3 | CANet-B4 is worse than unet |

Verdict: the decision rule fixed before the results were seen. CANet-B4 *improves on* a baseline if its mean test IoU is higher, the photograph-resampling 95% CI of ΔIoU excludes 0, and it is higher on at least 2 of 3 seeds; it *is worse* if that CI lies below 0; otherwise the two are *comparable*.

A CI that excludes 0 means the gap is larger than test-set sampling noise. Use the photograph interval: the test patches are cut from a few photographs, and patches of one photograph are not independent, so resampling patches understates the uncertainty. Neither bootstrap captures training-run variance; the seed-level t-test does, but has little power with 3 seeds.

## Efficiency (Tesla T4, 128x128 input, torch 2.10.0+cu128)

| Model | Params (M) | GFLOPs | latency_ms_bs1_fp32 | latency_ms_bs1_fp16 | throughput_img_s_bs8_fp16 | peak_mem_MB_bs8_fp16 | throughput_img_s_bs64_fp16 | peak_mem_MB_bs64_fp16 | CPU latency bs1 (ms) | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| canet_b4 | 29.39 | 2.3 | 22.6 | 24.2 | 308.6 | 262.2 | 868.5 | 646.3 | 67 | 3.1 |
| deeplabv3plus_r50 | 26.68 | 4.6 | 6.3 | 7.3 | 987.4 | 200.5 | 1322.7 | 802.4 | 57 | 1.2 |
| unet | 31.04 | 24.1 | 7.2 | 3.6 | 474.0 | 441.5 | 524.3 | 1540.0 | 156 | 1.7 |

## Models

- `canet_b4` — CANet, EfficientNet-B4 encoder (ours)
- `deeplabv3plus_r50` — DeepLabV3+, ResNet-50 encoder (baseline)
- `unet` — UNet, 4 encoder / 4 decoder blocks, from scratch (NWRD paper baseline)

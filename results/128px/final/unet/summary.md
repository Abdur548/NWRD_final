# unet: UNet, 4 encoder / 4 decoder blocks, from scratch (NWRD paper baseline)

Test split: 3840 patches, 722 with rust. Threshold chosen per run on validation; metrics pooled over all test pixels.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | IoU, rust patches | Val IoU | Threshold | Best epoch | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.7577 | 0.8621 | 0.8645 | 0.8598 | 0.7416 | 0.7982 | 0.8203 | 0.60 | 23 | 1.7 |
| 43 | 0.7515 | 0.8581 | 0.8463 | 0.8703 | 0.7310 | 0.8014 | 0.8220 | 0.69 | 21 | 1.7 |
| 44 | 0.7393 | 0.8501 | 0.8406 | 0.8597 | 0.7207 | 0.7951 | 0.8137 | 0.66 | 21 | 1.8 |
| **mean ± std** | 0.7495 ± 0.0094 | 0.8568 ± 0.0061 | 0.8505 ± 0.0124 | 0.8633 ± 0.0061 | 0.7311 ± 0.0104 | 0.7982 ± 0.0031 | 0.8187 ± 0.0044 | 0.65 | – | 1.7 |

## Against CANet-B4 (CANet-B4 minus baseline, pre-registered rule)

- vs `unet`: ΔIoU -0.0454, 95% CI [-0.0809, -0.0056] resampling test photographs; CANet-B4 higher on 0 of 3 seeds: CANet-B4 **is worse than** `unet`.

## Efficiency (Tesla T4, 128x128 input)

31.04M parameters, 24.08 GFLOPs; latency_ms_bs1_fp32 7.3, latency_ms_bs1_fp16 3.7, throughput_img_s_bs8_fp16 468.0, peak_mem_MB_bs8_fp16 288.6, throughput_img_s_bs64_fp16 529.8, peak_mem_MB_bs64_fp16 1540.0; CPU latency bs1 162 ms.

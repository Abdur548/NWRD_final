# canet_b4: CANet, EfficientNet-B4 encoder (ours)

Test split: 960 patches, 266 with rust. Threshold chosen per run on validation; metrics pooled over all test pixels.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | IoU, rust patches | Val IoU | Threshold | Best epoch | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.7184 | 0.8362 | 0.8262 | 0.8464 | 0.7069 | 0.7789 | 0.8193 | 0.76 | 12 | 3.3 |
| 43 | 0.7435 | 0.8529 | 0.8451 | 0.8609 | 0.7238 | 0.8000 | 0.8432 | 0.76 | 9 | 3.2 |
| 44 | 0.7237 | 0.8397 | 0.8493 | 0.8304 | 0.7069 | 0.7775 | 0.8097 | 0.86 | 23 | 3.1 |
| **mean ± std** | 0.7286 ± 0.0132 | 0.8429 ± 0.0088 | 0.8402 ± 0.0123 | 0.8459 ± 0.0153 | 0.7125 ± 0.0098 | 0.7855 ± 0.0126 | 0.8240 ± 0.0172 | 0.79 | – | 3.2 |

## Against CANet-B4 (CANet-B4 minus baseline, pre-registered rule)

- vs `deeplabv3plus_r50`: ΔIoU -0.0108, 95% CI [-0.0241, +0.0079] resampling test photographs; CANet-B4 higher on 0 of 3 seeds: CANet-B4 **is comparable to** `deeplabv3plus_r50`.
- vs `unet`: ΔIoU -0.0189, 95% CI [-0.0421, +0.0186] resampling test photographs; CANet-B4 higher on 0 of 3 seeds: CANet-B4 **is comparable to** `unet`.
- vs `unet_tuned`: ΔIoU +0.1019, 95% CI [+0.0056, +0.1886] resampling test photographs; CANet-B4 higher on 1 of 1 seeds: CANet-B4 **is comparable to** `unet_tuned`.

## Efficiency (Tesla T4, 512x512 input)

29.39M parameters, 36.87 GFLOPs; latency_ms_bs1_fp32 30.6, latency_ms_bs1_fp16 26.3, throughput_img_s_bs8_fp16 72.6, peak_mem_MB_bs8_fp16 743.7; CPU latency bs1 725 ms.

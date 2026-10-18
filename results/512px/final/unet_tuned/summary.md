# unet_tuned: UNet (paper architecture) with a from-scratch lr (config.unet_tuned_lr)

Test split: 960 patches, 266 with rust. Threshold chosen per run on validation; metrics pooled over all test pixels.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | IoU, rust patches | Val IoU | Threshold | Best epoch | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.6165 | 0.7628 | 0.6846 | 0.8610 | 0.6165 | 0.7283 | 0.7978 | 0.50 | 22 | 7.0 |
| **mean ± std** | 0.6165 | 0.7628 | 0.6846 | 0.8610 | 0.6165 | 0.7283 | 0.7978 | 0.50 | – | 7.0 |

## Against CANet-B4 (CANet-B4 minus baseline, pre-registered rule)

- vs `unet_tuned`: ΔIoU +0.1019, 95% CI [+0.0056, +0.1886] resampling test photographs; CANet-B4 higher on 1 of 1 seeds: CANet-B4 **is comparable to** `unet_tuned`.

## Efficiency (Tesla T4, 512x512 input)

31.04M parameters, 385.34 GFLOPs; latency_ms_bs1_fp32 96.3, latency_ms_bs1_fp16 41.0, throughput_img_s_bs8_fp16 27.7, peak_mem_MB_bs8_fp16 1872.6; CPU latency bs1 3024 ms.

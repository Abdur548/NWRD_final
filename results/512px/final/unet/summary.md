# unet: UNet, 4 encoder / 4 decoder blocks, from scratch (NWRD paper baseline)

Test split: 960 patches, 266 with rust. Threshold chosen per run on validation; metrics pooled over all test pixels.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | IoU, rust patches | Val IoU | Threshold | Best epoch | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.7368 | 0.8485 | 0.8233 | 0.8752 | 0.6980 | 0.7813 | 0.8164 | 0.76 | 28 | 7.3 |
| 43 | 0.7455 | 0.8542 | 0.8320 | 0.8777 | 0.7221 | 0.7935 | 0.8222 | 0.68 | 30 | 6.2 |
| 44 | 0.7599 | 0.8636 | 0.8495 | 0.8781 | 0.7432 | 0.7948 | 0.8206 | 0.66 | 30 | 6.2 |
| **mean ± std** | 0.7474 ± 0.0117 | 0.8554 ± 0.0076 | 0.8350 ± 0.0134 | 0.8770 ± 0.0015 | 0.7211 ± 0.0226 | 0.7899 ± 0.0074 | 0.8197 ± 0.0030 | 0.70 | – | 6.6 |

## Against CANet-B4 (CANet-B4 minus baseline, pre-registered rule)

- vs `unet`: ΔIoU -0.0189, 95% CI [-0.0421, +0.0186] resampling test photographs; CANet-B4 higher on 0 of 3 seeds: CANet-B4 **is comparable to** `unet`.

## Efficiency (Tesla T4, 512x512 input)

31.04M parameters, 385.34 GFLOPs; latency_ms_bs1_fp32 94.9, latency_ms_bs1_fp16 40.1, throughput_img_s_bs8_fp16 28.0, peak_mem_MB_bs8_fp16 1873.9; CPU latency bs1 3157 ms.

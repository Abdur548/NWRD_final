# deeplabv3plus_r50: DeepLabV3+, ResNet-50 encoder (baseline)

Test split: 3840 patches, 722 with rust. Threshold chosen per run on validation; metrics pooled over all test pixels.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | IoU, rust patches | Val IoU | Threshold | Best epoch | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.7350 | 0.8472 | 0.8427 | 0.8518 | 0.7224 | 0.7922 | 0.8131 | 0.68 | 8 | 1.2 |
| 43 | 0.7262 | 0.8414 | 0.8312 | 0.8517 | 0.7092 | 0.7929 | 0.8365 | 0.72 | 10 | 1.2 |
| 44 | 0.7348 | 0.8472 | 0.8594 | 0.8353 | 0.7215 | 0.7819 | 0.8233 | 0.72 | 16 | 1.2 |
| **mean ± std** | 0.7320 ± 0.0050 | 0.8452 ± 0.0034 | 0.8445 ± 0.0142 | 0.8462 ± 0.0095 | 0.7177 ± 0.0074 | 0.7890 ± 0.0062 | 0.8243 ± 0.0117 | 0.71 | – | 1.2 |

## Against CANet-B4 (CANet-B4 minus baseline, pre-registered rule)

- vs `deeplabv3plus_r50`: ΔIoU -0.0279, 95% CI [-0.0544, -0.0026] resampling test photographs; CANet-B4 higher on 0 of 3 seeds: CANet-B4 **is worse than** `deeplabv3plus_r50`.

## Efficiency (Tesla T4, 128x128 input)

26.68M parameters, 4.57 GFLOPs; latency_ms_bs1_fp32 6.3, latency_ms_bs1_fp16 7.3, throughput_img_s_bs8_fp16 987.4, peak_mem_MB_bs8_fp16 200.5, throughput_img_s_bs64_fp16 1322.7, peak_mem_MB_bs64_fp16 802.4; CPU latency bs1 57 ms.

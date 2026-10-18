# canet_b4: CANet, EfficientNet-B4 encoder (ours)

Test split: 3840 patches, 722 with rust. Threshold chosen per run on validation; metrics pooled over all test pixels.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | IoU, rust patches | Val IoU | Threshold | Best epoch | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.7008 | 0.8241 | 0.8035 | 0.8457 | 0.6749 | 0.7839 | 0.8282 | 0.75 | 13 | 3.1 |
| 43 | 0.7024 | 0.8252 | 0.8080 | 0.8431 | 0.6869 | 0.7822 | 0.8112 | 0.65 | 11 | 3.1 |
| 44 | 0.7092 | 0.8299 | 0.8386 | 0.8214 | 0.6896 | 0.7688 | 0.8000 | 0.75 | 10 | 3.1 |
| **mean ± std** | 0.7041 ± 0.0045 | 0.8264 ± 0.0031 | 0.8167 ± 0.0191 | 0.8367 ± 0.0134 | 0.6838 ± 0.0078 | 0.7783 ± 0.0083 | 0.8131 ± 0.0142 | 0.72 | – | 3.1 |

## Against CANet-B4 (CANet-B4 minus baseline, pre-registered rule)

- vs `unet`: ΔIoU -0.0454, 95% CI [-0.0809, -0.0056] resampling test photographs; CANet-B4 higher on 0 of 3 seeds: CANet-B4 **is worse than** `unet`.

## Efficiency (Tesla T4, 128x128 input)

29.39M parameters, 2.31 GFLOPs; latency_ms_bs1_fp32 23.8, latency_ms_bs1_fp16 25.4, throughput_img_s_bs8_fp16 298.7, peak_mem_MB_bs8_fp16 204.8, throughput_img_s_bs64_fp16 864.9, peak_mem_MB_bs64_fp16 646.9; CPU latency bs1 72 ms.

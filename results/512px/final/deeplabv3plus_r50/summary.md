# deeplabv3plus_r50: DeepLabV3+, ResNet-50 encoder (baseline)

Test split: 960 patches, 266 with rust. Threshold chosen per run on validation; metrics pooled over all test pixels.

| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | IoU, rust patches | Val IoU | Threshold | Best epoch | Train min/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.7193 | 0.8368 | 0.7857 | 0.8949 | 0.6882 | 0.7916 | 0.8242 | 0.69 | 17 | 2.3 |
| 43 | 0.7603 | 0.8638 | 0.8693 | 0.8584 | 0.7357 | 0.7964 | 0.8428 | 0.81 | 12 | 2.1 |
| 44 | 0.7385 | 0.8496 | 0.8457 | 0.8534 | 0.7326 | 0.7711 | 0.8150 | 0.60 | 16 | 2.1 |
| **mean ± std** | 0.7394 ± 0.0205 | 0.8500 ± 0.0135 | 0.8336 ± 0.0431 | 0.8689 ± 0.0227 | 0.7188 ± 0.0265 | 0.7864 ± 0.0134 | 0.8273 ± 0.0142 | 0.70 | – | 2.1 |

## Against CANet-B4 (CANet-B4 minus baseline, pre-registered rule)

- vs `deeplabv3plus_r50`: ΔIoU -0.0108, 95% CI [-0.0241, +0.0079] resampling test photographs; CANet-B4 higher on 0 of 3 seeds: CANet-B4 **is comparable to** `deeplabv3plus_r50`.

## Efficiency (Tesla T4, 512x512 input)

26.68M parameters, 73.12 GFLOPs; latency_ms_bs1_fp32 27.3, latency_ms_bs1_fp16 10.8, throughput_img_s_bs8_fp16 95.4, peak_mem_MB_bs8_fp16 834.7; CPU latency bs1 579 ms.

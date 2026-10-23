Tiles: 128px at 0.5× scale. Test: 3840 tiles (722 with rust) from 960 patches. Rust pixel share, train: 9.77% of patches, 9.79% of tiles.

## Accuracy at 128px (test tiles, threshold chosen on validation)

| Model | Seeds | IoU | Dice / F1 | Precision | Recall | IoU per seed |
|---|---|---|---|---|---|---|
| canet_b4 | 3 | 0.704 ± 0.004 | 0.826 ± 0.003 | 0.817 ± 0.019 | 0.837 ± 0.013 | 42: 0.701, 43: 0.702, 44: 0.709 |
| unet | 3 | 0.749 ± 0.009 | 0.857 ± 0.006 | 0.850 ± 0.012 | 0.863 ± 0.006 | 42: 0.758, 43: 0.752, 44: 0.739 |
| deeplabv3plus_r50 | 3 | 0.732 ± 0.005 | 0.845 ± 0.003 | 0.844 ± 0.014 | 0.846 ± 0.010 | 42: 0.735, 43: 0.726, 44: 0.735 |

CANet − the paper's UNet: ΔIoU -0.0454, 95% CI [-0.0809, -0.0056] resampling test photographs (p 0.028); CANet higher on 0 of 3 seeds; seed-level t-test p 0.030.
Pre-registered decision rule at 128px: CANet-B4 **is worse than** the paper's UNet.

CANet − DeepLabV3+ R50: ΔIoU -0.0279, 95% CI [-0.0544, -0.0026] resampling test photographs (p 0.033); CANet higher on 0 of 3 seeds; seed-level t-test p 0.013.
Pre-registered decision rule at 128px: CANet-B4 **is worse than** DeepLabV3+ R50.

## Against the paper's published UNet + APF

| Source | Test set | Precision | Recall | F1 / Dice | IoU |
|---|---|---|---|---|---|
| Paper, Table 3 | full test set | 0.506 | 0.624 | 0.557 | – |
| canet_b4, 128px | all test tiles | 0.817 | 0.837 | 0.826 | 0.704 |
| unet, 128px | all test tiles | 0.850 | 0.863 | 0.857 | 0.749 |
| deeplabv3plus_r50, 128px | all test tiles | 0.844 | 0.846 | 0.845 | 0.732 |
| Paper, Table 4 | background-only patches removed | 0.593 | 0.552 | 0.564 | 0.438 |
| canet_b4, 128px | rust tiles only | 0.918 | 0.837 | 0.875 | 0.778 |
| unet, 128px | rust tiles only | 0.914 | 0.863 | 0.888 | 0.798 |
| deeplabv3plus_r50, 128px | rust tiles only | 0.921 | 0.846 | 0.882 | 0.789 |

Same patch size and downsampling as the paper. Still different: the photographs (the paper's are the public NWRD release; this study's are not, docs/protocol.md §3.1), the paper's test split (random 80/10/10), its training (APF, RMSprop, 500 epochs, focal loss) and patch sampling, so these rows are context, not a paired comparison.

## Effect of scale on each model (test IoU, mean over seeds)

| Model | 512px native | 128px at 0.5× | Change | Seeds (512 / 128) |
|---|---|---|---|---|
| canet_b4 | 0.729 ± 0.013 | 0.704 ± 0.004 | -0.024 | 3 / 3 |
| unet | 0.747 ± 0.012 | 0.749 ± 0.009 | +0.002 | 3 / 3 |
| deeplabv3plus_r50 | 0.739 ± 0.021 | 0.732 ± 0.005 | -0.007 | 3 / 3 |

## Efficiency

Measured on Tesla T4 (torch 2.10.0+cu128), random weights.

| Scale | Model | Params (M) | GFLOPs | GPU fp32 bs1 (ms) | GPU fp16 bs1 (ms) | fp16 throughput (img/s) | CPU bs1 (ms) | Train min/epoch |
|---|---|---|---|---|---|---|---|---|
| 512px | canet_b4 | 29.4 | 36.87 | 30.6 | 26.3 | 73 @bs8 | 725 | 3.2 |
| 512px | unet | 31.0 | 385.34 | 94.9 | 40.1 | 28 @bs8 | 3157 | 6.6 |
| 512px | deeplabv3plus_r50 | 26.7 | 73.12 | 27.3 | 10.8 | 95 @bs8 | 579 | 2.1 |
| 128px | canet_b4 | 29.4 | 2.31 | 22.6 | 24.2 | 309 @bs8, 869 @bs64 | 67 | 3.1 |
| 128px | unet | 31.0 | 24.08 | 7.2 | 3.6 | 474 @bs8, 524 @bs64 | 156 | 1.7 |
| 128px | deeplabv3plus_r50 | 26.7 | 4.57 | 6.3 | 7.3 | 987 @bs8, 1323 @bs64 | 57 | 1.2 |

Per 6016×4000 photograph (96 native 512px patches, or 384 tiles of 128px at 0.5×, padding included; GPU batched time uses the largest batch measured):

| Scale | Model | TFLOPs / photo | GPU fp16 batched (ms / photo) | GPU fp32 bs1 (ms / photo) | CPU bs1 (s / photo) |
|---|---|---|---|---|---|
| 512px | canet_b4 | 3.54 | 1323 | 2934 | 69.6 |
| 512px | unet | 36.99 | 3432 | 9114 | 303.1 |
| 512px | deeplabv3plus_r50 | 7.02 | 1007 | 2620 | 55.6 |
| 128px | canet_b4 | 0.89 | 442 | 8679 | 25.7 |
| 128px | unet | 9.25 | 732 | 2759 | 60.0 |
| 128px | deeplabv3plus_r50 | 1.76 | 290 | 2408 | 21.9 |

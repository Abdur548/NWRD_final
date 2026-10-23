# Results

Curated outputs of the Kaggle runs: every number in `docs/` comes from a file in this folder.
Model weights are not kept here. The app model is `app/assets/best_model.pth`; checkpoints
stayed on Kaggle.

| Folder | Contents | Produced by |
|---|---|---|
| `512px/final/` | Main experiment on 512×512 native-resolution patches: `canet_b4`, `unet`, `deeplabv3plus_r50` × seeds 42, 43, 44, plus the `unet_tuned` learning-rate control, seed 42 | Kaggle kernel `abdur548/nwrd-canet-b4-vs-baselines`, version 2 (2026-09-26) resumed by version 4 (2026-09-28/29) |
| `512px/replicate/` | An earlier, independent session of the same protocol: seed 42 of `unet`, `deeplabv3plus_r50` and `unet_tuned`, and `canet_b4` seed 43. It was killed at Kaggle's 12 h limit, so it is partial. It is used only to measure run-to-run noise. | Same kernel, version 1 (2026-09-25) |
| `128px/final/` | The NWRD paper's scale: patches halved (Lanczos) and cut into 128×128 tiles; `canet_b4`, `unet`, `deeplabv3plus_r50` × seeds 42, 43, 44 | Kaggle kernel `abdur548/nwrd-128px-canet-vs-unet`: `canet_b4` and `unet` by version 1 (2026-09-29) resumed by version 2 (downloaded 2026-10-10); `deeplabv3plus_r50` added by version 3 (2026-10-18), which restored those runs unchanged and re-measured `benchmark.json` for all three models |
| `analysis/512px/`, `analysis/128px/` | Per-photograph, lesion-size, training, threshold and cost analysis with figures | `scripts/analyse_results.py` |
| `analysis/scales.md` | 128px against the paper and against 512px, with cost per photograph | `scripts/compare_scales.py` |

## Files in an experiment folder

| File | What it holds |
|---|---|
| `<model>/seed<N>/metrics.json` | Val-selected threshold; val and test metrics at that threshold and at 0.5; Table-4-style metrics on rust-positive test patches |
| `<model>/seed<N>/history.json` | Per-epoch train/val loss, val IoU and Dice, learning rate, epoch time |
| `<model>/seed<N>/test_counts.npz` | Per-patch test confusion counts (tp, fp, fn, tn) and file names: the input of every bootstrap |
| `<model>/seed<N>/val_threshold_sweep.json` | Validation metrics over the threshold grid 0.05–0.95 |
| `benchmark.json` | Parameters, FLOPs, T4 GPU latency and throughput, CPU latency (random weights) |
| `config.json` | The protocol settings the run used |
| `tiling.json` (128px) | How the tiles were made, and the rust-pixel share before and after downsampling |
| `log.txt` | The complete Kaggle log of every session, in order |
| `_cache/mask_stats_*.json` | Rust pixels per training and validation patch; they show the splits share no photograph, and `scripts/verify_dataset.py --counts` checks them, with the test counts, against the source masks |
| `comparison.md`, `comparison.json`, `figures/training_curves.png` | The aggregated comparison, including the pre-registered verdict |
| `<model>/summary.md` | One model on its own: every seed's metrics, mean ± std, its comparison with CANet-B4, and its cost |
| `figures/qualitative_results.png` | Test patches with each model's prediction (seed 42) |

`comparison.md`, `comparison.json`, `figures/training_curves.png` and the `summary.md` files were
regenerated with the current code (`python -m nwrd.run --stage compare --out
results/<scale>/final`). Their numbers are identical to the files written on Kaggle; the
regenerated files also state the verdict. Every other file is exactly as Kaggle produced it.

## Notes on the logs

- `512px/final/log.txt` ends with `INCOMPLETE`: the last session stopped at its time budget while
  training the optional `unet_tuned` control, seed 43. The control's seeds 43 and 44 were then
  dropped and the experiment was declared final (2026-10-17). On validation the shared-lr `unet`
  beats `unet_tuned` in every session and seed, so `unet` is the paper-architecture baseline.
- `128px/final/log.txt` has an `INCOMPLETE` line partway through, where the first session hit its
  time budget. It is followed by the second session's restore, which completed the `canet_b4` and
  `unet` runs, and by a third session (2026-10-18) that restored them and trained
  `deeplabv3plus_r50`.
- In both folders, a `--- new session` line marks each resumed Kaggle session.

## Reproducibility

Except the 128px `deeplabv3plus_r50` runs, these runs predate two changes, so their library
versions and augmentation streams are not fully recorded:

- Only PyTorch's version was logged (2.10.0+cu128, in `benchmark.json`). The pipeline now writes
  every library version at the top of each session's log.
- If Kaggle's preinstalled albumentations was version 2 (its version was not recorded), the
  training augmentation was not seeded: albumentations 2 seeds itself from the operating system.
  That adds run-to-run variation beyond GPU nondeterminism. Training now seeds augmentation from
  (seed, epoch, worker), so the data order and every augmentation follow from the seed. On a CPU
  a run is then bit-reproducible (tested). On a GPU, cuDNN's kernel choice can still vary results
  slightly between runs.

The 128px `deeplabv3plus_r50` runs were trained with both changes. Their session's log records
Python 3.12.13, PyTorch 2.10.0+cu128, NumPy 2.0.2, OpenCV 4.13.0, albumentations 2.0.8,
segmentation-models-pytorch 0.5.0 and EfficientNet-PyTorch 0.7.1 on a Tesla T4, and their
augmentation was seeded. The earlier runs were trained by the code this repository was cleaned
from, which was verified on a CPU to train identically apart from that seeding.

When `deeplabv3plus_r50` was added, `128px/final/benchmark.json` was re-measured for all three
models in one session, so the table compares like with like. CANet-B4's and UNet's latencies and
throughputs moved by at most 7% from the earlier measurement. Peak memory at batch 8 moved more
(CANet-B4 205 → 262 MB, UNet 289 → 442 MB), probably through cuDNN's per-session choice of
algorithms and their workspace; at batch 64, the figure `docs/results.md` reports, it was
unchanged.

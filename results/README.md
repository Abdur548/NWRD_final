# Results

Small files only: everything here is in git, and every number in `docs/` comes from a file here.

| Folder | What | Written by |
|---|---|---|
| `dataset/` | The dataset study: `summary.md` (the tables `docs/dataset.md` quotes), `photographs.json` (one record per photograph), `duplicates.json` (repeated photographs and how well their masks agree), `mask_101.png` (photo 101's mask outlined at the photograph's corner and at its best placement) | `scripts/describe_data.py` |
| `nwrd_final/` | Figures reused from `NWRD_final` (commit `61bf2fb`), the earlier study of the same models on other photographs: `benchmark_512px.json` and `benchmark_128px.json` are its T4 speed benchmarks (its `results/{512px,128px}/final/benchmark.json`), `epoch_seconds.json` its mean T4 epoch times per model (from its runs' `history.json`), and `reference.json` the figures the docs compare with, each with the file it comes from | Copied from `NWRD_final` |
| `512px/`, `128px/` | Per scale, one folder per model with `config.json`, `log.txt` and `windows.json` of its Kaggle kernel, and per seed `metrics.json`, `history.json`, `val_threshold_sweep.json`, `test_counts.npz` (per test photograph) and `paper_patches.npz` (the paper's 2,737 test patches); then `comparison.md` / `comparison.json` (the side-by-side log), `figures/training_curves.png` and each model's `summary.md` | `scripts/collect_results.py`, `python -m nwrd_public.compare results` |
| `summary.csv` | One row per (scale, model, seed) | `python -m nwrd_public.compare results` |

No checkpoint is ever kept here; the weights stay on Kaggle. Each run's `log.txt` records the
library versions and GPU it ran with. The `512px/` and `128px/` folders appear once the Kaggle runs
are collected.

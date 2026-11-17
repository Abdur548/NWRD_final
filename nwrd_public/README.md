# `nwrd_public/`: the pipeline

One run of one model at one scale: `python -m nwrd_public.run --model M --preset 512px`. The
Kaggle notebooks embed these modules (only the `.py` files), so after any edit here run
`python scripts/build_notebooks.py`. The map of the whole repository is in
[`../docs/codebase.md`](../docs/codebase.md).

| Module | What it holds |
|---|---|
| `config.py` | The protocol: every setting (`Config`), the `512px` and `128px` presets, and which models are compared (`COMPARED_MODELS`) or ablation-only (`ABLATION_MODELS`) |
| `dataset.py` | The release: finding it, checking every file against `data/manifest.json`, the fixed split, reading photographs and masks, the training `RustDataset` |
| `windows.py` | Windows on a stride fitted to each photograph (no padding, each pixel owned by one window), the 0.5× resize, the training windows' selection and cut, `pos_weight`, the NWRD paper's patch grid |
| `models.py` | CANet-B4, the NWRD paper's UNet, DeepLabV3+ R50 and the ablation's variants; `build_model` and safe `load_checkpoint` |
| `metrics.py` | Pixel metrics pooled over a split; `ProbHistogram` picks the IoU-maximising threshold (copied byte for byte from `NWRD_final`) |
| `train.py` | One (model, seed) training run: Dice + weighted BCE loss, encoder freeze, schedule, best checkpoint by validation IoU at 0.5, resumable |
| `evaluate.py` | Scoring a trained run: whole-photograph prediction, the validation threshold, every test pixel once at native resolution, the paper's 2,737 test patches |
| `compare.py` | The side-by-side report of one scale: bootstrap, seed t-test, the pre-registered decision rule, speed table, `summary.csv` (`python -m nwrd_public.compare results`) |
| `run.py` | The entry point: `cut → train → eval → summary`, each stage resumable; exit code 3 when the time budget is reached |

The tests for each module are in `tests/test_<module>.py`; the whole chain on a synthetic release
is in `tests/test_pipeline.py`.

# NWRD_final: pipeline, decisions and invariants

**Updated:** 2026-10-20

## What it is

A study of CANet with an ImageNet-pretrained EfficientNet-B4 encoder (CANet-B4, "ours") for
binary wheat-stripe-rust segmentation of 100 field photographs (not the public NWRD release:
`docs/protocol.md` §3.1), against the NWRD paper's UNet architecture and DeepLabV3+ R50, under
one protocol (`docs/protocol.md`). Two experiments, both **final**:

| Experiment | Preset | Kaggle kernel (slug from title) | Results |
|---|---|---|---|
| main, 512px native patches | `main` | `abdur548/nwrd-canet-b4-vs-baselines` | `results/512px/final/` (+ `replicate/`) |
| paper scale, 128px tiles | `128px` | `abdur548/nwrd-128px-canet-vs-unet` (title predates DeepLabV3+) | `results/128px/final/` |

Both experiments train `canet_b4`, `unet` and `deeplabv3plus_r50` × seeds 42–44; the main one also
has the `unet_tuned` control (seed 42). Verdicts (pre-registered rule): comparable to UNet and
DeepLabV3+ at 512px; worse than both at 128px. Efficiency: 10.4× fewer FLOPs than UNet and faster
than it at 512px on every device; DeepLabV3+ R50 is faster than CANet-B4 on every device at both
scales.

## Flow

```
nwrd/ ── scripts/build_notebooks.py ──> notebooks/*.ipynb (embed nwrd/, PRESET = main | 128px)
   scripts/run_on_kaggle.py [--experiment 128px] [--resume] [--budget-hours H] ──> Kaggle T4
   nwrd.run: [tile] → train → eval → bench → figures → compare → export (512px only)
   runs_kaggle/<exp>/<account>_v<N>/ (ignored) ──copy by hand──> results/<scale>/final/
   nwrd.run --stage compare | scripts/analyse_results.py | scripts/compare_scales.py → results/
   docs/*.md ── scripts/md_to_pdf.py ──> pdf/ (ignored)
   export/ → app/assets/ (best_model.pth in Git LFS) → app/backend (FastAPI) + app/frontend
   (React) + app/dashboard.py (Streamlit), all predicting through nwrd/inference.py
```

## Invariants

- `nwrd/` is the source of truth; notebooks are generated (`tests/test_notebooks.py`).
- Kernel titles in `scripts/run_on_kaggle.py` must not change (Kaggle derives the slug from them).
- Each notebook has exactly one `EXPECT_RESTORE = False` and one `KAGGLE_TIME_BUDGET_HOURS = 11.0`
  line (rewritten by `--resume` / `--budget-hours`).
- The protocol is identical for every model; the test split is never used for selection.
- Tile names keep `_patch_` (`compare.source_photo` groups by photograph for the bootstrap).
- CANet module names are frozen (checkpoints load strictly); context pooling is capped at the
  feature-map size (no effect at ≥256px, tested).
- Checkpoints load with `weights_only=True` everywhere.
- A model trained on 128px tiles is never exported to the app (enforced in `nwrd/run.py`).
- `--stage compare`/`export` alone never rewrites a results folder's `config.json` or `log.txt`.
- The bench stage re-measures the whole table when a model is missing from `benchmark.json`, so
  each table comes from one session.
- Text crossing a process boundary is UTF-8: `python -m nwrd.run` reconfigures stdout (reports
  contain ≈), and the Kaggle CLI runs with `PYTHONUTF8=1` (it saves the kernel log with the
  locale's encoding, which on Windows left it empty).
- The app predicts on native-resolution 512px patches, as the test set was cut, but pads by
  mirroring at the bottom and right where the dataset centred each photograph on a black canvas
  (`nwrd/inference.py`, `docs/protocol.md` §3.2); never resize a whole photograph to 512.
- Never call the photographs "the NWRD dataset": they are NWRDF (public Kaggle `manan369/nwrd-f`,
  copied byte for byte as the private `abdur548/nwrdf-unpatched`; all 6016×4000, Nikon D3200,
  2023), not the Zenodo release (`docs/protocol.md` §3.1). The patch recipe is checked by
  `scripts/verify_dataset.py`.

## Decisions

- 2026-10-17: replicated `Plant_Segmentation` into this repo (plan: `../NWRD_final_PLAN.md`).
  User decisions: keep both UIs (D1), main experiment final, `unet_tuned` seeds 43–44 dropped (D2),
  repo local only (D3), `canet_b4_twopass` removed (D4), Colab support dropped (D5), PDFs not
  versioned (D6).
- Equivalence gates (2026-10-17/18) proved the cleaned pipeline gives byte-identical outputs to
  the old one on CPU (training, models, aggregation of the published results).
- Augmentation is now seeded from (seed, epoch, worker) (`nwrd/train.py`); albumentations 2
  otherwise seeds from the OS. The published runs may have had unseeded augmentation.
- Patch bootstrap in the analysis now uses the pipeline's scheme (multinomial); the photograph
  bootstrap, which the decision rule uses, is unchanged.
- 2026-10-18, at the user's request: DeepLabV3+ R50 added to the 128px experiment (Kaggle
  version 3, a `--resume` that restored the earlier runs unchanged); its runs used seeded
  augmentation. Each model's results are also written on their own (`<model>/summary.md`,
  `docs/results.md` §8). `docs/protocol.md` describes `deeplabv3plus_r50` as the standard smp
  model standing in for the internship's "Custom DeepLabV3+", whose code was not preserved.
- 2026-10-19, at the user's request: the data's provenance was established and documented
  (`docs/protocol.md` §3.1–3.2). The patching script was lost; its recipe (centred black
  6144×4096 canvas, 12×8 grid of 512px patches, no resizing) was reconstructed and verified
  against the source (all 9,600 patch counts; every file of photograph 91). The photographs turned
  out not to be the NWRD release, so the docs no longer credit the margin over the paper's
  published scores to the protocol alone, and per-photograph costs use 6016×4000 (96 patches).
  2026-10-20 ("check kaggle"): the source traced to the public `manan369/nwrd-f` (2025-07-31),
  identical to the private copy; its uploader's notebooks test on it as a dataset separate from
  the NWRD release. Where its photographs were taken is still not stated anywhere.

## External dependencies

| What | Used for | Notes |
|---|---|---|
| Kaggle (T4 kernels, dataset `abdur548/nwrd-patched`) | Training | Token in `~/.kaggle/kaggle.json`; a second account via `KAGGLE_CONFIG_DIR`. The CLI clears its `-p` target folder on single-file downloads: give each download its own folder |
| Kaggle `manan369/nwrd-f` (public; private copy `abdur548/nwrdf-unpatched`) | The source photographs (NWRDF) | Needed only by `scripts/verify_dataset.py`; `--counts` needs just its masks (21 MB) |
| Zenodo, doi:10.5281/zenodo.21672928 (Kaggle mirror `bcap52/nwrd-wheat`) | The public NWRD release, for reference | Not used by the experiments |
| Git LFS | `app/assets/best_model.pth` (118 MB) | |
| jsDelivr | mermaid 10.9.1 for PDFs | SHA-256 pinned in `scripts/md_to_pdf.py` |

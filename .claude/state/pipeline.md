# NWRD_public: pipeline, decisions and invariants

**Updated:** 2026-10-22

Binary wheat-stripe-rust segmentation on the **public NWRD release** (Zenodo
doi:10.5281/zenodo.21672928; 100 photographs, 10 sizes, rust 0.57% of pixels). CANet-B4 (ours)
is compared with the NWRD paper's UNet architecture and DeepLabV3+ R50 under one protocol, at two
scales, three seeds each, with a decision rule fixed before any run. It backs a research paper and
is reviewed by the supervisor.

## Objective

A fair, reproducible side-by-side of the three models on the paper's own photographs and test
set, scored on every real pixel exactly once, despite photographs of 10 sizes, a repeated-scene
leak in the paper's split, and rust 16× sparser than in the earlier study (`../NWRD_final`).

## Flow

```
Zenodo NWRD.zip ── scripts/fetch_data.py (MD5, safe extract) ──> data/raw/NWRD + data/manifest.json
data/raw/NWRD ── scripts/describe_data.py ──> results/dataset/ ──> docs/dataset.md (by hand)
nwrd_public/ + data/manifest.json ── scripts/build_notebooks.py ──> notebooks/<model>.ipynb (3)
scripts/run_on_kaggle.py --model M [--preset 128px] [--resume] [--budget-hours H] ──> Kaggle T4,
   dataset bcap52/nwrd-wheat; in the kernel: python -m nwrd_public.run
      open_release (find under /kaggle/input, MD5 every file, split) ──┬─> mismatch: stop (RuntimeError)
      cut training windows to /tmp ──> train 3 seeds ──> eval ──> <model>/summary.md
      exit 3 = time budget hit ──> --resume (restore cell copies the previous runs/)
--status ──> runs_kaggle/<scale>/<model>/<account>_v<N>/ (ignored)
scripts/collect_results.py <that folder> ──> results/<scale>/<model>/ ──> python -m nwrd_public.compare results
   ──> results/<scale>/comparison.{md,json}, figures/, results/summary.csv ──> docs/results.md (after runs)
```

## Stages

| Stage | Does | Lives in |
|---|---|---|
| Protocol | Every setting; presets `512px` (window 512, scale 1) and `128px` (128, 0.5) | `nwrd_public/config.py` |
| Release | Finds the release, MD5-checks it against the manifest, the D3 split, reads photos/masks (101 cropped to its mask) | `nwrd_public/dataset.py` |
| Windows | Fitted-stride grid, ownership spans, resizing, rust counts, training sample and its cut | `nwrd_public/windows.py` |
| Models, metrics | Copied byte for byte from NWRD_final | `nwrd_public/models.py`, `metrics.py` |
| Training | NWRD_final's loop; per-epoch validation on whole photographs | `nwrd_public/train.py` |
| Evaluation | Whole-photo prediction, native scoring, threshold on val, the paper's 2,737 patches | `nwrd_public/evaluate.py` |
| Comparison | Side-by-side log, photograph bootstrap, verdict, paper numbers, speed, summary.csv | `nwrd_public/compare.py` |
| Entry point | `cut → train → eval → summary`, resumable, exit 3 at the time budget | `nwrd_public/run.py` |
| Scripts | fetch, describe, notebooks, Kaggle runner, collect | `scripts/` |

## Invariants

- `nwrd_public/` and `data/manifest.json` are the source of truth; the notebooks embed them and
  differ only in their `MODEL` line (`tests/test_notebooks.py` fails otherwise). Rebuild after
  every edit, or Kaggle silently runs old code.
- Each notebook has exactly one `PRESET = '512px'`, `EXPECT_RESTORE = False` and
  `KAGGLE_TIME_BUDGET_HOURS = 11.0` line: `run_on_kaggle.py` rewrites them by string.
- Kernel titles (`NWRD public <model> <preset>`) must not change: Kaggle derives the slug from the
  title, and `--resume` / `--status` find kernels by slug.
- No padding is ever scored and no edge is dropped: windows never extend past a photograph
  (`axis_starts` raises on a side shorter than the window), and ownership gives every pixel to
  exactly one window. A side shorter than a window must stay an error.
- Nothing is resized except the 0.5× of the `128px` scale (anti-aliased Pillow Lanczos for
  photographs, never OpenCV's INTER_LANCZOS4, which aliases when shrinking; area for
  masks); predictions are resized only for scoring (bilinear to native; area to half size for the
  paper's patches at 512px).
- The split is fixed: validation `VAL_IDS`, test = `test/`, training without `EXCLUDED_TWINS`
  (79, 86, 96, 99). No validation or test photograph may have a twin in training.
- Training windows are cut deterministically from (photos, window, scale, ratio, seed); the cache
  marker `windows.json` refuses other settings. A resumed Kaggle session re-cuts identically.
- Checkpoints load with `weights_only=True` and never enter git or `results/`.
- The decision rule (`compare.verdict`, `docs/protocol.md` §5) was fixed before any run.
- Every number in `docs/` traces to `results/`.
- The compromised GitHub repository `dll-ncai/NUST-Wheat-Rust-Disease-NWRD` is never cloned or
  opened.

## Decisions

### The user's D1–D8 (plan `../NWRD_public_PLAN.md` §5), approved 2026-10-22
D1 local git only, `E:\NWRD\NWRD_public`. D2 512px first, then 128px. D3 the paper's val/test,
training minus the four twins (77/9/10); in-train pairs keep both copies. D4 attach
`bcap52/nwrd-wheat` and MD5-check all 200 files every run (fallback: upload our own copy, needs
the user's OK). D5 no new speed benchmark (NWRD_final's T4 numbers, `results/nwrd_final/`). D6 three
models only. D7 NWRD_final's pre-registered rule, unchanged. D8 keep the `pos_weight` rule (63.1,
43.9) and widen the threshold grid to 0.01–0.99.

### Fitted stride instead of padding or resizing — 2026-10-22
⌈L/P⌉ windows spread evenly. Rejected: black padding (content never seen in a real photograph,
scored as easy negatives: 4.4% of each NWRD_final test photograph; up to 503 px here) and
resizing whole photographs (pustules are a few pixels wide). Consequence: overlaps of 0–72 px,
split at their midpoint for scoring.

### Whole photographs are the scoring unit, also during training — 2026-10-22
Per-epoch validation and final evaluation both predict whole photographs and score native
pixels, so checkpoint selection optimises exactly what is reported. Consequence: there is no
patch-level bootstrap; the photograph bootstrap is the only interval.

### The paper's patches are scored from our whole-photo predictions — 2026-10-22
The 2,737 patches are cut from each model's half-size probability map (the 128px model's own;
the 512px model's native map resized by area), not by re-running the model per patch. One
prediction is thus scored two ways. Masks halved by area (the authors' method is unknown).

## External dependencies

| What | Used for | Notes |
|---|---|---|
| Kaggle T4 kernels; dataset `bcap52/nwrd-wheat` (public mirror, license field "unknown") | Training | Token in `~/.kaggle/kaggle.json`; second account via `KAGGLE_CONFIG_DIR`. ~16 GPU-h for everything. At most 2 GPU sessions per account at once: a third push is refused |
| Zenodo record 21672928 | The reference data | `scripts/fetch_data.py`; slow (~0.2–0.4 MB/s) |
| `../NWRD_final` (commit `61bf2fb`) | Code origin; T4 speed, epoch times and reference numbers copied to `results/nwrd_final/` | Read-only from here |

## Known limitations

- 10 test photographs and 9 validation photographs: wide intervals.
- Annotation agreement between two masks of one photograph is IoU 0.40–0.63, which bounds every
  model's attainable IoU.
- APF, focal loss and the paper's 500-epoch schedule are not reproduced: the paper's published
  rows are context, not a paired comparison.

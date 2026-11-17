# NWRD_public: pipeline, decisions and invariants

**Updated:** 2026-11-15

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
nwrd_public/ + data/manifest.json ── scripts/build_notebooks.py ──> notebooks/<model>.ipynb (6)
scripts/run_on_kaggle.py --model M [--preset 128px] [--resume] [--budget-hours H] ──> Kaggle T4,
   dataset bcap52/nwrd-wheat; in the kernel: python -m nwrd_public.run
      open_release (find under /kaggle/input, MD5 every file, split) ──┬─> mismatch: stop (RuntimeError)
      cut training windows to /tmp ──> train 3 seeds ──> eval ──> <model>/summary.md
      exit 3 = time budget hit ──> --resume (restore cell copies the previous runs/)
--status ──> runs_kaggle/<scale>/<model>/<account>_v<N>/ (ignored)
scripts/collect_results.py <that folder> ──> results/<scale>/<model>/ ──> python -m nwrd_public.compare results
   ──> results/<scale>/comparison.{md,json}, figures/, results/summary.csv ──> docs/results.md (after runs)
results/ ── scripts/build_paper.py ──> results/<scale>/pairwise.json, paper/generated/, paper/figures/
   + paper/main.tex, references.bib (by hand) ──> --zip <path> (Overleaf "Upload project")
```

## Stages

| Stage | Does | Lives in |
|---|---|---|
| Protocol | Every setting; presets `512px` (window 512, scale 1) and `128px` (128, 0.5) | `nwrd_public/config.py` |
| Release | Finds the release, MD5-checks it against the manifest, the D3 split (101 in no split), reads photos/masks | `nwrd_public/dataset.py` |
| Windows | Fitted-stride grid, ownership spans, resizing, rust counts, training sample and its cut | `nwrd_public/windows.py` |
| Models, metrics | `metrics.py` byte for byte from NWRD_final; `models.py` from it plus `deeplabv3plus_r50_scratch` and `unet_r50` (ablation); an smp encoder is frozen only if it carries ImageNet weights | `nwrd_public/models.py`, `metrics.py` |
| Training | NWRD_final's loop; per-epoch validation on whole photographs | `nwrd_public/train.py` |
| Evaluation | Whole-photo prediction, native scoring, threshold on val, the paper's 2,737 patches | `nwrd_public/evaluate.py` |
| Comparison | Side-by-side log, photograph bootstrap, verdict, paper numbers, speed, summary.csv | `nwrd_public/compare.py` |
| Entry point | `cut → train → eval → summary`, resumable, exit 3 at the time budget | `nwrd_public/run.py` |
| Scripts | fetch, describe, patch (the release cut to disk on the fitted grid, with an index of owned regions; not used by the kernels, which cut their own), notebooks, Kaggle runner, collect | `scripts/` |
| Paper draft | Numbers (`\val{key}`), tables and figures of the LaTeX reference draft from `results/`; every model pair's photograph bootstrap (`pairwise.json`); the Overleaf zip. Missing results print red | `scripts/build_paper.py`, `paper/` |

## Invariants

- `nwrd_public/` and `data/manifest.json` are the source of truth; the notebooks embed them and
  differ only in their `MODEL` line (`tests/test_notebooks.py` fails otherwise). Rebuild after
  every edit, or Kaggle silently runs old code.
- Each notebook has exactly one `PRESET = '512px'`, `EXPECT_RESTORE = False` and
  `KAGGLE_TIME_BUDGET_HOURS = 11.0` line: `run_on_kaggle.py` rewrites them by string.
- `paper/generated/` is written by `scripts/build_paper.py`, never by hand; `paper/main.tex`
  takes model numbers only through `\val{key}` (`tests/test_paper.py` fails on an unknown key).
  The ablation's model names are fixed in `build_paper.ARCH`; new models must use them.
- Kernel titles (`NWRD public <model> <preset>`) must not change: Kaggle derives the slug from the
  title, and `--resume` / `--status` find kernels by slug.
- No padding is ever scored and no edge is dropped: windows never extend past a photograph
  (`axis_starts` raises on a side shorter than the window), and ownership gives every pixel to
  exactly one window. A side shorter than a window must stay an error.
- Nothing is resized except the 0.5× of the `128px` scale (anti-aliased Pillow Lanczos for
  photographs, never OpenCV's INTER_LANCZOS4, which aliases when shrinking; area for
  masks); predictions are resized only for scoring (bilinear to native; area to half size for the
  paper's patches at 512px).
- The split is fixed: validation `VAL_IDS` (the paper's `PAPER_VAL_IDS` without 101), test =
  `test/`, training without `EXCLUDED_TWINS` (79, 86, 96, 99) and `EXCLUDED_MISALIGNED` (101).
  No validation or test photograph may have a twin in training. A mask of another size than its
  photograph is an error, never cropped or placed by assumption.
- Training windows are cut deterministically from (photos, window, scale, ratio, seed); the cache
  marker `windows.json` refuses other settings. A resumed Kaggle session re-cuts identically.
- Checkpoints load with `weights_only=True` and never enter git or `results/`.
- The decision rule (`compare.verdict`, `docs/protocol.md` §5) was fixed before any run. It
  judges CANet-B4 against `config.COMPARED_MODELS` only (`compare.significance`); the
  `config.ABLATION_MODELS` (512px, exploratory, §9) get pairwise intervals from
  `build_paper.py`, never verdicts.
- Every number in `docs/` traces to `results/`.
- The compromised GitHub repository `dll-ncai/NUST-Wheat-Rust-Disease-NWRD` is never cloned or
  opened.

## Decisions

### The user's D1–D8 (plan `../NWRD_public_PLAN.md` §5), approved 2026-10-22
D1 local git only, `E:\NWRD\NWRD_public`. D2 512px first, then 128px. D3 the paper's val/test,
training minus the four twins (77/9/10; 77/8/10 since 101 left out, below); in-train pairs keep both copies. D4 attach
`bcap52/nwrd-wheat` and MD5-check all 200 files every run (fallback: upload our own copy, needs
the user's OK). D5 no new speed benchmark (NWRD_final's T4 numbers, `results/nwrd_final/`). D6 three
models only. D7 NWRD_final's pre-registered rule, unchanged. D8 keep the `pos_weight` rule (63.1,
43.9) and widen the threshold grid to 0.01–0.99.

### Fitted stride instead of padding or resizing — 2026-10-22
⌈L/P⌉ windows spread evenly. Rejected: black padding (content never seen in a real photograph,
scored as easy negatives: 4.4% of each NWRD_final test photograph; up to 360 px here, 503 px on
the excluded 101) and resizing whole photographs (pustules are a few pixels wide). Consequence:
overlaps of 0–46 px, split at their midpoint for scoring. Same window count as a padded grid, so
no cost difference. The user re-examined padding on 2026-10-23 (and another agent's case for it,
which assumed overlaps are scored twice) and kept the fitted stride.

### Photo 101 left out — 2026-10-23
Its mask (3593×3435 on a 3612×3456 photograph) was taken to line up with the top-left corner,
the best placement *inside* the photograph. Searching past the edges, it lines up shifted 8 px
left and 1 px up (rust contrast 21.1 vs 17.4; the six other cropped photographs peak at their
own corner), confirmed by eye on overlays across the photograph. The user chose to remove it
rather than re-register it: validation 8 photographs, 8.7% less validation rust. The first 512px
kernels (v1, which validated on it) were stopped. `read_photo` now refuses any mask of another
size.

### The paper is a re-benchmark centred on DeepLabV3+ — 2026-11-15 (reaffirmed 2026-11-16)
After the 512px results, the user chose to write the paper around DeepLabV3+ R50 beating the NWRD
paper's UNet (+ APF, published numbers) and to add an ablation of what causes it. DeepLabV3+ R50
is the stock smp model, so the contribution is the audit, the protocol and the ablation, not an
architecture. CANet-B4's pre-registered verdicts stay in the paper and the change of framing is
disclosed; the DeepLabV3+ comparisons and the ablation are labelled exploratory. The decision
rule (`compare.verdict`) is unchanged.

### Exploratory 512px ablation of DeepLabV3+ R50 — 2026-11-15
`deeplabv3plus_r50_scratch` (pretraining), `unet_r50` (decoder; smp UNet on the ImageNet
ResNet-50) and `deeplabv3plus_b4` (encoder; and CAM vs ASPP against CANet-B4), 3 seeds, the
protocol unchanged (`docs/protocol.md` §9, written before the ablation ran). The user chose a
Kaggle ablation ("B"); the three variants are Claude's pick, which the user did not object to.
Rejected: a DeepLabV3+ configuration search (too many runs on 10 test photographs). The
from-scratch encoder keeps the fine-tuning lr and schedule: it measures pretraining under this
protocol. Same-seed reruns are not reproducible on the GPU (`cudnn.benchmark`), so differences
under the run-to-run noise are not interpreted.

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

- 10 test photographs and 8 validation photographs: wide intervals.
- Annotation agreement between two masks of one photograph is IoU 0.40–0.63, which bounds every
  model's attainable IoU.
- APF, focal loss and the paper's 500-epoch schedule are not reproduced: the paper's published
  rows are context, not a paired comparison.

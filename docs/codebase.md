# Navigating the codebase

This repository trains and scores three segmentation models on the public NWRD release of wheat
stripe-rust photographs, under one protocol, and builds a paper draft from the results. **The paper
is built around DeepLabV3+ R50**: it beats the NWRD paper's UNet, and an ablation asks why.
CANet-B4 was the pre-registered candidate. Its verdicts, under a decision rule fixed before any
run, stay in the reports, which is why those reports still call it "ours".

This page is the map. The study itself is described in [`protocol.md`](protocol.md), and the data
in [`dataset.md`](dataset.md).

## Where to start

| You want to | Read |
|---|---|
| Understand the study | [`../README.md`](../README.md), then [`protocol.md`](protocol.md) §1–§5 and §9 (the ablation) |
| Check what is claimed about the data | [`dataset.md`](dataset.md), backed by `results/dataset/summary.md` |
| See the results | `results/512px/comparison.md` and `results/128px/comparison.md` |
| Trace a number in the paper to its source | [`../paper/README.md`](../paper/README.md) and [Tracing a number](#tracing-a-number) below |
| Change the code | This page, then [`../nwrd_public/README.md`](../nwrd_public/README.md) and [`../tests/README.md`](../tests/README.md) |
| Re-run a model | [Common tasks](#common-tasks) below, and [`../scripts/README.md`](../scripts/README.md) |

## Layout

```
nwrd_public/        the pipeline package: one run of one model at one scale (README inside)
scripts/            command-line tools around it: data, notebooks, Kaggle, results, paper (README inside)
notebooks/          one Kaggle notebook per model, generated from nwrd_public/ (README inside)
tests/              unit and end-to-end tests on a synthetic miniature release (README inside)
data/               manifest.json (MD5 of the release's 200 files); raw/ holds the release, not in git
results/            every reported number: dataset study, runs per scale, comparisons (README inside)
paper/              LaTeX draft; generated/ and figures/ are built from results/ (README inside)
docs/               protocol.md (the study), dataset.md (the data), codebase.md (this page)
CLAUDE.md, .claude/ conventions and session notes for AI coding assistants
build/, runs/, runs_kaggle/   local caches, logs and downloaded Kaggle output (not in git)
```

## How the pieces connect

```
Zenodo NWRD.zip ──scripts/fetch_data.py──> data/raw/NWRD/ (checked against data/manifest.json)
                                              │
              scripts/describe_data.py <──────┤──────> scripts/patch_release.py (patches on disk,
              results/dataset/ ─> docs/dataset.md       for inspection; the runs cut their own)
                                              │
nwrd_public/*.py + data/manifest.json ──scripts/build_notebooks.py──> notebooks/<model>.ipynb
                                              │
scripts/run_on_kaggle.py ──push──> Kaggle T4 kernel: python -m nwrd_public.run
                                   (cut → train → eval → summary, three seeds)
                         ──--status──> runs_kaggle/<scale>/<model>/<account>_v<N>/
                                              │
scripts/collect_results.py ──> results/<scale>/<model>/ + comparison.md/.json + summary.csv
                                              │
scripts/build_paper.py ──> results/<scale>/pairwise.json, paper/generated/, paper/figures/
                           + paper/main.tex (hand-written) ──--zip──> Overleaf
```

Kaggle runs the notebook, not the repository. Each notebook embeds the `nwrd_public/` modules and
the manifest as `%%writefile` cells, so a notebook built from old code trains old code.

## Inside one run

`python -m nwrd_public.run --model M --preset 512px` (`nwrd_public/run.py`) runs four stages,
each resumable:

| Stage | What happens | Code |
|---|---|---|
| `cut` | Find the release, check every file's MD5, apply the fixed split, cut the training windows to disk | `dataset.open_release`, `windows.cut_training_windows` |
| `train` | Per seed: train for 30 epochs and keep the checkpoint with the best pooled validation IoU at threshold 0.5 (`best.pt`; `last.pt` to resume) | `train.train_run`, `train.CombinedLoss` (Dice + BCE with `pos_weight`) |
| `eval` | Choose the threshold on the validation photographs, then score each test pixel once at native resolution, plus the NWRD paper's 2,737 test patches | `evaluate.evaluate_run`, `metrics.ProbHistogram.best_threshold`, `evaluate.paper_scores` |
| `summary` | The model's own `summary.md` across its seeds | `compare.write_model_summary` |

Output: `<out>/<model>/seed<N>/` with `metrics.json`, `history.json`,
`val_threshold_sweep.json`, `test_counts.npz` and `paper_patches.npz`, plus the weights, which
never leave Kaggle or enter git.

## Where each decision lives

| Question | File and symbol |
|---|---|
| Which photographs are training, validation, test? | `nwrd_public/dataset.py`: `splits`, `PAPER_VAL_IDS`, `VAL_IDS`, `EXCLUDED_TWINS`, `EXCLUDED_MISALIGNED` |
| Every hyperparameter; the 512px and 128px presets | `nwrd_public/config.py`: `Config`, `PRESETS` |
| Which models are compared, which are ablation | `nwrd_public/config.py`: `COMPARED_MODELS`, `ABLATION_MODELS` |
| How a model is built; which encoders are frozen | `nwrd_public/models.py`: `build_model` |
| Where windows sit on a photograph; who owns each pixel | `nwrd_public/windows.py`: `axis_starts`, `grid`, `owned_spans` |
| How the photographs are resized for 128px | `nwrd_public/windows.py`: `resize_photo`, `resize_mask` |
| Which background windows are trained on; the BCE rust weight | `nwrd_public/windows.py`: `select_windows`, `pos_weight` |
| IoU, Dice, precision and recall, pooled over pixels | `nwrd_public/metrics.py` |
| The NWRD paper's own test patches and averaging | `nwrd_public/windows.py`: `paper_grid`; `nwrd_public/evaluate.py`: `paper_scores`, `authors_average` |
| The photograph bootstrap, the seed t-test, the decision rule | `nwrd_public/compare.py`: `paired_bootstrap`, `seed_ttest`, `verdict`, `significance` |
| The ablation's pairwise intervals (DeepLabV3+ against each variant) | `scripts/build_paper.py`: `pairwise` → `results/512px/pairwise.json` |
| Speed figures (from the earlier study, not re-measured) | `results/nwrd_final/`; `nwrd_public/compare.py`: `speed_table` |
| Kaggle kernel names, push, resume, download | `scripts/run_on_kaggle.py`: `kernel`, `push`, `status` |

## Tracing a number

Every number in `docs/` and `paper/` comes from a file in `results/`:

1. `paper/main.tex` uses model numbers only as `\val{key}`, e.g. `\val{512.dlr.iou.pm}`
   (DeepLabV3+ R50's 512px test IoU, mean ± sd).
2. `paper/generated/numbers.tex` defines each key. `scripts/build_paper.py` writes it from
   `results/<scale>/comparison.json`, `pairwise.json` and `results/dataset/`.
3. Those come from each run's `results/<scale>/<model>/seed<N>/metrics.json`, which the Kaggle
   kernel wrote and `scripts/collect_results.py` copied.

Key names and model abbreviations (`dlr`, `unet`, `canet`, `dlrs`, `unetr`, `dlb`) are listed in
[`../paper/README.md`](../paper/README.md). `main.tex` ends with a section mapping each table to its
source file.

## Generated and hand-written files

Generated files are rebuilt by a script; edit the source, never the output.

| Generated | Built by | From |
|---|---|---|
| `notebooks/*.ipynb` | `scripts/build_notebooks.py` | `nwrd_public/*.py`, `data/manifest.json` |
| `results/dataset/` | `scripts/describe_data.py` | the release |
| `results/<scale>/<model>/` | `scripts/collect_results.py` | a Kaggle kernel's output |
| `results/<scale>/comparison.*`, `figures/`, `results/summary.csv` | `python -m nwrd_public.compare results` (also run by `collect_results.py`) | `results/<scale>/<model>/` |
| `results/<scale>/pairwise.json`, `paper/generated/`, `paper/figures/` | `scripts/build_paper.py` | `results/` |
| `data/manifest.json` | `scripts/fetch_data.py --write-manifest`, from a fresh extraction only | the Zenodo archive |

Hand-written: `nwrd_public/`, `scripts/`, `tests/`, `docs/`, `paper/main.tex`,
`paper/references.bib` and the README files.

## Common tasks

Run from the repository root, with the release in `data/raw/NWRD/` for anything that reads it.

```bash
python -m pytest -m "not slow"                        # fast tests (seconds)
python -m pytest                                       # all tests (~7 min on CPU)
python -m ruff check . && python -m ruff format --check .

python scripts/build_notebooks.py                      # after ANY edit in nwrd_public/ or the manifest

python scripts/run_on_kaggle.py --model deeplabv3plus_r50 --watch            # push one run (512px)
python scripts/run_on_kaggle.py --model unet --preset 128px --watch          # the paper's scale
python scripts/run_on_kaggle.py --model deeplabv3plus_r50 --resume --watch   # after a time-budget stop
python scripts/run_on_kaggle.py --model deeplabv3plus_r50 --status           # check, download output

python scripts/collect_results.py runs_kaggle/512px/<model>/<account>_v<N>    # into results/
python scripts/build_paper.py --zip <path>.zip                               # paper numbers + Overleaf zip
```

Kaggle allows two GPU sessions per account at once. A third push is refused with "Maximum batch
GPU session count"; `run_on_kaggle.py` reports the refusal and stops.

## Rules that are easy to break

- **Rebuild the notebooks after editing `nwrd_public/` or the manifest.**
  `tests/test_notebooks.py` fails on a stale notebook. Kaggle would otherwise run old code.
- **Don't edit generated files** (table above); rerun their script.
- **Don't rename the Kaggle kernels** (`NWRD public <model> <preset>`): `--resume` and `--status`
  find them by the slug Kaggle derives from the title.
- **Don't change the decision rule or the split.** Both were fixed before any run
  (`protocol.md` §5, `dataset.py`). The ablation's models get intervals, never verdicts.
- **Don't let padding or dropped edges into scoring.** `windows.axis_starts` raises on a side
  shorter than a window, and every pixel has exactly one owning window.
- **Collect only the reported runs.** `runs_kaggle/` (not in git) also holds discarded runs.
  `results/` holds exactly the runs reported.
- **Never clone or open the dataset's GitHub repository** (`dll-ncai/NUST-Wheat-Rust-Disease-NWRD`).
  It was compromised in October 2026 ([`dataset.md`](dataset.md) §1). The data comes from Zenodo
  or `bcap52/nwrd-wheat`, checked against the manifest.
- **Keep large files off C:** on the development machine (about 2 GB free). Windows, dry runs and
  pytest's `--basetemp` go on E:.

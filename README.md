# CANet-B4 against the NWRD paper's baselines on the public NWRD release

Pixel-level segmentation of wheat stripe rust in the photographs of the public NWRD release
(Anwar and Shafait, Zenodo doi:10.5281/zenodo.21672928, CC BY 4.0). Three models are trained and
scored under one protocol: **CANet-B4** (ours: CANet with an ImageNet-pretrained EfficientNet-B4
encoder), the **UNet** architecture of the NWRD paper (Anwar et al., *The NWRD Dataset*, Sensors
23(15):6942, 2023) and **DeepLabV3+ R50**. Each model is one generated Kaggle notebook; their
results are logged side by side, with a decision rule fixed before any run.

**Status (2026-10-22):** the pipeline, notebooks and dataset study are built and tested; the
Kaggle runs have not been made yet. Results will appear in `results/` and `docs/results.md`.

## What is different about this data

- **10 photograph sizes** (6016×4000 Nikon, 4608×3456 and 3456×4608 Huawei, 7 cropped ones), and
  no photograph has both sides a whole number of windows. Windows are therefore placed on a
  stride fitted to each photograph: ⌈L/512⌉ windows spread evenly over a side of L pixels, no
  padding, overlaps of 0–46 px. Every pixel of a validation or test photograph is scored exactly
  once, at native resolution, at both scales (512px native, and the paper's 128px windows of
  halved photographs).
- **Rust is sparse**: 0.57% of all pixels, and 0.61% of the training photographs' pixels against
  9.8% in the earlier study's training set.
- **The release repeats ten scenes** with different masks (twin masks agree at IoU 0.40–0.63),
  and under the paper's split two test and two validation photographs have their twin in
  training. This study keeps the paper's validation and test photographs and leaves those four
  twins out of training.
- **Photo 101's mask is misplaced**: it is smaller than the photograph and lines up with it only
  shifted 8 px left and 1 px up, past its edge. 101, one of the paper's validation photographs,
  is left out: 77 / 8 / 10 photographs.

Details: [`docs/dataset.md`](docs/dataset.md). Protocol and decision rule:
[`docs/protocol.md`](docs/protocol.md).

**Do not clone the dataset's GitHub repository** (`dll-ncai/NUST-Wheat-Rust-Disease-NWRD`): it was
compromised in October 2026 and opening it in VS Code runs a credential stealer
([`docs/dataset.md`](docs/dataset.md) §1). The data comes from Zenodo, or from the Kaggle copy
`bcap52/nwrd-wheat` (the same file names and sizes), each file checked against its MD5 in
`data/manifest.json` before every run.

## Repository layout

```
nwrd_public/   the pipeline: protocol (config), the release (dataset), fitted windows (windows),
               models and metrics (copied unchanged from NWRD_final), training, evaluation,
               side-by-side comparison, and the entry point (run)
notebooks/     canet_b4.ipynb, unet.ipynb, deeplabv3plus_r50.ipynb: generated, identical but MODEL
scripts/       fetch_data, describe_data, build_notebooks, run_on_kaggle, collect_results
data/          manifest.json (MD5 of the 200 files); raw/ holds the release (not in git)
results/       dataset/ (the dataset study), nwrd_final/ (the earlier study's T4 speed and other
               figures reused here), and per scale the runs and their comparison, plus
               summary.csv (see results/README.md)
docs/          dataset.md, protocol.md
tests/         unit and end-to-end tests on a synthetic miniature release (CPU)
```

## Installation

Python 3.10 or newer:

```bash
pip install -e ".[dev]"                   # or pip install -e . for the pipeline only
pip install -e ".[dev]" -c constraints.txt   # the exact versions last tested
```

## Usage

```bash
python scripts/fetch_data.py            # download from Zenodo, check, extract to data/raw/NWRD
python scripts/describe_data.py         # the dataset study -> results/dataset/
python -m pytest -m "not slow"          # fast tests; python -m pytest for all (minutes, CPU)

# Train and score on a Kaggle T4, one (model, scale) per kernel:
python scripts/run_on_kaggle.py --model canet_b4 --watch
python scripts/run_on_kaggle.py --model unet --preset 128px --watch
python scripts/run_on_kaggle.py --model canet_b4 --resume --watch     # after a time-budget stop

# Bring a finished kernel's runs into results/ and rebuild the side-by-side log:
python scripts/collect_results.py runs_kaggle/512px/canet_b4/<account>_v<N>
python -m nwrd_public.compare results   # every comparison and results/summary.csv
```

With a GPU and the release, a run also works locally:
`python -m nwrd_public.run --model canet_b4 --preset 512px --data-root data/raw/NWRD --out runs`.
After any change in `nwrd_public/`, rebuild the notebooks: `python scripts/build_notebooks.py`.

## Origin

The models, metrics, training loop, Kaggle runner and decision rule come from `NWRD_final`, the
same study on other photographs (100 Nikon photographs that are not this release). `models.py` and
`metrics.py` are byte-for-byte copies; the rest is adapted to whole photographs of mixed sizes.
`models.py` therefore also defines that study's `unet_tuned` control and `deeplabv3plus_b4`
ablation; they are not run here, and the configuration rejects them.

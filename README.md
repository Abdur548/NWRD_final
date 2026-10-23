# Wheat stripe-rust segmentation: CANet-B4 against the NWRD paper's baselines

Pixel-level segmentation of wheat stripe rust in 100 field photographs. The method studied is
**CANet with an ImageNet-pretrained EfficientNet-B4 encoder (CANet-B4)**. It is compared with the
UNet architecture of the NWRD paper (Anwar et al., *The NWRD Dataset*, Sensors 23(15):6942, 2023)
and with DeepLabV3+ R50 under one protocol: the same data, split by photograph, training schedule
and threshold rule, three seeds per model, and a decision rule fixed before the results were
seen. Accuracy and efficiency are both measured. The photographs are not those of the public NWRD
release ([Data](#data)). A FastAPI service with a React interface, and a Streamlit dashboard,
serve the trained model.

## Results

| | 512px (native resolution) | 128px (the paper's scale) |
|---|---|---|
| Test IoU, CANet-B4 (ours) | 0.729 ± 0.013 | 0.704 ± 0.004 |
| Test IoU, UNet (paper architecture, re-trained) | 0.747 ± 0.012 | 0.749 ± 0.009 |
| Test IoU, DeepLabV3+ R50 | 0.739 ± 0.021 | 0.732 ± 0.005 |
| CANet-B4 − UNet, 95% CI (resampling test photographs) | −0.019 [−0.042, +0.019] | −0.045 [−0.081, −0.006] |
| CANet-B4 − DeepLabV3+ R50, 95% CI (resampling test photographs) | −0.011 [−0.024, +0.008] | −0.028 [−0.054, −0.003] |
| Verdict against UNet / DeepLabV3+ R50 (pre-registered rule) | comparable / comparable | worse / worse |
| GFLOPs per input, CANet-B4 / UNet / DeepLabV3+ R50 | 36.9 / 385.3 / 73.1 | 2.3 / 24.1 / 4.6 |

CANet-B4 is not more accurate than either baseline: comparable to both at 512px and worse than
both at the paper's 128px scale. It needs a tenth of UNet's computation, and at 512px runs 3.1×
faster than UNet on a T4 GPU and 4.4× faster on a CPU; DeepLabV3+ R50, with twice CANet-B4's
FLOPs, is faster still on every device. All three re-trained models far exceed the paper's
published scores (F1 0.826–0.857 against 0.557 at the same scale), so that margin is not due to
CANet-B4's architecture; it reflects this study's protocol and photographs, which differ from the
paper's. Each model's results are also reported on their own
([`docs/results.md`](docs/results.md) §8). Full results: [`docs/results.md`](docs/results.md).
Protocol: [`docs/protocol.md`](docs/protocol.md). The baseline paper, with its figures and a
check of its claims: [`docs/paper_summary.md`](docs/paper_summary.md).

## Repository layout

```
nwrd/            the pipeline: models, data, tiling, training, evaluation, benchmark, comparison,
                 app export, whole-photograph inference
scripts/         Kaggle runner, notebook builder, dataset check, result analysis, cross-scale
                 report, PDF and paper-figure tools
notebooks/       the two Kaggle notebooks, generated from nwrd/ by scripts/build_notebooks.py
results/         curated outputs of the Kaggle runs and their analysis (see results/README.md)
app/             backend/ (FastAPI), frontend/ (React + Vite), dashboard.py (Streamlit),
                 assets/ (the served model, Git LFS, with its metrics and figures)
docs/            protocol, results and paper summary
tests/           unit and end-to-end tests on synthetic data (CPU)
```

## Installation

Python 3.10 or newer. The model file is stored with [Git LFS](https://git-lfs.com):

```bash
git lfs pull
pip install -e ".[app,dev,docs]"            # or pip install -e . for the pipeline only
pip install -e ".[app,dev,docs]" -c constraints.txt   # the exact versions last tested
```

## Usage

**Run the experiments.** Training runs on Kaggle GPUs. With a Kaggle API token in
`~/.kaggle/kaggle.json`:

```bash
python scripts/run_on_kaggle.py --watch                     # main experiment (512px)
python scripts/run_on_kaggle.py --experiment 128px --watch  # the paper's scale
python scripts/run_on_kaggle.py --resume --watch            # continue after a time-budget stop
```

Each notebook embeds `nwrd/`; after changing the package, rebuild them with
`python scripts/build_notebooks.py` (a test fails while one is stale). On a local GPU:
`python -m nwrd.run --preset main --data-root <path to wheat_rust_patches> --out runs`.

**Rebuild the reports** from the saved results, without a GPU:

```bash
python -m nwrd.run --stage compare --out results/512px/final
python -m nwrd.run --stage compare --out results/128px/final
python scripts/analyse_results.py results/512px/final --replicate results/512px/replicate
python scripts/analyse_results.py results/128px/final
python scripts/compare_scales.py results/128px/final results/512px/final --out results/analysis/scales.md
python scripts/md_to_pdf.py                                 # docs/*.md -> pdf/ (needs Edge or Chrome)
```

**Run the app.** The model predicts a photograph at native resolution in 512px patches, the way
the test set was scored (about 1.5 s per patch on a CPU).

```bash
uvicorn app.backend.main:app                                # API at http://localhost:8000
cd app/frontend && npm install && npm run dev               # interface at http://localhost:5173
streamlit run app/dashboard.py                              # or the Streamlit dashboard
```

The API's settings (model folder, allowed origins, upload limits) are environment variables,
documented in `app/backend/main.py`. The interface reads the API address from `VITE_API_URL`.

**Run the tests:**

```bash
python -m pytest -m "not slow"     # unit, app and notebook tests, about a minute
python -m pytest                   # all 82, with the end-to-end pipeline runs: about 4 minutes on a CPU
python -m ruff check . && python -m ruff format --check .
```

## Data

The experiments use the Kaggle dataset `abdur548/nwrd-patched`, cut from the Kaggle dataset
`manan369/nwrd-f`: 100 field photographs of wheat with stripe rust (Nikon D3200, 6016×4000) and
their masks, each centred on a black 6144×4096 canvas and cut, without resizing, into 96
non-overlapping 512×512 patches, split by photograph (train 1–85, validation 86–90, test
91–100). The script that cut them was not kept; `scripts/verify_dataset.py` checks the
reconstructed recipe against the source photographs.

These photographs are **not** those of the public NWRD release, which the NWRD paper's authors
publish on Zenodo ([doi:10.5281/zenodo.21672928](https://doi.org/10.5281/zenodo.21672928),
CC BY 4.0): 100 photographs from 2022, two-thirds of them from a phone, at several sizes.
Provenance, recipe and checks: [`docs/protocol.md`](docs/protocol.md) §3.

## Credits

- **Baseline:** Anwar, H. et al. *The NWRD Dataset: An Open-Source Annotated Segmentation
  Dataset of Diseased Wheat Crop.* Sensors 2023, 23(15), 6942. doi:10.3390/s23156942. Its
  photographs (Zenodo, doi:10.5281/zenodo.21672928) were not used here.
- **CANet design:** adapted from Tang, Q. et al. *Attention-guided Chained Context Aggregation for
  Semantic Segmentation*, arXiv:2002.12041; the project's reference is Shoaib, M. et al.,
  Frontiers in Plant Science 13:1095547 (2022). See [`docs/protocol.md`](docs/protocol.md) §2.1
  for what was changed.
- **Libraries:** EfficientNet-PyTorch (Apache-2.0) for the encoder and its ImageNet weights,
  segmentation-models-pytorch (MIT) for DeepLabV3+, PyTorch, albumentations.

**Project team:** Syed Muhammad Abdur Rahman, Abdul Hadi Sheikh, Abdullah Salim Nizami.
**Supervisor:** Dr. Imran Malik. NUST SEECS.

## License

No license has been chosen yet, so the code is not yet open source: until a `LICENSE` file is
added, all rights are reserved by the authors. MIT and Apache-2.0 both fit the dependencies;
Apache-2.0 matches the encoder library.

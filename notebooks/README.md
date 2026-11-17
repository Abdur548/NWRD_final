# `notebooks/`: one Kaggle notebook per model

**Generated; don't edit by hand.** `python scripts/build_notebooks.py` writes them from
`nwrd_public/*.py` and `data/manifest.json`. Each notebook embeds the whole package, so it runs on
Kaggle without the repository. The six are identical except for their `MODEL` line, which
`tests/test_notebooks.py` checks, along with their being up to date with the code.

| Notebook | Role |
|---|---|
| `deeplabv3plus_r50.ipynb` | DeepLabV3+ R50, the model the paper is built around |
| `unet.ipynb` | The NWRD paper's UNet architecture |
| `canet_b4.ipynb` | CANet-B4, the pre-registered candidate |
| `deeplabv3plus_r50_scratch.ipynb` | Ablation: DeepLabV3+ R50 without ImageNet pretraining |
| `unet_r50.ipynb` | Ablation: a UNet decoder on the same ImageNet ResNet-50 encoder |
| `deeplabv3plus_b4.ipynb` | Ablation: DeepLabV3+ on CANet-B4's EfficientNet-B4 encoder |

Push one with `python scripts/run_on_kaggle.py --model <name> [--preset 128px] --watch`.
`run_on_kaggle.py` rewrites the `PRESET`, `EXPECT_RESTORE` and `KAGGLE_TIME_BUDGET_HOURS` lines of
the copy it pushes, by exact string, so keep exactly one of each.

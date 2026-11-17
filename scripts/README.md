# `scripts/`: command-line tools

Run from the repository root. Scripts that read the release run with `python -P -E` (they add the
repository to `sys.path` themselves). Each script's `--help` lists its options, except
`build_notebooks.py`, which takes none and rebuilds every notebook when run. The map of the whole
repository is in [`../docs/codebase.md`](../docs/codebase.md).

In the order they are used:

| Script | Reads | Writes |
|---|---|---|
| `fetch_data.py` | Zenodo (or `--zip` a downloaded `NWRD.zip`) | `data/raw/NWRD/`, after checking the archive's MD5 and every file against `data/manifest.json` |
| `describe_data.py` | the release | `results/dataset/`: sizes, cameras, rust, repeated photographs, photo 101's mask, the paper's Table 2 (~6 min) |
| `patch_release.py` | the release | `build/patches/<preset>/`: every patch on the fitted grid, with `patches.csv`. For inspection; the Kaggle runs cut their own |
| `build_notebooks.py` | `nwrd_public/*.py`, `data/manifest.json` | `notebooks/<model>.ipynb` for all six models |
| `run_on_kaggle.py` | a notebook | Pushes it as a private GPU kernel; `--status` downloads its output to `runs_kaggle/<scale>/<model>/<account>_v<N>/` |
| `collect_results.py` | one `runs_kaggle/...` folder | `results/<scale>/<model>/` (no weights), then that scale's `comparison.md/.json` and `results/summary.csv` |
| `build_paper.py` | `results/` | `results/<scale>/pairwise.json`, `paper/generated/`, `paper/figures/`; `--zip` packs `paper/` for Overleaf |

`run_on_kaggle.py` needs a Kaggle API token in `~/.kaggle/kaggle.json` (or the folder
`KAGGLE_CONFIG_DIR` names). Kaggle allows two GPU sessions per account at once. A third push is
refused, and the script reports it and stops.

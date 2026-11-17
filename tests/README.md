# `tests/`

```bash
python -m pytest -m "not slow"   # fast: everything but the end-to-end runs
python -m pytest                 # all (~7 min on CPU)
```

Most tests run on a **synthetic miniature release** (`conftest.py`): every file name of the real
release, small photographs of mixed sizes, and its own manifest. Tests marked `data` use the real
release in `data/raw/NWRD/` and are skipped without it. Tests marked `slow` run the whole pipeline
on CPU. Keep pytest's temporary files off a small drive with `--basetemp=<dir>`.

| File | Covers |
|---|---|
| `test_config.py` | The protocol's settings and the two presets |
| `test_dataset.py` | The split, the authors' validation draw, the MD5 check, finding the release on Kaggle, reading photographs and masks |
| `test_windows.py` | Fitted-stride windows: every pixel covered without padding and owned once; resizing; rust counts; the paper's patch grid; the training sample |
| `test_models.py` | Output shapes at both window sizes, which encoders are frozen (including the ablation's), safe checkpoint loading |
| `test_metrics.py` | Pooled metrics: the histogram gives the same counts as thresholding every pixel |
| `test_evaluate.py` | Whole-photograph scoring, each pixel scored once; the paper's patches and the authors' averaging |
| `test_compare.py` | The photograph bootstrap, the decision rule (baselines only, never the ablation), the side-by-side report, the speed table |
| `test_run.py` | Run bookkeeping: refusing to mix settings across resumed stages |
| `test_pipeline.py` | End-to-end `cut → train → eval → summary` on CPU (`slow`) |
| `test_notebooks.py` | The notebooks embed the current code and manifest and differ only in `MODEL` |
| `test_kaggle_runner.py` | The Kaggle runner never pushes a resume that could restart silently, and never mixes kernels, scales, versions or accounts |
| `test_scripts.py` | Safe extraction and the manifest, collecting a kernel's output, the mask placement search, the patching script |
| `test_paper.py` | Every `\val{key}`, table, figure and citation in `paper/main.tex` exists; its intervals match `comparison.json` |
| `test_release.py` | The real release: its files, masks, and the window counts the docs quote (`data`) |

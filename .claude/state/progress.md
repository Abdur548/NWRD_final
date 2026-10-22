# Progress

**Status:** Phase 5 under way: the 512px Kaggle runs started 2026-10-23 on `abdur548` (the user's
go-ahead: "move ahead with the 512px runs for all 3 models"). **The 128px runs are on hold** until
the user says otherwise. The repository is `git init` only: nothing is committed (commit when the
user asks).
**Updated:** 2026-10-23
**Verified by:** the full suite, 135 passed (`python -m pytest -p no:cacheprovider
--basetemp=E:/NWRD/NWRD_public/build/pytest_tmp`, 7 min), before the push; the runner fix after it
(`tests/test_kaggle_runner.py`, 21 passed with `test_notebooks.py`); `ruff check` / `ruff format
--check` clean; notebooks rebuilt and current.

## Working

- The release is extracted to `data/raw/NWRD` and matches `data/manifest.json` (200 files) —
  `python -P -E scripts/fetch_data.py --zip E:/NWRD/datasets/nwrd_zenodo/NWRD.zip`.
- The dataset study, regenerated 2026-10-23 (only the stale `results/speed/` path changed; every
  number identical) — `results/dataset/summary.md`, `python -P -E scripts/describe_data.py`.
- Dry runs of the notebook's path on 2 photographs, 1 epoch (canet_b4 at both scales, unet and
  deeplabv3plus_r50 at 128px) ran end to end — `build/dryrun_*.log` (their scores mean nothing).
- The review's 18 findings are fixed (`build/review/findings.json`).
- `run_on_kaggle.push` now stops when Kaggle refuses a push (the CLI exits 0 on "Kernel push
  error"), instead of watching a kernel that never started.

## In progress

- **CANet-B4 512px**, kernel `abdur548/nwrd-public-canet-b4-512px` v1, pushed 2026-10-23 00:41
  (est. 3.0 h); watcher downloads to `runs_kaggle/512px/canet_b4/abdur548_v1/` when it stops.
- **DeepLabV3+ R50 512px**, `abdur548/nwrd-public-deeplabv3plus-r50-512px` v1, same time (est.
  2.0 h).
- **UNet 512px is queued**: an account runs at most 2 GPU sessions ("Maximum batch GPU session
  count of 2 reached"), so a background loop pushes `--model unet --watch` when either kernel above
  stops (est. 6.2 h after that). Its kernel already has a refused, cancelled version; the
  real one is the first the runner records in `build/kaggle_push/unet_512px/`.

## Next

1. When each 512px kernel finishes: `python scripts/collect_results.py
   runs_kaggle/512px/<model>/abdur548_v<N>`; an INCOMPLETE log line means `--resume --watch`.
2. After all three: `python -m nwrd_public.compare results` → `results/512px/comparison.md`.
3. The 128px kernels only when the user lifts the hold (two at a time).
4. `docs/results.md`, with the annotation-agreement ceiling beside the IoUs (`suggestions.md`).

## Open questions

- Commit the build before results come in (so they trace to a commit)? Needs: user.
- Qualitative prediction figures: `best.pt` stays in each kernel's output, so they can be made
  later from downloaded weights (`--status --with-checkpoints`) instead of a kernel stage. Needs:
  user.

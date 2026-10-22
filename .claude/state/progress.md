# Progress

**Status:** Phase 5 restarting. On 2026-10-23 photo 101's mask was found misplaced by 8 px, and
the user chose to remove 101 ("remove and restart"): validation is now 8 photographs (77/8/10).
The first 512px kernels (v1, which validated on 101) are discarded; the 512px runs are to be
pushed again once those are stopped. **The 128px runs are on hold** until the user says
otherwise. Committed: `eb813f2` (the build); the 101 change is to be committed with the restart.
**Updated:** 2026-10-23
**Verified by:** the fast tier after the change (126 passed, with the new placement tests);
`ruff check` / `ruff format --check` clean; notebooks rebuilt; the dataset study regenerated
(`results/dataset/`, new `mask_101.png`). The full suite is the last check before the push.

## Working

- The release in `data/raw/NWRD` matches `data/manifest.json` (200 files).
- The dataset study (`python -P -E scripts/describe_data.py`, ~6 min) now searches mask
  placements past the photograph's edges, for 101 and the six other cropped photographs: 101 fits
  at (y, x) = (−1, −8), contrast 21.07 vs 17.40 at its corner; the controls within 3 px of theirs.
- `dataset.splits`: validation `VAL_IDS` = `PAPER_VAL_IDS` without `EXCLUDED_MISALIGNED` (101);
  101 is in no split. `read_photo` refuses a mask of another size than its photograph.
- Validation: 563 windows (116 with rust) at 512px, 2,252 (155) at 128px; training unchanged
  (2,204 / 6,282 windows per epoch, `pos_weight` 63.1 / 43.9).
- `run_on_kaggle.push` stops when Kaggle refuses a push (the CLI exits 0 on "Kernel push error").

## In progress

- **Stopping the v1 512px kernels** (canet_b4, deeplabv3plus_r50; started 00:41): the user does
  it on their Kaggle pages, since the CLI has no cancel. UNet's v1 never started (refused push).
- Then: full suite, commit, push the three 512px kernels again (two at a time: an account runs
  at most 2 GPU sessions; the third waits for a free slot).

## Next

1. When each 512px kernel finishes: `python scripts/collect_results.py
   runs_kaggle/512px/<model>/abdur548_v<N>` (N is the version this push records in
   `build/kaggle_push/<model>_512px/`); an INCOMPLETE log line means `--resume --watch`.
2. After all three: `python -m nwrd_public.compare results` → `results/512px/comparison.md`.
3. The 128px kernels only when the user lifts the hold.
4. `docs/results.md`, with the annotation-agreement ceiling beside the IoUs (`suggestions.md`).

## Open questions

- Qualitative prediction figures: `best.pt` stays in each kernel's output, so they can be made
  later from downloaded weights (`--status --with-checkpoints`). Needs: user.

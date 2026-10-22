# Handoff — 2026-10-22 (session cut off by the usage limit)

**Status:** CONSUMED 2026-10-23 (tests passed, summary regenerated, 512px kernels pushed; see
`progress.md`)
**Reason:** usage limit

## Resume here

Run the full test suite: `python -m pytest -p no:cacheprovider --basetemp=E:/NWRD/NWRD_public/build/pytest_tmp`.
If it passes, push the three 512px Kaggle kernels. **The user has authorised the Kaggle runs**:
"go ahead with the kaggle runs once the review is done" (2026-10-22). The review is done and its
fixes are applied.

## State of the world

- The review workflow `review-nwrd-public` confirmed 18 findings (17 distinct; list in
  `build/review/findings.json`). All are fixed in code, docs and tests:
  - Pillow anti-aliased Lanczos in `windows.resize_photo`, with `PHOTO_RESIZE` in the cut's params.
  - `run.check_same_experiment`, the restore cell's model and preset check, and
    `collect_results`' per-run check.
  - Failed `cv2.imwrite` raises, and `windows.json` is written atomically.
  - `fetch_data --write-manifest` works only from a fresh extraction.
  - `results/speed/` was renamed to `results/nwrd_final/`, and `reference.json` was added.
  - The decision-rule text was clarified, and the Table 4 wording corrected.
- The notebooks were rebuilt after the last code edit. **Tests have NOT been re-run since the
  fixes** (new tests: `tests/test_run.py`, the side-by-side report in `test_compare.py`, and two
  in `test_windows.py`).
- `results/dataset/summary.md` still names `results/speed/nwrd_final_epoch_seconds.json` in its
  T4-estimate paragraph. Re-run `python -P -E scripts/describe_data.py` (3.5 min), which now also
  MD5-checks the release.
- `build/windows/` and `build/dryrun*` hold caches cut with the old OpenCV resize. They are safe
  to delete; the new params refuse them anyway.
- Nothing is committed (`git init` only); commit only when the user asks.

## Kaggle plan (D2: 512px first)

- Account `abdur548` (default `~/.kaggle`). `smautomateai` (`KAGGLE_CONFIG_DIR`) had GPU
  trouble before.
- Push all three in the background:
  `python scripts/run_on_kaggle.py --model {canet_b4,unet,deeplabv3plus_r50} --watch`.
- If one stops at its budget: `--resume`.
- When all three finish: `python scripts/collect_results.py runs_kaggle/512px/<model>/abdur548_v<N>`
  for each, then the same three with `--preset 128px`.
- Estimated time: 512px 3.0 / 6.2 / 2.0 h; 128px 2.6 / 1.5 / 1.0 h.
- Then write `docs/results.md`.

## Traps

- The Bash tool halves backslashes in heredocs: use Edit/Write for text containing `\`.
- C: has about 2 GB free. Keep pytest's basetemp, windows and dry runs on E:.

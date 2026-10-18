# Handoff

**Status:** ACTIVE (written 2026-10-18)

NWRD_final was built from `Plant_Segmentation` per `../NWRD_final_PLAN.md` (approved 2026-10-17),
verified (see progress.md), committed and pushed on 2026-10-18 to the public repository
`github.com/Abdur548/NWRD-Wheat-Rust` at the user's request.

What the next session needs that the files don't show:

- **In progress:** the user asked (2026-10-18) to run DeepLabV3+ at both scales and log the runs
  like the others. DeepLabV3+ R50 already has 3 seeds at 512px (`results/512px/final`); the new
  work is 128px. Done: `deeplabv3plus_r50` added to the `128px` preset; the bench stage now
  re-measures the whole table when a model is missing (`_benchmarked` in `nwrd/run.py`); kernel
  `abdur548/nwrd-128px-canet-vs-unet` **version 3** pushed with `--resume` (2026-10-18 ~09:35 UTC).
  Remaining, once it is COMPLETE (`python scripts/run_on_kaggle.py --experiment 128px --status`,
  output in `runs_kaggle/128px/abdur548_v3/`):
  1. Check `runs/log.txt` for CRASHED/Traceback and that `deeplabv3plus_r50/seed{42,43,44}` have
     `metrics.json`; note the logged library versions (first line of the new session).
  2. Copy the new `deeplabv3plus_r50/` runs, `benchmark.json` (re-measured), `log.txt`,
     `config.json`, `figures/qualitative_results.png` into `results/128px/final` (no `*.pt`,
     `TRAINED`, `export/`), then regenerate: `--stage compare`, `analyse_results.py`,
     `compare_scales.py` (extend it to list DeepLabV3+ in its tables).
  3. Update `docs/results.md` (§1, §3, §6.2), README results table, `docs/protocol.md` §1 table,
     `results/README.md`; note that the DLv3+ 128px runs used seeded augmentation; rerun tests;
     commit; ask the user before pushing to GitHub.
- **D1 was answered "keep"** and read as "keep both UIs". The user did not object; the dashboard
  is ported. If they say React-only, delete `app/dashboard.py`, `tests/test_dashboard.py` and
  streamlit from `pyproject.toml`/`constraints.txt`, and the dashboard lines in README/CLAUDE.md.
- **Open wording question:** `docs/protocol.md` calls `deeplabv3plus_r50` "the strongest earlier
  in-house model"; it is a standard re-implementation (the internship's "Custom DeepLabV3+" code
  was never preserved). The user was asked whether to reword it; no answer yet.

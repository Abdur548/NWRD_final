# Handoff

**Status:** CONSUMED (2026-10-18)

The DeepLabV3+ 128px work it described is done: Kaggle version 3 completed, its runs are in
`results/128px/final`, the reports and docs are regenerated, and each model's results are written
on their own (`<model>/summary.md`, `docs/results.md` §8). What remains is in progress.md
("Next"): pushing to GitHub waits for the user.

- **D1 was answered "keep"** and read as "keep both UIs". The user did not object; the dashboard
  is ported. If they say React-only, delete `app/dashboard.py`, `tests/test_dashboard.py` and
  streamlit from `pyproject.toml`/`constraints.txt`, and the dashboard lines in README/CLAUDE.md.

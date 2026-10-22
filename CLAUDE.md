# NWRD_public

CANet-B4 (ours) against the NWRD paper's UNet and DeepLabV3+ R50 on the **public NWRD release**
(Zenodo doi:10.5281/zenodo.21672928). It backs an official research paper and is reviewed by the
supervisor: keep it at professional standards (tested, linted, no dead code, accurate docs).
Plan: `../NWRD_public_PLAN.md` (approved 2026-10-22); build plan `../NWRD_public_BUILD_PLAN.md`.

## Running it

From this folder, with `python` (C:\Python313; torch CPU, pytest, ruff, kaggle):

- Tests: `python -m pytest -m "not slow"` (fast), `python -m pytest` (all, ~5 min CPU). The `data`
  tests need the release in `data/raw/NWRD` and are skipped without it.
- Lint: `python -m ruff check . && python -m ruff format --check .`
- After any `nwrd_public/` or `data/manifest.json` edit: `python scripts/build_notebooks.py`
  (a test fails on a stale notebook).
- The release: `python scripts/fetch_data.py` (or `--zip` an archive already downloaded).
  Large files go on E:, never C: (C: has about 2 GB free): pass `--window-dir` on E: for local
  runs; the default is under the temp folder on C:.
- Scripts that read the release run with `python -P -E`; they add the repository to `sys.path`
  themselves.

## Conventions

- No local GPU: training happens only on Kaggle (`scripts/run_on_kaggle.py`). Don't push to
  Kaggle, and don't commit or push to git, without the user's go-ahead.
- CANet-B4 is "ours" in every table. Report results honestly, even when it loses.
- **Never clone or open `github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD`**: it is compromised
  (`docs/dataset.md` §1). Data comes from Zenodo or `bcap52/nwrd-wheat`, MD5-checked.
- The split is fixed (`nwrd_public/dataset.py`): the paper's validation and test photographs,
  training without the twins 79, 86, 96, 99. The decision rule in `docs/protocol.md` §5 was
  fixed before any run; don't change it after results exist.
- Every number in `docs/` must trace to a file in `results/`.

<!-- pipeline-continuity:start -->
## Session continuity

State for this project lives in `.claude/state/`. Read it before doing anything else:

- `handoff.md` — the last session's notes, and the most specific thing here. Read it
  first. If its Status is ACTIVE, act on it, then mark it CONSUMED with today's date.
- `progress.md` — where the work stands and what comes next.
- `pipeline.md` — the architecture, the decisions behind it, and the invariants that
  must not break.
- `suggestions.md` — good ideas not yet built. Check before proposing new ones.

These describe the world as of their `Updated` date, not necessarily as it is now.
Check them against `git log` and `git status` before building on them. Where they
disagree with reality, reality wins — fix the file and say that you did.
<!-- pipeline-continuity:end -->
